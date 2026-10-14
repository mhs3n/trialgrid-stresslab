"""Search preserves evidence and charges all simulation work to finite budgets."""

import json
from dataclasses import replace
from random import Random

import pytest

from trialgrid.controllers import BaselineConfig, ForecastOptimizer
from trialgrid.controllers.llm import ControllerCallFailure
from trialgrid.metrics import FailureConfig
from trialgrid.runner import HoldController
from trialgrid.search.domain import ScenarioDomain, sample_scenario
from trialgrid.search.evaluation import (
    ControllerSpec,
    SearchConfig,
    run_search,
    verify_search_result,
)


def spec(factory=HoldController, *, horizon=2):
    return ControllerSpec("test", "1", horizon, "{}", factory)


def domain(**kwargs):
    return ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0, **kwargs)


def run(*, method="random", space=None, target=None, comparator=None, budget=None, fixed=()):
    return run_search(
        method,
        space or domain(),
        target or spec(),
        comparator or spec(),
        FailureConfig(),
        budget or SearchConfig(1000, 3, 1),
        fixed_scenarios=fixed,
    )


def test_random_deterministic_fresh_controllers_and_negative_evidence():
    created = []

    class FreshHold:
        def __init__(self):
            self.calls = 0
            created.append(self)

        def act(self, observation, rng):
            assert self.calls == observation.current.step_index
            self.calls += 1
            return HoldController().act(observation, rng)

    result = run(target=spec(FreshHold), comparator=spec(FreshHold))
    assert len(created) == 6
    assert all(controller.calls == 2 for controller in created)
    assert len(result.records) == 3
    assert all(
        record.report is not None and not record.report.findings for record in result.records
    )
    assert result.to_json() == run().to_json()
    assert verify_search_result(result.to_json()) == result.to_dict()
    assert result.to_dict()["summary"]["stop_reason"] == "candidate_cap"


def test_fixed_duplicates_consume_budget_and_deduplicate_summary():
    space = domain()
    scenario = sample_scenario(space, Random(1))
    result = run(method="fixed", space=space, fixed=(scenario,), budget=SearchConfig(12, 10, 1))
    assert [record.simulator_calls for record in result.records] == [4, 4, 4]
    summary = result.to_dict()["summary"]
    assert summary["attempted_candidates"] == 3
    assert summary["unique_environments"] == 1
    assert summary["used_simulator_calls"] == 12
    assert summary["stop_reason"] == "simulator_budget"


@pytest.mark.parametrize("limit,expected", [(1, (1,)), (3, (3,)), (5, (4, 1))])
def test_exact_budget_keeps_interrupted_candidate(limit, expected):
    result = run(budget=SearchConfig(limit, 10, 1))
    assert tuple(record.simulator_calls for record in result.records) == expected
    assert result.records[-1].cumulative_calls == limit
    assert result.records[-1].report is None
    assert "simulator_budget_exhausted" in (
        result.records[-1].target_error,
        result.records[-1].comparator_error,
    )
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_optimizer_internal_planning_is_counted_and_bounded():
    optimizer = spec(lambda: ForecastOptimizer(BaselineConfig(horizon=2)))
    complete = run(target=optimizer, budget=SearchConfig(10000, 1, 1))
    assert complete.records[0].simulator_calls > 4
    limited = run(target=optimizer, budget=SearchConfig(3, 2, 1))
    assert limited.records[0].simulator_calls == 3
    assert limited.records[0].target_error == "simulator_budget_exhausted"
    assert limited.records[0].target_episode is None


def test_reserve_classification_witnesses_are_counted_and_can_interrupt():
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_min_kwh=90, reserve_max_kwh=90)
    complete = run(space=space, budget=SearchConfig(10000, 1, 1))
    assert complete.records[0].simulator_calls > 4
    limited = run(space=space, budget=SearchConfig(5, 2, 1))
    record = limited.records[0]
    assert record.target_episode is not None and record.comparator_episode is not None
    assert record.evaluation_error == "simulator_budget_exhausted"
    assert record.report is None and record.simulator_calls == 5
    assert verify_search_result(limited.to_json()) == limited.to_dict()


def test_error_categories_retained_provider_messages_excluded_and_zero_work_capped():
    def model_failure():
        raise ControllerCallFailure("private provider text")

    def generic_failure():
        raise RuntimeError("secret-key-do-not-save")

    result = run(
        target=spec(model_failure), comparator=spec(generic_failure), budget=SearchConfig(10, 3, 1)
    )
    assert len(result.records) == 3
    assert all(record.target_error == "model_call_failure" for record in result.records)
    assert all(
        record.comparator_error == "controller_error:RuntimeError" for record in result.records
    )
    assert all(record.simulator_calls == 0 and record.report is None for record in result.records)
    assert "private provider" not in result.to_json()
    assert "secret-key" not in result.to_json()
    assert result.to_dict()["summary"]["stop_reason"] == "candidate_cap"
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_horizon_mismatch_and_missing_fixed_suite_rejected():
    with pytest.raises(ValueError, match="horizon"):
        run(target=spec(horizon=1))
    with pytest.raises(ValueError, match="fixed"):
        run(method="fixed")
    space = domain()
    outside = sample_scenario(replace(space, hours=3), Random(1))
    with pytest.raises(ValueError):
        run(method="fixed", fixed=(outside,))


@pytest.mark.parametrize("corruption", ["summary", "episode", "fingerprint", "report", "calls"])
def test_artifact_corruption_is_rejected(corruption):
    data = run().to_dict()
    record = data["records"][0]
    if corruption == "summary":
        data["summary"]["completed_evaluations"] += 1
    elif corruption == "episode":
        record["target_episode"]["metrics"]["profit_eur"] += 1
    elif corruption == "fingerprint":
        record["scenario_id"] = "bad"
    elif corruption == "report":
        record["report"]["profit_gap_eur"] += 1
    else:
        record["cumulative_calls"] += 1
    with pytest.raises(ValueError):
        verify_search_result(json.dumps(data))


@pytest.mark.parametrize(
    "settings",
    [
        {"simulator_call_budget": 0},
        {"simulator_call_budget": True},
        {"max_candidates": 0},
        {"max_candidates": 1.0},
        {"seed": -1},
        {"seed": 2**64},
    ],
)
def test_invalid_budgets(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


@pytest.mark.parametrize(
    "corruption",
    [
        "method",
        "metadata",
        "horizon",
        "identity",
        "deleted_report",
        "boolean_index",
        "lower_calls",
    ],
)
def test_verified_artifact_rejects_structural_and_semantic_tampering(corruption):
    result = run(budget=SearchConfig(1000, 1, 1))
    data = result.to_dict()
    if corruption == "method":
        data["method"] = "invented"
    elif corruption == "metadata":
        data["target_metadata_json"] = "not json"
    elif corruption in ("horizon", "identity"):
        metadata = json.loads(data["target_metadata_json"])
        metadata["horizon" if corruption == "horizon" else "name"] = (
            1 if corruption == "horizon" else "different_controller"
        )
        data["target_metadata_json"] = json.dumps(metadata)
    elif corruption == "deleted_report":
        from trialgrid.search.evaluation import summarize

        record = replace(result.records[0], report=None)
        data["records"][0]["report"] = None
        data["summary"] = summarize((record,), result.budget)
    elif corruption == "boolean_index":
        data["records"][0]["index"] = True
    else:
        from trialgrid.search.evaluation import summarize

        record = replace(result.records[0], simulator_calls=1, cumulative_calls=1)
        data["records"][0]["simulator_calls"] = 1
        data["records"][0]["cumulative_calls"] = 1
        data["summary"] = summarize((record,), result.budget)
    with pytest.raises(ValueError):
        verify_search_result(json.dumps(data))


def test_controller_cannot_read_longer_current_or_historical_forecasts():
    observations = []

    class ReadsAllForecasts:
        def act(self, observation, rng):
            observations.append(observation)
            return HoldController().act(observation, rng)

    space = ScenarioDomain(hours=4, forecast_horizon=4, reserve_max_kwh=0)
    result = run(
        space=space,
        target=spec(ReadsAllForecasts, horizon=1),
        comparator=spec(horizon=1),
        budget=SearchConfig(100, 1, 1),
    )
    assert len(result.records[0].scenario.forecast_prices_eur_per_mwh[0]) == 4
    assert len(observations) == 4
    assert all(
        len(observation.current.forecast_prices_eur_per_mwh) == 1 for observation in observations
    )
    assert all(
        len(frame.forecast_prices_eur_per_mwh) == 1
        for observation in observations
        for frame in observation.history
    )


def test_exhausted_budget_prevents_comparator_factory():
    def forbidden_factory():
        pytest.fail("comparator factory called after exhausted budget")

    result = run(comparator=spec(forbidden_factory), budget=SearchConfig(2, 1, 1))
    assert result.records[0].target_episode is not None
    assert result.records[0].comparator_error == "simulator_budget_exhausted"


def test_exhausted_budget_prevents_next_model_request():
    from trialgrid.controllers import LLMConfig, LLMController
    from trialgrid.controllers.llm_models import FakeModel

    model = FakeModel()
    target = spec(lambda: LLMController(model, LLMConfig("fake", horizon=2)))
    result = run(target=target, budget=SearchConfig(1, 1, 1))
    assert len(model.requests) == 1
    assert result.records[0].target_error == "simulator_budget_exhausted"
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_model_failure_fallback_is_separate_from_model_chosen_hold():
    from trialgrid.controllers import LLMConfig, LLMController
    from trialgrid.controllers.llm_models import FakeModel
    from trialgrid.controllers.llm_types import ModelCallError
    from trialgrid.domain import BatteryState
    from trialgrid.search.evaluation import actionable_score

    class FailedModel:
        def complete(self, request_json):
            raise ModelCallError("api_error", retryable=False)

    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_min_kwh=60, reserve_max_kwh=60)
    scenario = replace(sample_scenario(space, Random(1)), initial_state=BatteryState(50))
    config = LLMConfig("fake", horizon=2, max_attempts=1, failure_policy="hold")
    failed = run(
        method="fixed",
        space=space,
        fixed=(scenario,),
        target=spec(lambda: LLMController(FailedModel(), config)),
        budget=SearchConfig(1000, 1, 1),
    )
    report = failed.records[0].report
    assert {finding.classification for finding in report.findings} == {"model_failure_fallback"}
    assert actionable_score(report) == (0, 0.0, 0.0)
    summary = failed.to_dict()["summary"]
    assert summary["model_fallback_holds"] == summary["model_failed_decisions"] == 2
    assert summary["actionable_cases"] == 0
    assert verify_search_result(failed.to_json()) == failed.to_dict()

    chosen = run(
        method="fixed",
        space=space,
        fixed=(scenario,),
        target=spec(lambda: LLMController(FakeModel(), config)),
        budget=SearchConfig(1000, 1, 1),
    )
    assert {finding.classification for finding in chosen.records[0].report.findings} == {
        "locally_preventable"
    }
    assert actionable_score(chosen.records[0].report)[0] == 2
    assert chosen.to_dict()["summary"]["model_fallback_holds"] == 0
    assert verify_search_result(chosen.to_json()) == chosen.to_dict()


def test_failed_final_model_dispatch_cannot_back_completed_episode():
    from trialgrid.controllers.llm import LLMController
    from trialgrid.controllers.llm_models import FakeModel
    from trialgrid.controllers.llm_types import LLMConfig

    settings = LLMConfig("fake", max_attempts=1, horizon=2)
    model_spec = ControllerSpec("fake", "1", 2, "{}", lambda: LLMController(FakeModel(), settings))
    result = run(target=model_spec, budget=SearchConfig(1000, 1, 1))
    record = result.records[0]
    audit = json.loads(record.target_llm_audit_json)
    failed = audit["decisions"][-1]
    failed.update(outcome="call_failure", model_requested_action=None, returned_action=None)
    attempt = failed["attempts"][-1]
    attempt.update(
        response_json=None,
        usage=None,
        response_model_id=None,
        transport_error={"code": "api_status_401", "retryable": False},
    )
    altered = replace(result, records=(replace(record, target_llm_audit_json=json.dumps(audit)),))
    with pytest.raises(ValueError, match="failed model dispatch"):
        verify_search_result(altered.to_json())
