"""Targeted selection spends the same simulation budget and replays its lineage."""

import json
from dataclasses import replace
from math import inf, nan
from random import Random

import pytest

from trialgrid.metrics import FailureConfig, FailureFinding, FailureReport
from trialgrid.runner import HoldController
from trialgrid.search.domain import ScenarioDomain, sample_scenario
from trialgrid.search.evaluation import (
    ControllerSpec,
    SearchConfig,
    _next_generated_candidate,
    actionable_score,
    run_search,
    verify_search_result,
)


def run(method="targeted", *, budget=None, space=None, factory=HoldController, fixed=()):
    space = space or ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0)
    spec = ControllerSpec("hold", "1", 2, "{}", factory)
    return run_search(
        method,
        space,
        spec,
        spec,
        FailureConfig(),
        budget or SearchConfig(1000, 5, 7, random_exploration_probability=0),
        fixed_scenarios=fixed,
    )


def environmental_values(scenario):
    return (
        scenario.initial_state.energy_kwh,
        *scenario.prices_eur_per_mwh,
        *(value for row in scenario.forecast_prices_eur_per_mwh for value in row),
        *scenario.reserve_requirements_kwh,
    )


def test_initial_random_candidate_is_shared_and_paid():
    targeted, random = run(), run("random")
    assert targeted.records[0] == random.records[0]
    assert targeted.records[0].source == "random"
    assert targeted.records[0].parent_id is None
    assert targeted.records[0].simulator_calls == 4
    assert targeted.records[0].cumulative_calls == 4


def test_mutations_change_one_scalar_and_retain_known_parent():
    result = run()
    earlier = {}
    for record in result.records:
        if earlier:
            assert record.source == "mutation"
            parent = earlier[record.parent_id]
            changes = sum(
                a != b
                for a, b in zip(
                    environmental_values(record.scenario), environmental_values(parent), strict=True
                )
            )
            assert changes == 1
        earlier[record.scenario_id] = record.scenario
    assert all(record.report is not None for record in result.records)
    assert all(not record.report.findings for record in result.records)
    assert result.to_dict()["summary"]["actionable_cases"] == 0


def test_targeted_reproducibility_and_verified_serialization():
    result = run()
    assert result.to_json() == run().to_json()
    assert verify_search_result(result.to_json()) == result.to_dict()


@pytest.mark.parametrize("limit", [1, 3, 4, 5, 11, 12])
def test_all_methods_share_budget_including_interrupted_candidates_and_fixed_duplicates(limit):
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0)
    fixed = (sample_scenario(space, Random(7)),)
    budget = SearchConfig(limit, 100, 7, random_exploration_probability=0)
    results = [
        run(method, space=space, budget=budget, fixed=fixed if method == "fixed" else ())
        for method in ("targeted", "random", "fixed")
    ]
    expected_calls = [4] * (limit // 4) + ([limit % 4] if limit % 4 else [])
    for result in results:
        assert [record.simulator_calls for record in result.records] == expected_calls
        assert result.to_dict()["summary"]["used_simulator_calls"] == limit
        assert verify_search_result(result.to_json()) == result.to_dict()
        if limit % 4:
            assert result.records[-1].report is None
    assert results[-1].to_dict()["summary"]["unique_environments"] == 1


def test_full_exploration_has_no_mutation_lineage():
    result = run(budget=SearchConfig(1000, 8, 7, random_exploration_probability=1))
    assert all(record.source == "random" and record.parent_id is None for record in result.records)
    assert verify_search_result(result.to_json()) == result.to_dict()


def test_failed_candidates_are_saved_and_never_used_as_parents():
    def broken():
        raise RuntimeError("deliberate test failure")

    result = run(factory=broken)
    assert len(result.records) == 5
    assert all(record.report is None and record.source == "random" for record in result.records)
    assert all(record.simulator_calls == 0 for record in result.records)
    assert all(record.parent_id is None for record in result.records)
    assert verify_search_result(result.to_json()) == result.to_dict()


@pytest.mark.parametrize("corruption", ["seed", "parent", "exploration", "selected_scenario"])
def test_seed_selection_and_lineage_corruption_rejected(corruption):
    data = run().to_dict()
    if corruption == "seed":
        data["budget"]["seed"] += 1
    elif corruption == "parent":
        data["records"][1]["parent_id"] = "unknown-parent"
    elif corruption == "exploration":
        data["budget"]["random_exploration_probability"] = 1
    else:
        # Same environment but different episode seed must still fail generation replay.
        data["records"][1]["scenario"]["seed"] += 1
    with pytest.raises(ValueError, match="seeded generation/selection"):
        verify_search_result(json.dumps(data))


@pytest.mark.parametrize(
    "settings",
    [
        {"population_size": 0},
        {"population_size": 101},
        {"population_size": True},
        {"population_size": 1.0},
        {"random_exploration_probability": -0.1},
        {"random_exploration_probability": 1.1},
        {"random_exploration_probability": nan},
        {"random_exploration_probability": inf},
        {"random_exploration_probability": True},
    ],
)
def test_invalid_selection_configuration(settings):
    with pytest.raises(ValueError):
        SearchConfig(**settings)


def test_elite_selection_does_not_reward_impossible_unknown_or_fallback_reserves():
    original = run().records[0]
    classification = ("physically_impossible", "undetermined", "model_failure_fallback")
    ignored = tuple(FailureFinding("reserve_shortfall", 0, 1000, label) for label in classification)
    ignored_report = FailureReport(ignored, 0, False, 50, 50)
    assert actionable_score(ignored_report) == (0, 0, 0)
    useful_report = FailureReport(
        (FailureFinding("reserve_shortfall", 0, 1, "locally_preventable"),), 0, False, 50, 50
    )
    useless = replace(original, scenario_id="unactionable", report=ignored_report)
    useful = replace(original, scenario_id="actionable", report=useful_report)
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0)
    budget = SearchConfig(population_size=1, random_exploration_probability=0)
    for seed in range(10):
        _, source, parent = _next_generated_candidate(
            "targeted", space, budget, Random(seed), [useless, useful]
        )
        assert source == "mutation" and parent == "actionable"


def test_elite_population_excludes_lower_scores_and_incomplete_evaluations():
    original = run().records[0]

    def scored(count):
        findings = tuple(
            FailureFinding("invalid_action", index, 1, "invalid_request") for index in range(count)
        )
        return FailureReport(findings, 0, False, 50, 50)

    records = [
        replace(original, scenario_id=f"rank-{count}", report=scored(count)) for count in (1, 3, 2)
    ]
    records.append(replace(original, scenario_id="incomplete", report=None))
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0)
    budget = SearchConfig(population_size=2, random_exploration_probability=0)
    parents = {
        _next_generated_candidate("targeted", space, budget, Random(seed), records)[2]
        for seed in range(20)
    }
    assert parents == {"rank-3", "rank-2"}


def test_duplicate_targeted_proposals_are_evaluated_and_charged(monkeypatch):
    import trialgrid.search.evaluation as evaluation

    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_max_kwh=0)
    fixed = sample_scenario(space, Random(7))

    def repeated(method, domain, budget, rng, records):
        return fixed, "random", None

    monkeypatch.setattr(evaluation, "_next_generated_candidate", repeated)
    result = run(space=space, budget=SearchConfig(12, 100, 7))
    assert [record.simulator_calls for record in result.records] == [4, 4, 4]
    assert result.to_dict()["summary"]["unique_environments"] == 1
    assert result.to_dict()["summary"]["used_simulator_calls"] == 12


def test_interrupted_warmup_classification_has_no_free_retry():
    space = ScenarioDomain(hours=2, forecast_horizon=2, reserve_min_kwh=90, reserve_max_kwh=90)
    result = run(space=space, budget=SearchConfig(5, 100, 7))
    assert len(result.records) == 1
    record = result.records[0]
    assert record.source == "random" and record.parent_id is None
    assert record.target_episode is not None and record.comparator_episode is not None
    assert record.report is None
    assert record.evaluation_error == "simulator_budget_exhausted"
    assert record.simulator_calls == 5
    assert verify_search_result(result.to_json()) == result.to_dict()
