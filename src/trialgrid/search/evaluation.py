"""Budgeted offline evaluation; save successful, failed, duplicate and partial candidates."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields, replace
from random import Random

from trialgrid.controllers.llm import ControllerCallFailure, LLMController, canonical
from trialgrid.controllers.llm_audit import RecordedModel
from trialgrid.controllers.llm_types import LLMConfig, TokenPricing
from trialgrid.domain import Action, StepInput, _finite
from trialgrid.metrics import FailureConfig, FailureReport, evaluate_episode
from trialgrid.runner import (
    Controller,
    Episode,
    Observation,
    ObservationFrame,
    _matches_json,
    run_episode,
)
from trialgrid.scenario import Scenario, load_json, require_keys
from trialgrid.search.domain import (
    ScenarioDomain,
    mutate_scenario,
    sample_scenario,
    scenario_fingerprint,
    validate_scenario,
)
from trialgrid.simulation_budget import (
    SimulationBudgetExceeded,
    SimulationCounter,
    count_simulator_calls,
    remaining_simulator_calls,
)
from trialgrid.simulator import step


@dataclass(frozen=True, slots=True)
class ControllerSpec:
    name: str
    version: str
    horizon: int
    config_json: str
    factory: Callable[[], Controller]

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip() for value in (self.name, self.version)
        ):
            raise ValueError("controller name/version must be nonempty")
        if type(self.horizon) is not int or not 1 <= self.horizon <= 6:
            raise ValueError("controller horizon must be in [1, 6]")
        canonical(load_json(self.config_json))
        if not callable(self.factory):
            raise ValueError("controller factory must be callable")

    def metadata_json(self) -> str:
        return canonical(
            {
                "name": self.name,
                "version": self.version,
                "horizon": self.horizon,
                "configuration": load_json(self.config_json),
                "information_mode": "published_forecast_only",
            }
        )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulator_call_budget: int = 20_000
    max_candidates: int = 100
    seed: int = 42
    population_size: int = 4
    random_exploration_probability: float = 0.25

    def __post_init__(self) -> None:
        for name, low, high in (
            ("simulator_call_budget", 1, 10_000_000),
            ("max_candidates", 1, 100_000),
            ("seed", 0, 2**64 - 1),
            ("population_size", 1, 100),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        _finite("random_exploration_probability", self.random_exploration_probability)
        if not 0 <= self.random_exploration_probability <= 1:
            raise ValueError("random_exploration_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    index: int
    source: str
    scenario: Scenario
    scenario_id: str
    parent_id: str | None
    simulator_calls: int
    cumulative_calls: int
    target_episode: Episode | None
    comparator_episode: Episode | None
    target_error: str | None
    comparator_error: str | None
    evaluation_error: str | None
    target_llm_audit_json: str | None
    comparator_llm_audit_json: str | None
    report: FailureReport | None


def actionable_score(report: FailureReport | None) -> tuple[int, float, float]:
    """Predeclared lexicographic score; impossible/undetermined reserves earn no reward."""
    if report is None:
        return (0, 0.0, 0.0)
    actionable = [
        finding
        for finding in report.findings
        if finding.kind != "reserve_shortfall" or finding.classification == "locally_preventable"
    ]
    return (
        len(actionable),
        sum(f.severity for f in actionable if f.kind == "reserve_shortfall"),
        sum(f.severity for f in actionable if f.kind == "economic_underperformance"),
    )


def _bounded_observation(observation: Observation, horizon: int) -> Observation:
    def cap(frame: ObservationFrame) -> ObservationFrame:
        return replace(
            frame, forecast_prices_eur_per_mwh=frame.forecast_prices_eur_per_mwh[:horizon]
        )

    return replace(
        observation,
        current=cap(observation.current),
        history=tuple(cap(frame) for frame in observation.history),
    )


class _BoundedController:
    def __init__(self, controller: Controller, horizon: int) -> None:
        self.controller, self.horizon = controller, horizon

    def act(self, observation: Observation, rng: Random) -> Action:
        if remaining_simulator_calls() == 0:
            raise SimulationBudgetExceeded("simulator-call budget exhausted")
        return self.controller.act(_bounded_observation(observation, self.horizon), rng)


def _model_attribution(
    report: FailureReport, target_audit: str | None, comparator_audit: str | None
) -> FailureReport:
    target = load_json(target_audit)["decisions"] if target_audit else []
    comparator = load_json(comparator_audit)["decisions"] if comparator_audit else []
    fallback_steps = {d["step_index"] for d in target if d["outcome"] == "fallback_hold"}
    any_fallback = fallback_steps or any(d["outcome"] == "fallback_hold" for d in comparator)
    findings = tuple(
        replace(f, classification="model_failure_fallback")
        if f.kind == "reserve_shortfall" and f.step_index in fallback_steps
        else f
        for f in report.findings
        if not (f.kind == "economic_underperformance" and any_fallback)
    )
    return replace(
        report, findings=findings, economic_eligible=report.economic_eligible and not any_fallback
    )


def _run_spec(
    scenario: Scenario, spec: ControllerSpec
) -> tuple[Episode | None, str | None, str | None]:
    controller = None
    episode = None
    error = None
    try:
        controller = spec.factory()
        episode = run_episode(
            scenario,
            _BoundedController(controller, spec.horizon),
            controller_name=spec.name,
            controller_version=spec.version,
        )
    except SimulationBudgetExceeded:
        error = "simulator_budget_exhausted"
    except ControllerCallFailure:
        error = "model_call_failure"
    except Exception as exc:
        # Save the failure category without copying arbitrary provider/credential text.
        error = f"controller_error:{type(exc).__name__}"
    audit = None
    if isinstance(controller, LLMController):
        audit = canonical(
            {
                "config": asdict(controller.config),
                "decisions": [asdict(record) for record in controller.records],
                "interruption": error,
            }
        )
    return episode, error, audit


def _evaluate(
    scenario: Scenario,
    source: str,
    parent_id: str | None,
    index: int,
    target: ControllerSpec,
    comparator: ControllerSpec,
    criteria: FailureConfig,
    counter: SimulationCounter,
) -> EvaluationRecord:
    before = counter.used
    tested, tested_error, tested_audit = _run_spec(scenario, target)
    baseline = None
    baseline_error = baseline_audit = evaluation_error = None
    report = None
    if counter.used >= counter.limit:
        baseline_error = "simulator_budget_exhausted"
    elif tested_error != "simulator_budget_exhausted":
        baseline, baseline_error, baseline_audit = _run_spec(scenario, comparator)
    if tested is not None and baseline is not None:
        try:
            report = _model_attribution(
                evaluate_episode(tested, baseline, criteria), tested_audit, baseline_audit
            )
        except SimulationBudgetExceeded:
            evaluation_error = "simulator_budget_exhausted"
        except Exception as exc:
            evaluation_error = f"evaluation_error:{type(exc).__name__}"
    return EvaluationRecord(
        index,
        source,
        scenario,
        scenario_fingerprint(scenario),
        parent_id,
        counter.used - before,
        counter.used,
        tested,
        baseline,
        tested_error,
        baseline_error,
        evaluation_error,
        tested_audit,
        baseline_audit,
        report,
    )


def summarize(records: tuple[EvaluationRecord, ...], budget: SearchConfig) -> dict:
    complete = [record for record in records if record.report is not None]
    failures = [record for record in complete if record.report.findings]
    actionable = [record for record in complete if actionable_score(record.report)[0]]
    first = actionable[0] if actionable else None
    used = records[-1].cumulative_calls if records else 0
    decisions = [
        decision
        for record in records
        for audit in (record.target_llm_audit_json, record.comparator_llm_audit_json)
        if audit
        for decision in load_json(audit)["decisions"]
    ]
    return {
        "model_fallback_holds": sum(d["outcome"] == "fallback_hold" for d in decisions),
        "model_failed_decisions": sum(
            d["outcome"] in ("fallback_hold", "call_failure") for d in decisions
        ),
        "model_execution_errors": sum(
            error == "model_call_failure"
            for record in records
            for error in (record.target_error, record.comparator_error)
        ),
        "allocated_simulator_calls": budget.simulator_call_budget,
        "used_simulator_calls": used,
        "attempted_candidates": len(records),
        "completed_evaluations": len(complete),
        "incomplete_or_failed_evaluations": len(records) - len(complete),
        "unique_environments": len({record.scenario_id for record in records}),
        "failure_cases": len(failures),
        "unique_failure_cases": len({record.scenario_id for record in failures}),
        "actionable_cases": len(actionable),
        "unique_actionable_cases": len({record.scenario_id for record in actionable}),
        "worst_actionable_score": list(
            max((actionable_score(record.report) for record in complete), default=(0, 0.0, 0.0))
        ),
        "reserve_classifications": {
            label: sum(
                f.kind == "reserve_shortfall" and f.classification == label
                for record in complete
                for f in record.report.findings
            )
            for label in (
                "locally_preventable",
                "physically_impossible",
                "undetermined",
                "model_failure_fallback",
            )
        },
        "actionable_failure_types": sorted(
            {
                f.kind
                for record in complete
                for f in record.report.findings
                if f.kind != "reserve_shortfall" or f.classification == "locally_preventable"
            }
        ),
        "failure_types": sorted(
            {finding.kind for record in failures for finding in record.report.findings}
        ),
        "first_actionable_evaluation": first.index if first else None,
        "calls_at_first_actionable_evaluation": first.cumulative_calls if first else None,
        "stop_reason": "simulator_budget"
        if used == budget.simulator_call_budget
        else "candidate_cap",
        "deduplication": "exact environment, excluding episode seed; no near-duplicate claim",
    }


@dataclass(frozen=True, slots=True)
class SearchResult:
    method: str
    domain: ScenarioDomain
    criteria: FailureConfig
    budget: SearchConfig
    target_metadata_json: str
    comparator_metadata_json: str
    records: tuple[EvaluationRecord, ...]
    schema_version: int = 1

    def to_dict(self) -> dict:
        data = asdict(self)
        data["summary"] = summarize(self.records, self.budget)
        return json.loads(canonical(data))

    def to_json(self) -> str:
        return canonical(self.to_dict())


def _next_generated_candidate(
    method: str,
    domain: ScenarioDomain,
    budget: SearchConfig,
    rng: Random,
    records: list[EvaluationRecord],
) -> tuple[Scenario, str, str | None]:
    parents = [record for record in records if record.report is not None]
    if method != "targeted" or not parents or rng.random() < budget.random_exploration_probability:
        return sample_scenario(domain, rng), "random", None
    elites = sorted(parents, key=lambda record: actionable_score(record.report), reverse=True)
    parent = rng.choice(elites[: budget.population_size])
    return mutate_scenario(parent.scenario, domain, rng), "mutation", parent.scenario_id


def run_search(
    method: str,
    domain: ScenarioDomain,
    target: ControllerSpec,
    comparator: ControllerSpec,
    criteria: FailureConfig,
    budget: SearchConfig,
    *,
    fixed_scenarios: tuple[Scenario, ...] = (),
) -> SearchResult:
    if method not in ("fixed", "random", "targeted"):
        raise ValueError("method must be fixed, random or targeted")
    if target.horizon != comparator.horizon:
        raise ValueError("target and comparator must declare the same information horizon")
    if method == "fixed" and not fixed_scenarios:
        raise ValueError("fixed search requires scenarios")
    for scenario in fixed_scenarios:
        validate_scenario(scenario, domain)
    rng = Random(budget.seed)
    records = []
    with count_simulator_calls(budget.simulator_call_budget) as counter:
        while counter.used < counter.limit and len(records) < budget.max_candidates:
            if method == "fixed":
                scenario = fixed_scenarios[len(records) % len(fixed_scenarios)]
                source, parent_id = "fixed", None
            else:
                scenario, source, parent_id = _next_generated_candidate(
                    method, domain, budget, rng, records
                )
            records.append(
                _evaluate(
                    scenario,
                    source,
                    parent_id,
                    len(records) + 1,
                    target,
                    comparator,
                    criteria,
                    counter,
                )
            )
    return SearchResult(
        method,
        domain,
        criteria,
        budget,
        target.metadata_json(),
        comparator.metadata_json(),
        tuple(records),
    )


def _verify_model_audit(
    text: str | None,
    scenario: Scenario,
    episode: Episode | None,
    error: str | None,
    horizon: int,
) -> None:
    if text is None:
        return
    audit = require_keys(load_json(text), {"config", "decisions", "interruption"})
    config_data = dict(require_keys(audit["config"], {f.name for f in fields(LLMConfig)}))
    if config_data["pricing"] is not None:
        config_data["pricing"] = TokenPricing(
            **require_keys(config_data["pricing"], {f.name for f in fields(TokenPricing)})
        )
    config = LLMConfig(**config_data)
    decisions = audit["decisions"]
    if not isinstance(decisions, list) or len(decisions) > len(scenario.prices_eur_per_mwh):
        raise ValueError("invalid model decision count")
    attempts = []
    for decision in decisions:
        if not isinstance(decision, dict) or not isinstance(decision.get("attempts"), list):
            raise ValueError("invalid model decision record")
        attempts.extend(decision["attempts"])
    if any(not isinstance(attempt, dict) for attempt in attempts):
        raise ValueError("invalid model attempt record")
    model = RecordedModel(attempts)
    controller = LLMController(model, config, sleep_fn=lambda _: None)
    history = []
    state = scenario.initial_state
    rng = Random(scenario.seed)
    terminal_failure = False
    for index in range(len(decisions)):
        frame = ObservationFrame(
            index,
            state,
            scenario.prices_eur_per_mwh[index],
            scenario.forecast_prices_eur_per_mwh[index],
            scenario.reserve_requirements_kwh[index],
        )
        observation = Observation(scenario.config, frame, tuple(history))
        try:
            action = controller.act(_bounded_observation(observation, horizon), rng)
        except ControllerCallFailure:
            terminal_failure = True
            break
        if episode is not None and action != episode.trace[index].result.requested_action:
            raise ValueError("model audit action differs from episode")
        state = step(scenario.config, state, action, StepInput(frame.price_eur_per_mwh)).state
        history.append(frame)
    model.assert_consumed()
    if terminal_failure and (episode is not None or error != "model_call_failure"):
        raise ValueError("failed model dispatch cannot accompany a completed episode")
    if error == "model_call_failure" and not terminal_failure:
        raise ValueError("model-call failure requires a failed recorded decision")
    expected = {
        "config": asdict(config),
        "decisions": [asdict(d) for d in controller.records],
        "interruption": error,
    }
    if not _matches_json(audit, json.loads(canonical(expected))):
        raise ValueError("model audit does not match response replay")
    if episode is not None and len(decisions) != len(episode.trace):
        raise ValueError("completed episode requires every model decision")


def verify_search_result(text: str) -> dict:
    """Replay episodes/responses and verify findings, metadata and accounting.

    Internal planning counts and exception provenance remain reported measurements;
    no cryptographic authenticity or controller-source verification is claimed.
    """
    data = require_keys(load_json(text), {f.name for f in fields(SearchResult)} | {"summary"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported search schema_version")
    if data["method"] not in ("fixed", "random", "targeted"):
        raise ValueError("unsupported search method")
    domain = ScenarioDomain.from_dict(data["domain"])
    criteria = FailureConfig(
        **require_keys(data["criteria"], {f.name for f in fields(FailureConfig)})
    )
    budget = SearchConfig(**require_keys(data["budget"], {f.name for f in fields(SearchConfig)}))
    metadata = []
    for key in ("target_metadata_json", "comparator_metadata_json"):
        value = require_keys(
            load_json(data[key]),
            {"name", "version", "horizon", "configuration", "information_mode"},
        )
        ControllerSpec(
            value["name"],
            value["version"],
            value["horizon"],
            canonical(value["configuration"]),
            lambda: None,
        )
        if value["information_mode"] != "published_forecast_only":
            raise ValueError("unsupported information mode")
        metadata.append(value)
    if metadata[0]["horizon"] != metadata[1]["horizon"]:
        raise ValueError("controller information horizons differ")
    if not isinstance(data["records"], list) or not data["records"]:
        raise ValueError("search records must be a nonempty array")
    records = []
    cumulative = 0
    replay_rng = Random(budget.seed)
    for index, value in enumerate(data["records"], 1):
        item = dict(require_keys(value, {f.name for f in fields(EvaluationRecord)}))
        scenario = Scenario.from_dict(item["scenario"])
        validate_scenario(scenario, domain)
        calls = item["simulator_calls"]
        if type(calls) is not int or calls < 0:
            raise ValueError("invalid simulator call count")
        cumulative += calls
        if (
            type(item["index"]) is not int
            or type(item["cumulative_calls"]) is not int
            or item["index"] != index
            or item["cumulative_calls"] != cumulative
        ):
            raise ValueError("inconsistent evaluation index/call accounting")
        if cumulative > budget.simulator_call_budget or index > budget.max_candidates:
            raise ValueError("search exceeds declared budget")
        if item["scenario_id"] != scenario_fingerprint(scenario):
            raise ValueError("environment fingerprint mismatch")
        if data["method"] == "fixed":
            if item["source"] != "fixed" or item["parent_id"] is not None:
                raise ValueError("invalid fixed candidate source/parent")
        else:
            expected_scenario, expected_source, expected_parent = _next_generated_candidate(
                data["method"], domain, budget, replay_rng, records
            )
            if (
                scenario != expected_scenario
                or item["source"] != expected_source
                or item["parent_id"] != expected_parent
            ):
                raise ValueError("candidate does not match seeded generation/selection")
        tested = (
            Episode.from_dict(item["target_episode"])
            if item["target_episode"] is not None
            else None
        )
        baseline = (
            Episode.from_dict(item["comparator_episode"])
            if item["comparator_episode"] is not None
            else None
        )
        for prefix, episode, meta in zip(
            ("target", "comparator"), (tested, baseline), metadata, strict=True
        ):
            error = item[f"{prefix}_error"]
            if error is not None and (
                not isinstance(error, str)
                or not (
                    error in ("simulator_budget_exhausted", "model_call_failure")
                    or error.startswith("controller_error:")
                )
            ):
                raise ValueError("invalid controller error category")
            if episode is not None:
                if episode.scenario != scenario or error is not None:
                    raise ValueError("episode scenario/error mismatch")
                if (
                    episode.controller_name != meta["name"]
                    or episode.controller_version != meta["version"]
                ):
                    raise ValueError("episode controller identity mismatch")
            elif error is None:
                raise ValueError("missing episode requires an error")
            _verify_model_audit(
                item[f"{prefix}_llm_audit_json"], scenario, episode, error, meta["horizon"]
            )
        if calls < sum(len(episode.trace) for episode in (tested, baseline) if episode is not None):
            raise ValueError("call count cannot be below completed episode steps")
        evaluation_error = item["evaluation_error"]
        if evaluation_error is not None and (
            not isinstance(evaluation_error, str)
            or not (
                evaluation_error == "simulator_budget_exhausted"
                or evaluation_error.startswith("evaluation_error:")
            )
        ):
            raise ValueError("invalid evaluation error category")
        errors = (item["target_error"], item["comparator_error"], evaluation_error)
        if "simulator_budget_exhausted" in errors and (
            cumulative != budget.simulator_call_budget or index != len(data["records"])
        ):
            raise ValueError("budget interruption must end search at its limit")
        report = None
        if tested is not None and baseline is not None and evaluation_error is None:
            report = _model_attribution(
                evaluate_episode(tested, baseline, criteria),
                item["target_llm_audit_json"],
                item["comparator_llm_audit_json"],
            )
        expected_report = json.loads(canonical(asdict(report))) if report is not None else None
        if not _matches_json(item["report"], expected_report):
            raise ValueError("failure report does not match episode evidence")
        item.update(
            scenario=scenario, target_episode=tested, comparator_episode=baseline, report=report
        )
        records.append(EvaluationRecord(**item))
    if cumulative < budget.simulator_call_budget and len(records) != budget.max_candidates:
        raise ValueError("search stopped before either declared limit")
    if not _matches_json(data["summary"], summarize(tuple(records), budget)):
        raise ValueError("search summary does not match records")
    return data
