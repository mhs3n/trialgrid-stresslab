"""Deterministic shrinking of replayable controller failures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from math import isfinite

from trialgrid.metrics import FailureConfig, FailureReport, evaluate_episode
from trialgrid.runner import run_episode
from trialgrid.scenario import Scenario
from trialgrid.search.evaluation import ControllerSpec, EvaluationRecord, _BoundedController

FailureSignature = tuple[tuple[str, str], ...]
ReportEvaluator = Callable[[Scenario], FailureReport]


def actionable_signature(report: FailureReport) -> FailureSignature:
    """Return the sorted actionable failure kinds and classifications."""
    return tuple(
        sorted(
            (finding.kind, finding.classification)
            for finding in report.findings
            if finding.kind != "reserve_shortfall"
            or finding.classification == "locally_preventable"
        )
    )


@dataclass(frozen=True, slots=True)
class ShrinkConfig:
    """Bounded, reproducible shrinking settings."""

    max_attempts: int = 500
    repetitions: int = 1

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100_000:
            raise ValueError("max_attempts must be an integer in [1, 100000]")
        if type(self.repetitions) is not int or not 1 <= self.repetitions <= 20:
            raise ValueError("repetitions must be an integer in [1, 20]")


@dataclass(frozen=True, slots=True)
class ShrinkResult:
    original_scenario: Scenario
    minimized_scenario: Scenario
    original_report: FailureReport
    minimized_report: FailureReport
    signature: FailureSignature
    attempts: int
    repetitions: int
    complete: bool

    @property
    def changed(self) -> bool:
        return self.original_scenario != self.minimized_scenario

    def to_dict(self) -> dict:
        return {
            "original_scenario": self.original_scenario.to_dict(),
            "minimized_scenario": self.minimized_scenario.to_dict(),
            "original_report": asdict(self.original_report),
            "minimized_report": asdict(self.minimized_report),
            "signature": [list(item) for item in self.signature],
            "attempts": self.attempts,
            "repetitions": self.repetitions,
            "complete": self.complete,
        }


def _toward(value: float, target: float) -> Iterator[float]:
    """Yield a midpoint and the simplest endpoint in stable order."""
    if value == target:
        return
    if target != value:
        yield target
    midpoint = (value + target) / 2
    if isfinite(midpoint) and midpoint != value and midpoint != target:
        yield midpoint


def _candidates(scenario: Scenario) -> Iterator[Scenario]:
    """Yield candidates in a stable coordinate order without changing shape."""
    config = scenario.config
    for candidate in _toward(scenario.initial_state.energy_kwh, config.energy_floor_kwh):
        yield replace(scenario, initial_state=replace(scenario.initial_state, energy_kwh=candidate))
    for index, value in enumerate(scenario.prices_eur_per_mwh):
        for candidate in _toward(value, 0.0):
            prices = list(scenario.prices_eur_per_mwh)
            prices[index] = candidate
            yield replace(scenario, prices_eur_per_mwh=tuple(prices))
    for row_index, row in enumerate(scenario.forecast_prices_eur_per_mwh):
        for offset, value in enumerate(row):
            for candidate in _toward(value, 0.0):
                forecasts = [list(item) for item in scenario.forecast_prices_eur_per_mwh]
                forecasts[row_index][offset] = candidate
                yield replace(
                    scenario,
                    forecast_prices_eur_per_mwh=tuple(tuple(item) for item in forecasts),
                )
    for index, value in enumerate(scenario.reserve_requirements_kwh):
        for candidate in _toward(value, 0.0):
            reserves = list(scenario.reserve_requirements_kwh)
            reserves[index] = candidate
            yield replace(scenario, reserve_requirements_kwh=tuple(reserves))


def shrink_scenario(
    scenario: Scenario,
    evaluate: ReportEvaluator,
    *,
    config: ShrinkConfig | None = None,
) -> ShrinkResult:
    """Greedily shrink a scenario while preserving its actionable signature.

    The evaluator must execute the target/comparator and return a fresh report.
    Candidates retain the original seed and timeline shape. ``complete`` is
    false when the attempt budget ends before a full pass reaches a fixpoint.
    """
    config = config or ShrinkConfig()
    original = evaluate(scenario)
    signature = actionable_signature(original)
    if not signature:
        raise ValueError("scenario has no actionable findings to shrink")
    current = scenario
    current_report = original
    attempts = 0
    complete = True
    while True:
        changed = False
        for candidate in _candidates(current):
            if attempts >= config.max_attempts:
                complete = False
                break
            attempts += 1
            try:
                report = evaluate(candidate)
            except (ValueError, TypeError):
                continue
            if actionable_signature(report) == signature:
                current, current_report, changed = candidate, report, True
                break
        if not complete or not changed:
            break
    return ShrinkResult(
        scenario,
        current,
        original,
        current_report,
        signature,
        attempts,
        config.repetitions,
        complete,
    )


def shrink_failure(
    record: EvaluationRecord,
    target: ControllerSpec,
    comparator: ControllerSpec,
    criteria: FailureConfig,
    *,
    config: ShrinkConfig | None = None,
) -> ShrinkResult:
    """Shrink a completed M5 record by replaying its declared controllers."""
    config = config or ShrinkConfig()
    if record.report is None or record.target_episode is None or record.comparator_episode is None:
        raise ValueError("cannot shrink an incomplete evaluation record")
    if (
        record.target_episode.controller_name != target.name
        or record.target_episode.controller_version != target.version
        or record.comparator_episode.controller_name != comparator.name
        or record.comparator_episode.controller_version != comparator.version
    ):
        raise ValueError("controller specification does not match evaluation record")
    if record.report != evaluate_episode(
        record.target_episode, record.comparator_episode, criteria
    ):
        raise ValueError("evaluation record report is inconsistent")

    def evaluate(scenario: Scenario) -> FailureReport:
        reports = []
        for _ in range(config.repetitions):
            tested = run_episode(
                scenario,
                _BoundedController(target.factory(), target.horizon),
                controller_name=target.name,
                controller_version=target.version,
            )
            baseline = run_episode(
                scenario,
                _BoundedController(comparator.factory(), comparator.horizon),
                controller_name=comparator.name,
                controller_version=comparator.version,
            )
            reports.append(evaluate_episode(tested, baseline, criteria))
        signature = actionable_signature(reports[0])
        if not signature or any(
            actionable_signature(report) != signature for report in reports[1:]
        ):
            return FailureReport((), 0.0, False, 0.0, 0.0)
        return reports[0]

    return shrink_scenario(record.scenario, evaluate, config=config)

