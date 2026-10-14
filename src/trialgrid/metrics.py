"""Predeclared failure predicates over completed, recorded episodes."""

from dataclasses import dataclass

from trialgrid.controllers._planning import candidates
from trialgrid.domain import _finite
from trialgrid.runner import Episode


@dataclass(frozen=True, slots=True)
class FailureConfig:
    schema_version: int = 1
    economic_threshold_eur: float = 1.0

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported failure schema_version")
        _finite("economic_threshold_eur", self.economic_threshold_eur)
        if self.economic_threshold_eur <= 0:
            raise ValueError("economic_threshold_eur must be positive")


@dataclass(frozen=True, slots=True)
class FailureFinding:
    kind: str
    step_index: int | None
    severity: float
    classification: str


@dataclass(frozen=True, slots=True)
class FailureReport:
    findings: tuple[FailureFinding, ...]
    profit_gap_eur: float
    economic_eligible: bool
    tested_ending_energy_kwh: float
    comparator_ending_energy_kwh: float


def evaluate_episode(episode: Episode, comparator: Episode, config: FailureConfig) -> FailureReport:
    """Classify outcomes without giving either controller future information.

    Reserve witnesses establish only current-step preventability, not whether a
    past decision should have anticipated a hidden reserve. Economic dominance
    uses realized profit and requires a compliant comparator with no less ending
    energy; it does not claim optimal regret or assign inventory a guessed price.
    Callers supply completed runner/replay episodes and a predeclared comparator.
    """
    if episode.scenario != comparator.scenario:
        raise ValueError("comparison requires identical scenarios")
    findings = []
    battery = episode.scenario.config
    for trace in episode.trace:
        current = trace.observation.current
        if trace.result.violations:
            findings.append(
                FailureFinding("invalid_action", current.step_index, 1.0, "invalid_request")
            )
        if trace.reserve_shortfall_kwh:
            reserve = current.reserve_requirement_kwh
            if reserve > battery.energy_ceiling_kwh:
                classification = "physically_impossible"
            elif any(
                result.state.energy_kwh >= reserve
                for result in candidates(
                    battery, current.state, current.price_eur_per_mwh, reserve, (1.0,)
                )
            ):
                classification = "locally_preventable"
            else:
                classification = "undetermined"
            findings.append(
                FailureFinding(
                    "reserve_shortfall",
                    current.step_index,
                    trace.reserve_shortfall_kwh,
                    classification,
                )
            )
    tested_energy = episode.trace[-1].result.state.energy_kwh
    comparator_energy = comparator.trace[-1].result.state.energy_kwh
    gap = comparator.metrics.profit_eur - episode.metrics.profit_eur
    _finite("profit_gap_eur", gap)
    eligible = (
        comparator.metrics.rejected_action_count == 0
        and comparator.metrics.reserve_violation_count == 0
        and comparator_energy >= tested_energy
    )
    if eligible and gap >= config.economic_threshold_eur:
        findings.append(
            FailureFinding("economic_underperformance", None, gap, "feasible_comparator_dominance")
        )
    return FailureReport(tuple(findings), gap, eligible, tested_energy, comparator_energy)
