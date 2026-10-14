"""Independent failure examples and conservative economic eligibility."""

from dataclasses import FrozenInstanceError, replace
from math import inf, nan

import pytest

from trialgrid.domain import Action, BatteryConfig, BatteryState
from trialgrid.metrics import FailureConfig, evaluate_episode
from trialgrid.runner import run_episode
from trialgrid.scenario import Scenario


class Script:
    def __init__(self, powers):
        self.powers = iter(powers)

    def act(self, observation, rng):
        return Action(next(self.powers))


def scenario(prices=(100,), reserves=None, energy=50, battery=None):
    return Scenario(
        1,
        7,
        battery or BatteryConfig(100, 10, 90, 20, 20),
        BatteryState(energy),
        prices,
        tuple(() for _ in prices),
        reserves or tuple(0 for _ in prices),
    )


def report(case, tested, comparator, threshold=1):
    return evaluate_episode(
        run_episode(case, Script(tested)),
        run_episode(case, Script(comparator)),
        FailureConfig(economic_threshold_eur=threshold),
    )


@pytest.mark.parametrize("threshold", [0, -1, inf, -inf, nan, True, "1", None])
def test_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        FailureConfig(economic_threshold_eur=threshold)


@pytest.mark.parametrize("version", [0, 2, True, 1.0, "1", None])
def test_invalid_version(version):
    with pytest.raises(ValueError):
        FailureConfig(schema_version=version)


def test_configuration_and_report_are_immutable():
    config = FailureConfig()
    with pytest.raises(FrozenInstanceError):
        config.economic_threshold_eur = 2
    result = report(scenario(), [0], [0])
    with pytest.raises(FrozenInstanceError):
        result.economic_eligible = False
    assert result.findings == ()


def test_one_invalid_attempt_even_when_multiple_limits_violated():
    result = report(scenario(), [200], [0])
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert (finding.kind, finding.step_index, finding.severity, finding.classification) == (
        "invalid_action",
        0,
        1,
        "invalid_request",
    )


@pytest.mark.parametrize(
    ("reserve", "classification"),
    [
        (60, "locally_preventable"),
        (80, "undetermined"),
        (100, "physically_impossible"),
    ],
)
def test_reserve_classifications(reserve, classification):
    result = report(scenario(reserves=(reserve,)), [0], [0])
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.kind == "reserve_shortfall"
    assert finding.step_index == 0
    assert finding.severity == reserve - 50
    assert finding.classification == classification


def test_preventability_uses_efficiency_and_current_power_limits():
    battery = BatteryConfig(100, 10, 90, 20, 20, eta_charge=0.5)
    result = report(scenario(reserves=(65,), battery=battery), [0], [0])
    assert result.findings[0].classification == "undetermined"


def test_previously_depleted_state_is_not_called_globally_unavoidable():
    result = report(scenario(prices=(100, 100), reserves=(0, 70)), [20, 0], [0, -20])
    reserve = next(f for f in result.findings if f.kind == "reserve_shortfall")
    assert reserve.step_index == 1
    assert reserve.classification == "undetermined"


def test_material_economics_with_identical_terminal_inventory():
    # Tested buys at EUR 100/MWh and sells at zero: a EUR 2 loss.
    result = report(scenario(prices=(100, 0)), [-20, 20], [0, 0], threshold=2)
    assert result.profit_gap_eur == 2
    assert result.economic_eligible
    assert result.tested_ending_energy_kwh == result.comparator_ending_energy_kwh == 50
    assert len(result.findings) == 1
    assert result.findings[0].kind == "economic_underperformance"
    assert result.findings[0].classification == "feasible_comparator_dominance"
    assert result.findings[0].step_index is None
    assert result.findings[0].severity == 2


def test_threshold_and_negative_gap_do_not_flag():
    case = scenario(prices=(100, 0))
    assert report(case, [-20, 20], [0, 0], threshold=2.01).findings == ()
    result = report(case, [0, 0], [-20, 20])
    assert result.profit_gap_eur == -2
    assert result.findings == ()


def test_liquidating_more_inventory_does_not_establish_dominance():
    result = report(scenario(), [0], [20])
    assert result.profit_gap_eur == 2
    assert not result.economic_eligible
    assert result.findings == ()


def test_more_inventory_and_more_profit_dominates():
    result = report(scenario(prices=(-100,)), [0], [-20])
    assert result.economic_eligible
    assert result.findings[0].kind == "economic_underperformance"
    assert result.comparator_ending_energy_kwh == 70


@pytest.mark.parametrize(("reserves", "comparator"), [((0, 0), [200, 0]), ((60, 0), [0, 0])])
def test_comparator_must_meet_reserves_and_physical_limits(reserves, comparator):
    result = report(scenario(prices=(100, 0), reserves=reserves), [-20, 20], comparator)
    assert result.profit_gap_eur == 2
    assert not result.economic_eligible
    assert not any(f.kind == "economic_underperformance" for f in result.findings)


def test_identical_scenario_required_including_seed():
    case = scenario()
    with pytest.raises(ValueError, match="identical scenarios"):
        evaluate_episode(
            run_episode(case, Script([0])),
            run_episode(replace(case, seed=8), Script([0])),
            FailureConfig(),
        )


def test_overflowing_profit_difference_rejected():
    case = scenario(prices=(0,))
    episode = run_episode(case, Script([0]))
    tested = replace(episode, metrics=replace(episode.metrics, profit_eur=-1e308))
    comparator = replace(episode, metrics=replace(episode.metrics, profit_eur=1e308))
    with pytest.raises(ValueError, match="profit_gap_eur"):
        evaluate_episode(tested, comparator, FailureConfig())
