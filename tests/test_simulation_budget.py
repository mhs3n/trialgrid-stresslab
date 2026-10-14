"""Count actual simulator invocations, including baseline planning and rejected requests."""

from random import Random

import pytest

from trialgrid import Action, BatteryConfig, BatteryState, StepInput, step
from trialgrid.controllers import ForecastOptimizer
from trialgrid.runner import Observation, ObservationFrame
from trialgrid.simulation_budget import SimulationBudgetExceeded, count_simulator_calls


def transition(action=None):
    return step(
        BatteryConfig(10, 0, 10, 2, 2),
        BatteryState(5),
        action if action is not None else Action(),
        StepInput(100),
    )


def test_budget_counts_rejected_commands_and_stops_before_extra_transition():
    with count_simulator_calls(2) as counter:
        transition()
        assert transition(Action(100)).violations
        with pytest.raises(SimulationBudgetExceeded):
            transition()
        assert counter.used == 2
    assert transition().state == BatteryState(5)


def test_controller_planning_is_included_in_budget():
    obs = Observation(
        BatteryConfig(10, 0, 10, 2, 2), ObservationFrame(0, BatteryState(5), 100, (100, 200), 0), ()
    )
    with count_simulator_calls(1) as counter:
        with pytest.raises(SimulationBudgetExceeded):
            ForecastOptimizer().act(obs, Random(0))
        assert counter.used == 1


def test_context_restored_after_exception_and_nested_budget_rejected():
    with pytest.raises(RuntimeError, match="test"):
        with count_simulator_calls(10):
            raise RuntimeError("test")
    with count_simulator_calls(3) as counter:
        with pytest.raises(ValueError, match="nested"):
            with count_simulator_calls(1):
                pass
        transition()
        assert counter.used == 1


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, 10_000_001))
def test_invalid_budget(limit):
    with pytest.raises(ValueError):
        with count_simulator_calls(limit):
            pass
