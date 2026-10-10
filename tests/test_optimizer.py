"""Independent arithmetic and bounded-search checks for the forecast optimizer."""

from itertools import product
from random import Random

import pytest

from trialgrid.controllers import BaselineConfig
from trialgrid.controllers.optimizer import ForecastOptimizer
from trialgrid.domain import Action, BatteryConfig, BatteryState
from trialgrid.runner import Observation, ObservationFrame, replay_episode, run_episode
from trialgrid.scenario import Scenario


def observation(price=10.0, forecast=(10.0, 100.0), energy=0.0, reserve=0.0, **kwargs):
    config = BatteryConfig(1, 0, 1, 1, 1, **kwargs)
    frame = ObservationFrame(0, BatteryState(energy), price, forecast, reserve)
    return Observation(config, frame, ())


def optimizer(horizon=3):
    return ForecastOptimizer(BaselineConfig(horizon=horizon, power_fractions=(1.0,)))


def test_two_hour_arbitrage_manual_accounting():
    obs = observation()
    plan = optimizer().plan(obs)
    assert plan.actions == (Action(-1), Action(1))
    assert plan.forecast_profit_eur == pytest.approx(0.09)
    assert optimizer().act(obs, Random(9)) == plan.actions[0]


@pytest.mark.parametrize("prices", list(product((-100.0, 0.0, 100.0), repeat=3)))
@pytest.mark.parametrize("initial", (0.0, 1.0))
def test_matches_independent_exhaustive_unit_battery(prices, initial):
    # This independently enumerates integer schedules, without sharing candidate
    # generation, recurrence, or simulator accounting with the implementation.
    scores = []
    for powers in product((-1, 0, 1), repeat=3):
        energy = initial
        profit = 0.0
        for power, price in zip(powers, prices, strict=True):
            energy -= power
            if not 0 <= energy <= 1:
                break
            profit += power * price / 1000
        else:
            scores.append(profit)
    plan = optimizer().plan(observation(prices[0], prices, initial))
    assert plan.forecast_profit_eur == pytest.approx(max(scores))
    energy = initial
    for action in plan.actions:
        energy -= action.p_kw
        assert 0 <= energy <= 1


def test_efficiency_and_both_throughput_costs():
    obs = observation(10, (10, 100), eta_charge=0.8, eta_discharge=0.5,
                      degradation_eur_per_kwh=0.001)
    plan = optimizer().plan(obs)
    assert plan.actions[0].p_kw == pytest.approx(-1)
    assert plan.actions[1].p_kw == pytest.approx(0.4)
    assert plan.forecast_profit_eur == pytest.approx(-0.01 + 0.04 - 0.0014)


def test_degradation_can_make_arbitrage_unprofitable():
    plan = optimizer().plan(observation(10, (10, 20), degradation_eur_per_kwh=0.01))
    assert plan.actions == (Action(), Action())
    assert plan.forecast_profit_eur == 0


def test_negative_prices_pay_to_charge_even_without_terminal_value():
    plan = optimizer().plan(observation(-100, (999,)))
    assert plan.actions == (Action(-1),)
    assert plan.forecast_profit_eur == pytest.approx(0.1)


def test_zero_terminal_value_liquidates_initial_energy():
    plan = optimizer().plan(observation(100, (), 1))
    assert plan.actions == (Action(1),)
    assert plan.forecast_profit_eur == pytest.approx(0.1)


def test_current_actual_replaces_forecast_zero_and_horizon_caps_future():
    obs = observation(10, (1e6, 100, 1e9))
    plan = optimizer(horizon=2).plan(obs)
    assert plan.actions == (Action(-1), Action(1))
    assert plan.forecast_profit_eur == pytest.approx(0.09)
    assert optimizer(horizon=1).plan(obs).actions == (Action(),)


def test_no_forecast_and_single_forecast_both_use_current_actual():
    for forecast in ((), (-1e6,)):
        assert optimizer().plan(observation(100, forecast, 1)).actions == (Action(1),)


def test_reserve_priority_over_economics_and_unknown_future_reserves():
    plan = optimizer().plan(observation(100, (100, 200), 0, 0.3))
    assert -plan.actions[0].p_kw >= 0.3
    assert sum(action.p_kw for action in plan.actions) == pytest.approx(0)
    # Current reserve is not silently repeated into the future projection.
    assert plan.actions[-1].p_kw > 0


def test_unreachable_reserve_uses_maximum_feasible_charge():
    config = BatteryConfig(2, 0, 1, 0.25, 1)
    obs = Observation(config, ObservationFrame(0, BatteryState(0), 100, (), 2), ())
    assert optimizer().plan(obs).actions == (Action(-0.25),)


def test_reserve_boundary_candidate_allows_non_grid_discharge():
    plan = optimizer().plan(observation(100, (), 1, 0.3))
    assert plan.actions[0].p_kw == pytest.approx(0.7)
    assert 1 - plan.actions[0].p_kw >= 0.3


def test_zero_profit_ties_prefer_hold_deterministically():
    obs = observation(0, (0, 0, 0), 0.5)
    expected = (Action(), Action(), Action())
    assert optimizer().plan(obs).actions == expected
    assert optimizer().plan(obs).actions == expected


def test_episode_determinism_and_json_replay():
    obs = observation()
    scenario = Scenario(1, 77, obs.config, obs.current.state, (10, 100, -20),
                        ((10, 90, -10), (100, -10), (-20,)), (0, 0, 0))
    first = run_episode(scenario, optimizer(), controller_name="forecast", controller_version="1")
    second = run_episode(scenario, optimizer(), controller_name="forecast", controller_version="1")
    assert first == second == replay_episode(first)
    assert first.metrics.rejected_action_count == 0


def test_forecast_total_overflow_is_explicit():
    config = BatteryConfig(1000, 0, 1000, 1000, 1000)
    obs = Observation(config, ObservationFrame(0, BatteryState(0), -1e308,
                                               (-1e308, 1e308), 0), ())
    with pytest.raises(ValueError, match="forecast plan profit"):
        optimizer().plan(obs)
