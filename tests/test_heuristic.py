"""Independent dispatch expectations for the forecast-value heuristic."""

from dataclasses import replace
from math import inf, nan
from random import Random

import pytest

from trialgrid.controllers._planning import BaselineConfig
from trialgrid.controllers.heuristic import HeuristicController
from trialgrid.domain import BatteryConfig, BatteryState, StepInput
from trialgrid.runner import Observation, ObservationFrame
from trialgrid.simulator import step


def dispatch(*, battery=None, energy=50, price=100, forecast=(), reserve=0, config=None):
    battery = battery or BatteryConfig(100, 10, 90, 20, 20)
    observation = Observation(
        battery, ObservationFrame(0, BatteryState(energy), price, forecast, reserve), ()
    )
    controller = HeuristicController(config or BaselineConfig())
    action = controller.act(observation, Random(7))
    result = step(battery, observation.current.state, action, StepInput(price))
    assert not result.violations
    assert result.requested_action == result.applied_action
    return result


def test_future_sale_value_accounts_for_both_efficiencies_and_degradation():
    battery = BatteryConfig(100, 10, 90, 20, 20, 0.8, 0.5, degradation_eur_per_kwh=0.01)
    # Future stored-energy value = (0.3 - 0.01) * 0.5 = EUR 0.145/kWh.
    # Charging buys 20 kWh for EUR 2 plus EUR .2 wear, adds 16 kWh worth EUR 2.32.
    result = dispatch(battery=battery, price=100, forecast=(9999, 300))
    assert result.applied_action.p_kw == -20
    assert result.state.energy_kwh == 66
    assert result.revenue_eur == -2
    assert result.degradation_cost_eur == pytest.approx(0.2)
    # Below break-even future value, hold beats either direction.
    assert dispatch(battery=battery, price=100, forecast=(9999, 250)).applied_action.p_kw == 0


def test_current_actual_supersedes_forecast_zero_and_horizon_truncates():
    assert dispatch(price=100, forecast=(-99999, 200)).applied_action.p_kw == -20
    assert dispatch(price=100, forecast=(99999, 200)).applied_action.p_kw == -20
    assert dispatch(price=100, forecast=(-99999,)).applied_action.p_kw == 20
    assert dispatch(
        price=100, forecast=(-99999, 200), config=BaselineConfig(horizon=1)
    ).applied_action.p_kw == 20


def test_heuristic_uses_mean_future_value():
    assert dispatch(price=125, forecast=(125, 0, 250)).applied_action.p_kw == 0
    assert dispatch(price=100, forecast=(100, 20, 200)).applied_action.p_kw == -20


@pytest.mark.parametrize("forecast", [(), (500,)])
def test_no_future_forecast_liquidates_at_positive_price(forecast):
    assert dispatch(forecast=forecast).applied_action.p_kw == 20


def test_negative_price_charges_and_degradation_can_prevent_it():
    assert dispatch(price=-100).applied_action.p_kw == -20
    battery = BatteryConfig(100, 10, 90, 20, 20, degradation_eur_per_kwh=0.2)
    assert dispatch(battery=battery, price=-100).applied_action.p_kw == 0
    assert dispatch(battery=battery, price=100).applied_action.p_kw == 0


def test_reserve_priority_and_exact_boundary_outside_fraction_menu():
    result = dispatch(energy=50, price=10000, reserve=53)
    assert result.applied_action.p_kw == -3
    assert result.state.energy_kwh == 53
    result = dispatch(energy=50, price=10000, reserve=47)
    assert result.applied_action.p_kw == 3
    assert result.state.energy_kwh == 47


@pytest.mark.parametrize("reserve", [80, 100])
def test_unreachable_reserve_requests_maximum_feasible_charge(reserve):
    result = dispatch(energy=50, price=10000, reserve=reserve)
    assert result.applied_action.p_kw == -20
    assert result.state.energy_kwh == 70
    assert reserve - result.state.energy_kwh > 0


def test_reserve_above_energy_ceiling_reaches_ceiling_without_invalid_request():
    result = dispatch(energy=85, price=10000, reserve=100)
    assert result.applied_action.p_kw == -5
    assert result.state.energy_kwh == 90


@pytest.mark.parametrize(("energy", "price"), [(10, 100), (90, -100)])
def test_hold_at_physical_bound(energy, price):
    assert dispatch(energy=energy, price=price).applied_action.p_kw == 0


def test_disabled_power_holds_even_with_reserve_shortfall():
    battery = BatteryConfig(100, 10, 90, 0, 0)
    result = dispatch(battery=battery, reserve=70)
    assert result.applied_action.p_kw == 0
    assert result.state.energy_kwh == 50


@pytest.mark.parametrize("price", [-100, 100])
def test_nonbinary_efficiency_boundaries_remain_physically_feasible(price):
    battery = BatteryConfig(10, 0.1, 9.7, 30, 30, 0.73, 0.83)
    result = dispatch(battery=battery, energy=3.14159, price=price)
    assert battery.energy_floor_kwh <= result.state.energy_kwh <= battery.energy_ceiling_kwh
    expected = battery.energy_ceiling_kwh if price < 0 else battery.energy_floor_kwh
    assert result.state.energy_kwh == pytest.approx(expected)


def test_repeated_decision_is_deterministic_and_does_not_consume_rng():
    battery = BatteryConfig(100, 10, 90, 20, 20)
    observation = Observation(
        battery, ObservationFrame(0, BatteryState(50), 100, (100, 200), 0), ()
    )
    controller, rng = HeuristicController(), Random(19)
    before = rng.getstate()
    assert controller.act(observation, rng) == controller.act(observation, rng)
    assert before == rng.getstate()
    # History does not override the currently published forecast.
    changed = replace(observation, history=(replace(observation.current, price_eur_per_mwh=-999),))
    assert controller.act(observation, rng) == controller.act(changed, rng)


@pytest.mark.parametrize("horizon", [True, 0, 7, 1.5, "4"])
def test_invalid_horizon(horizon):
    with pytest.raises(ValueError):
        BaselineConfig(horizon=horizon)


@pytest.mark.parametrize(
    "fractions", [(), (0, 1), (-1, 1), (0.5,), (1, 0.5), (1, 1), (True,),
                  (nan, 1), (inf, 1), (0.1, 0.2, 0.3, 1), "1"]
)
def test_invalid_power_menu(fractions):
    with pytest.raises(ValueError):
        BaselineConfig(power_fractions=fractions)


def test_power_menu_is_immutable_copy_and_controller_requires_typed_config():
    values = [0.5, 1]
    config = BaselineConfig(power_fractions=values)
    values[0] = 0.1
    assert config.power_fractions == (0.5, 1)
    with pytest.raises(ValueError):
        HeuristicController(config={})
