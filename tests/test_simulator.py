"""Independent expected values for physics, constraints, and accounting."""

from dataclasses import FrozenInstanceError, replace
from math import inf, nextafter

import pytest

from trialgrid.domain import Action, BatteryConfig, BatteryState, StepInput
from trialgrid.simulator import step


@pytest.fixture
def config():
    return BatteryConfig(100, 10, 90, 40, 30, 0.8, 0.5, 0.25, 0.03)


@pytest.mark.parametrize(
    ("power", "expected_energy", "energy_in", "energy_out"),
    [(-20, 54, 4, 0), (20, 40, 0, 10), (0, 50, 0, 0)],
)
@pytest.mark.parametrize("price", [-200, 0, 200])
def test_physics_and_grid_side_accounting(
    config, power, expected_energy, energy_in, energy_out, price
):
    state, action = BatteryState(50), Action(power)
    result = step(config, state, action, StepInput(price))
    assert result.state.energy_kwh == pytest.approx(expected_energy)
    assert result.energy_in_kwh == pytest.approx(energy_in)
    assert result.energy_out_kwh == pytest.approx(energy_out)
    assert result.state.energy_kwh == pytest.approx(50 + energy_in - energy_out)
    assert result.revenue_eur == pytest.approx(power * 0.25 / 1000 * price)
    assert result.degradation_cost_eur == pytest.approx(abs(power * 0.25) * 0.03)
    assert result.profit_eur == pytest.approx(result.revenue_eur - result.degradation_cost_eur)
    assert result.requested_action == result.applied_action == action
    assert result.violations == ()
    assert state.energy_kwh == 50


@pytest.mark.parametrize("duration", [0.125, 0.5, 1, 2])
@pytest.mark.parametrize("power", [-5, 5])
def test_duration_scales_transition_and_accounting(config, duration, power):
    result = step(replace(config, dt_h=duration), BatteryState(50), Action(power), StepInput(100))
    delta = -power * duration * 0.8 if power < 0 else -power * duration / 0.5
    assert result.state.energy_kwh == pytest.approx(50 + delta)
    assert result.revenue_eur == pytest.approx(power * duration * 0.1)
    assert result.degradation_cost_eur == pytest.approx(abs(power) * duration * 0.03)


@pytest.mark.parametrize(
    ("energy", "power", "violations"),
    [
        (50, 31, ("discharge_power_limit",)),
        (50, -41, ("charge_power_limit",)),
        (10, 1, ("energy_floor",)),
        (90, -1, ("energy_ceiling",)),
        (10, 100, ("discharge_power_limit", "energy_floor")),
        (90, -100, ("charge_power_limit", "energy_ceiling")),
    ],
)
def test_rejected_requests_apply_hold_and_preserve_evidence(config, energy, power, violations):
    action = Action(power)
    result = step(config, BatteryState(energy), action, StepInput(-123))
    assert result.requested_action == action
    assert result.applied_action == Action(0)
    assert result.violations == violations
    assert result.state == BatteryState(energy)
    assert result.energy_in_kwh == result.energy_out_kwh == 0
    assert result.revenue_eur == result.degradation_cost_eur == result.profit_eur == 0


@pytest.mark.parametrize(("energy", "power", "expected"), [(25, 30, 10), (82, -40, 90)])
def test_exact_power_and_energy_limits_are_feasible(config, energy, power, expected):
    result = step(config, BatteryState(energy), Action(power), StepInput(100))
    assert result.state.energy_kwh == expected
    assert result.violations == ()
    assert result.applied_action.p_kw == power


@pytest.mark.parametrize(("energy", "power"), [(25, 30 + 1e-10), (82, -40 - 1e-10)])
def test_small_exceedances_are_rejected_without_clamping(config, energy, power):
    result = step(config, BatteryState(energy), Action(power), StepInput(0))
    assert result.violations
    assert result.state.energy_kwh == energy
    assert result.applied_action.p_kw == 0


@pytest.mark.parametrize("power", [nextafter(30, inf), nextafter(-40, -inf)])
def test_single_float_power_limit_exceedance_is_rejected(config, power):
    result = step(config, BatteryState(50), Action(power), StepInput(0))
    assert result.violations
    assert result.applied_action.p_kw == 0


@pytest.mark.parametrize("power", [nextafter(10, inf), nextafter(-10, -inf)])
def test_single_float_energy_limit_exceedance_is_rejected(config, power):
    cfg = replace(
        config, eta_charge=1, eta_discharge=1, dt_h=1, energy_floor_kwh=40, energy_ceiling_kwh=60
    )
    # A one-ULP command exceedance can round away at larger state magnitudes;
    # anchor the lower boundary at zero so the excess remains representable.
    if power > 0:
        cfg = replace(cfg, energy_floor_kwh=0)
        energy = 10
    else:
        cfg = replace(cfg, energy_floor_kwh=0, energy_ceiling_kwh=10)
        energy = 0
    result = step(cfg, BatteryState(energy), Action(power), StepInput(0))
    assert result.violations
    assert result.applied_action.p_kw == 0


@pytest.mark.parametrize("energy", [0, 9.99, 90.01, 101])
def test_invalid_initial_state_rejected_even_for_hold(config, energy):
    with pytest.raises(ValueError, match="initial energy"):
        step(config, BatteryState(energy), Action(), StepInput(0))


@pytest.mark.parametrize("energy", [10, 90])
def test_hold_at_limits_with_disabled_power(config, energy):
    result = step(
        replace(config, max_charge_kw=0, max_discharge_kw=0),
        BatteryState(energy),
        Action(),
        StepInput(-100),
    )
    assert result.state.energy_kwh == energy
    assert result.violations == ()
    assert result.profit_eur == 0


@pytest.mark.parametrize(
    ("charge_efficiency", "discharge_efficiency"), [(1, 1), (0.8, 1), (1, 0.5), (0.8, 0.5)]
)
def test_round_trip_never_creates_energy(config, charge_efficiency, discharge_efficiency):
    cfg = replace(config, eta_charge=charge_efficiency, eta_discharge=discharge_efficiency)
    charged = step(cfg, BatteryState(50), Action(-20), StepInput(0))
    discharged = step(cfg, charged.state, Action(20), StepInput(0))
    assert discharged.state.energy_kwh <= 50
    assert discharged.state.energy_kwh == pytest.approx(
        50 + 5 * charge_efficiency - 5 / discharge_efficiency
    )


def test_repeated_sequence_is_deterministic_and_conserves_energy(config):
    def run():
        state = BatteryState(50)
        results = []
        for power, price in [(-20, -100), (10, 100), (999, 200), (0, 0)] * 20:
            result = step(config, state, Action(power), StepInput(price))
            assert 10 <= result.state.energy_kwh <= 90
            assert result.state.energy_kwh == pytest.approx(
                state.energy_kwh + result.energy_in_kwh - result.energy_out_kwh
            )
            results.append(result)
            state = result.state
        return results

    assert run() == run()


def test_result_and_nested_state_are_immutable(config):
    result = step(config, BatteryState(50), Action(10), StepInput(1))
    with pytest.raises(FrozenInstanceError):
        result.profit_eur = 123
    with pytest.raises(FrozenInstanceError):
        result.state.energy_kwh = 123


@pytest.mark.parametrize("field", ["price_eur_per_mwh", "degradation_eur_per_kwh"])
def test_nonfinite_accounting_is_not_returned(field):
    cfg = BatteryConfig(1e308, 0, 1e308, 1e308, 1e308)
    price = StepInput(1e308 if field == "price_eur_per_mwh" else 0)
    if field == "degradation_eur_per_kwh":
        cfg = replace(cfg, degradation_eur_per_kwh=1e308)
    with pytest.raises(ValueError, match="finite numeric range"):
        step(cfg, BatteryState(1e308), Action(1e308), price)


def test_overflowing_infeasible_energy_request_is_rejected():
    cfg = BatteryConfig(100, 0, 100, 1e308, 1e308, dt_h=2)
    result = step(cfg, BatteryState(50), Action(1e308), StepInput(1))
    assert result.violations == ("energy_floor",)
    assert result.state.energy_kwh == 50
    assert result.applied_action.p_kw == result.profit_eur == 0


@pytest.mark.parametrize("power", [10**300, -(10**300)])
def test_large_integer_arithmetic_rejects_like_floats(power):
    cfg = BatteryConfig(100, 0, 100, 10**300, 10**300, dt_h=10**300)
    result = step(cfg, BatteryState(50), Action(power), StepInput(1))
    float_result = step(
        replace(cfg, dt_h=float(cfg.dt_h), max_charge_kw=1e300, max_discharge_kw=1e300),
        BatteryState(50),
        Action(float(power)),
        StepInput(1),
    )
    assert result.state.energy_kwh == float_result.state.energy_kwh == 50
    assert result.violations == float_result.violations
    assert result.applied_action.p_kw == result.profit_eur == 0
