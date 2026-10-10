"""Validation at every public numeric model boundary."""

from dataclasses import FrozenInstanceError, fields, replace

import pytest

from trialgrid.domain import Action, BatteryConfig, BatteryState, StepInput


def config():
    return BatteryConfig(100, 10, 90, 20, 30)


@pytest.mark.parametrize("field", [field.name for field in fields(BatteryConfig)])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True, "1", None])
def test_config_requires_finite_numbers(field, invalid):
    with pytest.raises(ValueError):
        replace(config(), **{field: invalid})


@pytest.mark.parametrize(
    "changes",
    [
        {"capacity_kwh": 0},
        {"capacity_kwh": -1},
        {"energy_floor_kwh": -1},
        {"energy_floor_kwh": 90},
        {"energy_floor_kwh": 91},
        {"energy_ceiling_kwh": 101},
        {"energy_ceiling_kwh": 9},
        {"max_charge_kw": -1},
        {"max_discharge_kw": -1},
        {"eta_charge": 0},
        {"eta_charge": -0.1},
        {"eta_charge": 1.01},
        {"eta_discharge": 0},
        {"eta_discharge": -0.1},
        {"eta_discharge": 1.01},
        {"dt_h": 0},
        {"dt_h": -1},
        {"degradation_eur_per_kwh": -0.01},
    ],
)
def test_invalid_configuration_relationships(changes):
    with pytest.raises(ValueError):
        replace(config(), **changes)


def test_valid_extreme_boundaries_and_defaults():
    value = BatteryConfig(100, 0, 100, 0, 0)
    assert value.eta_charge == value.eta_discharge == value.dt_h == 1
    assert value.degradation_eur_per_kwh == 0
    assert BatteryState(0).energy_kwh == 0
    assert Action().p_kw == 0
    assert StepInput(-100).price_eur_per_mwh == -100


@pytest.mark.parametrize("model", [BatteryState, Action, StepInput])
@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), False, "0", None, 10**1000]
)
def test_scalar_models_reject_invalid_numbers(model, invalid):
    with pytest.raises(ValueError):
        model(invalid)


def test_negative_state_is_invalid():
    with pytest.raises(ValueError):
        BatteryState(-1)


@pytest.mark.parametrize(
    ("model", "attribute"),
    [
        (config(), "capacity_kwh"),
        (BatteryState(10), "energy_kwh"),
        (Action(1), "p_kw"),
        (StepInput(1), "price_eur_per_mwh"),
    ],
)
def test_models_are_immutable(model, attribute):
    with pytest.raises(FrozenInstanceError):
        setattr(model, attribute, 2)
