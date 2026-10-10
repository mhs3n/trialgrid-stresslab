"""Scenario boundaries and strict, deterministic JSON round trips."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from trialgrid.domain import BatteryConfig, BatteryState
from trialgrid.scenario import Scenario


@pytest.fixture
def scenario():
    return Scenario(
        schema_version=1,
        seed=42,
        config=BatteryConfig(100, 10, 90, 20, 20),
        initial_state=BatteryState(50),
        prices_eur_per_mwh=(-50, 100, 0),
        forecast_prices_eur_per_mwh=((-45, 95, 10), (110,), ()),
        reserve_requirements_kwh=(0, 60, 100),
    )


def test_json_roundtrip_and_independent_dictionary(scenario):
    encoded = scenario.to_json()
    assert Scenario.from_json(encoded) == scenario
    assert Scenario.from_dict(json.loads(encoded)) == scenario
    assert Scenario.from_json(encoded).to_json() == encoded
    data = scenario.to_dict()
    data["config"]["capacity_kwh"] = 200
    data["forecast_prices_eur_per_mwh"][0][0] = 999
    assert scenario.config.capacity_kwh == 100
    assert scenario.forecast_prices_eur_per_mwh[0][0] == -45


def test_list_inputs_are_copied_to_immutable_tuples(scenario):
    prices, forecasts, reserves = [1, 2, 3], [[4], [], []], [0, 0, 0]
    value = replace(
        scenario, prices_eur_per_mwh=prices,
        forecast_prices_eur_per_mwh=forecasts, reserve_requirements_kwh=reserves,
    )
    prices[0] = 99
    forecasts[0][0] = 99
    reserves[0] = 99
    assert value.prices_eur_per_mwh == (1, 2, 3)
    assert value.forecast_prices_eur_per_mwh == ((4,), (), ())
    assert value.reserve_requirements_kwh == (0, 0, 0)
    with pytest.raises(FrozenInstanceError):
        value.seed = 0


@pytest.mark.parametrize("version", [True, 1.0, "1", 0, 2, None])
def test_unsupported_version(scenario, version):
    with pytest.raises(ValueError):
        replace(scenario, schema_version=version)


@pytest.mark.parametrize("seed", [True, 1.0, "1", -1, 2**64, None])
def test_invalid_seed(scenario, seed):
    with pytest.raises(ValueError):
        replace(scenario, seed=seed)


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_boundaries(scenario, seed):
    assert Scenario.from_json(replace(scenario, seed=seed).to_json()).seed == seed


@pytest.mark.parametrize(
    "changes",
    [
        {"config": {}}, {"initial_state": {}},
        {"initial_state": BatteryState(9)}, {"initial_state": BatteryState(91)},
        {"prices_eur_per_mwh": ()}, {"prices_eur_per_mwh": (1,)},
        {"forecast_prices_eur_per_mwh": ()}, {"reserve_requirements_kwh": (0,)},
        {"forecast_prices_eur_per_mwh": ((1, 2, 3, 4), (), ())},
        {"forecast_prices_eur_per_mwh": ((), (), (1, 2))},
        {"forecast_prices_eur_per_mwh": ((), None, ())},
    ],
)
def test_structure_validation(scenario, changes):
    with pytest.raises(ValueError):
        replace(scenario, **changes)


@pytest.mark.parametrize("duration", [0.5, 2])
def test_hourly_resolution_required(scenario, duration):
    with pytest.raises(ValueError, match="1 hour"):
        replace(scenario, config=replace(scenario.config, dt_h=duration))


@pytest.mark.parametrize(
    "field", ["prices_eur_per_mwh", "forecast_prices_eur_per_mwh", "reserve_requirements_kwh"]
)
@pytest.mark.parametrize("value", [None, "123", {0: 1}, 3])
def test_timelines_require_sequences(scenario, field, value):
    with pytest.raises(ValueError):
        replace(scenario, **{field: value})


@pytest.mark.parametrize(
    "bad", [True, None, "1", float("nan"), float("inf"), -float("inf"), 10**1000]
)
@pytest.mark.parametrize("kind", ["actual", "forecast", "reserve"])
def test_numeric_validation(scenario, bad, kind):
    changes = {
        "actual": {"prices_eur_per_mwh": (bad, 0, 0)},
        "forecast": {"forecast_prices_eur_per_mwh": ((bad,), (), ())},
        "reserve": {"reserve_requirements_kwh": (bad, 0, 0)},
    }
    with pytest.raises(ValueError):
        replace(scenario, **changes[kind])


@pytest.mark.parametrize("reserve", [-1, 101])
def test_reserve_bounds(scenario, reserve):
    with pytest.raises(ValueError):
        replace(scenario, reserve_requirements_kwh=(reserve, 0, 0))


def test_reserve_above_operating_ceiling_is_representable(scenario):
    assert scenario.reserve_requirements_kwh[-1] > scenario.config.energy_ceiling_kwh


@pytest.mark.parametrize("nested", [None, "config", "initial_state"])
@pytest.mark.parametrize("operation", ["extra", "missing"])
def test_unknown_and_missing_fields(scenario, nested, operation):
    data = scenario.to_dict()
    target = data if nested is None else data[nested]
    if operation == "extra":
        target["unexpected"] = 1
    else:
        target.pop(next(iter(target)))
    with pytest.raises(ValueError):
        Scenario.from_dict(data)


@pytest.mark.parametrize("bad", [None, [], "text", 123])
@pytest.mark.parametrize("nested", [None, "config", "initial_state"])
def test_dictionary_object_types(scenario, bad, nested):
    data = scenario.to_dict()
    if nested is None:
        data = bad
    else:
        data[nested] = bad
    with pytest.raises(ValueError):
        Scenario.from_dict(data)


@pytest.mark.parametrize("payload", ["[]", "null", "3", "{}", "{", '{"seed":1,"seed":2}'])
def test_malformed_json(payload):
    with pytest.raises(ValueError):
        Scenario.from_json(payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_json(scenario, constant):
    payload = scenario.to_json().replace('"seed": 42', f'"seed": {constant}')
    with pytest.raises(ValueError):
        Scenario.from_json(payload)


def test_nested_duplicate_json_field(scenario):
    payload = scenario.to_json().replace('"energy_kwh": 50', '"energy_kwh": 50, "energy_kwh": 60')
    with pytest.raises(ValueError, match="duplicate"):
        Scenario.from_json(payload)


@pytest.mark.parametrize("payload", [None, b"{}", 1])
def test_json_requires_text(payload):
    with pytest.raises(ValueError):
        Scenario.from_json(payload)
