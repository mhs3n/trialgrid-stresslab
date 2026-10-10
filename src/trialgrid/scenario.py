"""Validated, versioned scenario inputs for deterministic hourly episodes."""

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

from trialgrid.domain import BatteryConfig, BatteryState, _finite


def require_keys(value: Any, keys: set[str]) -> dict:
    """Require a JSON object with exactly the specified fields."""
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"expected object fields: {sorted(keys)}")
    return value


def load_json(text: str) -> dict:
    """Read a JSON object without ambiguous duplicate fields or nonfinite numbers."""
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON number: {value}")

    if not isinstance(text, str):
        raise ValueError("JSON input must be text")
    value = json.loads(text, object_pairs_hook=object_pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _sequence(name: str, value: Any) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Scenario:
    """Forecast row t is published at start t; its first value forecasts hour t."""

    schema_version: int
    seed: int
    config: BatteryConfig
    initial_state: BatteryState
    prices_eur_per_mwh: tuple[float, ...]
    forecast_prices_eur_per_mwh: tuple[tuple[float, ...], ...]
    reserve_requirements_kwh: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("schema_version must be integer 1")
        if type(self.seed) is not int or not 0 <= self.seed <= 2**64 - 1:
            raise ValueError("seed must be an integer in [0, 2**64 - 1]")
        if not isinstance(self.config, BatteryConfig):
            raise ValueError("config must be BatteryConfig")
        if not isinstance(self.initial_state, BatteryState):
            raise ValueError("initial_state must be BatteryState")
        if self.config.dt_h != 1:
            raise ValueError("scenario dt_h must be 1 hour")
        if not (
            self.config.energy_floor_kwh
            <= self.initial_state.energy_kwh
            <= self.config.energy_ceiling_kwh
        ):
            raise ValueError("initial energy must be within configured bounds")
        prices = _sequence("prices_eur_per_mwh", self.prices_eur_per_mwh)
        forecasts = _sequence("forecast_prices_eur_per_mwh", self.forecast_prices_eur_per_mwh)
        reserves = _sequence("reserve_requirements_kwh", self.reserve_requirements_kwh)
        if not prices or len(forecasts) != len(prices) or len(reserves) != len(prices):
            raise ValueError("scenario timelines must have equal nonzero length")
        for price in prices:
            _finite("price_eur_per_mwh", price)
        forecast_rows = []
        for index, forecast in enumerate(forecasts):
            row = _sequence("forecast row", forecast)
            if len(row) > len(prices) - index:
                raise ValueError("forecast row exceeds remaining horizon")
            for price in row:
                _finite("forecast price", price)
            forecast_rows.append(row)
        for reserve in reserves:
            _finite("reserve requirement", reserve)
            if not 0 <= reserve <= self.config.capacity_kwh:
                raise ValueError("reserve requirement must be within capacity")
        object.__setattr__(self, "prices_eur_per_mwh", prices)
        object.__setattr__(self, "forecast_prices_eur_per_mwh", tuple(forecast_rows))
        object.__setattr__(self, "reserve_requirements_kwh", reserves)

    def to_dict(self) -> dict:
        """Return an independent JSON-compatible object."""
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "config": asdict(self.config),
            "initial_state": asdict(self.initial_state),
            "prices_eur_per_mwh": list(self.prices_eur_per_mwh),
            "forecast_prices_eur_per_mwh": [list(row) for row in self.forecast_prices_eur_per_mwh],
            "reserve_requirements_kwh": list(self.reserve_requirements_kwh),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, value: dict) -> "Scenario":
        data = require_keys(value, {field.name for field in fields(cls)})
        config = require_keys(data["config"], {field.name for field in fields(BatteryConfig)})
        state = require_keys(data["initial_state"], {field.name for field in fields(BatteryState)})
        return cls(
            schema_version=data["schema_version"],
            seed=data["seed"],
            config=BatteryConfig(**config),
            initial_state=BatteryState(**state),
            prices_eur_per_mwh=data["prices_eur_per_mwh"],
            forecast_prices_eur_per_mwh=data["forecast_prices_eur_per_mwh"],
            reserve_requirements_kwh=data["reserve_requirements_kwh"],
        )

    @classmethod
    def from_json(cls, text: str) -> "Scenario":
        return cls.from_dict(load_json(text))
