"""Bounded synthetic scenarios with independently sampled published forecasts."""

import hashlib
import json
from dataclasses import asdict, dataclass, fields, replace
from math import nextafter
from random import Random

from trialgrid.domain import BatteryConfig, BatteryState, _finite
from trialgrid.scenario import Scenario, load_json, require_keys


@dataclass(frozen=True, slots=True)
class ScenarioDomain:
    config: BatteryConfig = BatteryConfig(100, 10, 90, 20, 20, 0.9, 0.8)
    hours: int = 24
    forecast_horizon: int = 4
    price_min: float = -100.0
    price_max: float = 200.0
    reserve_min_kwh: float = 0.0
    reserve_max_kwh: float = 90.0

    def __post_init__(self) -> None:
        if not isinstance(self.config, BatteryConfig) or self.config.dt_h != 1:
            raise ValueError("domain requires an hourly BatteryConfig")
        for name, maximum in (("hours", 168), ("forecast_horizon", 6)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in [1, {maximum}]")
        for name in ("price_min", "price_max", "reserve_min_kwh", "reserve_max_kwh"):
            _finite(name, getattr(self, name))
        if self.price_min > self.price_max:
            raise ValueError("price range must be ordered")
        if not 0 <= self.reserve_min_kwh <= self.reserve_max_kwh <= self.config.capacity_kwh:
            raise ValueError("reserve range must be ordered and within capacity")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, value: dict) -> "ScenarioDomain":
        data = dict(require_keys(value, {field.name for field in fields(cls)}))
        data["config"] = BatteryConfig(**require_keys(
            data["config"], {field.name for field in fields(BatteryConfig)}
        ))
        return cls(**data)

    @classmethod
    def from_json(cls, text: str) -> "ScenarioDomain":
        return cls.from_dict(load_json(text))


def validate_scenario(scenario: Scenario, domain: ScenarioDomain) -> None:
    """Require membership, including the exact publication lengths and battery."""
    if not isinstance(domain, ScenarioDomain) or not isinstance(scenario, Scenario):
        raise ValueError("expected a Scenario and ScenarioDomain")
    if scenario.config != domain.config or len(scenario.prices_eur_per_mwh) != domain.hours:
        raise ValueError("scenario battery or horizon differs from domain")
    for index, row in enumerate(scenario.forecast_prices_eur_per_mwh):
        if len(row) != min(domain.forecast_horizon, domain.hours - index):
            raise ValueError("scenario forecast publication length differs from domain")
    for value in (*scenario.prices_eur_per_mwh,
                  *(p for row in scenario.forecast_prices_eur_per_mwh for p in row)):
        if not domain.price_min <= value <= domain.price_max:
            raise ValueError("scenario price outside domain")
    if any(not domain.reserve_min_kwh <= value <= domain.reserve_max_kwh
           for value in scenario.reserve_requirements_kwh):
        raise ValueError("scenario reserve outside domain")


def _uniform(rng: Random, low: float, high: float) -> float:
    # Weighted endpoints avoid overflow in high-low for wide finite domains.
    weight = rng.random()
    return min(high, max(low, (1.0 - weight) * low + weight * high))


def sample_scenario(domain: ScenarioDomain, rng: Random) -> Scenario:
    """Sample actuals and every forecast value independently, never copy actuals."""
    if not isinstance(domain, ScenarioDomain) or not isinstance(rng, Random):
        raise ValueError("expected a ScenarioDomain and Random")
    prices = tuple(_uniform(rng, domain.price_min, domain.price_max) for _ in range(domain.hours))
    forecasts = tuple(
        tuple(_uniform(rng, domain.price_min, domain.price_max)
              for _ in range(min(domain.forecast_horizon, domain.hours - index)))
        for index in range(domain.hours)
    )
    reserves = tuple(_uniform(rng, domain.reserve_min_kwh, domain.reserve_max_kwh)
                     for _ in range(domain.hours))
    energy = _uniform(rng, domain.config.energy_floor_kwh, domain.config.energy_ceiling_kwh)
    return Scenario(1, rng.getrandbits(64), domain.config, BatteryState(energy),
                    prices, forecasts, reserves)


def mutate_scenario(parent: Scenario, domain: ScenarioDomain, rng: Random) -> Scenario:
    """Resample exactly one mutable environmental scalar and renew the episode seed."""
    validate_scenario(parent, domain)
    if not isinstance(rng, Random):
        raise ValueError("expected Random")
    coordinates = [("energy", 0, 0)]
    if domain.price_min < domain.price_max:
        coordinates.extend(("price", index, 0) for index in range(domain.hours))
        coordinates.extend(("forecast", index, offset)
                           for index, row in enumerate(parent.forecast_prices_eur_per_mwh)
                           for offset in range(len(row)))
    if domain.reserve_min_kwh < domain.reserve_max_kwh:
        coordinates.extend(("reserve", index, 0) for index in range(domain.hours))
    kind, index, offset = rng.choice(coordinates)
    prices = list(parent.prices_eur_per_mwh)
    forecasts = [list(row) for row in parent.forecast_prices_eur_per_mwh]
    reserves = list(parent.reserve_requirements_kwh)
    energy = parent.initial_state.energy_kwh
    if kind == "energy":
        old, low, high = energy, domain.config.energy_floor_kwh, domain.config.energy_ceiling_kwh
    elif kind == "reserve":
        old, low, high = reserves[index], domain.reserve_min_kwh, domain.reserve_max_kwh
    else:
        old = prices[index] if kind == "price" else forecasts[index][offset]
        low, high = domain.price_min, domain.price_max
    value = _uniform(rng, low, high)
    if value == old:
        value = nextafter(old, high if old < high else low)
    if kind == "energy":
        energy = value
    elif kind == "reserve":
        reserves[index] = value
    elif kind == "price":
        prices[index] = value
    else:
        forecasts[index][offset] = value
    return replace(parent, seed=rng.getrandbits(64), initial_state=BatteryState(energy),
                   prices_eur_per_mwh=tuple(prices),
                   forecast_prices_eur_per_mwh=tuple(tuple(row) for row in forecasts),
                   reserve_requirements_kwh=tuple(reserves))


def scenario_fingerprint(scenario: Scenario) -> str:
    """Exact environmental deduplication, excluding seed; not approximate similarity."""
    data = scenario.to_dict()
    del data["seed"]

    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        if type(value) in (int, float):
            return float(value) if value else 0.0
        return value

    normalized = normalize(data)
    normalized["schema_version"] = scenario.schema_version
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()
