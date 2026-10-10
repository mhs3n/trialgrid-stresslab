"""Immutable model boundaries. All energy is kWh and power is grid-side kW."""

from dataclasses import dataclass, fields
from math import isfinite


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        finite = isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True, slots=True)
class BatteryConfig:
    """Physical limits; floor/ceiling are absolute energy within usable capacity."""

    capacity_kwh: float
    energy_floor_kwh: float
    energy_ceiling_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    eta_charge: float = 1.0
    eta_discharge: float = 1.0
    dt_h: float = 1.0
    degradation_eur_per_kwh: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            _finite(field.name, getattr(self, field.name))
        if self.capacity_kwh <= 0 or self.dt_h <= 0:
            raise ValueError("capacity_kwh and dt_h must be positive")
        if not 0 <= self.energy_floor_kwh < self.energy_ceiling_kwh <= self.capacity_kwh:
            raise ValueError("require 0 <= energy_floor_kwh < energy_ceiling_kwh <= capacity_kwh")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("power limits must be nonnegative")
        if not 0 < self.eta_charge <= 1 or not 0 < self.eta_discharge <= 1:
            raise ValueError("efficiencies must be in (0, 1]")
        if self.degradation_eur_per_kwh < 0:
            raise ValueError("degradation_eur_per_kwh must be nonnegative")


@dataclass(frozen=True, slots=True)
class BatteryState:
    energy_kwh: float

    def __post_init__(self) -> None:
        _finite("energy_kwh", self.energy_kwh)
        if self.energy_kwh < 0:
            raise ValueError("energy_kwh must be nonnegative")


@dataclass(frozen=True, slots=True)
class Action:
    """Positive p_kw discharges to grid; negative charges; zero holds."""

    p_kw: float = 0.0

    def __post_init__(self) -> None:
        _finite("p_kw", self.p_kw)


@dataclass(frozen=True, slots=True)
class StepInput:
    """Current realized price only; no future information or controller state."""

    price_eur_per_mwh: float

    def __post_init__(self) -> None:
        _finite("price_eur_per_mwh", self.price_eur_per_mwh)


@dataclass(frozen=True, slots=True)
class StepResult:
    """A rejected request applies hold and incurs zero revenue and throughput cost."""

    state: BatteryState
    requested_action: Action
    applied_action: Action
    violations: tuple[str, ...]
    energy_in_kwh: float
    energy_out_kwh: float
    revenue_eur: float
    degradation_cost_eur: float
    profit_eur: float
