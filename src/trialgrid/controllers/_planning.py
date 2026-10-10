"""Shared information horizon and feasible candidates for the M3 baselines."""

from dataclasses import dataclass
from math import nextafter

from trialgrid.domain import Action, BatteryConfig, BatteryState, StepInput, StepResult, _finite
from trialgrid.runner import Observation
from trialgrid.simulator import step


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Bound work explicitly; both baselines use the same horizon/action menu.

    Fractions scale each state's maximum feasible charge/discharge power, not
    nameplate power. Hold and a current-reserve boundary action are added.
    """

    horizon: int = 4
    power_fractions: tuple[float, ...] = (0.5, 1.0)

    def __post_init__(self) -> None:
        if type(self.horizon) is not int or not 1 <= self.horizon <= 6:
            raise ValueError("horizon must be an integer from 1 to 6")
        if not isinstance(self.power_fractions, (tuple, list)):
            raise ValueError("power_fractions must be a list or tuple")
        fractions = tuple(self.power_fractions)
        if not 1 <= len(fractions) <= 3:
            raise ValueError("require 1 to 3 power fractions")
        for value in fractions:
            _finite("power fraction", value)
            if not 0 < value <= 1:
                raise ValueError("power fractions must be in (0, 1]")
        if tuple(sorted(set(fractions))) != fractions or fractions[-1] != 1:
            raise ValueError("power fractions must be increasing, unique, and end in 1")
        object.__setattr__(self, "power_fractions", fractions)


def planning_prices(observation: Observation, config: BaselineConfig) -> tuple[float, ...]:
    """Current actual supersedes forecast row[0]; never read later publications."""
    return (observation.current.price_eur_per_mwh,) + (
        observation.current.forecast_prices_eur_per_mwh[1 : config.horizon]
    )


def _feasible_result(
    config: BatteryConfig,
    state: BatteryState,
    power: float,
    price: float,
) -> StepResult | None:
    # Inverse bound calculations can round outside strict M1 bounds. Move only
    # planning candidates toward hold, never repair a command sent to the runner.
    for _ in range(64):
        result = step(config, state, Action(power), StepInput(price))
        if not result.violations:
            return result
        power = nextafter(power, 0.0)
    return None


def candidates(
    config: BatteryConfig,
    state: BatteryState,
    price: float,
    reserve: float,
    fractions: tuple[float, ...],
) -> tuple[StepResult, ...]:
    """Feasible actions minimizing current reserve shortfall before economics.

    All physics/accounting comes from M1. Unreachable commitments retain their
    shortfall in the real episode; maximal feasible charging is best effort.
    Future planning steps pass zero reserve because no future schedule is known.
    """
    charge = min(
        config.max_charge_kw,
        (config.energy_ceiling_kwh - state.energy_kwh) / config.eta_charge / config.dt_h,
    )
    discharge = min(
        config.max_discharge_kw,
        (state.energy_kwh - config.energy_floor_kwh) * config.eta_discharge / config.dt_h,
    )
    results = [step(config, state, Action(), StepInput(price))]
    for fraction in fractions:
        for power in (-charge * fraction, discharge * fraction):
            result = _feasible_result(config, state, power, price)
            if result is not None:
                results.append(result)

    # Include an action aimed at the exact reserve, independent of power grid.
    target = max(config.energy_floor_kwh, min(reserve, config.energy_ceiling_kwh))
    delta = state.energy_kwh - target
    power = delta * config.eta_discharge if delta >= 0 else delta / config.eta_charge
    power = max(-charge, min(discharge, power / config.dt_h))
    result = _feasible_result(config, state, power, price)
    if result is not None:
        # Rounding may land just below reserve. Bias toward stored energy, but
        # never accept a power/energy violation or move the simulator's bounds.
        for _ in range(64):
            if result.state.energy_kwh >= target:
                results.append(result)
                break
            power = nextafter(result.requested_action.p_kw, -float("inf"))
            adjusted = step(config, state, Action(power), StepInput(price))
            if adjusted.violations:
                break
            result = adjusted

    unique = {result.requested_action.p_kw: result for result in results}
    shortfall = min(max(0.0, reserve - result.state.energy_kwh) for result in unique.values())
    # Stable tie preference: hold, lower absolute power, then charge.
    return tuple(
        sorted(
            (
                result
                for result in unique.values()
                if max(0.0, reserve - result.state.energy_kwh) == shortfall
            ),
            key=lambda result: (abs(result.requested_action.p_kw), result.requested_action.p_kw),
        )
    )
