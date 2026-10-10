"""Pure deterministic transitions; infeasible commands are rejected, never clipped."""

from math import isfinite

from trialgrid.domain import Action, BatteryConfig, BatteryState, StepInput, StepResult


def step(
    config: BatteryConfig, state: BatteryState, action: Action, step_input: StepInput
) -> StepResult:
    """Apply one command without mutating inputs.

    Comparisons are strict: even a small bound exceedance is rejected. Callers
    should leave numerical headroom when deriving boundary commands. Invalid
    initial state raises ValueError; infeasible finite requests produce a trace.
    Arithmetic overflow raises ValueError rather than returning nonfinite results.
    """
    if not config.energy_floor_kwh <= state.energy_kwh <= config.energy_ceiling_kwh:
        raise ValueError("initial energy must be within configured bounds")

    violations: list[str] = []
    if action.p_kw > config.max_discharge_kw:
        violations.append("discharge_power_limit")
    if -action.p_kw > config.max_charge_kw:
        violations.append("charge_power_limit")

    # Convert before multiplication so large, valid integer inputs follow the
    # same overflow/rejection path as floats instead of creating unbounded ints.
    grid_kwh = float(action.p_kw) * config.dt_h
    energy_out = max(grid_kwh, 0.0) / config.eta_discharge
    energy_in = max(-grid_kwh, 0.0) * config.eta_charge
    next_energy = state.energy_kwh - energy_out + energy_in
    if next_energy < config.energy_floor_kwh:
        violations.append("energy_floor")
    if next_energy > config.energy_ceiling_kwh:
        violations.append("energy_ceiling")

    applied = Action() if violations else action
    if violations:
        grid_kwh = energy_in = energy_out = 0.0
        next_energy = state.energy_kwh
    revenue = (grid_kwh / 1000.0) * step_input.price_eur_per_mwh
    cost = abs(grid_kwh) * config.degradation_eur_per_kwh
    profit = revenue - cost
    if not all(isfinite(value) for value in (next_energy, revenue, cost, profit)):
        raise ValueError("step arithmetic exceeds finite numeric range")
    return StepResult(
        state=BatteryState(next_energy),
        requested_action=action,
        applied_action=applied,
        violations=tuple(violations),
        energy_in_kwh=energy_in,
        energy_out_kwh=energy_out,
        revenue_eur=revenue,
        degradation_cost_eur=cost,
        profit_eur=profit,
    )
