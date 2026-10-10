"""Receding-horizon optimization over feasible actions and published forecasts."""

from dataclasses import dataclass
from functools import cache
from random import Random

from trialgrid.controllers._planning import BaselineConfig, candidates, planning_prices
from trialgrid.domain import Action, BatteryState, _finite
from trialgrid.runner import Observation


@dataclass(frozen=True, slots=True)
class ForecastPlan:
    """A forecast projection, never a realized result or an oracle upper bound."""

    actions: tuple[Action, ...]
    forecast_profit_eur: float


@dataclass(frozen=True, slots=True)
class ForecastOptimizer:
    """Maximize forecast net profit over a bounded, state-dependent action tree.

    Exact M1 transitions; approximate relative to continuous-action optimization.
    Current reserve is prioritized, future obligations are unknown, and terminal
    stored energy has zero value. Only the first action is executed, then the
    next observation triggers a new plan. No scenario or future realized inputs
    are accepted. Ties prefer the first lower-power action at each step.
    """

    config: BaselineConfig = BaselineConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.config, BaselineConfig):
            raise ValueError("config must be a BaselineConfig")

    def plan(self, observation: Observation) -> ForecastPlan:
        prices = planning_prices(observation, self.config)

        @cache
        def solve(index: int, energy: float) -> ForecastPlan:
            if index == len(prices):
                return ForecastPlan((), 0.0)
            options = candidates(
                observation.config,
                BatteryState(energy),
                prices[index],
                observation.current.reserve_requirement_kwh if index == 0 else 0.0,
                self.config.power_fractions,
            )
            best: ForecastPlan | None = None
            for result in options:
                continuation = solve(index + 1, result.state.energy_kwh)
                profit = result.profit_eur + continuation.forecast_profit_eur
                _finite("forecast plan profit", profit)
                if best is None or profit > best.forecast_profit_eur:
                    best = ForecastPlan(
                        (result.requested_action,) + continuation.actions,
                        profit,
                    )
            # candidates always includes at least one physically feasible option.
            if best is None:
                raise ValueError("no feasible planning action")
            return best

        return solve(0, observation.current.state.energy_kwh)

    def act(self, observation: Observation, rng: Random) -> Action:
        return self.plan(observation).actions[0]
