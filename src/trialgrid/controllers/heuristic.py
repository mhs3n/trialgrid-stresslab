"""A one-step forecast-value heuristic, deliberately simpler than horizon search."""

from dataclasses import dataclass
from math import fsum
from random import Random

from trialgrid.controllers._planning import BaselineConfig, candidates, planning_prices
from trialgrid.domain import Action, _finite
from trialgrid.runner import Observation


@dataclass(frozen=True, slots=True)
class HeuristicController:
    """Score immediate profit plus the change in energy at mean forecast value.

    Stored-energy value is the nonnegative mean future sale price, net of
    discharge degradation and efficiency. No future forecast means zero value.
    This ignores timing, future charge opportunities, and future power limits.
    Current achievable reserve has priority over this economic score.
    """

    config: BaselineConfig = BaselineConfig()

    def __post_init__(self) -> None:
        if not isinstance(self.config, BaselineConfig):
            raise ValueError("config must be a BaselineConfig")

    def act(self, observation: Observation, rng: Random) -> Action:
        prices = planning_prices(observation, self.config)
        battery = observation.config
        future = prices[1:]
        reference = fsum(price / len(future) for price in future) / 1000 if future else 0.0
        value = max(0.0, reference - battery.degradation_eur_per_kwh) * battery.eta_discharge
        options = candidates(
            battery,
            observation.current.state,
            prices[0],
            observation.current.reserve_requirement_kwh,
            self.config.power_fractions,
        )
        best_action = options[0].requested_action
        best_score = -float("inf")
        for result in options:
            score = (
                result.profit_eur
                + (result.state.energy_kwh - observation.current.state.energy_kwh) * value
            )
            _finite("heuristic score", score)
            if score > best_score:
                best_action, best_score = result.requested_action, score
        return best_action
