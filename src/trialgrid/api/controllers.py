"""Small allowlist of offline controllers available to the local M7 API."""

from dataclasses import asdict

from trialgrid.controllers import BaselineConfig, ForecastOptimizer, HeuristicController
from trialgrid.controllers.llm import canonical
from trialgrid.runner import Controller, HoldController

CONTROLLERS = ("hold", "heuristic", "forecast_optimizer")
CONTROLLER_VERSIONS = {
    "hold": "m2-hold-v1",
    "heuristic": "m3-heuristic-v1",
    "forecast_optimizer": "m3-forecast-optimizer-v1",
}


def build_controller(name: str, horizon: int) -> tuple[Controller, str]:
    if name not in CONTROLLERS:
        raise ValueError("unsupported controller")
    if type(horizon) is not int or not 1 <= horizon <= 6:
        raise ValueError("horizon must be an integer in [1, 6]")
    if name == "hold":
        controller = HoldController()
        configuration = {"horizon": horizon}
    else:
        settings = BaselineConfig(horizon=horizon)
        configuration = asdict(settings)
        controller = (
            HeuristicController(settings)
            if name == "heuristic"
            else ForecastOptimizer(settings)
        )
    metadata = canonical(
        {
            "name": name,
            "version": CONTROLLER_VERSIONS[name],
            "horizon": horizon,
            "configuration": configuration,
            "information_mode": "published_forecast_only",
        }
    )
    return controller, metadata
