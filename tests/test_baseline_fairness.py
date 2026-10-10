"""Metamorphic information-boundary tests against both real M3 controllers."""

from dataclasses import replace

import pytest

from trialgrid import BatteryConfig, BatteryState, Episode, Scenario, replay_episode, run_episode
from trialgrid.controllers import BaselineConfig, ForecastOptimizer, HeuristicController

CONTROLLERS = (HeuristicController, ForecastOptimizer)


def market() -> Scenario:
    return Scenario(
        1,
        42,
        BatteryConfig(10, 0, 10, 3, 3, 0.9, 0.8, degradation_eur_per_kwh=0.005),
        BatteryState(4),
        (30, 80, -20, 100),
        ((999, 70, 10, 60), (999, 50, 100), (999, 70), (999,)),
        (2, 3, 4, 1),
    )


@pytest.mark.parametrize("controller_type", CONTROLLERS)
@pytest.mark.parametrize("boundary", (1, 2, 3))
def test_hidden_future_changes_cannot_change_earlier_decisions(controller_type, boundary):
    original = market()
    altered = replace(
        original,
        prices_eur_per_mwh=original.prices_eur_per_mwh[:boundary] + (10000,) * (4 - boundary),
        reserve_requirements_kwh=(
            original.reserve_requirements_kwh[:boundary] + (10,) * (4 - boundary)
        ),
        forecast_prices_eur_per_mwh=(
            original.forecast_prices_eur_per_mwh[:boundary]
            + tuple((-10000,) * (4 - i) for i in range(boundary, 4))
        ),
    )
    first = run_episode(original, controller_type())
    second = run_episode(altered, controller_type())
    # Full prefix equality includes requests, applied actions, observations, and economics.
    assert first.trace[:boundary] == second.trace[:boundary]


@pytest.mark.parametrize("controller_type", CONTROLLERS)
def test_published_forecast_can_change_action_but_realized_future_cannot(controller_type):
    original = Scenario(
        1,
        0,
        BatteryConfig(1, 0, 1, 1, 1),
        BatteryState(0),
        (50, -500),
        ((12345, 200), (12345,)),
        (0, 0),
    )
    wrong_forecast = run_episode(original, controller_type())
    changed_actual = run_episode(
        replace(original, prices_eur_per_mwh=(50, 500)),
        controller_type(),
    )
    changed_published = run_episode(
        replace(original, forecast_prices_eur_per_mwh=((12345, 20), (12345,))),
        controller_type(),
    )
    assert wrong_forecast.trace[0] == changed_actual.trace[0]
    assert wrong_forecast.trace[0].result.requested_action.p_kw == -1
    assert changed_published.trace[0].result.requested_action.p_kw == 0
    assert wrong_forecast.metrics.profit_eur < 0  # No concealed perfect-price fallback.


@pytest.mark.parametrize("controller_type", CONTROLLERS)
def test_serialized_scenario_and_action_replay_reproduce_baseline(controller_type):
    original = market()
    baseline = controller_type(BaselineConfig(horizon=3))
    first = run_episode(original, baseline, controller_name=controller_type.__name__)
    restored = run_episode(
        Scenario.from_json(original.to_json()),
        baseline,
        controller_name=controller_type.__name__,
    )
    assert first.to_json() == restored.to_json()
    assert Episode.from_json(first.to_json()) == replay_episode(first) == first
    assert first.metrics.rejected_action_count == 0


def test_comparison_publishes_identical_market_information_to_both_controllers():
    first = run_episode(market(), HeuristicController())
    second = run_episode(market(), ForecastOptimizer())
    for a, b in zip(first.trace, second.trace, strict=True):
        assert a.observation.config == b.observation.config
        # Policy choices may cause battery state to diverge; published market data cannot.
        assert replace(a.observation.current, state=BatteryState(0)) == replace(
            b.observation.current,
            state=BatteryState(0),
        )
