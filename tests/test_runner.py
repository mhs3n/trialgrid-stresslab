"""Episode accounting, information boundaries and auditable replay."""

import random
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import pytest

from trialgrid.domain import Action, BatteryConfig, BatteryState
from trialgrid.runner import Episode, HoldController, replay_episode, run_episode
from trialgrid.scenario import Scenario


def scenario():
    return Scenario(
        schema_version=1,
        seed=42,
        config=BatteryConfig(100, 10, 90, 20, 20, 0.8, 0.5, degradation_eur_per_kwh=0.1),
        initial_state=BatteryState(50),
        prices_eur_per_mwh=(100, 200, 300),
        forecast_prices_eur_per_mwh=((101, 202, 303), (204, 305), (306,)),
        reserve_requirements_kwh=(60, 50, 50),
    )


class Commands:
    def __init__(self, powers):
        self.powers = powers

    def act(self, observation, rng):
        return Action(self.powers[observation.current.step_index])


def test_complete_episode_accounting_rejection_and_post_step_reserves():
    episode = run_episode(scenario(), Commands((-10, 5, 30)))
    assert len(episode.trace) == 3
    assert [row.observation.current.state.energy_kwh for row in episode.trace] == [50, 58, 48]
    assert [row.result.state.energy_kwh for row in episode.trace] == [58, 48, 48]
    assert [row.result.requested_action.p_kw for row in episode.trace] == [-10, 5, 30]
    assert [row.result.applied_action.p_kw for row in episode.trace] == [-10, 5, 0]
    assert [row.reserve_shortfall_kwh for row in episode.trace] == [2, 2, 2]
    assert episode.trace[0].result.violations == ()
    assert episode.trace[0].violations == ("reserve_shortfall",)
    assert episode.trace[2].violations == (
        "discharge_power_limit",
        "energy_floor",
        "reserve_shortfall",
    )
    assert episode.trace[0].result.revenue_eur == -1
    assert episode.trace[1].result.revenue_eur == 1
    assert episode.metrics.revenue_eur == 0
    assert episode.metrics.degradation_cost_eur == 1.5
    assert episode.metrics.profit_eur == -1.5
    assert episode.metrics.rejected_action_count == 1
    assert episode.metrics.reserve_violation_count == 3


def test_rejected_action_does_not_end_episode_and_reserve_equality_passes():
    value = replace(scenario(), reserve_requirements_kwh=(50, 40, 40))
    episode = run_episode(value, Commands((30, 5, 0)))
    assert episode.trace[0].result.applied_action == Action()
    assert episode.trace[1].result.applied_action == Action(5)
    assert episode.trace[-1].result.state.energy_kwh == 40
    assert episode.metrics.reserve_violation_count == 0


def test_only_current_forecast_and_past_observations_are_published():
    value = scenario()
    episode = run_episode(value, HoldController())
    for index, row in enumerate(episode.trace):
        observation = row.observation
        assert observation.current.step_index == index
        assert observation.current.price_eur_per_mwh == value.prices_eur_per_mwh[index]
        assert (
            observation.current.forecast_prices_eur_per_mwh
            == (value.forecast_prices_eur_per_mwh[index])
        )
        assert observation.history == tuple(
            item.observation.current for item in episode.trace[:index]
        )
        assert {field.name for field in fields(observation)} == {"config", "current", "history"}


def test_future_realized_prices_reserves_and_unpublished_forecasts_do_not_leak():
    class Responsive:
        def act(self, observation, rng):
            return Action(-1 if observation.current.forecast_prices_eur_per_mwh[0] > 100 else 1)

    original = scenario()
    changed = replace(
        original,
        prices_eur_per_mwh=(100, 200, -999),
        forecast_prices_eur_per_mwh=((101, 202, 303), (204, 305), (-999,)),
        reserve_requirements_kwh=(60, 50, 90),
    )
    first = run_episode(original, Responsive())
    second = run_episode(changed, Responsive())
    assert first.trace[:2] == second.trace[:2]
    assert first.trace[2].observation != second.trace[2].observation


def test_observations_and_their_nested_state_are_immutable():
    observation = run_episode(scenario(), HoldController()).trace[1].observation
    with pytest.raises(FrozenInstanceError):
        observation.current.state.energy_kwh = 0
    with pytest.raises(FrozenInstanceError):
        observation.config.capacity_kwh = 0
    with pytest.raises(FrozenInstanceError):
        observation.current.price_eur_per_mwh = 0
    with pytest.raises(TypeError):
        observation.history[0] = observation.current
    with pytest.raises(TypeError):
        observation.current.forecast_prices_eur_per_mwh[0] = 0


def test_seeded_controller_is_repeatable_and_does_not_touch_global_random():
    class RandomController:
        def act(self, observation, rng):
            return Action(rng.uniform(-1, 1))

    value = scenario()
    global_state = random.getstate()
    first = run_episode(value, RandomController())
    second = run_episode(value, RandomController())
    assert random.getstate() == global_state
    assert first == second
    changed = run_episode(replace(value, seed=value.seed + 1), RandomController())
    assert first.trace[0].result.requested_action != changed.trace[0].result.requested_action
    assert replay_episode(first) == first


def test_json_roundtrip_and_recorded_replay_include_rejected_requests():
    value = Scenario.from_json(scenario().to_json())
    episode = run_episode(
        value, Commands((-10, 5, 30)), controller_name="commands", controller_version="1"
    )
    assert Episode.from_json(episode.to_json()) == episode
    assert Episode.from_dict(episode.to_dict()) == episode
    assert replay_episode(episode) == episode
    assert (
        run_episode(
            Scenario.from_json(value.to_json()),
            Commands((-10, 5, 30)),
            controller_name="commands",
            controller_version="1",
        )
        == episode
    )


@pytest.mark.parametrize(
    "field", ["state", "requested_action", "applied_action", "revenue_eur", "violations"]
)
def test_import_rejects_corrupt_result_evidence(field):
    data = run_episode(scenario(), Commands((-10, 5, 30))).to_dict()
    result = data["trace"][0]["result"]
    replacements = {
        "state": {"energy_kwh": 59},
        "requested_action": {"p_kw": -11},
        "applied_action": {"p_kw": 0},
        "revenue_eur": 999,
        "violations": ["invented"],
    }
    result[field] = replacements[field]
    with pytest.raises(ValueError):
        Episode.from_dict(data)


@pytest.mark.parametrize(
    "target", ["observation", "metrics", "shortfall", "trace", "version", "simulator", "boolean"]
)
def test_import_rejects_corrupt_trace_metadata_and_metrics(target):
    data = run_episode(scenario(), HoldController()).to_dict()
    if target == "observation":
        data["trace"][0]["observation"]["current"]["price_eur_per_mwh"] = 9
    elif target == "metrics":
        data["metrics"]["profit_eur"] = 9
    elif target == "shortfall":
        data["trace"][0]["reserve_shortfall_kwh"] = 9
    elif target == "trace":
        data["trace"].pop()
    elif target == "version":
        data["schema_version"] = 2
    elif target == "simulator":
        data["simulator_version"] = "future"
    else:
        data["metrics"]["revenue_eur"] = False
    with pytest.raises(ValueError):
        Episode.from_dict(data)


def test_wrong_controller_action_fails_explicitly():
    class WrongController:
        def act(self, observation, rng):
            return 0

    with pytest.raises(ValueError, match="Action"):
        run_episode(scenario(), WrongController())


def test_episode_rejects_aggregate_overflow_even_when_steps_are_finite():
    value = Scenario(
        1,
        0,
        BatteryConfig(1e6, 0, 1e6, 1000, 1000),
        BatteryState(10000),
        (1e308, 1e308),
        ((), ()),
        (0, 0),
    )
    with pytest.raises(ValueError, match="finite"):
        run_episode(value, Commands((1000, 1000)))


def test_sample_24_hour_scenario_runs_and_replays():
    path = Path(__file__).resolve().parents[1] / "examples/scenarios/hold_24h.json"
    value = Scenario.from_json(path.read_text())
    episode = run_episode(value, HoldController())
    assert len(episode.trace) == 24
    assert all(row.result.state == value.initial_state for row in episode.trace)
    assert all(row.result.applied_action == Action() for row in episode.trace)
    assert replay_episode(Episode.from_json(episode.to_json())) == episode
