"""Hourly episodes with a narrow observation boundary and verified action replay."""

import json
from dataclasses import asdict, dataclass
from random import Random
from typing import Protocol

from trialgrid.domain import Action, BatteryConfig, BatteryState, StepInput, StepResult, _finite
from trialgrid.scenario import Scenario, load_json, require_keys
from trialgrid.simulator import step

SIMULATOR_VERSION = "m1-v1"


@dataclass(frozen=True, slots=True)
class ObservationFrame:
    """Information published at the start of one hour, before its action.

    Forecast index zero predicts this hour. Reserves are post-step minimum
    energy commitments; future reserve requirements are not published here.
    """

    step_index: int
    state: BatteryState
    price_eur_per_mwh: float
    forecast_prices_eur_per_mwh: tuple[float, ...]
    reserve_requirement_kwh: float


@dataclass(frozen=True, slots=True)
class Observation:
    config: BatteryConfig
    current: ObservationFrame
    history: tuple[ObservationFrame, ...]


class Controller(Protocol):
    """Use only observation and this episode's seeded RNG to choose an action.

    Stateful controllers must be freshly constructed/reset before each run.
    The interface prevents accidental lookahead, not hostile Python inspection.
    """

    def act(self, observation: Observation, rng: Random) -> Action: ...


class HoldController:
    def act(self, observation: Observation, rng: Random) -> Action:
        return Action()


@dataclass(frozen=True, slots=True)
class StepTrace:
    observation: Observation
    result: StepResult
    reserve_shortfall_kwh: float
    violations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    revenue_eur: float
    degradation_cost_eur: float
    profit_eur: float
    rejected_action_count: int
    reserve_violation_count: int


@dataclass(frozen=True, slots=True)
class Episode:
    scenario: Scenario
    controller_name: str
    controller_version: str
    trace: tuple[StepTrace, ...]
    metrics: EpisodeMetrics
    schema_version: int = 1
    simulator_version: str = SIMULATOR_VERSION

    def to_dict(self) -> dict:
        # Convert immutable tuples into ordinary JSON arrays, with strict numbers.
        return json.loads(self.to_json())

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Episode":
        return cls.from_dict(load_json(text))

    @classmethod
    def from_dict(cls, value: dict) -> "Episode":
        """Import evidence only if replay reproduces every observation and outcome."""
        data = require_keys(
            value,
            {
                "scenario",
                "controller_name",
                "controller_version",
                "trace",
                "metrics",
                "schema_version",
                "simulator_version",
            },
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported episode schema_version")
        if data["simulator_version"] != SIMULATOR_VERSION:
            raise ValueError("unsupported simulator_version")
        scenario = Scenario.from_dict(data["scenario"])
        trace = data["trace"]
        if not isinstance(trace, list) or len(trace) != len(scenario.prices_eur_per_mwh):
            raise ValueError("trace length must match scenario horizon")
        actions = []
        for item in trace:
            if not isinstance(item, dict) or not isinstance(item.get("result"), dict):
                raise ValueError("trace must contain step results")
            action = require_keys(item["result"].get("requested_action"), {"p_kw"})
            actions.append(Action(action["p_kw"]))
        episode = run_episode(
            scenario,
            _RecordedController(tuple(actions)),
            controller_name=data["controller_name"],
            controller_version=data["controller_version"],
        )
        if not _matches_json(data, episode.to_dict()):
            raise ValueError("episode evidence does not match deterministic replay")
        return episode


def _matches_json(actual: object, expected: object) -> bool:
    """Compare strict JSON shapes, rejecting bools masquerading as numeric evidence."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_matches_json(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches_json(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if type(expected) in (int, float):
        return type(actual) in (int, float) and actual == expected
    return type(actual) is type(expected) and actual == expected


class _RecordedController:
    def __init__(self, actions: tuple[Action, ...]) -> None:
        self.actions = actions

    def act(self, observation: Observation, rng: Random) -> Action:
        return self.actions[observation.current.step_index]


def replay_episode(episode: Episode) -> Episode:
    """Recompute from recorded requests; fail if any stored evidence was altered."""
    return Episode.from_dict(episode.to_dict())


def run_episode(
    scenario: Scenario,
    controller: Controller,
    *,
    controller_name: str = "unspecified",
    controller_version: str = "unspecified",
) -> Episode:
    """Execute all hours; reserve misses are evidence, never physical corrections.

    Only current realized inputs and the current published forecast reach the
    controller. No future disturbances are modeled in M2. Randomness is local
    to each episode; the global random generator is never seeded or consumed.
    """
    if not isinstance(scenario, Scenario):
        raise ValueError("scenario must be a Scenario")
    for name, value in (
        ("controller_name", controller_name),
        ("controller_version", controller_version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    rng = Random(scenario.seed)
    state = scenario.initial_state
    history: list[ObservationFrame] = []
    trace: list[StepTrace] = []
    revenue = cost = profit = 0.0
    rejected = reserve_misses = 0
    for index, price in enumerate(scenario.prices_eur_per_mwh):
        current = ObservationFrame(
            index,
            state,
            price,
            scenario.forecast_prices_eur_per_mwh[index],
            scenario.reserve_requirements_kwh[index],
        )
        observation = Observation(scenario.config, current, tuple(history))
        action = controller.act(observation, rng)
        if not isinstance(action, Action):
            raise ValueError("controller must return an Action")
        result = step(scenario.config, state, action, StepInput(price))
        shortfall = max(0.0, current.reserve_requirement_kwh - result.state.energy_kwh)
        violations = result.violations + (("reserve_shortfall",) if shortfall else ())
        trace.append(StepTrace(observation, result, shortfall, violations))
        revenue += result.revenue_eur
        cost += result.degradation_cost_eur
        profit += result.profit_eur
        for name, value in (
            ("revenue_eur", revenue),
            ("degradation_cost_eur", cost),
            ("profit_eur", profit),
        ):
            _finite(f"episode {name}", value)
        rejected += bool(result.violations)
        reserve_misses += bool(shortfall)
        history.append(current)
        state = result.state
    return Episode(
        scenario,
        controller_name,
        controller_version,
        tuple(trace),
        EpisodeMetrics(revenue, cost, profit, rejected, reserve_misses),
    )
