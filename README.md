# TrialGrid StressLab

**Experimental adversarial testing for AI controllers in simulated industrial systems.**

TrialGrid explores operating scenarios, identifies controller failures, shrinks counterexamples, and exports replayable regression tests. The first proving ground is a simulated battery energy storage system (BESS).

## Core hypothesis

Under an equal simulation budget, targeted scenario search can discover **distinct, reproducible, operationally meaningful failures** that a fixed suite or uniform random testing misses. This is a hypothesis to test, not a claimed result.

## MVP boundaries

- Deterministic battery simulation, typed scenario definition, physical constraints and economic accounting.
- A simple heuristic controller, an optimization baseline, and later a tool-using LLM controller.
- Fixed scenarios, random exploration, targeted/evolutionary search, counterexample shrinking.
- Stored traces, replay, regression tests, and honest comparison using equal experiment budgets.
- Thin API and interactive replay UI after the experiment works.

**Not in MVP:** live equipment connectivity, grid dispatch, safety certification, multi-asset optimization, a generic agent framework, or claims of guaranteed safe operation.

## Build order

See [Implementation Plan](docs/IMPLEMENTATION_PLAN.md), [Product Requirements](docs/PRD.md), [Architecture](docs/ARCHITECTURE.md), and [Codex Instructions](AGENTS.md).

## Current milestone

M3 adds a heuristic and a forecast-only optimization baseline on the same
observation interface. Controller review comes before M4/LLM integration. No LLM,
frontend, database server, or cloud infrastructure is included.

## Quick start

Requires Python 3.11 or newer. The simulation has no runtime dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
ruff check .
```

```python
from trialgrid import Action, BatteryConfig, BatteryState, StepInput, step

config = BatteryConfig(
    capacity_kwh=100,
    energy_floor_kwh=10,
    energy_ceiling_kwh=90,
    max_charge_kw=20,
    max_discharge_kw=20,
    eta_charge=0.9,
    eta_discharge=0.8,
    dt_h=0.5,
    degradation_eur_per_kwh=0.01,
)
result = step(config, BatteryState(50), Action(16), StepInput(100))
assert result.state.energy_kwh == 40
assert result.applied_action == Action(16)
assert result.violations == ()
```

Power is grid-side kW: positive discharges, negative charges, and zero holds.
Energy is kWh; prices are EUR/MWh. Degradation is charged on absolute grid-side
throughput, not cell-side throughput. Inputs are immutable and each call returns
a new result. Infeasible finite commands apply hold, retain the requested action,
and report all violated power/energy limits; they earn no revenue or degradation
cost. Malformed inputs and invalid starting states raise `ValueError`.

Physical comparisons are strict, with no clipping or tolerance-based acceptance.
Leave numerical headroom when computing boundary commands in floating point.
Nonfinite input values are invalid; arithmetic that cannot produce finite
accounting raises `ValueError`.

## Run a complete episode

After installing the project with the quick start above:

```bash
python examples/run_hold.py --output /tmp/hold_episode.json
```

The sample in `examples/scenarios/hold_24h.json` is an illustrative, hand-authored
24-hour fixture, not market data or an experimental result. Hold keeps 50 kWh,
earns zero revenue, incurs zero degradation cost, and misses the four 60 kWh
reserve commitments. The demo exports the full trace and verifies action replay.

```python
from pathlib import Path
from trialgrid import Episode, HoldController, Scenario, replay_episode, run_episode

scenario = Scenario.from_json(Path("examples/scenarios/hold_24h.json").read_text())
episode = run_episode(
    scenario, HoldController(), controller_name="hold", controller_version="1"
)
restored = Episode.from_json(episode.to_json())
assert restored == replay_episode(episode) == episode
assert Scenario.from_json(scenario.to_json()) == scenario
```

### Scenario and observation contract

Schema version 1 requires a battery configuration, initial state, unsigned 64-bit
integer seed, and three equal-length, nonempty hourly timelines:

- `prices_eur_per_mwh`: realized prices, including negative prices.
- `forecast_prices_eur_per_mwh`: one forecast row published at the start of each
  hour. Row `t` starts with a prediction for hour `t`, followed by predictions
  for subsequent hours, up to the remaining horizon. Empty rows mean no forecast.
- `reserve_requirements_kwh`: minimum post-step stored energy, between zero and
  capacity. Requirements above the usable energy ceiling are allowed to describe
  impossible commitments; the runner does not classify their preventability.

Scenario `dt_h` must be exactly 1; M1's standalone `step` still supports other
positive durations. JSON import rejects unsupported versions, missing/unknown
fields, duplicate keys, nonfinite values, invalid seeds, and inconsistent timelines.

A controller implements `act(observation: Observation, rng: random.Random) -> Action`.
`observation.current` contains the hour index, starting state, current realized
price, current reserve requirement, and only the forecast published this hour.
`observation.history` contains prior immutable observation frames; `config` supplies
static battery limits. Future realized prices, future reserve requirements, future
forecast publications, the scenario, and the seed are not passed to the controller.
Forecast data is supplied explicitly, never synthesized from future actual prices.
There is no separate physical disturbance model in M2. This is an information
boundary for cooperative controllers, not a sandbox for hostile Python code.

Each run creates its own RNG from the scenario seed, leaving global randomness
untouched. Use the supplied RNG for repeatability, and construct/reset stateful
controllers for each new episode. Recorded-action replay needs no live controller.

### Evidence and reserves

Every step records its observation (including prior observations), starting state,
requested/applied action, ending state, physical violations, energy flows, revenue,
degradation cost, profit, and reserve shortfall. Physical violations retain M1's
hold-on-rejection policy. Reserve shortfalls are checked **after** the transition,
recorded as `reserve_shortfall`, and never change state or override an action.
Episode metrics sum revenue, degradation cost, and profit and count rejected
actions and reserve misses. Nonfinite aggregate accounting raises `ValueError`.

Episode JSON contains the full scenario/seed, controller name/version supplied by
the caller, episode schema version, and simulator version. No wall-clock timestamp
is included, so identical runs export identical JSON. Import/replay recomputes the
full trace from recorded **requested** actions and rejects inconsistent evidence;
it checks internal consistency, not artifact authenticity. Malformed controller
returns raise an error rather than silently becoming hold actions.

## Compare the M3 baselines

```bash
python examples/compare_baselines.py --output-dir /tmp/trialgrid-baselines
```

This runs `HeuristicController` and `ForecastOptimizer` on the same serialized
scenario, using the same forecast horizon and action fractions. Both get only
M2 observations; neither sees future realized prices, future reserve requirements,
or later forecast publications. Forecast index zero is replaced by the known
current price. Empty forecasts produce a current-hour decision.

```python
from trialgrid.controllers import BaselineConfig, ForecastOptimizer, HeuristicController

settings = BaselineConfig(horizon=4, power_fractions=(0.5, 1.0))
heuristic = HeuristicController(settings)
optimizer = ForecastOptimizer(settings)
```

The heuristic values stored energy using the mean published future price. The
optimizer searches feasible action sequences, executes only the first action,
and replans next hour. It is a **finite-action forecast approximation**, with zero
terminal energy value, not an oracle or a continuous optimum. Bad forecasts can
make it perform worse than the heuristic. Both prioritize achievable current
reserves; future commitments are unknown. There is no oracle controller in M3.

The demo exports verified episode JSON plus `comparison.json` containing the
shared configuration, information assumptions, realized metrics, and final stored
energy. Keep this manifest with the episodes to reproduce the controller settings;
the M2 episode schema remains unchanged. Results on the hand-authored fixture
illustrate behavior only, not measured superiority across a scenario population.
See [baseline contracts](docs/BASELINES.md) for objectives, limits, and ties, and
[M3 review](docs/M3_REVIEW.md) for the checks performed before any LLM work.

## Status

M0–M3 provide the project foundation, deterministic battery core, versioned
scenarios, episode runner/replay, and reviewed heuristic/forecast optimization
baselines. M4/LLM work awaits explicit approval after review. Search and
API/frontend remain later milestones; no population-level evaluation exists yet.
