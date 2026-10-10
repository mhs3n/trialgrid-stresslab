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

## Day-one goal

Implement the pure Python BESS simulator with deterministic tests. No LLM, frontend, database server, or cloud infrastructure is required for this milestone.

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

## Status

M0/M1 provide the project foundation and deterministic single-step battery core.
Scenario serialization, reserve timelines, episode runners and controllers begin
in M2 and later milestones. No experimental results have been produced yet.
