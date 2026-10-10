# Implementation Plan — Codex execution order

Work in small PR-sized increments. For each milestone write tests, run them, summarize outcomes, and update this plan.

## M0 — Repository foundation
**Status:** Complete (issue #1). Added src-layout packaging, development dependencies,
README quick start and Python 3.11/3.12 lint/test CI; retained the existing standard
`.gitignore`. Validated locally on Python 3.12; CI has not yet run remotely.

- Add `pyproject.toml` for Python 3.11+ with package under `src/`, pytest and Ruff.
- Add standard `.gitignore`, a focused README quick start, and test layout.
- Add CI running lint and tests when it becomes useful.
**Done when:** `python -m pytest` and `ruff check .` succeed locally, including at least one meaningful unit test.

## M1 — Deterministic battery simulator (FIRST IMPLEMENTATION TASK)
**Status:** Complete (issue #2). Immutable typed models and a pure step transition
implement signed grid power, distinct efficiencies, strict physical rejection,
requested/applied evidence and per-step accounting. Tests cover numerical
transitions, boundaries, validation, overflow, immutability and repeatability.
Validation: `python -m pytest` and `ruff check .` in the project virtual environment.
Reserve timelines and scenario/runner work remain deferred to M2.

- Typed `BatteryConfig`, `BatteryState`, `Action`, `StepInput`, `StepResult`.
- Explicit signed power convention and `dt_h`.
- Implement charge/discharge/hold with efficiencies, physical bounds, power limits and separate requested/applied actions.
- Default rejection of infeasible actions with violation reason, hold as applied action, never silently clamp.
- Per-step economic accounting and degradation proxy.
- Unit tests: energy conservation in both directions, efficiency, positive/negative prices, power and energy bounds, no energy creation, rejected actions, repeated-run determinism.
**Done when:** test suite asserts exact transitions within numerical tolerances and input validation rejects invalid parameters.

## M2 — Scenarios + runner
**Status:** Complete locally (issue #3), ready for PR review. Added immutable
version-1 hourly scenarios, strict JSON validation, per-publication forecast rows,
reserve timelines and a 64-bit seed. The typed controller protocol receives only
current/past published observations and an episode-local RNG. M1 owns physical
transitions; post-step reserve shortfalls are recorded separately. Full episode
JSON includes observations, requested/applied actions, states, energy flows,
accounting, violations and aggregate metrics. Import/replay recomputes all
recorded requests and rejects inconsistent evidence. A hand-authored 24-hour
fixture and hold-controller CLI demo are included.

Validation on Python 3.12.3:
- `.venv/bin/python -m pytest tests/test_scenarios.py`: 97 passed.
- `.venv/bin/python -m pytest tests/test_runner.py -q`: 22 passed.
- `.venv/bin/python -m pytest`: 266 passed, including all 147 existing tests.
- `.venv/bin/ruff check .`: all checks passed.
- `.venv/bin/python -O examples/run_hold.py --output /tmp/trialgrid-m2-hold-episode.json`:
  completed 24 hours, EUR 0.00 profit, four reserve misses, verified JSON replay.
- `git diff --check`: passed. Remote CI has not yet been verified for M2.

Limits: hourly scenario resolution; no physical disturbance model, failure
classification, or wall-clock run timestamps. Replay verifies internal evidence
consistency, not authenticity. Stateful controllers must be recreated/reset for
new runs. M3 and all later features remain deferred pending explicit approval.

- Validated versioned serializable scenarios, observed vs forecast prices, reserve requirement timeline, scenario seed.
- Common controller protocol and deterministic episode runner returning full traces and metrics.
- No lookahead: controller receives only permitted observations.
**Done when:** fixture can be serialized, reloaded and replayed with identical traces for deterministic controller.

## M3 — Baselines
- Heuristic dispatch policy to expose known weaknesses.
- Feasible optimization baseline for comparison, with stated information assumptions.
**Done when:** both run the same scenario, satisfy controller protocol, and have documented objectives.

## M4 — LLM controller
- Typed action tools/structured output, retries, call tracing, configurable model, safe action-validation boundary.
- Record responses for replay and compare cost.
**Done when:** one complete example episode can be audited/replayed without dependence on current model stochasticity.

## M5 — Failure predicates + scenario search
- Explicit economic, constraint and reserve failure functions.
- Fixed test suite and seeded random generator.
- Targeted evolutionary search with mutation/selection only after random baseline works.
**Done when:** equal-budget comparison can be run from one command and produces honest, reproducible summary.

## M6 — Shrinking and regression export
- Shrink disturbances while preserving predicate; export case JSON and pytest regression.
**Done when:** a detected failure reproduces from fixture and is flagged against the defective controller.

## M7 — Evidence report and demo
- API to run and replay cases; React plots for price, SoC, actions, commitments and violation events.
- Technical evaluation report and repeatable one-command demonstration.
**Done when:** a reviewer can reproduce a failure, inspect its cause, apply a fix and re-run regression tests.

## Stop/go gates

- After M1: physics and accounting verified, otherwise do not build AI layers.
- After M5: targeted search evaluated against random; if no advantage, document negative result and revisit search.
- After M7: validate actual buyer demand before expanding to other industries.

## First Codex prompt

> Read README.md, AGENTS.md, docs/PRD.md, docs/ARCHITECTURE.md and docs/IMPLEMENTATION_PLAN.md. Implement ONLY M0 and M1. Use Python 3.11+, a small src-layout package, pytest and Ruff. Implement the documented signed power convention, rejection policy, accounting and physically valid transitions. Include comprehensive numerical and boundary tests. Do not build an agent, dashboard or database. Run lint/tests and report exact commands and outcomes. Keep the diff reviewable.
