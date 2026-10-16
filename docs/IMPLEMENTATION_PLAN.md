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
**Status:** Complete (issue #3); merged in PR #9. Added immutable
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
new runs. M3 and M4 were subsequently authorized; M5 and later features remain deferred.

- Validated versioned serializable scenarios, observed vs forecast prices, reserve requirement timeline, scenario seed.
- Common controller protocol and deterministic episode runner returning full traces and metrics.
- No lookahead: controller receives only permitted observations.
**Done when:** fixture can be serialized, reloaded and replayed with identical traces for deterministic controller.

## M3 — Baselines
**Status:** Complete (issue #4); merged in PR #10. Separate
incremental commits add the heuristic, forecast-only optimizer, and comparison/
review evidence. The existing M2 runner, observation contract and schema are unchanged.

- `HeuristicController`: immediate net profit plus mean-forecast stored-energy value.
- `ForecastOptimizer`: receding-horizon finite-action search, using current actual
  price and only the current published forecast; no oracle or realized-price lookahead.
- Shared configuration and M1-checked candidates, current-reserve priority, explicit
  unknown future reserves, bounded forecast horizon and zero optimizer terminal value.
- Same-scenario comparison exports replay-verified episodes and a manifest containing
  controller settings, information assumptions, realized metrics and final energy.
- Review: no outstanding actionable correctness/fairness findings; see
  `docs/BASELINES.md` and `docs/M3_REVIEW.md` for limitations and review evidence.

Validation on Python 3.12.3:
- `.venv/bin/python -m pytest tests/test_heuristic.py -q`: 33 passed.
- `.venv/bin/python -m pytest tests/test_optimizer.py -q`: 67 passed.
- `.venv/bin/python -m pytest tests/test_baseline_fairness.py -q`: 11 passed.
- `.venv/bin/python -m pytest`: 377 passed, retaining all 266 M2 tests.
- `.venv/bin/ruff check .`: all checks passed; `git diff --check`: passed.
- `.venv/bin/python examples/compare_baselines.py --output-dir /tmp/trialgrid-m3-baselines`:
  both sample episodes completed with verified replay, no rejected actions or reserve
  misses. Demo profits: heuristic EUR 5.14, optimizer EUR 4.37; both ended at 40 kWh.
  These illustrative outputs do not establish superiority. Remote CI not yet verified.

**Review gate:** M3 was reviewed and merged before M4 was explicitly authorized.

- Heuristic dispatch policy to expose known weaknesses.
- Feasible optimization baseline for comparison, with stated information assumptions.
**Done when:** both run the same scenario, satisfy controller protocol, and have documented objectives.

## M4 — LLM controller
**Status:** Complete (issue #5); merged in PR #11.
Strict Responses dispatch tools implement the unchanged M2 controller protocol.
A configurable 1–6 hour forecast cap and whitelisted prompt fields preserve M3's
information boundary; historical forecasts cannot bypass the cap. Finite infeasible
requests remain agent decisions and are passed unchanged to M1.

Immutable attempt/decision records capture exact prompts/observations, model and
prompt versions, raw responses, usage, latency, requested/returned actions, errors
and retries. Versioned companion JSON replays responses/errors through the same
parser and simulator without API calls. Default exhausted-call behavior raises;
optional fallback hold is labeled separately. Optional OpenAI SDK transport disables
SDK retries, sanitizes errors and remains absent from the core runtime dependencies.
The fake demo is deterministic; real model use requires explicit mode/environment.

Validation on Python 3.12.3:
- `.venv/bin/python -m pytest tests/test_llm.py tests/test_llm_audit.py -q`: 89 passed.
- `.venv/bin/python -m pytest tests/test_llm_models.py -q`: 38 passed.
- `.venv/bin/python -m pytest tests/test_llm_sdk.py -q`: 6 passed with OpenAI SDK 2.54.0,
  using mocked HTTP only. CI retains dependency-free fake tests and a separate SDK smoke job.
- Pytest with `sys.modules["openai"] = None`: 504 passed, SDK module skipped; no optional
  SDK is required for fake/controller/audit coverage.
- `.venv/bin/python -m pytest`: 510 passed, retaining all 377 M3 tests.
- `.venv/bin/ruff check .`: passed; `git diff --check`: passed.
- `python examples/run_llm.py` and its `--replay` mode, via `.venv/bin/python` with
  output directories under `/tmp`: 24 hours, 50 kWh ending energy, EUR 0 profit,
  four reserve misses, zero rejected actions. Synthetic usage/latency/cost are zero;
  this is fixture verification, not a real LLM performance result.

Read-only review found and resolved malformed usage metadata being treated as known
billing, and malformed successful HTTP responses bypassing audit/retry. Regressions
cover both. Tests also cover malformed dispatches, API/retry/fallback distinctions,
no-lookahead, prompt caps, audit corruption, unknown costs, and replay without SDK calls.

Limits: no live model request was made; provider behavior/performance remains untested.
Costs are caller-priced USD estimates, separate from EUR profit, and unknown without
complete billing evidence. Aborted runs keep call records but not a complete physical
trace/economic summary. Replay verifies consistency, not authenticity. See
`docs/LLM_CONTROLLER.md`. Remote CI has not yet been verified for that milestone.
M5 was subsequently merged and the M7 viewer was separately authorized.

- Typed action tools/structured output, retries, call tracing, configurable model, safe action-validation boundary.
- Record responses for replay and compare cost.
**Done when:** one complete example episode can be audited/replayed without dependence on current model stochasticity.

## M5 — Failure predicates + scenario search
**Status:** Complete (issue #6); merged in PR #12. Incremental commits first
validated fixed/random evaluation, then introduced charged targeted mutation.

Added predeclared invalid-action, classified reserve-shortfall and conservative
economic-dominance predicates; bounded independent forecast/actual generators;
three fixed fixtures; seeded random and elitist mutation with exploration. A
context-local M1 hook counts actual calls including planning and witnesses, with
hard caps and explicit incomplete final candidates. Search preserves every case,
controller/model failure attribution and replay evidence. Verification reconstructs
episodes, LLM responses, generated proposals, findings, lineage and summaries.

Validation on Python 3.12.3:
- `.venv/bin/python -m pytest`: 631 passed, retaining all 510 M4 tests.
- `.venv/bin/ruff check .` and `git diff --check`: passed.
- `examples/compare_search.py --calls 20000 --seeds 11 22 33`, run via the venv:
  all three methods used exactly 20,000 calls per seed, with three completed
  evaluations each and zero actionable heuristic cases. No targeted advantage found.
- Same command with `--target fake-llm --seeds 11`: fixed one unique actionable
  environment, random three, targeted three; synthetic holds, no provider calls.

Read-only review findings on horizon enforcement, budget-boundary model calls,
report deletion, fallback attribution and final failed-dispatch replay were fixed
with regression tests. See `docs/SEARCH.md` for the exact criteria/budget definition,
result limits and negative finding. Remote CI passed after merge.

**Stop/go gate:** No search advantage established in this small synthetic check;
revisit declared domains/criteria and larger held-out evaluation before changing
the search method or making comparative claims.

- Explicit economic, constraint and reserve failure functions.
- Fixed test suite and seeded random generator.
- Targeted evolutionary search with mutation/selection only after random baseline works.
**Done when:** equal-budget comparison can be run from one command and produces honest, reproducible summary.

## M6 — Shrinking and regression export
**Status:** Complete. Added bounded deterministic shrinking over initial energy,
realized prices, published forecasts and reserve requirements. Candidates preserve
the original seed and timeline shape and are accepted only when the same actionable
finding kind/classification signature reproduces. Repeated verification is available
for stochastic controllers. Added version-1 JSON regression fixtures and generated
offline pytest modules for the allowlisted hold, heuristic and forecast optimizer
controllers; live LLM/controller factories are rejected during export.

Validation:
- `.venv/bin/python -m pytest tests/test_shrink.py -q`: 4 passed.
- `.venv/bin/python -m pytest`: 652 passed, including all 648 M7 tests.
- `.venv/bin/ruff check .` and `git diff --check`: passed.

- Shrink disturbances while preserving predicate; export case JSON and pytest regression.
**Done when:** a detected failure reproduces from fixture and is flagged against the defective controller.

## M7 — Evidence report and demo
**Status:** Complete (merged in PR #13). The evidence viewer/API now sits above the
replayable M6 cases; M6 regression export is available as a separate library path.

- Versioned content-addressed target/comparator artifacts replay both episodes and
  recompute the failure report before exposing metrics.
- Local FastAPI service lists, runs, replays and reruns cases with an allowlist of
  offline controllers. It uses one configured artifact root and never constructs a
  live model transport.
- React/TypeScript viewer compares metrics and plots actual prices, published
  forecasts, energy, reserves, requested/applied actions and violation events.
- Deterministic one-command fixed-case demo and `docs/EVIDENCE_REPORT.md` retain the
  M5 negative result and label the forecast optimizer as a non-oracle approximation.
- No authentication, database, or cloud deployment.

Validation on Python 3.12.3 and Node 20.19.6:
- `.venv/bin/python -m pytest`: 648 passed, retaining all 631 M5 tests.
- `.venv/bin/ruff check .` and `git diff --check`: passed.
- `npm test --prefix frontend`: 6 passed; `npm run build --prefix frontend`: passed.
- `npm audit --prefix frontend --audit-level=high`: zero vulnerabilities reported.
- `trialgrid-demo --generate-only` produced and reloaded the verified 24-hour
  artifact; browser smoke verified list/detail/replay/rerun and the built UI.

- API to run and replay cases; React plots for price, SoC, actions, commitments and violation events.
- Technical evaluation report and repeatable one-command demonstration.
**Done when:** a reviewer can reproduce a failure, inspect its cause, apply a fix and re-run regression tests. The API/UI, current-controller rerun, shrinking and standalone pytest regression export are present.

## Stop/go gates

- After M1: physics and accounting verified, otherwise do not build AI layers.
- After M5: targeted search evaluated against random; if no advantage, document negative result and revisit search.
- After M7: validate actual buyer demand before expanding to other industries.

## First Codex prompt

> Read README.md, AGENTS.md, docs/PRD.md, docs/ARCHITECTURE.md and docs/IMPLEMENTATION_PLAN.md. Implement ONLY M0 and M1. Use Python 3.11+, a small src-layout package, pytest and Ruff. Implement the documented signed power convention, rejection policy, accounting and physically valid transitions. Include comprehensive numerical and boundary tests. Do not build an agent, dashboard or database. Run lint/tests and report exact commands and outcomes. Keep the diff reviewable.
