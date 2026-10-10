# Product Requirements — TrialGrid StressLab

## User and problem

Target early adopter: an engineering team testing an AI-assisted operational controller, initially in battery energy storage. Such a team must determine whether a controller violates operating constraints, misses commitments, or loses money under unusual combinations of conditions. Conventional scripted test cases do not necessarily explore those combinations efficiently.

## MVP proposition

For a controller adapter and scenario model, automatically search for failure-inducing operating conditions, save complete evidence, minimize the scenario, replay it, and export an executable regression test.

## Core user journey

1. Choose a controller version and scenario configuration.
2. Run a fixed reference suite.
3. Run uniform random and targeted search with comparable simulation budgets.
4. Inspect ranked failure cases and detailed time-series traces.
5. Minimize one case while preserving its original failure predicate.
6. Replay the case and export a regression fixture.
7. Re-run the fixture after changing the controller, and see pass/fail.

## Requirements

**R1 Simulator:** Fixed discrete time step (initially one hour), battery state in kWh, charge/discharge power in kW, usable capacity, efficiency, SoC bounds, degradation throughput penalty, electricity price, and optional reserve obligation. State updates enforce conservation and power/energy limits. Never silently turn infeasible requested actions into feasible successes; record attempts and applied action separately.

**R2 Scenario:** Versioned JSON-serializable scenario with initial state, price schedule, observed/forecast values, external disturbances, reserve obligations, and random seed. Full simulation deterministic given a controller that is deterministic or recorded in replay mode.

**R3 Controller interface:** Receives only allowed observation history, returns a typed charge/discharge/hold request. Simulator owns validation; controller cannot mutate state. Separate actual market price from price forecast to avoid lookahead.

**R4 Failure predicates:** attempted invalid command; unmet reserve obligation; economically material regret against a feasible baseline; and any later additional predicates declared *before* the evaluation. Keep unavoidable outcomes distinct from preventable errors.

**R5 Search:** Compare fixed suite, uniform random generation, and targeted search. Define scenario domains, validity rules and failure budget; save unsuccessful attempts too.

**R6 Evidence:** Capture seed, scenario, simulator version, controller config/version, prompts/tool calls when applicable, full step trace, metrics, failure predicate and severity, and run timestamp. Use recorded agent outputs for deterministic replay if live LLM behavior varies.

**R7 Shrinking:** Remove/relax scenario disturbances until a failure no longer reproduces; respect scenario validity and keep failure type constant. Verify candidate repeatedly for stochastic controllers.

**R8 Regression:** Persist minimal scenario as JSON fixture plus automated assertion checking the original failure or its resolution.

**R9 UI/API:** A run list, metric comparison and scenario replay timeline; no authentication or multi-tenancy in MVP.

## Evaluation methodology

- A predefined scenario space and fixed compute/simulator-call budget for each search method.
- Report distinct failure *types* and unique cases (deduplicate near-identical cases), time-to-first discovery, reproduction rate, cost and economic severity.
- Compare optimization and heuristic baselines as appropriate; optimize search failure yield rather than arguing that LLM dispatch should outperform classical optimization.
- Run several independent seeds, and separate the scenario/search tuning set from held-out test configurations.
- Do not claim statistical superiority without sufficient repeated measurements.

## Exit criteria

- Simulator passes conservation, bound, efficiency, limit, and determinism tests.
- Each controller finishes a scenario without modifying simulator internals.
- At least one discovered case can be exported, replayed and asserted in automated tests.
- Search comparisons are run with equal budgets and accurately report negative findings.
- A technical user can inspect every parameter involved in a result.

## Commercial validation

Interview at least five practitioners before calling this a commercial product. Validate integration burden, existing tools, target buyers, regulatory context and willingness to pilot. The BESS example is an experimental wedge, not evidence of market demand.
