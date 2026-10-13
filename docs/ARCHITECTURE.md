# Architecture — TrialGrid StressLab

## Principle

Use a **pure deterministic simulation core**, explicit controller adapters, and a separate evaluation/search layer. Nothing in the LLM/controller can modify physical simulator state directly.

## Planned layout

```text
src/trialgrid/
  domain.py             # typed state, parameters, actions, observations
  simulator.py          # deterministic transitions and constraints
  scenario.py           # serializable scenario fixtures, input validation
  runner.py             # run episodes and collect step traces
  metrics.py            # failure predicates and economics
  controllers/
    heuristic.py
    optimizer.py
    llm.py
  search/
    random.py
    evolutionary.py
    shrink.py
  results/
    artifacts.py        # storage/replay
  api/                  # later FastAPI surface
tests/
  test_simulator.py
  test_scenarios.py
  test_runner.py
  fixtures/
frontend/               # later React UI
docs/
```

Directories are intended structure, not a requirement to generate empty placeholder modules.

## Physics and accounting (initial single-asset model)

Choose discrete step duration `dt_h`. Action `p_kw` is signed: positive means discharge to grid, negative means charging from grid.

For feasible applied action:
- If discharge `p_kw >= 0`, energy out from cell `= (p_kw * dt_h) / eta_discharge`.
- If charging `p_kw < 0`, energy into cell `= (-p_kw * dt_h) * eta_charge`.
- Updated energy `E_next = E_now - energy_out + energy_in`.
- Cash-flow for that step `revenue_eur = (p_kw * dt_h / 1000) * price_eur_per_mwh`.
- Throughput penalty `cost_eur = abs(p_kw * dt_h) * degradation_eur_per_kwh` (explicit modeling convention, not physically complete degradation).
- Step profit `= revenue_eur - cost_eur`.

Parameters: usable capacity, energy floor/ceiling, charge/discharge power limits, distinct charge/discharge efficiencies in `(0,1]`, step duration in hours. The applied command must satisfy energy bounds; record invalid requests and the explicit policy (reject or clip). **Default MVP policy: reject infeasible actions, apply hold, and record a violation.** Reserve obligation check should specify whether requirement is checked before or after step; start with post-step.

## Leakage and fairness

- Agents see observations and published forecasts, never realized future prices or future hidden disturbances.
- Optimization baseline's information horizon must be clearly labeled (oracle upper-bound vs forecast-based deployable baseline).
- A common runner evaluates all controllers on an identical scenario instance.
- API keys belong in ignored local environment variables; raw prompts/responses may contain sensitive information and should be redacted before external sharing.

## Search loop

```text
Scenario domain --> generator --> runner(controller, scenario)
                                      |
                            validator + metrics
                                      |
                            score / novelty / evidence
                                      |
                      mutation / selection / shrinking
                                      |
                         fixture + replay + report
```

Initial search: parameter sweeps and seeded uniform random. Add evolutionary methods only after baseline comparisons work. Scenario shrinking must preserve the same failure predicate while maintaining valid environmental conditions.

## Proposed implementation strategy

Python 3.11+, dataclasses or Pydantic for model boundaries, pytest, Ruff, simple JSON fixtures. Use NumPy/SciPy/CVXPY only where their value is clear. Avoid DB server and framework scaffolding until core code exists.

## Security and scope

This is an offline simulation experiment, not a validated industrial safety system or live equipment controller.

## M2 implementation

`scenario.py` validates immutable version-1 hourly scenarios and strict JSON.
`runner.py` owns the `Controller` protocol, published observation frames, episode
traces/metrics, the hold demo controller, and replay. The execution path is
scenario → current observation → requested action → M1 `step` → trace and totals.
M1 remains the only physical transition and hard-constraint boundary.

Each forecast row is published at its row's hour and begins with a forecast for
that hour. History contains only previous published frames; realized future
prices and unpublished rows never enter observations. There is no physical
disturbance model in M2. The runner provides an episode-local seeded RNG.

Reserves are post-step minimum stored-energy commitments. Misses add evidence
without changing the physical result. JSON episode import reruns recorded
requests, then compares all observations, outcomes, and metrics. Static simulator
and schema versions plus caller-supplied controller identity accompany the full
scenario; timestamps are excluded from deterministic episode artifacts.

## M3 implementation and comparison boundary

`controllers/_planning.py` defines bounded shared configuration and M1-verified
candidate actions. `heuristic.py` scores one-step profit plus an average forecast
energy value. `optimizer.py` searches feasible trajectories using only the current
published forecast; it executes the first action and replans next hour. Both
implement the unchanged M2 protocol. No scenario object enters either controller.

Current actual price supersedes forecast entry zero. Both prioritize achievable
current post-step reserve, and neither assumes knowledge of later reserve rows.
The optimizer assumes zero terminal energy value and optimizes a finite candidate
tree, not continuous powers. It is not an oracle upper bound. Configuration and
information assumptions accompany replayable episodes in the comparison manifest;
M2 schemas and physical enforcement remain unchanged. See `docs/BASELINES.md`.

M3 controller review was completed before the separately authorized M4 work.
M3 introduced no LLM controller, API, UI, search, or oracle.

## M4 controller and evidence

`controllers/llm_types.py` defines validated configuration, transport replies and
immutable call/decision records. `llm.py` projects the M2 observation and converts
one strict dispatch tool call into `Action`. `llm_models.py` provides a deterministic
fake and a lazily imported optional OpenAI Responses transport with zero SDK retries.
`llm_audit.py` wraps the existing runner, records explicit call failures/fallbacks,
exports versioned companion evidence, and replays responses/errors without network.

M1, M2 domain/runner/scenario schemas, and M3 baselines are unchanged. The model's
prompt only uses the capped current publication and past actual/state/reserve data;
historical forecast rows are excluded. Physical invalidity is an agent decision
for M1 to reject, not a model-call retry. Only API/output failures enter retries.

Complete runs include the existing M2 episode. Aborted runs preserve call evidence
but no completed physical trace/episode metrics. Reports distinguish fallback
holds, output failures and transport failures from model-chosen actions and physical
rejections. The companion schema records original latency/usage and caller-supplied
pricing; replay checks consistency rather than authenticity. See `docs/LLM_CONTROLLER.md`.
