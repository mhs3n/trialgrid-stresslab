# M4 tool-using LLM controller

## Boundary and decision semantics

`LLMController.act(observation, rng) -> Action` implements the unchanged M2
protocol. The path is observation → projected JSON prompt → strict `dispatch`
function call → locally typed `Action` → M1 simulator. No model receives a
scenario or can mutate battery state. The adapter uses the Responses API with a
forced function choice, `strict: true`, `additionalProperties: false`, and
`parallel_tool_calls: false`; local parsing still validates every result. See the
[official function-calling guide](https://developers.openai.com/api/docs/guides/function-calling).

`LLMConfig` sets model ID, forecast horizon (1–6, default 4), prompt version,
maximum attempts (1–5, default 3), output token cap, retry delay, failure policy,
and optional user-supplied token pricing. Only prompt version `dispatch-v1` is
supported. The prompt's objective is current reserve first, then forecast net
profit with zero terminal energy value, consistent with the M3 comparison model.
That is an instruction, not a guarantee that a model obeys it.

The prompt includes static configuration, current state/price/reserve, planning
prices `[current actual] + current published forecast[1:horizon]`, and past states,
actual prices, and reserves. Historical forecasts are omitted so they cannot
bypass the configured cap. No future actual price, future reserve, later forecast
publication, scenario seed, or oracle result is available. No missing forecasts
are filled in. Each API request is independent: no previous response ID or
provider conversation is used, and `store=False` is set. This is not a promise of
zero provider retention; consult the provider's data policies for real usage.

M3 and M4 have the same permitted observation source and configurable horizon.
M3 baselines ignore history and restrict themselves to a finite action menu; the
LLM may use permitted past observations and request any finite signed power.
Equal information does not mean equal action search or compute. This remains an
offline comparison, not a live industrial-control integration.

`dispatch` has exactly one numeric argument, `p_kw` (positive discharge, negative
charge, zero hold). Boolean, nonfinite, duplicate, extra, missing, or malformed
arguments are model-output failures. Missing/wrong/multiple tools, refusals and
incomplete responses are also output failures. Responses may include reasoning
items, but no reasoning text is needed or extracted separately.

A **finite infeasible action is a model decision**: it goes unchanged to M1, which
records the request and applies its existing rejection policy. It is never clamped
or retried by the LLM controller. Reserve misses likewise remain runner evidence.
The dispatch request ends the model turn; the simulator executes it in the caller.
No additional model call is made to narrate a result.

## Retries and distinct failures

Each attempted API call has its own record. SDK automatic retries are disabled;
only this controller retries. Connection/timeout, response decoding/validation,
HTTP 408/409/429 and 5xx errors are retryable; other HTTP errors stop immediately.
Malformed model output can also retry, using the same observation and a fixed
error code in the next request. A fixed configurable delay separates attempts.
Unexpected programming exceptions propagate rather than being labeled decisions.

The default `failure_policy="raise"` raises `ControllerCallFailure` after exhaustion
or a permanent API failure. `run_llm_episode` catches this specific error and returns
an audit with `failure="model_call_failure"`, no complete episode, and null complete-
episode economic metrics. Earlier requests/observations remain in the audit and
can be replayed; the generic runner's partial physical trace is not retained.

Optional `failure_policy="hold"` continues the episode with `outcome="fallback_hold"`,
`model_requested_action=null`, and a zero returned action. This is **not** a model-
chosen hold or a simulator rejection. The default generic `run_episode` behavior
and episode schema are unchanged; use the companion audit when interpreting LLM
runs to avoid attributing fallback actions to the model.

## Audit and replay

`llm_run.json` is a version-1 companion artifact containing the full scenario,
configuration/pricing source, and the unchanged M2 episode (or null on abort).
Each decision records the exact projected observation, outcome, model-requested
and returned actions. Each attempt records the complete request/response JSON,
requested model ID (in the request), returned model ID, prompt version, token
usage when valid, measured call latency, sanitized error code, and attempt number.
Provider error bodies/headers and API keys are never copied into error records.
Raw model responses are retained for reproducibility; review them before sharing.

`LLMRun.from_json` and `replay_llm_run` regenerate each prompt, replay saved raw
responses **and transport errors** through the same parser/retry flow, rerun M1,
and compare the complete resulting evidence. Missing, reordered, surplus and
inconsistent attempts/observations/actions/results are rejected. Replay neither
constructs an SDK client nor sleeps or contacts an API. It retains original call
latencies/cost estimates; these are not new replay costs. This verifies internal
consistency, not cryptographic authenticity or the truth of a provider's metadata.

A new controller instance is required for each episode. Scenario seeds do not make
live model responses deterministic; recorded-response replay is the reproducibility
mechanism. The deterministic fake model returns synthetic hold calls with zero
usage and latency solely for CI/demo verification, not as LLM performance evidence.

## Reports and billing

`summarize_llm_run` reports realized profit in EUR, ending energy, reserve misses,
physical rejected actions, model decisions, fallback holds, failed decisions,
transport/output failures, attempts/retries, total call latency and token totals.
Latency includes unsuccessful API attempts but excludes retry sleep, local planning,
and filesystem time. Provider usage includes reasoning within output token totals;
it is not billed a second time.

API cost is an **estimate in USD**, kept separate from EUR operating profit. It
uses all successful response telemetry, including malformed decisions that incurred
usage, with caller-supplied ordinary-input/cached-input/output tariffs and provenance.
Cost remains null when rates or any attempt's usage are missing/invalid, including
transport failures, or when cache-write tokens require an unsupported tariff.
Do not interpret missing costs as zero. The accounting basis follows the provider's
[usage/cost example](https://developers.openai.com/cookbook/articles/per_run_spending_controller_responses_api).
This is not a provider invoice; tariffs must match the chosen model/service tier.

## Run without credentials

```bash
python examples/run_llm.py --output-dir /tmp/trialgrid-llm-fake
python examples/run_llm.py --replay /tmp/trialgrid-llm-fake/llm_run.json \
  --output-dir /tmp/trialgrid-llm-replay
```

The demo writes `llm_run.json`, `episode.json` (null if aborted), and `summary.json`.
Replay uses the saved configuration. The summary identifies fake, OpenAI, or replay
mode and distinguishes original call metrics from new API calls. Aborted runs save
their audit and exit nonzero. The fixture is the existing hand-authored 24-hour
scenario. Hold ends at 50 kWh with zero profit, four reserve misses, and zero
physical rejections; fake API cost/latency is zero by construction.

## Optional real model

Install the optional dependency; core simulation and CI do not require it:

```bash
python -m pip install -e '.[llm]'
```

Set `OPENAI_API_KEY` and `TRIALGRID_LLM_MODEL` in your local environment, then run:

```bash
python examples/run_llm.py --real --horizon 4 --max-attempts 3 \
  --failure-policy raise --output-dir /tmp/trialgrid-llm-real
```

Choose a model that supports strict Responses function calling. No model or tariff
is hardcoded. `TRIALGRID_LLM_TIMEOUT_S` optionally sets the timeout (default 30).
For a known cost estimate, provide all of `TRIALGRID_INPUT_USD_PER_MILLION`,
`TRIALGRID_CACHED_INPUT_USD_PER_MILLION`, `TRIALGRID_OUTPUT_USD_PER_MILLION`, and
`TRIALGRID_PRICING_SOURCE`, or leave all unset for unknown cost. The prompt/output
budget may need adjustment for reasoning models through `LLMConfig.max_output_tokens`.
No `.env` file is automatically loaded and no credentials belong in git.

M4 does not add M5 search, failure predicates, a frontend, or a server API. Real
provider behavior and comparative economic performance require separate runs;
offline fake/mock tests do not establish either.

## M4 validation and review

On Python 3.12.3, the complete suite passed **510 tests** with Ruff clean, retaining
all 377 M3 tests. Blocking the optional SDK import still passed **504 tests** with
only the SDK smoke module skipped. Six installed-SDK checks used OpenAI 2.54.0
with mocked HTTP, including retries after malformed non-object responses.

Review identified and fixed falsey malformed usage metadata being accepted as
known billing, and malformed HTTP response decoding/shape errors escaping audit.
Regression tests cover both, with unexpected programming errors still propagated.
The fake 24-hour demo and recorded-response replay completed without provider calls.
No live model request, real billing validation, or economic comparison was performed.
