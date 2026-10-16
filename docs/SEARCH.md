# M5 failure evaluation and scenario search

## Declared criteria

`FailureConfig` version 1 declares the economic threshold before evaluation
(default EUR 1, positive and finite). Every result stores that configuration,
controller identity/configuration and effective information horizon.

- **Invalid action:** an M1 physical rejection, using `StepResult.violations`.
  The requested and applied actions remain in the episode evidence.
- **Reserve shortfall:** every missed post-step commitment is recorded. It is
  `locally_preventable` only if an M1-verified alternative could meet the currently
  published commitment from the actual starting state. Above the usable ceiling
  is `physically_impossible`. Other misses are `undetermined`: earlier depletion
  might matter, but future reserve requirements were unpublished. No global
  preventability claim is inferred from an unreachable current state.
- **Economic underperformance:** comparator realized profit exceeds tested profit
  by at least the threshold, the comparator has no physical rejections or reserve
  misses, and its ending energy is at least the tested controller's ending energy.
  This conservative dominance gate avoids rewarding liquidation of extra inventory.
  It neither assigns an invented inventory price nor claims optimal/oracle regret.
  Raw profit gaps and ending energies are reported even when the gate is ineligible.

A model-failure hold is classified as `model_failure_fallback`, excluded from the
agent mistake score, and counted separately. Economic failure attribution is
ineligible when either run includes a fallback. Actual model decisions, including
finite infeasible commands, retain ordinary simulator-based classification.
Transport/output failures remain in model audit records; an aborted run has no
fabricated successful episode or zero-profit economic score.

## Domain and information fairness

`ScenarioDomain` validates fixed hourly duration, battery configuration, price and
reserve ranges, and forecast publication length. The example domain uses the
24-hour sample battery, prices from -100 to 200 EUR/MWh, reserves from 0 to 100 kWh,
and publications up to four hours. Prices, forecast entries, reserves and initial
energy are sampled independently with a local seeded RNG. These synthetic
forecasts are not fitted to market data and are not copied from future actuals.

Mutations resample exactly one environmental scalar within the same domain and
renew the episode seed. Every evaluated scenario is saved. Three hand-authored
fixtures under `examples/scenarios/fixed/` define the fixed suite before evaluation.
Their explicit forecasts include forecast error and hidden reserve surprises.

Each episode constructs a fresh controller. A shared wrapper caps current **and
historical** forecast rows at the declared information horizon for target and
comparator. The runner still supplies only M2 observations. No scenario, future
realized prices, unpublished forecasts or future reserve schedule reaches the
controllers. The comparator is the named forecast-only M3 optimizer, with its
finite action menu and zero terminal energy value; it is not an oracle.
Factories and controller implementations are trusted cooperative Python code;
this is not a sandbox against arbitrary introspection or external state.

## Budgets, selection and deduplication

`simulation_budget.py` counts every call entering M1 `step` in the evaluation
context, including controller planning, comparator planning, executed transitions,
rejections and reserve-witness checks. M1 physical arithmetic is unchanged. The
counter is context-local, rejects nesting, restores itself on errors, and raises
before executing a call beyond the limit. The controller wrapper checks remaining
allowance before another decision/model call. Verification after search is outside
the discovery budget and cannot feed selection.

All methods receive the same simulator-call cap and candidate-attempt cap. Every
warm-up, mutation, duplicate, failure and incomplete final candidate consumes its
actual allowance. There are no free replacements or uncounted parent evaluations.
The candidate cap also prevents zero-simulation failure loops. Reports distinguish
allocated and consumed calls and explain which limit stopped a method. Equal caps
do not guarantee equal used calls if the candidate cap stops a failing run first.
Wall time and completed episode counts can differ.

- **Fixed:** cycle the declared fixtures until a limit is reached. Repeated
  evaluations consume budget and do not inflate unique-environment counts.
- **Random:** independently sample the declared domain for each candidate.
- **Targeted:** charge a random initial sample, select among the best completed
  evaluations, and mutate one scalar. The elite population defaults to four;
  random exploration defaults to 25%. Incomplete evaluations cannot be parents.

Selection uses a predeclared lexicographic score: number of actionable findings,
locally preventable shortfall kWh, then material economic gap EUR. Impossible,
undetermined and fallback reserve outcomes earn no search reward. This policy and
all settings are stored in the artifact. Random evaluation/tests were implemented
and validated before introducing targeted selection.

Environment fingerprints exclude episode seed and normalize numeric representation.
Deduplication is **exact only**; it makes no near-duplicate/novelty claim. All repeated
attempts remain in the artifact. Seed exclusion groups stochastic replications of
one environment; each actual episode seed and response remains available separately.

## Evidence and verification

Search JSON contains domain, criteria, settings, method, seeds, metadata, every
candidate/parent, actual call counts, episodes/errors, LLM response records when
present, findings and summary. Successful nonfailure cases are retained too.
`verify_search_result` replays completed episodes and model responses/errors,
regenerates random/targeted proposals and selection, and validates derived findings,
lineage, controller identities, count bounds and summaries. Partial candidates
remain explicitly incomplete. Internal planner counts and exception provenance
are reported measurements, not authenticated by recorded dispatch actions.
Verification checks internal consistency, not cryptographic authenticity.

## M6 shrinking and regression export

`trialgrid.search.shrink.shrink_failure` accepts a completed `EvaluationRecord` and
re-runs its target and comparator for every candidate. It keeps the original seed,
hour count and forecast publication shape, and greedily moves initial energy toward
the floor, prices/forecasts toward zero, and reserves toward zero. A candidate is
accepted only when the same actionable `(finding kind, classification)` signature
reproduces. The attempt budget is explicit; `complete=false` means the result is not
eligible for export. `repetitions` can require the predicate to reproduce on every
fresh controller run, which is useful for stochastic controllers.

`trialgrid.regression.RegressionFixture` stores the minimized scenario, controller
metadata, criteria, signature and repetition policy as version-1 JSON. Exported
pytest modules resolve only the allowlisted offline `hold`, `heuristic` and
`forecast_optimizer` versions. Unknown versions and live LLM factories are rejected.
The generated test reruns the fixture and fails if the original actionable signature
no longer appears; this tests the failure against the current controller code.

Summaries report unique raw/actionable cases, predicate types, reserve classifications,
worst actionable score, incomplete evaluations, model failures/fallbacks, and the
completed evaluation/cumulative calls at first actionable discovery. This is not
within-episode discovery timing or a measured stochastic reproduction rate.
Deterministic artifacts exclude timestamps; the CLI comparison manifest carries
UTC time and measured search wall time separately.

## Run and inspect

```bash
python examples/compare_search.py --calls 20000 --seeds 11 22 33 \
  --output-dir /tmp/trialgrid-m5
python examples/compare_search.py --target fake-llm --calls 20000 --seeds 11 \
  --output-dir /tmp/trialgrid-m5-fake
```

The CLI has only offline heuristic, hold and deterministic fake-LLM targets.
It writes each method/seed result and `comparison.json`, verifies saved evidence,
and reports paired targeted-minus-random unique actionable counts. Fake-model
results are synthetic behavior checks, not evidence about live LLM performance.
No provider calls are made by these commands.

## Recorded local results and limitations

Python 3.12.3 validation: **631 tests passed**, including all 510 M4 tests;
`ruff check .` and `git diff --check` passed. Focused M5 tests cover predicate
arithmetic/gates, generation, mutation, exact budgets, failures, duplicates,
information caps, fallback attribution, lineage and corrupted evidence.

With the default heuristic target, EUR 1 threshold and seeds 11/22/33, all three
methods consumed exactly 20,000 M1 calls per seed and completed three evaluations
per run. Each found **zero actionable cases**. The fake-hold check at seed 11 used
the same caps: fixed found one unique actionable environment, random and targeted
found three each. Targeted-minus-random was zero. These tiny synthetic experiments
establish no targeted-search advantage or population-level superiority. The fixed
suite is identical across seeds and is not three independent datasets.

Before expanding search, revisit domains/criteria and evaluate predeclared larger
budgets with held-out configurations. Any tuning must be reported separately. The
shrinker does not claim a globally minimal case: it is bounded and greedy, and its
predicate signature intentionally preserves failure type/classification rather than
exact severity or failure hour. The requested paid smoke test remains pending
model/budget details and local API configuration; it is separate from these offline
experiments.
