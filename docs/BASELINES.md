# M3 baseline contracts

## Shared information and feasible actions

Both baselines implement M2's `act(observation, rng) -> Action` interface. They
receive identical kinds of information from the same runner: static battery
limits, current state/price/reserve, current published forecast, and previous
observation frames. Their states can diverge after they choose different actions.
Neither accepts a `Scenario`, future realized prices, future reserve requirements,
or later forecast publications. No oracle controller is included.

Planning prices are the current **realized** price followed by entries `1:` of
the current published forecast row. Entry zero forecasts the already-known current
hour and is superseded, not appended. The horizon defaults to four hours, is
limited to the available published row, and is configurable from one to six.
Empty or one-entry rows provide only a current-hour decision. No missing values
are filled from realized future prices. Past observations are available but these
stateless baselines do not use them. The seeded RNG is unused.

`BaselineConfig` supplies the same bounded horizon and action menu to both
controllers. Default power fractions `(0.5, 1.0)` scale the maximum *currently
feasible* charge and discharge power, respecting energy bounds and efficiencies.
Candidates also include hold and an action targeting the current reserve boundary.
One to three increasing fractions are permitted, with the final fraction exactly
one. Every candidate transition and its accounting is evaluated by M1 `step`.
Floating-point boundary candidates may be moved a few representable values toward
hold or toward meeting reserve **before** a controller selects its request. The
simulator and runner never repair a submitted request.

Both policies first minimize achievable **current** post-step reserve shortfall,
then make their economic choice among the remaining candidates. This meets
reachable commitments even when charging costs money, and uses maximal feasible
charging for unreachable commitments. Actual reserve misses remain in the episode
trace. Future reserves are unpublished and are not inferred: planning assumes
only the physical floor in later hours. This does not guarantee future commitments.
Exact floating-point score ties prefer hold, then lower absolute power, then charging before discharging.

## Heuristic controller

`HeuristicController` scores each current candidate as:

```
immediate profit + (next energy - current energy) * stored-energy value
stored-energy value = max(0, mean(future forecast prices) / 1000 - degradation) * eta_discharge
```

The mean uses only future entries within the shared published horizon. With none,
stored-energy value is zero. Profit and degradation use M1's grid-side accounting.
This rule uses an average sale value rather than scheduling future dispatch; it
ignores future power bottlenecks, the order of prices, and later opportunities to
buy. It is intentionally a simple comparison baseline, not a claim of optimality.

## Forecast optimizer

`ForecastOptimizer` performs receding-horizon search over the shared feasible
candidate menu at every projected state. It maximizes the sum of forecast net
profits (market revenue minus degradation), conditional on minimizing the current
reserve shortfall. Only the first action is sent to the runner; a new forecast
and actual state produce a new plan next hour. A local memo caches subproblems by
horizon index and exact stored energy; no rounded state grid or interpolated
physics is used. It permits only one signed action per step, preventing fictitious
simultaneous charging/discharging at negative prices.

Terminal energy has **zero value** in the planning objective. This can encourage
selling initial inventory or draining near the forecast horizon, and it should
not be confused with sustainable long-run arbitrage returns. `plan(observation)`
returns the candidate actions and their **forecast** profit for inspection.
Realized performance always comes from the episode runner, never that projection.

The optimizer is optimal over its finite, state-dependent candidate tree using
floating-point M1 accounting, not over all continuous actions. Default horizon is
four; the cap of six and maximum three power fractions bound combinatorial work.
Coarse candidates can miss profitable intermediate powers. Different information
horizons, inaccurate forecasts, unknown future commitments, and zero terminal
value can all limit performance. It is neither an oracle nor a theoretical upper
bound, and it is not guaranteed to outperform the heuristic in realized profit.
An oracle, if added later, must be a separately named result with an explicit
future-information label; it must never replace this deployable-information
baseline in ordinary comparisons.

## Reproducible comparison evidence

`examples/compare_baselines.py` creates fresh controllers using one shared
`BaselineConfig`, runs both through the same M2 runner/scenario, and verifies both
exported traces with recorded-action replay. `comparison.json` records the horizon,
power fractions, information mode, reserve policy, zero terminal valuation,
controller names/versions, realized accounting and final energy. Keep it beside
the two episode files; the existing M2 episode schema is unchanged. The comparison
uses realized simulator accounting and never substitutes `ForecastPlan` profit.

These are offline policies under M2's assumption that the current price is known
at dispatch. Published forecasts may be wrong. Performance is not an oracle bound,
and equal information does not imply equal computation or equal realized outcomes.
The heuristic performs one-step scoring; the optimizer spends more computation
searching. Re-evaluate those costs when adding future agent controllers.
