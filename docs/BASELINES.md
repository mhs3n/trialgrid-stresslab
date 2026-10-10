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
