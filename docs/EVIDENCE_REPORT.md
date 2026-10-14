# M7 evidence report and local demo

## Purpose and scope

M7 makes existing TrialGrid evidence inspectable without weakening the simulator or
controller information boundary. It adds a local API, a React timeline and a repeatable
offline demonstration. It does not add authentication, multi-tenancy, a database,
cloud deployment, live model calls, shrinking or regression-test export.

The complete path is:

```text
validated scenario
  -> allowlisted offline target and comparator
  -> existing M2 runner and M1 simulator
  -> predeclared M5 failure report
  -> versioned content-addressed artifact
  -> replay verification on every load
  -> normalized API projection
  -> React metrics and timeline
```

Replay reconstructs transitions from recorded requested actions and compares the full
trace. It detects inconsistent evidence but does not authenticate who created a file.

## Demonstration case

`trialgrid-demo` generates a fixed artifact from the hand-authored
`examples/scenarios/fixed/price_cycle.json` fixture. The target is hold and the
comparator is the forecast-only finite-action optimizer with a four-hour published
forecast horizon. The comparator is not an oracle and receives no future realized
prices or future reserve requirements.

The deterministic demonstration produces:

| Measure | Hold target | Forecast-only comparator |
|---|---:|---:|
| Realized profit | EUR 0.00 | EUR 4.37 |
| Ending energy | 50 kWh | 40 kWh |
| Reserve violations | 4 | 0 |
| Rejected actions | 0 | 0 |

The four target shortfalls occur at hours 16–19 and are each classified as locally
preventable with severity 10 kWh. Economic dominance is ineligible because the
comparator ends with less energy than the target. This is a workflow example, not a
discovered or minimized counterexample and not evidence that one search method wins.

## What the viewer shows

- A verified run list and target/comparator metrics, including ending energy.
- Actual hourly prices and each hour's published forecast values.
- Initial energy and every post-step energy state, measured in kWh.
- Post-step reserve commitments and reserve-shortfall markers.
- Requested and applied signed power, with positive discharge and negative charge.
- Physical rejections separately from reserve findings.
- Failure kind, severity and preventability classification.
- Explicit replay verification and current-controller rerun actions.

The browser never renders raw HTML from evidence. Raw model prompts and responses are
not exposed by this initial viewer because they may contain sensitive information.

## M5 evaluation result retained

M5's recorded small synthetic evaluation did not establish a targeted-search
advantage. For heuristic seeds 11, 22 and 33, fixed, random and targeted methods each
completed only three evaluations per run and found zero actionable cases. In the
synthetic fake-hold check, random and targeted each found three unique actionable
environments, so targeted minus random was zero. These are descriptive results with no
statistical superiority claim.

## Security and operational boundary

The API has no authentication and must remain on loopback. It uses one configured
artifact directory, fixed-format artifact IDs, size bounds, symlink rejection and a
hardcoded offline controller registry. It accepts no remote URLs, arbitrary Python
imports, shell commands, live-provider settings or client-selected filesystem paths.
The service never connects to industrial equipment.

## Remaining limitation

M6 is still absent. A reviewer can reproduce a full case, inspect the trace, change a
registered controller and rerun the same scenario. The workflow cannot yet shrink the
case or export it as a standalone pytest regression. Therefore the final M7 roadmap
criterion that includes regression export remains incomplete until M6 is implemented.
