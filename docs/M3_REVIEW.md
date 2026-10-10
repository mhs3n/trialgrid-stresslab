# M3 controller review

Status: code review and local validation complete; human PR review precedes any
M4 implementation. This is an offline software review, not operating approval for
industrial equipment.

## Findings and information fairness

Read-only controller review found no outstanding actionable correctness, regression,
or information-leakage issues. The review covered candidate feasibility, reserve
handling, forecast indexing, objective/accounting consistency, replay, and the
limits of the optimizer's claims. During implementation, the heuristic mean was
computed before converting EUR/MWh to EUR/kWh to reduce rounding sensitivity;
tie preference is explicitly for exact floating-point score ties.

Both controllers use the unchanged M2 `Observation`; neither accepts the scenario.
Tests change future actual prices, future reserves and unpublished forecast rows
while holding the observable prefix fixed, and require identical earlier traces
for each controller. Separate tests confirm that a newly published forecast *can*
change the action. Known current price replaces forecast entry zero. The shared
horizon/action configuration is recorded in the comparison manifest.

## Numerical and behavior validation

- Heuristic: 33 focused tests cover manually calculated value scores, efficiencies,
  degradation, price signs, forecast timing, reserve priorities and boundary actions.
- Optimizer: 67 focused tests include independent exhaustive integer schedules
  for tiny batteries, manual two-step accounting, negative-price behavior, terminal
  liquidation, unavailable future reserves, deterministic ties and overflow errors.
- Fairness/integration: 11 tests cover hidden-future independence, published-forecast
  sensitivity, identical public market inputs, deterministic JSON and action replay.
- Full suite: `.venv/bin/python -m pytest` — **377 passed**, retaining all 266 M2 tests.
- `.venv/bin/ruff check .` and `git diff --check` — passed.
- `.venv/bin/python examples/compare_baselines.py --output-dir /tmp/trialgrid-m3-baselines`
  — both 24-hour episodes completed and replay verification passed.

The hand-authored sample produced EUR 5.14 realized profit for the heuristic and
EUR 4.37 for the forecast optimizer. Both ended at 40 kWh with zero reserve misses
and zero rejected actions. These are reproducible demo outputs, not evidence of
population-level superiority. Initial energy was 50 kWh, so profit also reflects
use of initial inventory. No oracle result was produced.

## Limitations and M4 gate

The optimizer is approximate over continuous actions, bounded to at most six
forecast hours, and uses zero terminal inventory value. Future reserve schedules
are unknown to both policies. Equal information does not imply equal compute;
optimizer search is more expensive than heuristic scoring. Neither algorithm
claims superiority under forecast error. Replay proves evidence consistency,
not authenticity or that an arbitrary third-party controller is honest.

Before approving M4, review the baseline objectives and the shared observation
contract. A future LLM adapter must use that same contract and reported horizon
for an equal-information comparison. Any later oracle must have a separate name,
explicit future-information label, and a distinct reporting category. M4 and all
later milestones remain unimplemented. Remote CI status is not established by
these local checks.
