# Codex instructions for TrialGrid

Read `README.md`, `docs/PRD.md`, `docs/ARCHITECTURE.md`, and `docs/IMPLEMENTATION_PLAN.md` before coding.

## Working agreement

1. Work on the current milestone only. Ask for explicit approval before expanding scope.
2. Prefer small, modular and typed Python code; avoid clever abstractions until warranted.
3. Tests are mandatory for all simulator arithmetic, invariants and edge cases.
4. The simulator—not the controller—owns physical state and hard constraint enforcement.
5. Record requested actions separately from applied actions. Never silently fix an invalid action.
6. Keep deterministic seeds and serialized fixtures; never fabricate experiment results.
7. Avoid lookahead: future realized prices/disturbances must not leak to the controller.
8. Label oracle optimizers clearly if they see future realized information.
9. Run `python -m pytest` and `ruff check .` before declaring a milestone complete. If unavailable, state that honestly.
10. Avoid unnecessary dependencies, cloud deployments, databases and UI until core evaluation exists.
11. Never connect to or control live industrial equipment.
12. No credentials in git. Keep API keys in ignored local environment variables.
13. In final implementation summaries include changed files, tested behavior, commands/results, limitations and suggested next step.

## Immediate priority

M0–M4 are implemented. Review the M4 controller, audit/replay artifacts and optional
transport before expanding scope. Do not implement M5, frontend or server APIs
without explicit approval. Preserve equal observation access, distinguish model
call failures from decisions, and label any future oracle separately.
