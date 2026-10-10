# TrialGrid StressLab

**Experimental adversarial testing for AI controllers in simulated industrial systems.**

TrialGrid explores operating scenarios, identifies controller failures, shrinks counterexamples, and exports replayable regression tests. The first proving ground is a simulated battery energy storage system (BESS).

## Core hypothesis

Under an equal simulation budget, targeted scenario search can discover **distinct, reproducible, operationally meaningful failures** that a fixed suite or uniform random testing misses. This is a hypothesis to test, not a claimed result.

## MVP boundaries

- Deterministic battery simulation, typed scenario definition, physical constraints and economic accounting.
- A simple heuristic controller, an optimization baseline, and later a tool-using LLM controller.
- Fixed scenarios, random exploration, targeted/evolutionary search, counterexample shrinking.
- Stored traces, replay, regression tests, and honest comparison using equal experiment budgets.
- Thin API and interactive replay UI after the experiment works.

**Not in MVP:** live equipment connectivity, grid dispatch, safety certification, multi-asset optimization, a generic agent framework, or claims of guaranteed safe operation.

## Build order

See [Implementation Plan](docs/IMPLEMENTATION_PLAN.md), [Product Requirements](docs/PRD.md), [Architecture](docs/ARCHITECTURE.md), and [Codex Instructions](AGENTS.md).

## Day-one goal

Implement the pure Python BESS simulator with deterministic tests. No LLM, frontend, database server, or cloud infrastructure is required for this milestone.

## Status

Planning and engineering specification. No experimental results have been produced yet.
