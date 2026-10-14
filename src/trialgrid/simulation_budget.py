"""Context-local accounting for every M1 call, including controller planning."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


class SimulationBudgetExceeded(RuntimeError):
    """Raised before executing a transition beyond the allocated call budget."""


@dataclass(slots=True)
class SimulationCounter:
    limit: int
    used: int = 0


_active: ContextVar[SimulationCounter | None] = ContextVar(
    "trialgrid_simulation_counter", default=None
)


def remaining_simulator_calls() -> int | None:
    counter = _active.get()
    return counter.limit - counter.used if counter is not None else None


def record_simulator_call() -> None:
    counter = _active.get()
    if counter is not None:
        if counter.used >= counter.limit:
            raise SimulationBudgetExceeded("simulator-call budget exhausted")
        counter.used += 1


@contextmanager
def count_simulator_calls(limit: int) -> Iterator[SimulationCounter]:
    if type(limit) is not int or not 1 <= limit <= 10_000_000:
        raise ValueError("simulator call limit must be an integer in [1, 10000000]")
    if _active.get() is not None:
        raise ValueError("nested simulator budgets are not supported")
    counter = SimulationCounter(limit)
    token = _active.set(counter)
    try:
        yield counter
    finally:
        _active.reset(token)
