"""Transport-neutral LLM configuration and immutable audit records."""

from dataclasses import dataclass
from typing import Protocol

from trialgrid.domain import Action, _finite

PROMPT_VERSION = "dispatch-v1"


def nonnegative(name: str, value: float) -> None:
    _finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def integer(name: str, value: int, lower: int = 0, upper: int = 10**12) -> None:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


@dataclass(frozen=True, slots=True)
class TokenPricing:
    """Caller-supplied USD rates, not a provider invoice or built-in price table."""

    input_usd_per_million: float
    cached_input_usd_per_million: float
    output_usd_per_million: float
    source: str

    def __post_init__(self) -> None:
        for name in (
            "input_usd_per_million",
            "cached_input_usd_per_million",
            "output_usd_per_million",
        ):
            nonnegative(name, getattr(self, name))
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("pricing source must be nonempty")


@dataclass(frozen=True, slots=True)
class LLMConfig:
    model_id: str
    horizon: int = 4
    prompt_version: str = PROMPT_VERSION
    max_attempts: int = 3
    failure_policy: str = "raise"
    max_output_tokens: int = 1024
    retry_delay_s: float = 0.0
    pricing: TokenPricing | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be nonempty")
        integer("horizon", self.horizon, 1, 6)
        integer("max_attempts", self.max_attempts, 1, 5)
        integer("max_output_tokens", self.max_output_tokens, 16, 32768)
        if self.prompt_version != PROMPT_VERSION:
            raise ValueError("unsupported prompt_version")
        if self.failure_policy not in ("raise", "hold"):
            raise ValueError("failure_policy must be raise or hold")
        nonnegative("retry_delay_s", self.retry_delay_s)
        if self.retry_delay_s > 5:
            raise ValueError("retry_delay_s must be at most 5")
        if self.pricing is not None and not isinstance(self.pricing, TokenPricing):
            raise ValueError("pricing must be TokenPricing")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        for name in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens"):
            integer(name, getattr(self, name))
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached tokens exceed input tokens")


@dataclass(frozen=True, slots=True)
class ModelReply:
    response_json: str
    latency_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.response_json, str):
            raise ValueError("response_json must be text")
        nonnegative("latency_s", self.latency_s)


@dataclass(frozen=True, slots=True)
class CallError:
    """Safe error code only: no exception body, credentials, or HTTP headers."""

    code: str
    retryable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("error code must be nonempty")
        if type(self.retryable) is not bool:
            raise ValueError("retryable must be a boolean")


class ModelCallError(Exception):
    def __init__(self, code: str, retryable: bool = True, latency_s: float = 0.0) -> None:
        self.error = CallError(code, retryable)
        nonnegative("latency_s", latency_s)
        self.latency_s = latency_s
        super().__init__(code)


class Model(Protocol):
    def complete(self, request_json: str) -> ModelReply: ...


@dataclass(frozen=True, slots=True)
class AttemptRecord:
    attempt_number: int
    request_json: str
    response_json: str | None
    latency_s: float
    usage: TokenUsage | None
    transport_error: CallError | None
    output_error: str | None
    response_model_id: str | None


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    step_index: int
    observation_json: str
    attempts: tuple[AttemptRecord, ...]
    model_requested_action: Action | None
    returned_action: Action | None
    outcome: str
