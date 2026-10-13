"""Versioned LLM evidence and strict offline response/error replay around M2."""

from dataclasses import asdict, dataclass, fields
from math import fsum

from trialgrid.controllers.llm import ControllerCallFailure, LLMController, canonical
from trialgrid.controllers.llm_types import (
    DecisionRecord,
    LLMConfig,
    Model,
    ModelCallError,
    ModelReply,
    TokenPricing,
    nonnegative,
)
from trialgrid.domain import _finite
from trialgrid.runner import Episode, _matches_json, run_episode
from trialgrid.scenario import Scenario, load_json, require_keys


class RecordedModel:
    """Consume original responses/errors only when the complete request matches."""

    def __init__(self, attempts: list[dict]) -> None:
        self.attempts = attempts
        self.position = 0

    def complete(self, request_json: str) -> ModelReply:
        if self.position >= len(self.attempts):
            raise ValueError("recorded responses exhausted")
        record = self.attempts[self.position]
        if record.get("request_json") != request_json:
            raise ValueError("recorded request does not match observation/config/retry")
        nonnegative("recorded latency_s", record.get("latency_s"))
        self.position += 1
        error = record.get("transport_error")
        if error is not None:
            error = require_keys(error, {"code", "retryable"})
            if record.get("response_json") is not None:
                raise ValueError("transport failure cannot contain a successful response")
            raise ModelCallError(error["code"], error["retryable"], record["latency_s"])
        return ModelReply(record.get("response_json"), record["latency_s"])

    def assert_consumed(self) -> None:
        if self.position != len(self.attempts):
            raise ValueError("unused recorded responses")


@dataclass(frozen=True, slots=True)
class LLMRun:
    scenario: Scenario
    config: LLMConfig
    decisions: tuple[DecisionRecord, ...]
    episode: Episode | None
    failure: str | None
    schema_version: int = 1

    def to_dict(self) -> dict:
        return load_json(self.to_json())

    def to_json(self) -> str:
        return canonical(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "LLMRun":
        return cls.from_dict(load_json(text))

    @classmethod
    def from_dict(cls, value: dict) -> "LLMRun":
        data = require_keys(value, {field.name for field in fields(cls)})
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("unsupported LLM audit schema_version")
        config_data = dict(
            require_keys(data["config"], {field.name for field in fields(LLMConfig)})
        )
        if config_data["pricing"] is not None:
            config_data["pricing"] = TokenPricing(
                **require_keys(
                    config_data["pricing"],
                    {field.name for field in fields(TokenPricing)},
                )
            )
        config = LLMConfig(**config_data)
        scenario = Scenario.from_dict(data["scenario"])
        if not isinstance(data["decisions"], list):
            raise ValueError("decisions must be a JSON array")
        attempts = []
        for record in data["decisions"]:
            if not isinstance(record, dict) or not isinstance(record.get("attempts"), list):
                raise ValueError("decision must contain recorded attempts")
            if any(not isinstance(attempt, dict) for attempt in record["attempts"]):
                raise ValueError("attempts must contain objects")
            attempts.extend(record["attempts"])
        model = RecordedModel(attempts)
        # Replay re-parses responses and errors; no network model or SDK is instantiated.
        replayed = run_llm_episode(scenario, model, config, replay=True)
        model.assert_consumed()
        if not _matches_json(data, replayed.to_dict()):
            raise ValueError("LLM evidence does not match response replay")
        return replayed


def run_llm_episode(
    scenario: Scenario,
    model: Model,
    config: LLMConfig,
    *,
    replay: bool = False,
) -> LLMRun:
    """Retain failed-call evidence without changing the generic runner's behavior."""
    controller = (
        LLMController(model, config, sleep_fn=lambda _: None)
        if replay
        else LLMController(model, config)
    )
    episode = None
    failure = None
    try:
        episode = run_episode(
            scenario,
            controller,
            controller_name=f"llm:{config.model_id}",
            controller_version=config.prompt_version,
        )
    except ControllerCallFailure:
        failure = "model_call_failure"
    return LLMRun(scenario, config, controller.records, episode, failure)


def replay_llm_run(run: LLMRun) -> LLMRun:
    return LLMRun.from_json(run.to_json())


def summarize_llm_run(run: LLMRun) -> dict:
    """Unknown billing is null. USD call estimates are separate from EUR profit."""
    attempts = [attempt for record in run.decisions for attempt in record.attempts]
    usage_known = all(attempt.usage is not None for attempt in attempts)
    usage = None
    if usage_known:
        usage = {
            name: sum(getattr(attempt.usage, name) for attempt in attempts)
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cache_write_tokens",
            )
        }
    cost = None
    rates = run.config.pricing
    if usage_known and rates is not None and usage["cache_write_tokens"] == 0:
        cost = (
            ((usage["input_tokens"] - usage["cached_input_tokens"]) / 1_000_000)
            * rates.input_usd_per_million
            + (usage["cached_input_tokens"] / 1_000_000) * rates.cached_input_usd_per_million
            + (usage["output_tokens"] / 1_000_000) * rates.output_usd_per_million
        )
        _finite("API cost estimate", cost)
    latency = fsum(attempt.latency_s for attempt in attempts)
    _finite("total call latency", latency)
    episode = run.episode
    return {
        "status": "completed" if episode is not None else "model_call_failure",
        "model_id": run.config.model_id,
        "prompt_version": run.config.prompt_version,
        "horizon": run.config.horizon,
        "realized_profit_eur": episode.metrics.profit_eur if episode else None,
        "ending_energy_kwh": episode.trace[-1].result.state.energy_kwh if episode else None,
        "reserve_violations": episode.metrics.reserve_violation_count if episode else None,
        "rejected_actions": episode.metrics.rejected_action_count if episode else None,
        "model_decisions": sum(record.outcome == "model_decision" for record in run.decisions),
        "fallback_holds": sum(record.outcome == "fallback_hold" for record in run.decisions),
        "failed_decisions": sum(record.outcome == "call_failure" for record in run.decisions),
        "call_attempts": len(attempts),
        "retry_attempts": sum(len(record.attempts) - 1 for record in run.decisions),
        "transport_failures": sum(attempt.transport_error is not None for attempt in attempts),
        "output_failures": sum(attempt.output_error is not None for attempt in attempts),
        "total_call_latency_s": latency,
        "token_usage": usage,
        "api_cost_estimate_usd": cost,
        "pricing": asdict(rates) if rates is not None else None,
        "cost_note": (
            "Caller-supplied text-token tariffs; missing usage or cache writes => unknown."
        ),
    }
