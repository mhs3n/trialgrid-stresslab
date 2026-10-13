"""Tool-using controller with explicit call failures and an auditable retry boundary."""

import json
from collections.abc import Callable
from dataclasses import asdict
from random import Random
from time import sleep

from trialgrid.controllers.llm_types import (
    AttemptRecord,
    DecisionRecord,
    LLMConfig,
    Model,
    ModelCallError,
    TokenUsage,
)
from trialgrid.domain import Action
from trialgrid.runner import Observation
from trialgrid.scenario import load_json, require_keys

INSTRUCTIONS = """TrialGrid dispatch-v1. You control only an offline simulated battery.
Request exactly one dispatch tool call with signed grid-side power p_kw:
positive discharges, negative charges, zero holds. Energy is kWh, time is hours,
prices are EUR/MWh. Prefer meeting the current post-step reserve, then maximize
net market revenue minus grid-side throughput degradation over the published
forecast horizon. Terminal energy has zero assigned value. Charge energy added
is -p_kw * dt_h * eta_charge; discharge energy removed is p_kw * dt_h /
eta_discharge. Observe configured power and energy bounds. The simulator enforces
physics and records infeasible requests unchanged before rejecting them.
Only current and past published information is provided. Planning-price index
zero is current actual price; later entries are forecasts, not realized prices.
Future reserve obligations are unknown; do not invent future observations.
This call ends at the dispatch request; no narrative or further tool is needed.
"""


def canonical(value: object) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def project_observation(observation: Observation, horizon: int) -> dict:
    """Whitelist allowed fields; old forecasts cannot bypass the horizon cap."""
    current = observation.current
    return {
        "config": asdict(observation.config),
        "current": {
            "step_index": current.step_index,
            "state": asdict(current.state),
            "price_eur_per_mwh": current.price_eur_per_mwh,
            "reserve_requirement_kwh": current.reserve_requirement_kwh,
            "planning_prices_eur_per_mwh": [current.price_eur_per_mwh]
            + list(current.forecast_prices_eur_per_mwh[1:horizon]),
        },
        "history": [
            {
                "step_index": frame.step_index,
                "state": asdict(frame.state),
                "price_eur_per_mwh": frame.price_eur_per_mwh,
                "reserve_requirement_kwh": frame.reserve_requirement_kwh,
            }
            for frame in observation.history
        ],
    }


def make_request(config: LLMConfig, observation_json: str, attempt: int, error: str | None) -> str:
    return canonical(
        {
            "model": config.model_id,
            "instructions": INSTRUCTIONS,
            "input": canonical(
                {
                    "prompt_version": config.prompt_version,
                    "observation": load_json(observation_json),
                    "attempt_number": attempt,
                    "previous_error": error,
                }
            ),
            "tools": [
                {
                    "type": "function",
                    "name": "dispatch",
                    "description": "Request signed battery grid power in kW for this step.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"p_kw": {"type": "number"}},
                        "required": ["p_kw"],
                        "additionalProperties": False,
                    },
                }
            ],
            "tool_choice": {"type": "function", "name": "dispatch"},
            "parallel_tool_calls": False,
            "max_output_tokens": config.max_output_tokens,
            "store": False,
        }
    )


def response_usage(data: dict) -> TokenUsage | None:
    """Missing or invalid provider telemetry is unknown, never fabricated zero."""
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    details = usage.get("input_tokens_details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        return None
    try:
        result = TokenUsage(
            usage["input_tokens"],
            usage["output_tokens"],
            details.get("cached_tokens", 0),
            details.get("cache_write_tokens", 0),
        )
        if result.cached_input_tokens + result.cache_write_tokens > result.input_tokens:
            return None
        return result
    except (KeyError, ValueError):
        return None


def parse_action(data: dict) -> Action:
    """Validate syntax/types only. Finite physically infeasible requests are decisions."""
    if data.get("status") != "completed":
        raise ValueError("response_not_completed")
    output = data.get("output")
    if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
        raise ValueError("invalid_output_items")
    calls = []
    for item in output:
        kind = item.get("type")
        if kind == "function_call":
            calls.append(item)
        elif kind == "message":
            content = item.get("content", [])
            if not isinstance(content, list) or any(not isinstance(c, dict) for c in content):
                raise ValueError("invalid_message")
            if any(c.get("type") == "refusal" for c in content):
                raise ValueError("model_refusal")
        elif kind != "reasoning":
            raise ValueError("unexpected_output_item")
    if len(calls) != 1 or calls[0].get("name") != "dispatch":
        raise ValueError("expected_one_dispatch_call")
    call = calls[0]
    if call.get("status", "completed") != "completed":
        raise ValueError("dispatch_not_completed")
    if not isinstance(call.get("call_id"), str) or not call["call_id"]:
        raise ValueError("missing_call_id")
    try:
        arguments = require_keys(load_json(call.get("arguments")), {"p_kw"})
        return Action(arguments["p_kw"])
    except (ValueError, TypeError):
        raise ValueError("invalid_dispatch_arguments") from None


class ControllerCallFailure(RuntimeError):
    """No model action was produced; the recorded failure must not count as a decision."""


class LLMController:
    def __init__(
        self, model: Model, config: LLMConfig, *, sleep_fn: Callable[[float], None] = sleep
    ) -> None:
        if not isinstance(config, LLMConfig):
            raise ValueError("config must be LLMConfig")
        self.model = model
        self.config = config
        self._sleep = sleep_fn
        self._records: list[DecisionRecord] = []

    @property
    def records(self) -> tuple[DecisionRecord, ...]:
        return tuple(self._records)

    def act(self, observation: Observation, rng: Random) -> Action:
        if observation.current.step_index != len(self._records):
            raise ValueError("create a fresh LLMController for each episode")
        observation_json = canonical(project_observation(observation, self.config.horizon))
        attempts: list[AttemptRecord] = []
        previous_error = None
        action = None
        for number in range(1, self.config.max_attempts + 1):
            request = make_request(self.config, observation_json, number, previous_error)
            response_json = None
            usage = None
            transport_error = None
            output_error = None
            response_model = None
            retryable = True
            try:
                reply = self.model.complete(request)
                response_json, latency = reply.response_json, reply.latency_s
            except ModelCallError as error:
                transport_error, latency = error.error, error.latency_s
                retryable = transport_error.retryable
            else:
                try:
                    data = load_json(response_json)
                except ValueError:
                    output_error = "invalid_response_json"
                else:
                    usage = response_usage(data)
                    response_model = (
                        data.get("model") if isinstance(data.get("model"), str) else None
                    )
                    try:
                        action = parse_action(data)
                    except ValueError as error:
                        output_error = str(error)  # Only fixed parser codes, never provider text.
            attempts.append(
                AttemptRecord(
                    number,
                    request,
                    response_json,
                    latency,
                    usage,
                    transport_error,
                    output_error,
                    response_model,
                )
            )
            if action is not None:
                break
            previous_error = transport_error.code if transport_error else output_error
            if not retryable:
                break
            if number < self.config.max_attempts and self.config.retry_delay_s:
                self._sleep(self.config.retry_delay_s)
        outcome = (
            "model_decision"
            if action is not None
            else ("fallback_hold" if self.config.failure_policy == "hold" else "call_failure")
        )
        returned = (
            action if action is not None else (Action() if outcome == "fallback_hold" else None)
        )
        self._records.append(
            DecisionRecord(
                observation.current.step_index,
                observation_json,
                tuple(attempts),
                action,
                returned,
                outcome,
            )
        )
        if returned is None:
            raise ControllerCallFailure(
                f"model call failed at step {observation.current.step_index}"
            )
        return returned
