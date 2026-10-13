"""Offline dispatch parsing, retries, and model-visible information boundaries."""

import json
from dataclasses import replace
from random import Random

import pytest

from trialgrid import Action, BatteryConfig, BatteryState, Scenario, run_episode
from trialgrid.controllers.llm import (
    ControllerCallFailure,
    LLMController,
    parse_action,
    project_observation,
    response_usage,
)
from trialgrid.controllers.llm_types import LLMConfig, ModelCallError, ModelReply, TokenUsage
from trialgrid.runner import Observation, ObservationFrame


def response(arguments='{"p_kw":0}'):
    return {
        "status": "completed",
        "model": "fake-snapshot",
        "output": [
            {"type": "function_call", "name": "dispatch", "call_id": "c1", "arguments": arguments}
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "input_tokens_details": {"cached_tokens": 2},
        },
    }


class ScriptedModel:
    def __init__(self, *items):
        self.items = list(items)
        self.requests = []

    def complete(self, request_json):
        self.requests.append(json.loads(request_json))
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return ModelReply(json.dumps(item) if isinstance(item, dict) else item, 0.25)


def market():
    return Scenario(
        1,
        42,
        BatteryConfig(10, 0, 10, 3, 3),
        BatteryState(4),
        (30, 80, -20, 100),
        ((999, 70, 10, 60), (999, 50, 100), (999, 70), (999,)),
        (2, 3, 4, 1),
    )


def observation():
    return Observation(
        market().config, ObservationFrame(0, BatteryState(4), 30, (999, 70, 10, 60), 2), ()
    )


@pytest.mark.parametrize(
    "arguments",
    [
        '{"p_kw":true}',
        '{"p_kw":null}',
        '{"p_kw":"1"}',
        '{"p_kw":NaN}',
        '{"p_kw":Infinity}',
        '{"p_kw":1e999}',
        '{"p_kw":0,"p_kw":1}',
        '{"p_kw":0,"other":1}',
        "{}",
        "[]",
        "not JSON",
        "{",
    ],
)
def test_malformed_arguments_are_not_dispatch_decisions(arguments):
    with pytest.raises(ValueError, match="invalid_dispatch_arguments"):
        parse_action(response(arguments))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(status="incomplete"),
        lambda d: d.update(output=[]),
        lambda d: d.update(output=None),
        lambda d: d.update(output=[None]),
        lambda d: d["output"].append(d["output"][0].copy()),
        lambda d: d["output"][0].update(name="other"),
        lambda d: d["output"][0].update(call_id=""),
        lambda d: d["output"][0].update(status="in_progress"),
        lambda d: d["output"].append({"type": "message", "content": [{"type": "refusal"}]}),
        lambda d: d["output"].append({"type": "web_search_call"}),
    ],
)
def test_invalid_response_envelopes(mutation):
    data = response()
    mutation(data)
    with pytest.raises(ValueError):
        parse_action(data)


def test_reasoning_item_does_not_hide_single_dispatch():
    data = response('{"p_kw":-2.5}')
    data["output"].insert(0, {"type": "reasoning", "summary": []})
    assert parse_action(data) == Action(-2.5)


@pytest.mark.parametrize(
    "failure", [ModelCallError("rate_limit", latency_s=0.1), response('{"p_kw":true}'), "not JSON"]
)
def test_retry_recovers_and_preserves_all_attempts(failure):
    model = ScriptedModel(failure, response('{"p_kw":1}'))
    controller = LLMController(model, LLMConfig("fake"))
    assert controller.act(observation(), Random(1)) == Action(1)
    record = controller.records[0]
    assert record.outcome == "model_decision"
    assert len(record.attempts) == 2
    first, second = record.attempts
    assert first.transport_error is not None or first.output_error is not None
    assert second.usage == TokenUsage(12, 3, 2)
    assert second.response_model_id == "fake-snapshot"
    retry_input = json.loads(model.requests[1]["input"])
    assert retry_input["attempt_number"] == 2
    assert retry_input["previous_error"]
    assert retry_input["observation"] == json.loads(model.requests[0]["input"])["observation"]


@pytest.mark.parametrize("retryable,expected_attempts", [(True, 3), (False, 1)])
@pytest.mark.parametrize("policy", ["raise", "hold"])
def test_exhaustion_and_permanent_errors_are_distinct_from_decisions(
    retryable, expected_attempts, policy
):
    model = ScriptedModel(*(ModelCallError("api_error", retryable, 0.1) for _ in range(3)))
    controller = LLMController(model, LLMConfig("fake", failure_policy=policy))
    if policy == "raise":
        with pytest.raises(ControllerCallFailure):
            controller.act(observation(), Random(0))
    else:
        assert controller.act(observation(), Random(0)) == Action()
    record = controller.records[0]
    assert record.model_requested_action is None
    assert record.returned_action == (Action() if policy == "hold" else None)
    assert record.outcome == ("fallback_hold" if policy == "hold" else "call_failure")
    assert len(record.attempts) == len(model.requests) == expected_attempts
    assert all(a.transport_error.code == "api_error" for a in record.attempts)


def test_finite_infeasible_decision_is_not_retried_or_clamped():
    model = ScriptedModel(*(response('{"p_kw":100}') for _ in range(4)))
    controller = LLMController(model, LLMConfig("fake"))
    episode = run_episode(market(), controller)
    assert episode.metrics.rejected_action_count == 4
    assert len(model.requests) == 4
    for trace, record in zip(episode.trace, controller.records, strict=True):
        assert trace.result.requested_action == record.model_requested_action == Action(100)
        assert record.returned_action == Action(100)
        assert trace.result.applied_action == Action()
        assert record.outcome == "model_decision"


def test_prompt_projection_caps_forecasts_and_drops_stale_forecasts():
    old = observation().current
    current = replace(
        old, step_index=1, price_eur_per_mwh=80, forecast_prices_eur_per_mwh=(123456, 50, 987654)
    )
    obs = Observation(market().config, current, (old,))
    projected = project_observation(obs, 2)
    assert projected["current"]["planning_prices_eur_per_mwh"] == [80, 50]
    assert projected["history"] == [
        {
            "step_index": 0,
            "state": {"energy_kwh": 4},
            "price_eur_per_mwh": 30,
            "reserve_requirement_kwh": 2,
        }
    ]
    altered = replace(
        obs,
        history=(replace(old, forecast_prices_eur_per_mwh=(1, 2, 3, 4)),),
        current=replace(current, forecast_prices_eur_per_mwh=(0, 50, -999)),
    )
    assert project_observation(altered, 2) == projected


@pytest.mark.parametrize("forecast", [(), (999,)])
def test_missing_future_forecast_is_not_synthesized(forecast):
    obs = replace(
        observation(), current=replace(observation().current, forecast_prices_eur_per_mwh=forecast)
    )
    assert project_observation(obs, 4)["current"]["planning_prices_eur_per_mwh"] == [30]


class ReactiveModel:
    def complete(self, request_json):
        payload = json.loads(json.loads(request_json)["input"])
        prices = payload["observation"]["current"]["planning_prices_eur_per_mwh"]
        power = -1 if len(prices) > 1 and prices[1] > prices[0] else 1
        return ModelReply(json.dumps(response(json.dumps({"p_kw": power}))), 0.0)


@pytest.mark.parametrize("boundary", [1, 2, 3])
def test_hidden_future_cannot_change_model_requests_or_decisions(boundary):
    original = market()
    altered = replace(
        original,
        prices_eur_per_mwh=original.prices_eur_per_mwh[:boundary] + (10000,) * (4 - boundary),
        reserve_requirements_kwh=original.reserve_requirements_kwh[:boundary]
        + (10,) * (4 - boundary),
        forecast_prices_eur_per_mwh=original.forecast_prices_eur_per_mwh[:boundary]
        + tuple((-10000,) * (4 - i) for i in range(boundary, 4)),
    )
    first = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    second = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    a, b = run_episode(original, first), run_episode(altered, second)
    assert a.trace[:boundary] == b.trace[:boundary]
    assert first.records[:boundary] == second.records[:boundary]


def test_offline_model_is_deterministic_and_request_is_strict():
    first = LLMController(ReactiveModel(), LLMConfig("fake"))
    second = LLMController(ReactiveModel(), LLMConfig("fake"))
    assert run_episode(market(), first) == run_episode(market(), second)
    assert first.records == second.records
    request = json.loads(first.records[0].attempts[0].request_json)
    assert request["parallel_tool_calls"] is False
    assert request["tool_choice"] == {"type": "function", "name": "dispatch"}
    tool = request["tools"][0]
    assert tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["required"] == ["p_kw"]
    with pytest.raises(ValueError, match="fresh"):
        run_episode(market(), first)


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        [],
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 2, "output_tokens": 1.5},
        {"input_tokens": 2, "output_tokens": 1, "input_tokens_details": "bad"},
        {"input_tokens": 2, "output_tokens": 1, "input_tokens_details": {"cached_tokens": 3}},
        {
            "input_tokens": 2,
            "output_tokens": 1,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
        },
    ],
)
def test_bad_usage_is_unknown_not_zero(usage):
    assert response_usage({"usage": usage}) is None


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": " "},
        {"horizon": True},
        {"horizon": 0},
        {"horizon": 7},
        {"max_attempts": 0},
        {"max_attempts": 6},
        {"max_attempts": 1.5},
        {"failure_policy": "clip"},
        {"prompt_version": "unknown"},
        {"retry_delay_s": -1},
        {"retry_delay_s": float("nan")},
        {"retry_delay_s": 6},
        {"max_output_tokens": 0},
        {"pricing": {}},
    ],
)
def test_invalid_controller_configuration(changes):
    with pytest.raises(ValueError):
        LLMConfig(**({"model_id": "fake"} | changes))


def test_malformed_output_exhaustion_records_response_usage_and_no_decision():
    bad = response('{"p_kw":false}')
    controller = LLMController(ScriptedModel(bad, bad), LLMConfig("fake", max_attempts=2))
    with pytest.raises(ControllerCallFailure):
        controller.act(observation(), Random(0))
    record = controller.records[0]
    assert record.outcome == "call_failure"
    assert record.model_requested_action is record.returned_action is None
    assert len(record.attempts) == 2
    assert all(a.response_json == json.dumps(bad) for a in record.attempts)
    assert all(a.usage == TokenUsage(12, 3, 2) for a in record.attempts)
    assert all(a.latency_s == 0.25 for a in record.attempts)
    assert all(a.output_error == "invalid_dispatch_arguments" for a in record.attempts)


def test_unknown_usage_does_not_turn_valid_action_into_call_failure():
    data = response('{"p_kw":0}')
    data.pop("usage")
    controller = LLMController(ScriptedModel(data), LLMConfig("fake"))
    assert controller.act(observation(), Random(0)) == Action()
    record = controller.records[0]
    assert record.outcome == "model_decision"
    assert record.model_requested_action == Action()
    assert record.attempts[0].usage is None


def test_current_published_forecast_can_influence_model_action():
    original = market()
    altered = replace(
        original,
        forecast_prices_eur_per_mwh=(
            (999, -100, 10, 60),
            *original.forecast_prices_eur_per_mwh[1:],
        ),
    )
    first = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    second = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    a, b = run_episode(original, first), run_episode(altered, second)
    assert a.trace[0].result.requested_action == Action(-1)
    assert b.trace[0].result.requested_action == Action(1)


def test_out_of_horizon_forecasts_cannot_change_any_prompt_even_after_publication():
    original = market()
    altered = replace(
        original,
        forecast_prices_eur_per_mwh=tuple(
            row[:2] + tuple(123456 for _ in row[2:]) for row in original.forecast_prices_eur_per_mwh
        ),
    )
    first = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    second = LLMController(ReactiveModel(), LLMConfig("fake", horizon=2))
    a, b = run_episode(original, first), run_episode(altered, second)
    assert first.records == second.records
    assert [t.result for t in a.trace] == [t.result for t in b.trace]


@pytest.mark.parametrize("details", [False, [], "", 0])
def test_falsey_invalid_usage_details_are_unknown(details):
    assert (
        response_usage(
            {
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 10,
                    "input_tokens_details": details,
                }
            }
        )
        is None
    )
