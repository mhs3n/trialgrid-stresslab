"""Optional installed-SDK wire smoke tests; mocked HTTP, never a live API call."""

import json

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from trialgrid import BatteryConfig, BatteryState, Scenario  # noqa: E402
from trialgrid.controllers.llm_audit import replay_llm_run, run_llm_episode  # noqa: E402
from trialgrid.controllers.llm_models import OpenAIModel  # noqa: E402
from trialgrid.controllers.llm_types import LLMConfig  # noqa: E402


@pytest.mark.parametrize(
    "failure",
    (
        "http_error",
        "malformed_json",
        "plain_text",
        "json_array",
        "json_null",
        "json_string",
    ),
)
def test_installed_sdk_single_attempts_and_audited_recovery(failure):
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1/responses"
        assert payload["tool_choice"] == {"type": "function", "name": "dispatch"}
        assert payload["parallel_tool_calls"] is False
        assert payload["store"] is False
        if len(requests) == 1:
            if failure == "plain_text":
                return httpx.Response(200, text="hello", headers={"content-type": "text/plain"})
            if failure in ("json_array", "json_null", "json_string"):
                body = {"json_array": "[]", "json_null": "null", "json_string": '"hello"'}[failure]
                return httpx.Response(200, text=body, headers={"content-type": "application/json"})
            if failure == "malformed_json":
                return httpx.Response(
                    200, text="not JSON", headers={"content-type": "application/json"}
                )
            return httpx.Response(429, json={"error": {"message": "synthetic transient"}})
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "test-snapshot",
                "status": "completed",
                "error": None,
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "c_test",
                        "status": "completed",
                        "name": "dispatch",
                        "arguments": '{"p_kw":1}',
                    }
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        client = openai.OpenAI(api_key="test-only-placeholder", http_client=http_client)
        model = OpenAIModel("test-only-placeholder", client=client)
        scenario = Scenario(
            1, 0, BatteryConfig(2, 0, 2, 1, 1), BatteryState(1), (100,), ((),), (0,)
        )
        run = run_llm_episode(scenario, model, LLMConfig("test-model"))
    assert len(requests) == 2  # SDK must not hide retries from the audit.
    assert len(run.decisions[0].attempts) == 2
    assert run.decisions[0].attempts[0].transport_error is not None
    assert run.decisions[0].attempts[1].usage.input_tokens == 10
    assert run.episode.metrics.profit_eur == pytest.approx(0.1)
    assert replay_llm_run(run) == run
