"""Offline transport contracts: no credentials, SDK installation, or network needed."""

import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trialgrid.controllers.llm_models import FakeModel, OpenAIModel
from trialgrid.controllers.llm_types import ModelCallError, ModelReply


@pytest.fixture
def sdk(monkeypatch):
    class APIConnectionError(Exception):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class APIResponseValidationError(Exception):
        pass

    class APIStatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code
            super().__init__("private response and API secret")

    client = SimpleNamespace(responses=SimpleNamespace(create=Mock()))
    client.with_options = Mock(return_value=client)
    client.responses.create.return_value = SimpleNamespace(
        model_dump_json=lambda: '{"status":"completed"}'
    )
    module = SimpleNamespace(
        OpenAI=Mock(return_value=client),
        APIConnectionError=APIConnectionError,
        APITimeoutError=APITimeoutError,
        APIStatusError=APIStatusError,
        APIResponseValidationError=APIResponseValidationError,
    )
    monkeypatch.setitem(sys.modules, "openai", module)
    return module, client


def test_adapter_preserves_wire_request_and_disables_hidden_retries(sdk):
    module, client = sdk
    model = OpenAIModel("test-key", timeout_s=12.5)
    module.OpenAI.assert_called_once_with(api_key="test-key", max_retries=0, timeout=12.5)
    payload = {
        "model": "test-model",
        "input": "bounded observation",
        "store": False,
        "parallel_tool_calls": False,
        "tool_choice": {"type": "function", "name": "dispatch"},
        "tools": [{"type": "function", "name": "dispatch", "strict": True}],
    }
    reply = model.complete(json.dumps(payload))
    client.responses.create.assert_called_once_with(**payload)
    assert reply.response_json == '{"status":"completed"}'
    assert reply.latency_s >= 0


@pytest.mark.parametrize(
    "status,retryable",
    [
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (408, True),
        (409, True),
        (422, False),
        (429, True),
        (500, True),
        (503, True),
    ],
)
def test_api_status_sanitization_and_retry_classification(sdk, status, retryable):
    module, client = sdk
    client.responses.create.side_effect = module.APIStatusError(status)
    with pytest.raises(ModelCallError) as error:
        OpenAIModel("test-key").complete('{"model":"test"}')
    assert str(error.value) == f"api_status_{status}"
    assert error.value.error.retryable is retryable
    assert error.value.latency_s >= 0
    assert error.value.__suppress_context__
    assert client.responses.create.call_count == 1


@pytest.mark.parametrize(
    "kind,code",
    [
        ("APITimeoutError", "api_timeout"),
        ("APIConnectionError", "api_connection"),
    ],
)
def test_connection_errors_are_sanitized(sdk, kind, code):
    module, client = sdk
    client.responses.create.side_effect = getattr(module, kind)("secret")
    with pytest.raises(ModelCallError) as error:
        OpenAIModel("test-key").complete("{}")
    assert str(error.value) == code
    assert error.value.error.retryable


def test_programming_errors_propagate(sdk):
    _, client = sdk
    client.responses.create.side_effect = TypeError("programming defect")
    with pytest.raises(TypeError, match="programming defect"):
        OpenAIModel("test-key").complete("{}")


def test_injected_client_is_used_without_constructing_sdk_client(sdk):
    module, client = sdk
    model = OpenAIModel("test-key", client=client)
    model.complete("{}")
    module.OpenAI.assert_not_called()
    client.with_options.assert_called_once_with(max_retries=0, timeout=30.0)
    client.responses.create.assert_called_once_with()


@pytest.mark.parametrize("stage", ["create", "serialize"])
@pytest.mark.parametrize("kind", ["sdk_validation", "json_decode"])
def test_response_decoding_failures_are_sanitized_and_measured(sdk, monkeypatch, stage, kind):
    module, client = sdk
    error = (
        module.APIResponseValidationError("private response")
        if kind == "sdk_validation"
        else json.JSONDecodeError("private response", "secret", 0)
    )
    if stage == "create":
        client.responses.create.side_effect = error
    else:
        client.responses.create.return_value.model_dump_json = Mock(side_effect=error)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr("trialgrid.controllers.llm_models.perf_counter", lambda: next(ticks))
    with pytest.raises(ModelCallError) as caught:
        OpenAIModel("test-key").complete("{}")
    assert str(caught.value) == "api_invalid_response"
    assert caught.value.error.retryable
    assert caught.value.latency_s == 2.5
    assert caught.value.__suppress_context__


def test_serialization_programming_error_propagates(sdk):
    _, client = sdk
    client.responses.create.return_value.model_dump_json = Mock(side_effect=TypeError("bug"))
    with pytest.raises(TypeError, match="bug"):
        OpenAIModel("test-key").complete("{}")


def test_injected_options_copy_is_the_client_used(sdk):
    module, client = sdk
    original = SimpleNamespace(with_options=Mock(return_value=client))
    OpenAIModel("test-key", timeout_s=7, client=original).complete("{}")
    original.with_options.assert_called_once_with(max_retries=0, timeout=7)
    module.OpenAI.assert_not_called()
    client.responses.create.assert_called_once_with()


@pytest.mark.parametrize("payload", ["[]", '{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_invalid_json_never_calls_sdk(sdk, payload):
    _, client = sdk
    with pytest.raises(ValueError):
        OpenAIModel("test-key").complete(payload)
    client.responses.create.assert_not_called()


def test_optional_sdk_absence_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    with pytest.raises(ImportError, match=r"trialgrid\[llm\]"):
        OpenAIModel("test-key")


def test_import_and_fake_need_no_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['openai']=None; "
            "from trialgrid.controllers.llm_models import FakeModel; "
            'assert FakeModel().complete(\'{"model":"fake"}\').latency_s == 0',
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_fake_repeated_requests_are_deterministic_and_auditable():
    fake = FakeModel()
    request = '{"model":"fake-hold","input":"hour 4"}'
    first = fake.complete(request)
    assert first == fake.complete(request) == FakeModel().complete(request)
    assert fake.requests == [request, request]
    response = json.loads(first.response_json)
    assert response["model"] == "fake-hold"
    assert response["status"] == "completed"
    assert json.loads(response["output"][0]["arguments"]) == {"p_kw": 0.0}
    assert response["usage"]["input_tokens"] == response["usage"]["output_tokens"] == 0
    assert first.latency_s == 0
    assert isinstance(first, ModelReply)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_timeout_validation(sdk, timeout):
    with pytest.raises(ValueError):
        OpenAIModel("test-key", timeout_s=timeout)


@pytest.mark.parametrize("response", [None, [], "plain text", 1, {"model_dump_json": "bad"}])
def test_invalid_response_shapes_enter_recorded_error_path(sdk, response):
    _, client = sdk
    client.responses.create.return_value = response
    with pytest.raises(ModelCallError) as caught:
        OpenAIModel("test-key").complete("{}")
    assert caught.value.error.code == "api_invalid_response"
    assert caught.value.error.retryable is True
