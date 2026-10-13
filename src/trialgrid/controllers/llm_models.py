"""Optional OpenAI transport and a deterministic, synthetic offline hold model."""

import hashlib
import importlib
import json
from time import perf_counter

from trialgrid.controllers.llm_types import ModelCallError, ModelReply, nonnegative
from trialgrid.scenario import load_json


def _request(text: str) -> dict:
    payload = load_json(text)
    # JSON's exponent syntax can overflow a float despite rejecting NaN literals.
    json.dumps(payload, allow_nan=False)
    return payload


class OpenAIModel:
    """One observable SDK attempt per call; import the optional SDK only on construction."""

    def __init__(self, api_key: str, timeout_s: float = 30.0, *, client=None) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be nonempty")
        nonnegative("timeout_s", timeout_s)
        if timeout_s == 0:
            raise ValueError("timeout_s must be positive")
        try:
            self._sdk = importlib.import_module("openai")
        except ModuleNotFoundError as exc:
            if exc.name != "openai":
                raise
            raise ImportError("Install trialgrid[llm] to use OpenAIModel") from None
        self._client = (
            client.with_options(max_retries=0, timeout=timeout_s)
            if client is not None
            else self._sdk.OpenAI(api_key=api_key, max_retries=0, timeout=timeout_s)
        )

    def complete(self, request_json: str) -> ModelReply:
        payload = _request(request_json)
        started = perf_counter()
        try:
            response = self._client.responses.create(**payload)
            serialize = getattr(response, "model_dump_json", None)
            if not callable(serialize):
                raise ModelCallError("api_invalid_response", True, perf_counter() - started)
            response_json = serialize()
        except (self._sdk.APIResponseValidationError, json.JSONDecodeError):
            raise ModelCallError("api_invalid_response", True, perf_counter() - started) from None
        except self._sdk.APITimeoutError:
            raise ModelCallError("api_timeout", True, perf_counter() - started) from None
        except self._sdk.APIConnectionError:
            raise ModelCallError("api_connection", True, perf_counter() - started) from None
        except self._sdk.APIStatusError as exc:
            status = exc.status_code
            retryable = status in (408, 409, 429) or status >= 500
            raise ModelCallError(
                f"api_status_{status}", retryable, perf_counter() - started
            ) from None
        return ModelReply(response_json, perf_counter() - started)


class FakeModel:
    """Synthetic hold responses with zero tokens/latency; not model performance evidence."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def complete(self, request_json: str) -> ModelReply:
        payload = _request(request_json)
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("request model must be nonempty")
        self.requests.append(request_json)
        identity = hashlib.sha256(request_json.encode()).hexdigest()[:24]
        response = {
            "id": f"fake_{identity}",
            "model": model,
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": "dispatch",
                    "call_id": f"fake_call_{identity}",
                    "arguments": '{"p_kw":0.0}',
                }
            ],
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
            },
        }
        return ModelReply(json.dumps(response, sort_keys=True, allow_nan=False), 0.0)
