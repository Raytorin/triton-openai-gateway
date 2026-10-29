"""Real generation adapters consume optional engine metadata without retries."""
import asyncio
import functools
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from fastapi import HTTPException

from gateway import triton_client as tc
from gateway.generation_metadata import GenerationOutput, metadata_of
from gateway.prompt import build_usage
from gateway.schemas import ChatCompletionRequest
from test_reasoning import FakeTokenizer, reasoning_settings, parse_sse


def run_async(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def envelope(count=1, reason="stop", finished=True):
    return {"version": 1, "finished": finished, "prompt_tokens": 42,
            "outputs": [{"index": 0, "finish_reason": reason, "stop_reason": 248046,
                         "completion_tokens": count, "first_token_id": 7,
                         "last_token_id": 248046}]}


class Result:
    def __init__(self, text=b"", metadata=None):
        self.data = {"text_output": np.asarray([text], dtype=object)}
        if metadata is not None:
            self.data["generation_metadata"] = np.asarray([json.dumps(metadata).encode()], dtype=object)

    def as_numpy(self, name):
        return self.data.get(name)


class Iterator:
    def __init__(self, results):
        self.results = iter(results)
        self.cancelled = False

    def __aiter__(self):return self

    async def __anext__(self):
        try:return next(self.results), None
        except StopIteration:raise StopAsyncIteration

    def cancel(self):self.cancelled = True


class Client:
    def __init__(self, results, supported=True):
        self.results = results
        self.supported = supported
        self.metadata_calls = 0
        self.inference_calls = 0
        self.requests = []
        self.close = AsyncMock()

    async def get_model_metadata(self, *args, **kwargs):
        self.metadata_calls += 1
        return {"inputs": [{"name": "return_generation_metadata"}] if self.supported else [],
                "outputs": [{"name": "generation_metadata"}] if self.supported else []}

    def stream_infer(self, request_iterator, **kwargs):
        self.inference_calls += 1
        self.requests.append(request_iterator)
        return Iterator(self.results)


def chat_request():
    return ChatCompletionRequest(model="test-metadata", messages=[{"role": "user", "content": "hi"}], stream=True)


@run_async
async def test_nonstream_empty_eos_preserves_facts_and_requests_metadata():
    client = Client([Result(metadata=envelope())])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        output = await tc.call_triton_multimodal("test-metadata", "prompt", {"max_tokens": 99}, [])
        wire = [item async for item in client.requests[0]][0]
    assert output == ""
    assert metadata_of(output).finish_reason == "stop"
    assert build_usage(FakeTokenizer(), "prompt", output)["completion_tokens"] == 1
    assert build_usage(FakeTokenizer(), "prompt", output)["prompt_tokens"] == 42
    assert "return_generation_metadata" in [item.name() for item in wire["inputs"]]
    assert client.inference_calls == 1


@run_async
async def test_stream_keeps_repeated_wire_deltas_and_metadata_only_final():
    client = Client([Result(b"a", envelope(1, None, False)),
                     Result(b"a", envelope(2, None, False)), Result(b"", envelope(2))])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        events = [x async for x in tc.stream_triton_multimodal_to_openai(
            chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 1}, [],
            reasoning_settings=reasoning_settings("disabled"))]
    payloads = parse_sse(events)
    assert "".join(p["choices"][0]["delta"].get("content", "") for p in payloads) == "aa"
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-1]["usage"] == {"prompt_tokens": 42, "completion_tokens": 2, "total_tokens": 44}
    assert client.inference_calls == 1


@run_async
@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_empty_reasoning_uses_actual_engine_finish(reason):
    client = Client([Result(metadata=envelope(reason=reason))])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        events = [x async for x in tc.stream_triton_multimodal_to_openai(
            chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 99}, [], reasoning_settings=reasoning_settings())]
    last = parse_sse(events)[-1]
    assert last["choices"][0]["finish_reason"] == reason
    assert last["usage"]["completion_tokens"] == 1


@run_async
@pytest.mark.parametrize("metadata", [envelope(reason="abort"), {"version": 99}, None, envelope(count=None),
    {"version": 1, "finished": True, "prompt_tokens": 42, "outputs": [{"index": 0, "finish_reason": "stop"}]}])
async def test_bad_or_missing_advertised_metadata_is_not_success(metadata):
    client = Client([Result(metadata=metadata)])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        with pytest.raises(HTTPException) as error:
            await tc.call_triton_multimodal("test-metadata", "prompt", {}, [])
    assert error.value.status_code == 502
    assert client.inference_calls == 1


@run_async
async def test_legacy_backend_request_shape_stays_text_only_and_empty_is_stop():
    client = Client([Result()], supported=False)
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        events = [x async for x in tc.stream_triton_multimodal_to_openai(
            chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 99}, [], reasoning_settings=reasoning_settings())]
        wire = [item async for item in client.requests[0]][0]
    assert [item.name() for item in wire["outputs"]] == ["text_output"]
    assert "return_generation_metadata" not in [item.name() for item in wire["inputs"]]
    assert parse_sse(events)[-1]["choices"][0]["finish_reason"] == "stop"


@run_async
async def test_capability_cache_expires_after_model_upgrade():
    client = Client([Result(b"ok")], supported=False)
    clock = [1000.0]
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)), patch.object(tc.time, "monotonic", side_effect=lambda: clock[0]):
        await tc.call_triton_multimodal("test-metadata", "prompt", {}, [])
        client.supported = True
        await tc.call_triton_multimodal("test-metadata", "prompt", {}, [])
        client.results = [Result(b"ok", envelope(2))]
        clock[0] += 31
        result = await tc.call_triton_multimodal("test-metadata", "prompt", {}, [])
    assert client.metadata_calls == 2
    assert metadata_of(result).completion_tokens == 2
    assert client.inference_calls == 3


@run_async
async def test_http_generation_negotiates_and_decodes_same_contract():
    requests = []
    async def handler(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": [{"name": "return_generation_metadata"}],
                                           "outputs": [{"name": "generation_metadata"}]})
        assert json.loads(request.content)["return_generation_metadata"] is True
        return httpx.Response(200, json={"text_output": "", "generation_metadata": [json.dumps(envelope())]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(tc, "get_http_client", return_value=client):
            result = await tc.call_triton("test-metadata", "prompt", {})
    assert result == "" and metadata_of(result).completion_tokens == 1
    assert [r.method for r in requests] == ["GET", "POST"]


@run_async
@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_http_stream_preserves_metadata_only_completion(reason):
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": [{"name": "return_generation_metadata"}],
                                           "outputs": [{"name": "generation_metadata"}]})
        data = {"text_output": "", "generation_metadata": [json.dumps(envelope(reason=reason))]}
        return httpx.Response(200, headers={"content-type": "text/event-stream"},
                              text="data: " + json.dumps(data) + "\n\n")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(tc, "get_http_client", return_value=client):
            events = [x async for x in tc.stream_triton_to_openai(
                chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 99}, reasoning_settings())]
    last = parse_sse(events)[-1]
    assert last["choices"][0]["finish_reason"] == reason
    assert last["usage"]["completion_tokens"] == 1


@run_async
@pytest.mark.parametrize("reason,expected", [("stop", "tool_calls"), ("length", "length")])
async def test_tool_stream_retains_metadata_through_parser(reason, expected):
    from gateway.generation_metadata import decode_generation_metadata
    raw = '<tool_call>{"name":"weather","arguments":{"city":"London"}}</tool_call>'
    generated = GenerationOutput(raw, decode_generation_metadata(envelope(7, reason)))
    tools = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    with patch.object(tc, "call_triton_multimodal", AsyncMock(return_value=generated)):
        events = [x async for x in tc.stream_tool_aware_multimodal_response(
            chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 2}, [], tools)]
    last = parse_sse(events)[-1]
    assert last["choices"][0]["finish_reason"] == expected
    assert last["usage"]["completion_tokens"] == 7


@run_async
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("reason", ["stop", "length", "abort"])
async def test_chat_http_boundary_exposes_true_finish_or_typed_error(tmp_path, stream, reason):
    from gateway.admission import AdmissionLease
    import gateway.app as gateway_app
    import gateway.generation as generation
    (tmp_path / "model.json").write_text('{"max_model_len":1024}')
    (tmp_path / "config.json").write_text('{"model_type":"qwen3"}')
    (tmp_path / "tokenizer_config.json").write_text('{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}')
    (tmp_path / "gateway.json").write_text('{"reasoning":{"mode":"separate"}}')
    client = Client([Result(metadata=envelope(reason=reason))])
    with (patch.object(gateway_app.registry, "resolve", return_value=tmp_path),
          patch.object(gateway_app.registry, "validate_route"),
          patch.object(gateway_app.registry, "get_backend", return_value="vllm_multimodal"),
          patch.object(gateway_app.registry, "get_tokenizer_async", AsyncMock(return_value=(FakeTokenizer(), tmp_path))),
          patch.object(gateway_app.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))),
          patch.object(tc, "get_grpc_client", AsyncMock(return_value=client))):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app.app), base_url="http://test") as api:
            response = await api.post("/v1/chat/completions", json={"model": "test-metadata", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 99, "stream": stream})
    if reason == "abort":
        if stream:
            assert response.status_code == 200
            payloads = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
            assert any(p.get("error", {}).get("code") == 502 for p in payloads)
            assert not any(c.get("finish_reason") for p in payloads for c in p.get("choices", []))
        else:
            assert response.status_code == 502
    else:
        assert response.status_code == 200
        payload = ([json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")][-1]
                   if stream else response.json())
        assert payload["choices"][0]["finish_reason"] == reason
        assert payload["usage"]["prompt_tokens"] == 42
        assert payload["usage"]["completion_tokens"] == 1
        assert response.headers["x-token-usage-source"] == "backend-preferred"


@run_async
@pytest.mark.parametrize("payload", [None, {"outputs": None}, {"outputs": [None]}, "invalid-json"])
async def test_invalid_http_metadata_response_is_a_typed_upstream_error(payload):
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": [{"name": "return_generation_metadata"}],
                                           "outputs": [{"name": "generation_metadata"}]})
        if payload == "invalid-json":
            return httpx.Response(200, text="not json")
        return httpx.Response(200, content=json.dumps(payload).encode(), headers={"content-type": "application/json"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(tc, "get_http_client", return_value=client):
            with pytest.raises(HTTPException) as error:
                await tc.call_triton("bad-metadata", "prompt", {})
    assert error.value.status_code == 502


@run_async
async def test_invalid_http_stream_metadata_event_is_a_typed_upstream_error():
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": [{"name": "return_generation_metadata"}],
                                           "outputs": [{"name": "generation_metadata"}]})
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text="data: null\n\n")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(tc, "get_http_client", return_value=client):
            with pytest.raises(HTTPException) as error:
                _ = [x async for x in tc.stream_triton_to_openai(
                    chat_request(), FakeTokenizer(), "prompt", {"max_tokens": 99}, reasoning_settings())]
    assert error.value.status_code == 502
