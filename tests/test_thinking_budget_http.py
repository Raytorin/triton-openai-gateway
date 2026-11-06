# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""HTTP budget transport against the pinned Qwen Jinja and real gRPC serializer.

Only registry/tokenizer loading, admission, and the remote client are replaced.
The tokenizer facade proves rendering/policy; this is not a native inference test.
"""
import asyncio
from contextlib import ExitStack
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import pytest
from tritonclient.utils import deserialize_bytes_tensor

import gateway.app as app_module
from gateway.admission import AdmissionLease
from gateway import generation
from gateway import triton_client as tc
from test_qwen_templates import QwenTemplateTokenizer


REASONING = "Read evidence carefully."
ANSWER = "Concise summary."


class Tokenizer(QwenTemplateTokenizer):
    def __init__(self):
        super().__init__("qwen38")

    def encode(self, text, **kwargs):
        return [7] * len(str(text).split())

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text, **kwargs))


def metadata(*, finished, count):
    return {"version": 1, "finished": finished, "prompt_tokens": 42,
        "outputs": [{"index": 0, "finish_reason": "stop" if finished else None,
            "stop_reason": 2 if finished else None, "completion_tokens": count,
            "first_token_id": 7, "last_token_id": 2 if finished else 7}]}


class Result:
    def __init__(self, text, *, finished, count):
        self.values = {"text_output": np.asarray([text.encode()], dtype=object),
            "generation_metadata": np.asarray([json.dumps(metadata(finished=finished, count=count)).encode()], dtype=object)}

    def as_numpy(self, name):
        return self.values.get(name)


class RemoteIterator:
    def __init__(self, client, requests):
        self.client = client
        self.requests = requests
        self.results = None
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.results is None:
            wire = await anext(self.requests)
            tensors = {tensor.name(): tensor for tensor in wire["inputs"]}
            sampling = json.loads(deserialize_bytes_tensor(tensors["sampling_parameters"]._get_content())[0])
            prompt = deserialize_bytes_tensor(tensors["text_input"]._get_content())[0].decode()
            stream = bool(np.frombuffer(tensors["stream"]._get_content(), dtype=np.bool_)[0])
            self.client.records.append({"sampling": sampling, "prompt": prompt, "stream": stream,
                "metadata_requested": "return_generation_metadata" in tensors})
            # Metadata-capable streaming backends send deltas. Split the closing
            # tag so the actual gateway parser must join it before final content.
            chunks = [REASONING, "</thi", "nk>", ANSWER] if stream else [REASONING + "</think>" + ANSWER]
            self.results = iter(Result(text, finished=i == len(chunks) - 1, count=(i + 1) * 8)
                for i, text in enumerate(chunks))
        await asyncio.sleep(0)
        try:
            return next(self.results), None
        except StopIteration:
            raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True


class RemoteClient:
    def __init__(self):
        self.records = []

    async def get_model_metadata(self, *args, **kwargs):
        return {"inputs": [{"name": "return_generation_metadata"}],
            "outputs": [{"name": "generation_metadata"}]}

    def stream_infer(self, requests, **kwargs):
        return RemoteIterator(self, requests)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def endpoint(tmp_path, monkeypatch):
    for name in ("GATEWAY_REASONING_MODE", "GATEWAY_REASONING_PARSER", "GATEWAY_REASONING_RESPONSE_FIELD"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_8"}')
    (tmp_path / "model.json").write_text(json.dumps({"max_model_len": 16384,
        "reasoning_config": {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}}))
    (tmp_path / "gateway.json").write_text('{"reasoning":{"mode":"separate","parser":"qwen3"},"context_compression":{"mode":"disabled"}}')
    client = RemoteClient()
    with ExitStack() as stack:
        for name, replacement in {
            "resolve": lambda *args: tmp_path,
            "validate_route": lambda *args: None,
            "get_backend": lambda *args: "vllm",
            "get_tokenizer_async": AsyncMock(return_value=(Tokenizer(), tmp_path)),
        }.items():
            stack.enter_context(patch.object(generation.registry, name, replacement))
        stack.enter_context(patch.object(generation.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))))
        stack.enter_context(patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)))
        yield SimpleNamespace(client=client, path=tmp_path)


def request(protocol, stream):
    common = {"model": "budget-http", "thinking_token_budget": 128, "stream": stream}
    if protocol == "chat":
        return "/v1/chat/completions", {**common, "messages": [{"role": "user", "content": "Summarize this."}], "max_tokens": 4096}
    return "/v1/responses", {**common, "input": "Summarize this.", "max_output_tokens": 4096}


def sse(response):
    return [json.loads(line[5:]) for line in response.text.splitlines()
        if line.startswith("data:") and line[5:].strip() != "[DONE]"]


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_budget_reaches_real_transport_and_final_answer_survives(endpoint, protocol, stream):
    path, body = request(protocol, stream)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://test") as client:
        response = await client.post(path, json=body)
    assert response.status_code == 200, response.text
    assert len(endpoint.client.records) == 1
    wire = endpoint.client.records[0]
    assert wire["sampling"]["thinking_token_budget"] == 128
    assert wire["sampling"]["max_tokens"] == 4096
    assert wire["sampling"]["skip_special_tokens"] is False
    assert wire["stream"] is stream
    assert wire["metadata_requested"]
    assert wire["prompt"].endswith("<think>\n")
    if protocol == "chat":
        if stream:
            packets = sse(response)
            content = "".join(packet["choices"][0]["delta"].get("content", "") for packet in packets)
            thought = "".join(packet["choices"][0]["delta"].get("reasoning_content", "") for packet in packets)
            assert content == ANSWER and thought == REASONING
            assert packets[-1]["choices"][0]["finish_reason"] == "stop"
            assert "data: [DONE]" in response.text
        else:
            choice = response.json()["choices"][0]
            assert choice["message"]["content"] == ANSWER
            assert choice["message"]["reasoning_content"] == REASONING
            assert choice["finish_reason"] == "stop"
    else:
        assert REASONING not in response.text
        if stream:
            packets = sse(response)
            assert packets[-1]["type"] == "response.completed"
            final = packets[-1]["response"]
            assert "".join(packet.get("delta", "") for packet in packets if packet["type"] == "response.output_text.delta") == ANSWER
        else:
            final = response.json()
        assert final["status"] == "completed"
        assert final["output"][0]["content"][0]["text"] == ANSWER


@pytest.mark.anyio
@pytest.mark.parametrize("protocol", ["chat", "responses"])
async def test_streaming_budget_without_engine_config_is_http_400_before_transport(endpoint, protocol):
    (endpoint.path / "model.json").write_text('{"max_model_len":16384}')
    path, body = request(protocol, True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://test") as client:
        response = await client.post(path, json=body)
    assert response.status_code == 400
    detail = response.json()["detail"] if protocol == "chat" else response.json()["error"]["message"]
    assert "reasoning_config" in detail
    assert endpoint.client.records == []
    assert "text/event-stream" not in response.headers.get("content-type", "")
