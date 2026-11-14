# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from gateway import triton_client
from gateway.app import app
from gateway.generation_types import GenerationEvent, GenerationStream, generation_stream
from gateway.schemas import ChatCompletionRequest


CHAT = "/v1/chat/completions"
RESPONSES = "/v1/responses"
DIAGNOSTIC = "backend-only diagnostic from synthetic test"


def post_stream(monkeypatch, route, stream):
    generate = AsyncMock(return_value=stream)
    monkeypatch.setattr("gateway.app.generate", generate)
    monkeypatch.setattr("gateway.responses.generation.generate", generate)
    body = {"model": "fixture", "stream": True}
    if route == CHAT:
        body["messages"] = [{"role": "user", "content": "Hello"}]
    else:
        body["input"] = "Hello"
    return TestClient(app).post(route, json=body)


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"]


@pytest.mark.parametrize("route", [CHAT, RESPONSES])
def test_buffered_python_backend_failure_is_not_an_assistant_answer(monkeypatch, route):
    request = ChatCompletionRequest(
        model="fixture", messages=[{"role": "user", "content": "Hello"}], stream=True,
    )
    backend = AsyncMock(side_effect=HTTPException(502, DIAGNOSTIC))
    monkeypatch.setattr(triton_client, "call_triton_python_chat", backend)
    def tokenizer(text, **_kwargs):
        return SimpleNamespace(input_ids=text.split())

    stream = generation_stream(triton_client.stream_python_chat_to_openai(
        request, tokenizer, "Hello", [], {},
    ), media_type="text/event-stream", headers={})
    stream.lease = AsyncMock()

    response = post_stream(monkeypatch, route, stream)

    backend.assert_awaited_once()
    assert stream.closed
    stream.lease.release.assert_awaited_once()
    assert DIAGNOSTIC not in response.text
    if route == CHAT:
        assert response.status_code == 200
        assert events(response) == [{"error": {
            "message": "Generation failed", "type": "server_error",
            "param": None, "code": "server_error",
        }}]
        assert response.text.count("data: [DONE]") == 1
    else:
        assert response.status_code == 502
        assert response.json()["error"]["message"] == "Generation failed"


@pytest.mark.parametrize("route", [CHAT, RESPONSES])
@pytest.mark.parametrize("after_finish", [False, True])
def test_late_failure_hides_diagnostics_and_preserves_partial_output(monkeypatch, route, after_finish):
    async def backend():
        yield GenerationEvent("fixture-id", 1, "fixture", {"content": "Partial answer"})
        if after_finish:
            yield GenerationEvent("fixture-id", 1, "fixture", {}, "stop",
                                  {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})
        raise HTTPException(502, DIAGNOSTIC)

    stream = GenerationStream(backend(), {})
    stream.lease = AsyncMock()
    response = post_stream(monkeypatch, route, stream)

    assert response.status_code == 200
    assert DIAGNOSTIC not in response.text
    parsed = events(response)
    if route == CHAT:
        assert parsed[0]["choices"][0]["delta"]["content"] == "Partial answer"
        assert parsed[0]["choices"][0]["finish_reason"] is None
        assert parsed[1]["error"]["message"] == "Generation failed"
        assert len(parsed) == 2
    else:
        assert [event["type"] for event in parsed[-2:]] == ["error", "response.failed"]
        terminal = parsed[-1]["response"]
        assert terminal["error"]["message"] == "Generation failed"
        assert terminal["usage"] is None
        assert terminal["output"][0]["status"] == "incomplete"
        assert terminal["output"][0]["content"][0]["text"] == "Partial answer"
    stream.lease.release.assert_awaited_once()


def test_unexpected_responses_backend_failure_before_headers_is_json(monkeypatch):
    async def backend():
        yield GenerationEvent("fixture-id", 1, "fixture", {"role": "assistant"})
        raise RuntimeError(DIAGNOSTIC)

    stream = GenerationStream(backend(), {})
    stream.lease = AsyncMock()
    response = post_stream(monkeypatch, RESPONSES, stream)

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Generation failed"
    assert DIAGNOSTIC not in response.text
    stream.lease.release.assert_awaited_once()
