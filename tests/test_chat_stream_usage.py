# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Public Chat SSE usage negotiation."""

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway import app as gateway_app
from gateway.generation_types import GenerationEvent, GenerationStream


USAGE = {
    "prompt_tokens": 7,
    "completion_tokens": 2,
    "total_tokens": 9,
    "completion_tokens_details": {"reasoning_tokens": 1},
}


@pytest.fixture
def chat_stream(monkeypatch):
    def post(*, fail=False, incomplete=False, **options):
        async def generate(request):
            async def events():
                yield GenerationEvent("chatcmpl-fixture", 12, request.model, {"role": "assistant"})
                yield GenerationEvent("chatcmpl-fixture", 12, request.model, {"content": "Hi"})
                if not incomplete:
                    yield GenerationEvent("chatcmpl-fixture", 12, request.model, {}, "stop", USAGE)
                if fail:
                    raise HTTPException(502, "Synthetic backend failure")

            return GenerationStream(events(), {})

        monkeypatch.setattr(gateway_app, "generate", generate)
        response = TestClient(gateway_app.app).post(
            "/v1/chat/completions",
            json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                **options,
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        return [json.loads(frame) if frame != "[DONE]" else frame for frame in frames]

    return post


def test_explicit_usage_adds_a_separate_final_chunk(chat_stream):
    *choices, usage, done = chat_stream(stream_options={"include_usage": True})
    assert done == "[DONE]"
    assert len(choices) == 3
    assert [chunk["usage"] for chunk in choices] == [None, None, None]
    assert [chunk["choices"][0]["finish_reason"] for chunk in choices] == [None, None, "stop"]
    assert usage == {
        "id": "chatcmpl-fixture",
        "object": "chat.completion.chunk",
        "created": 12,
        "model": "test",
        "choices": [],
        "usage": USAGE,
    }


def test_explicit_false_omits_stream_usage(chat_stream):
    *chunks, done = chat_stream(stream_options={"include_usage": False})
    assert done == "[DONE]"
    assert len(chunks) == 3
    assert all("usage" not in chunk for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.parametrize("options", [
    {}, {"stream_options": None}, {"stream_options": {}},
    {"stream_options": {"include_usage": None}},
])
def test_missing_usage_preference_retains_legacy_chat_stream(chat_stream, options):
    *chunks, done = chat_stream(**options)
    assert done == "[DONE]"
    assert len(chunks) == 3
    assert all("usage" not in chunk for chunk in chunks[:-1])
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"] == USAGE


@pytest.mark.parametrize("failure", ["backend error", "missing finish"])
@pytest.mark.parametrize("include_usage", [None, True, False])
def test_interrupted_stream_cannot_emit_a_final_usage_chunk(chat_stream, failure, include_usage):
    frames = chat_stream(
        fail=failure == "backend error",
        incomplete=failure == "missing finish",
        stream_options={"include_usage": include_usage},
    )
    chunks = [frame for frame in frames if isinstance(frame, dict)]
    assert any("error" in chunk for chunk in chunks)
    assert not any(choice.get("finish_reason") for chunk in chunks for choice in chunk.get("choices", []))
    assert not any(chunk.get("usage") is not None for chunk in chunks)
