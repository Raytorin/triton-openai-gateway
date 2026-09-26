# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from gateway.app import app
from gateway.generation_types import GenerationResult
from gateway.responses import ResponsesRequest, to_chat


def result(message=None, finish="stop"):
    return GenerationResult("chat-123", 123, "test", message or {"content": "Hello"}, finish,
        {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        headers={"X-Output-Token-Limit": "4096", "X-Token-Usage-Source": "estimated"})


@pytest.mark.parametrize("path", ["/v1/responses", "/responses"])
def test_json_contract_and_defaults(path):
    with patch("gateway.responses.generation.generate", AsyncMock(return_value=result())) as generate:
        response = TestClient(app).post(path, json={"model": "test", "input": "Hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "response" and data["id"].startswith("resp_")
    assert data["output"][0]["content"][0]["text"] == "Hello"
    assert data["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
        "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}
    assert data["store"] is False and data["max_output_tokens"] == 4096
    assert response.headers["x-token-usage-source"] == "estimated"
    assert response.headers["x-request-id"]
    assert generate.call_args.kwargs == {"context_mode": "disabled"}
    assert generate.call_args.args[0].max_completion_tokens is None


@pytest.mark.parametrize("extra", [
    {"store": True}, {"background": True}, {"previous_response_id": "resp_a"},
    {"conversation": "conv_a"}, {"reasoning": {"effort": "high"}},
    {"tools": [{"type": "web_search"}]}, {"max_output_tokens": True},
    {"max_output_tokens": 0}, {"max_output_tokens": -1}, {"max_output_tokens": 1.5},
    {"max_output_tokens": "123"}, {"text": {"verbosity": "high"}},
    {"max_tokens": 100}, {"tool_choice": "required"},
    {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "x"}]}]},
    {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "x"}]}]},
    {"input": [{"type": "function_call_output", "call_id": "x", "output": "done"}]},
    {"text": {"format": {"type": "json_schema"}}},
])
def test_unsupported_or_invalid_rejected_before_generation(extra):
    with patch("gateway.responses.generation.generate", AsyncMock()) as generate:
        response = TestClient(app).post("/v1/responses", json={"model": "test", "input": "Hi", **extra})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["type"] == "invalid_request_error"
    generate.assert_not_called()


def test_function_roundtrip_and_structured_output_mapping():
    request = ResponsesRequest(model="test", instructions="Be precise", input=[
        {"role": "user", "content": "Weather?"},
        {"type": "function_call", "call_id": "call_a", "name": "weather", "arguments": '{"city":"Paris"}'},
        {"type": "function_call", "call_id": "call_b", "name": "weather", "arguments": '{"city":"London"}'},
        {"type": "function_call_output", "call_id": "call_b", "output": "Rain"},
        {"type": "function_call_output", "call_id": "call_a", "output": "Sun"}],
        tools=[{"type": "function", "name": "weather", "parameters": {"type": "object"}}],
        tool_choice={"type": "function", "name": "weather"}, lora_name="adapter",
        text={"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": True}})
    chat = to_chat(request)
    assert [m.role for m in chat.messages] == ["system", "user", "assistant", "tool", "tool"]
    assert [t["id"] for t in chat.messages[2].tool_calls] == ["call_a", "call_b"]
    assert [m.tool_call_id for m in chat.messages[3:]] == ["call_b", "call_a"]
    assert chat.response_format.json_schema.json_schema == {"type": "object"}
    assert chat.lora_name == "adapter"


@pytest.mark.parametrize("url", ["https://example.org/a.png", "data:image/png;base64,aGVsbG8="])
def test_image_mapping(url):
    request = ResponsesRequest(model="test", input=[{"role": "user", "content": [
        {"type": "input_text", "text": "Describe"}, {"type": "input_image", "image_url": url}]}])
    assert to_chat(request).messages[0].content[1] == {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}


def test_incomplete_hides_raw_reasoning():
    with patch("gateway.responses.generation.generate", AsyncMock(return_value=result(
        {"content": "", "reasoning_content": "secret chain"}, "length"))):
        response = TestClient(app).post("/responses", json={"model": "test", "input": "Hello"})
    assert response.json()["status"] == "incomplete"
    assert response.json()["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response.json()["output"] == []
    assert "secret chain" not in response.text
