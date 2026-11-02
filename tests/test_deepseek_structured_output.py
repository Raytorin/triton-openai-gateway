# SPDX-License-Identifier: Apache-2.0
"""DeepSeek JSON protocol tests; the remote engine is a deterministic CPU double.

These tests verify request forwarding and response parsing, not GPU grammar
enforcement. The same adversarial cases belong in actual engine qualification.
"""
import json

import pytest

import gateway.triton_client as tc
from gateway.deepseek import EOS
from test_deepseek_protocol import TOOLS, body, chunks, endpoint, post
from test_deepseek_finish_metadata import engine_response, metadata_engine


SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]},
        "count": {"type": "integer", "enum": [1, 2, 3]},
        "marker": {"type": "string"},
    },
    "required": ["color", "count", "marker"],
    "additionalProperties": False,
}
VALUE = {"color": "blue", "count": 2,
         "marker": 'Русский текст: "<think>" \\ </think> <｜DSML｜ calls> </｜DSML｜ calls> ' + EOS}
TEXT = json.dumps(VALUE, ensure_ascii=False)
PROMPT = "Write a long prose story. Do not output JSON, numbers or code."


@pytest.fixture
def anyio_backend():
    return "asyncio"


def request(route, stream, **extra):
    if route == "/v1/chat/completions":
        return body(messages=[{"role": "user", "content": PROMPT}], stream=stream,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "answer", "strict": True, "schema": SCHEMA}}, **extra)
    return {"model": "deepseek", "input": PROMPT, "stream": stream,
            "text": {"format": {"type": "json_schema", "name": "answer",
                                  "strict": True, "schema": SCHEMA}}, **extra}


def install_output(monkeypatch, endpoint, text, *, reason="stop", eos=False):
    async def wire(*args, **kwargs):
        item = args[2][0]
        endpoint.records.append((item.prompt, item.sampling, item.stream))
        raw = text + (EOS if eos else "")
        try:
            for index, start in enumerate(range(0, len(raw), endpoint.step), 1):
                yield engine_response(raw[start:start + endpoint.step], completion_tokens=index)
            yield engine_response("", finished=True, reason=reason, completion_tokens=max(101, len(raw) + 1),
                                  stop_reason=1 if reason == "stop" else None)
        finally:
            endpoint.closed += 1
    monkeypatch.setattr(tc, "_stream_grpc_results", wire)


def result_text(response, route, stream, reason="stop"):
    assert response.status_code == 200, response.text
    if stream:
        events = chunks(response)
        assert not [event for event in events if "error" in event], response.text
        if route == "/v1/chat/completions":
            choices = [choice for event in events for choice in event.get("choices", [])]
            assert [choice["finish_reason"] for choice in choices if choice.get("finish_reason")] == [reason]
            assert not [choice for choice in choices if choice["delta"].get("reasoning_content")]
            assert not [choice for choice in choices if choice["delta"].get("tool_calls")]
            return "".join(choice["delta"].get("content", "") for choice in choices)
        final_type = "response.incomplete" if reason == "length" else "response.completed"
        assert events[-1]["type"] == final_type, response.text
        assert not [event for event in events if event["type"] in {
            "response.failed", "response.function_call_arguments.delta"}], response.text
        return "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta")
    data = response.json()
    if route == "/v1/chat/completions":
        choice = data["choices"][0]
        assert choice["finish_reason"] == reason
        assert not choice["message"].get("reasoning_content")
        assert not choice["message"].get("tool_calls")
        return choice["message"]["content"]
    assert data["status"] == ("incomplete" if reason == "length" else "completed")
    return "".join(part["text"] for item in data["output"] if item["type"] == "message"
                   for part in item["content"] if part["type"] == "output_text")


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("step", [1, 7, 10000])
async def test_json_schema_preserves_literal_markers_and_forwards_grammar(
        endpoint, metadata_engine, monkeypatch, route, stream, step):
    endpoint.step = step
    install_output(monkeypatch, endpoint, TEXT, eos=True)
    response = await post(request(route, stream), route)
    visible = result_text(response, route, stream)
    assert visible == TEXT
    value = json.loads(visible)
    assert value == VALUE
    assert set(value) == {"color", "count", "marker"}
    assert value["color"] in {"red", "green", "blue"}
    assert type(value["count"]) is int and value["count"] in {1, 2, 3}
    prompt, sampling, remote_stream = endpoint.records[0]
    assert PROMPT in prompt
    assert not prompt.endswith("<think>")
    assert json.loads(sampling["structured_outputs"]) == {"json": SCHEMA}
    assert remote_stream is True


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_json_object_keeps_service_marker_strings(endpoint, metadata_engine, monkeypatch, stream):
    install_output(monkeypatch, endpoint, TEXT)
    response = await post(body(stream=stream, response_format={"type": "json_object"}))
    assert result_text(response, "/v1/chat/completions", stream) == TEXT
    assert json.loads(endpoint.records[0][1]["structured_outputs"]) == {"json_object": True}


@pytest.mark.anyio
@pytest.mark.parametrize("raw", ['{"color":', '{"count":NaN}', '{"count":1,"count":2}',
                                  '{"marker":"unterminated', '{"color":"blue"}' + EOS[:8]])
@pytest.mark.parametrize("stream", [False, True])
async def test_normal_stop_cannot_successfully_finish_invalid_json(
        endpoint, metadata_engine, monkeypatch, raw, stream):
    install_output(monkeypatch, endpoint, raw)
    response = await post(request("/v1/chat/completions", stream))
    if stream:
        events = chunks(response)
        assert any("error" in event for event in events), response.text
        assert not [choice for event in events for choice in event.get("choices", [])
                    if choice.get("finish_reason")]
    else:
        assert response.status_code == 502, response.text


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_engine_length_preserves_truncated_json(endpoint, metadata_engine, monkeypatch, route, stream):
    raw = '{"marker":"literal <think> and '
    install_output(monkeypatch, endpoint, raw, reason="length")
    response = await post(request(route, stream), route)
    assert result_text(response, route, stream, reason="length") == raw


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("terminal", ["abort", "unexpected", "missing"])
async def test_json_eos_does_not_hide_engine_terminal_failure(
        endpoint, metadata_engine, monkeypatch, route, stream, terminal):
    drained = []

    async def wire(*args, **kwargs):
        try:
            yield engine_response(TEXT + EOS, completion_tokens=4)
            drained.append("tail")
            yield engine_response("ignored after parser EOS", completion_tokens=5)
            drained.append("terminal")
            if terminal != "missing":
                yield engine_response("", finished=True, reason=terminal, completion_tokens=7)
        finally:
            endpoint.closed += 1

    monkeypatch.setattr(tc, "_stream_grpc_results", wire)
    response = await post(request(route, stream), route)
    assert drained == ["tail", "terminal"]
    assert endpoint.closed == 1
    if not stream:
        assert response.status_code == 502, response.text
        return

    assert response.status_code == 200, response.text
    events = chunks(response)
    assert any("error" in event or event.get("type") == "error" for event in events), response.text
    if route == "/v1/chat/completions":
        choices = [choice for event in events for choice in event.get("choices", [])]
        assert "".join(choice["delta"].get("content", "") for choice in choices) == TEXT
        assert not any(choice.get("finish_reason") for choice in choices)
    else:
        assert "".join(event["delta"] for event in events
                       if event.get("type") == "response.output_text.delta") == TEXT
        assert events[-1]["type"] == "response.failed", response.text
        assert events[-1]["response"]["status"] == "failed"
        assert not any(event.get("type") in {"response.completed", "response.incomplete"}
                       for event in events)


@pytest.mark.anyio
@pytest.mark.parametrize("choice", ["auto", "none"])
async def test_json_schema_with_optional_tools_returns_content(endpoint, metadata_engine, monkeypatch, choice):
    install_output(monkeypatch, endpoint, TEXT)
    response = await post(request("/v1/chat/completions", False, tools=TOOLS, tool_choice=choice))
    assert result_text(response, "/v1/chat/completions", False) == TEXT


@pytest.mark.anyio
@pytest.mark.parametrize("choice", ["required", {"type": "function", "function": {"name": "typed"}}])
async def test_json_schema_rejects_required_tool_call_before_engine(endpoint, choice):
    response = await post(request("/v1/chat/completions", False, tools=TOOLS, tool_choice=choice))
    assert response.status_code == 400, response.text
    assert not endpoint.records


@pytest.mark.anyio
async def test_json_schema_rejects_explicit_thinking_before_engine(endpoint):
    response = await post(request("/v1/chat/completions", False,
                                  chat_template_kwargs={"enable_thinking": True}))
    assert response.status_code == 400, response.text
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
async def test_json_object_rejects_non_object_normal_stop(endpoint, metadata_engine, monkeypatch, stream):
    install_output(monkeypatch, endpoint, '["valid JSON but not an object"]')
    response = await post(body(stream=stream, response_format={"type": "json_object"}))
    if stream:
        assert any("error" in event for event in chunks(response)), response.text
    else:
        assert response.status_code == 502, response.text


@pytest.mark.anyio
@pytest.mark.parametrize("value,schema", [
    (["<think>", EOS], {"type": "array", "items": {"type": "string"}}),
    ("</think> <｜DSML｜ calls>", {"type": "string"}),
])
async def test_json_schema_allows_valid_non_object_roots(endpoint, metadata_engine, monkeypatch, value, schema):
    raw = json.dumps(value, ensure_ascii=False)
    install_output(monkeypatch, endpoint, raw, eos=True)
    payload = request("/v1/chat/completions", False)
    payload["response_format"]["json_schema"]["schema"] = schema
    response = await post(payload)
    assert result_text(response, "/v1/chat/completions", False) == raw
    assert json.loads(endpoint.records[0][1]["structured_outputs"]) == {"json": schema}


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
async def test_malformed_schema_container_is_rejected_before_engine(endpoint, route):
    payload = request(route, False)
    if route == "/v1/chat/completions":
        payload["response_format"]["json_schema"]["schema"] = "not a schema object"
    else:
        payload["text"]["format"]["schema"] = "not a schema object"
    response = await post(payload, route)
    assert response.status_code == (422 if route == "/v1/chat/completions" else 400), response.text
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_engine_grammar_failure_is_not_a_success(endpoint, metadata_engine, monkeypatch, route, stream):
    calls = []
    async def wire(*args, **kwargs):
        calls.append(args[2][0].sampling)
        raise RuntimeError("Structured output compilation failed")
        yield  # makes the failing remote boundary an async generator
    monkeypatch.setattr(tc, "_stream_grpc_results", wire)
    response = await post(request(route, stream), route)
    assert len(calls) == 1  # Do not retry generation after a compile failure.
    if stream and route == "/v1/chat/completions":
        events = chunks(response)
        assert any("error" in event for event in events), response.text
        assert not [choice for event in events for choice in event.get("choices", [])
                    if choice.get("finish_reason")]
    else:
        # Responses waits for the first visible event before committing SSE headers.
        assert response.status_code == 502, response.text
