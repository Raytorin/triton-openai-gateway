# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP clients -> gateway ASGI -> deterministic generation fixture.

Run with the isolated requirements-client.txt environment:
python tests/client_compatibility.py --server-python /path/to/test-env/bin/python
No model, credentials, or remote inference required. Does not claim runtime proof.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serve():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from gateway.app import app
    from gateway.generation_types import GenerationResult, GenerationStream, GenerationEvent

    captured = []
    async def generate(request, **kwargs):
        if request.model in {"invalid-request-fixture", "media-limit-fixture"}:
            from fastapi import HTTPException
            if request.model == "media-limit-fixture":
                raise HTTPException(413, "Media item exceeds the configured size limit")
            raise HTTPException(400, "Malformed test parameter")
        if request.tools and not any(m.role == "tool" for m in request.messages):
            message = {"tool_calls": [{"id": "call_weather", "type": "function", "function": {
                "name": "weather", "arguments": '{"city":"Paris"}'}}]}
            if any(m.content == "Mixed weather?" for m in request.messages):
                message["content"] = "I will check the weather."
            finish = "tool_calls"
        else:
            message = {"content": '{"answer":"sunny"}' if request.response_format else "It is sunny"}
            finish = "stop"
        usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        if any(m.content == "visual-usage" for m in request.messages):
            usage.update({
                "vision_usage": {
                    "main": {"status": "available", "source": "engine", "unit": "embedding_positions",
                             "by_source": {"image": 3}, "calls": 1},
                    "preprocessing": {"status": "available", "source": "engine", "unit": "embedding_positions",
                                      "by_source": {"pdf": 7}, "calls": 2},
                },
                "vision_context_reservation": {"source": "estimated", "unit": "context_positions",
                                               "by_scope": {"main": 1024}},
            })
        if any(m.content == "exhaust-output" for m in request.messages):
            # Model a backend that exhausts its budget before visible output.
            message, finish = {"content": ""}, "length"
            count = request.max_completion_tokens or 4096
            usage = {"prompt_tokens": 5, "completion_tokens": count, "total_tokens": 5 + count}
        headers = {"X-Output-Token-Limit": str(request.max_completion_tokens or 4096)}
        if not request.stream:
            return GenerationResult("chat-1", 123, request.model, message, finish, usage, headers=headers)
        async def events():
            delta = dict(message)
            if "tool_calls" in delta:
                delta["tool_calls"] = [dict(t, index=i) for i, t in enumerate(delta["tool_calls"])]
            yield GenerationEvent("chat-1", 123, request.model, delta)
            yield GenerationEvent("chat-1", 123, request.model, {}, finish, usage)
        return GenerationStream(events(), headers)

    def deepseek_response(body, record):
        # Reuse the pinned Jinja fixture renderer, not the deterministic API
        # result double above. Only gRPC/model execution is replaced here.
        from contextlib import ExitStack
        import tempfile
        from types import SimpleNamespace
        sys.path.insert(0, str(ROOT / "tests"))
        from test_deepseek_protocol import Tokenizer
        from deepseek_backend_wire import production_text_wire
        from gateway.admission import AdmissionLease
        from gateway import generation, triton_client as tc
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory(prefix="deepseek-client-") as folder:
            model_path = Path(folder)
            (model_path / "model.json").write_text(json.dumps({"model": "/fixed/DeepSeek-V4.1-Flash", "max_model_len": 32768}))
            (model_path / "gateway.json").write_text('{"reasoning":{"mode":"separate"}}')
            def inputs(prompt, sampling, images, stream, **kwargs):
                record["backend_prompt"] = prompt
                record["backend_sampling"] = sampling
                return [SimpleNamespace(prompt=prompt)]
            async def backend(operation, model, inputs, outputs, **kwargs):
                prompt = inputs[0].prompt
                text = "It is sunny"
                if "Weather?" in prompt and "<tool_result>" not in prompt:
                    text = ('<｜DSML｜ calls><｜DSML｜ invoke name="weather">'
                            '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>'
                            '</｜DSML｜ invoke></｜DSML｜ calls>')
                if prompt.endswith("<think>"):
                    text = "Inspect first.</think>" + text
                with production_text_wire() as serialize:
                    for end in range(3, len(text) + 3, 3):
                        yield serialize(text[:end])
            with ExitStack() as stack:
                for name, replacement in {
                    "resolve": lambda *args: model_path,
                    "get_backend": lambda *args: "vllm_deepseek",
                    "get_tokenizer_async": AsyncMock(return_value=(tokenizer, model_path)),
                }.items():
                    stack.enter_context(patch.object(generation.registry, name, replacement))
                stack.enter_context(patch.object(generation.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))))
                stack.enter_context(patch.object(tc, "_build_grpc_generate_inputs", inputs))
                stack.enter_context(patch.object(tc, "get_grpc_client", AsyncMock(return_value=object())))
                stack.enter_context(patch.object(tc, "_stream_grpc_results", backend))
                return TestClient(app).post("/v1/chat/completions", json=body)

    def context_response(path, body):
        # Keep generation, prompt measurement, context rejection, and HTTP error
        # rendering real. Replace only unavailable model/registry resources.
        from contextlib import ExitStack
        import tempfile
        from types import SimpleNamespace
        from gateway import generation
        from gateway.admission import AdmissionLease

        class WhitespaceTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return " ".join(str(message.get("content", "")) for message in messages)
            def encode(self, text, **kwargs):
                return text.split()
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=self.encode(text))

        with tempfile.TemporaryDirectory(prefix="context-client-") as folder:
            model_path = Path(folder)
            (model_path / "model.json").write_text('{"max_model_len":128}')
            (model_path / "gateway.json").write_text(json.dumps({
                "context_compression": {"mode": "disabled", "safety_margin_tokens": 2},
            }))
            with ExitStack() as stack:
                for name, replacement in {
                    "resolve": lambda *args: model_path,
                    "validate_route": lambda *args: None,
                    "get_backend": lambda *args: "vllm",
                    "get_tokenizer_async": AsyncMock(return_value=(WhitespaceTokenizer(), model_path)),
                }.items():
                    stack.enter_context(patch.object(generation.registry, name, replacement))
                stack.enter_context(patch.object(generation.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))))
                stack.enter_context(patch.object(generation, "call_triton_multimodal", AsyncMock(
                    side_effect=AssertionError("Overflow must be rejected before inference"))))
                return TestClient(app).post(path, json=body)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(captured).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({"path": self.path, "body": body})
            if body.get("model") == "deepseek-fixture":
                response = deepseek_response(body, captured[-1])
            elif body.get("model") == "context-overflow-fixture":
                response = context_response(self.path, body)
            else:
                with patch("gateway.responses.generation.generate", generate), patch("gateway.app.generate", generate):
                    response = TestClient(app).post(self.path, json=body)
            self.send_response(response.status_code)
            for k,v in response.headers.items():
                if k.lower() not in {"transfer-encoding", "connection"}:
                    self.send_header(k,v)
            self.end_headers()
            self.wfile.write(response.content)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_port, flush=True)
    server.serve_forever()


def check_clients(url):
    # Disable optional remote model-price/tokenizer lookups. Test traffic stays local.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import httpx
    import openai
    import litellm
    from importlib.metadata import version
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    tool = {"type": "function", "name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    text = {"format": {"type": "json_schema", "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"], "additionalProperties": False}}}
    client = openai.OpenAI(base_url=url+"/v1", api_key="fixture", max_retries=0)
    def lite(**kwargs):
        return litellm.responses(model="openai/test", api_base=url+"/v1", api_key="fixture", **kwargs)
    client_starts = []
    for name, create in [("openai", lambda **kw: client.responses.create(model="test", **kw)), ("litellm", lite)]:
        client_starts.append(len(httpx.get(url+"/captured").json()))
        response = create(input="Hello", store=False)
        assert response.output_text == "It is sunny", (name, response)
        events = list(create(input="Hello", stream=True, max_output_tokens=77, store=False))
        assert events[-1].type == "response.completed", (name, events)
        assert events[-1].response.output[0].content[0].text == "It is sunny"
        assert json.loads(create(input="JSON", text=text, store=False).output_text) == {"answer": "sunny"}
        history = [{"role": "user", "content": "Weather?"}]
        first = create(input=history, tools=[tool], store=False)
        call = first.output[0]
        assert call.type == "function_call" and call.name == "weather"
        history += [call.model_dump(exclude_none=True), {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"}]
        second = create(input=history, tools=[tool], store=False)
        assert second.output_text == "It is sunny"
        tools_stream = list(create(input="Weather?", tools=[tool], stream=True, store=False))
        assert tools_stream[-1].response.output[0].arguments == '{"city":"Paris"}'
        schema_stream = list(create(input="JSON", text=text, stream=True, store=False))
        assert json.loads(schema_stream[-1].response.output[0].content[0].text) == {"answer": "sunny"}
        mixed = list(create(input="Mixed weather?", tools=[tool], stream=True, store=False))[-1].response
        assert [item.type for item in mixed.output] == ["message", "function_call"]
        mixed_history = [{"role": "user", "content": "Mixed weather?"},
            *(item.model_dump(exclude_none=True) for item in mixed.output),
            {"type": "function_call_output", "call_id": mixed.output[1].call_id, "output": "sunny"}]
        assert create(input=mixed_history, tools=[tool], store=False).output_text == "It is sunny"
        for streaming in (False, True):
            visual = create(input="visual-usage", stream=streaming, store=False)
            if streaming:
                visual = list(visual)[-1].response
            visual_usage = visual.usage.model_dump(exclude_none=True)
            assert visual_usage["input_tokens"] == 5 and visual_usage["total_tokens"] == 9
            assert visual_usage["vision_usage"]["main"]["by_source"] == {"image": 3}, (name, visual_usage)
            assert visual_usage["vision_usage"]["preprocessing"]["by_source"] == {"pdf": 7}, (name, visual_usage)
            assert visual_usage["vision_context_reservation"]["unit"] == "context_positions"
        print(f"{name}: text, streaming, function roundtrip, mixed text/tool stream roundtrip, JSON Schema and schema stream passed")
    # Exercise the SDK event accumulator, not just event iteration.
    with client.responses.stream(model="test", input="Hello", store=False) as stream:
        response = stream.get_final_response()
        assert response.output_text == "It is sunny"
    captured = httpx.get(url+"/captured").json()
    assert all(item["path"] == "/v1/responses" for item in captured)
    for start in client_starts:
        assert "max_output_tokens" not in captured[start]["body"]
        assert captured[start+1]["body"]["max_output_tokens"] == 77
    # Characterize the provider adapter boundary: LiteLLM raises small limits
    # before they reach our API. This is not gateway compatibility for 1..15.
    boundary_start = len(captured)
    direct = client.responses.create(model="test", input="Hello", max_output_tokens=1, store=False)
    adapted = lite(input="Hello", max_output_tokens=1, store=False)
    assert direct.max_output_tokens == 1
    assert adapted.max_output_tokens == 16
    for invalid in (0, -1, True, False):
        try:
            client.responses.create(model="test", input="Hello", max_output_tokens=invalid, store=False)
        except openai.BadRequestError:
            pass
        else:
            raise AssertionError(f"Gateway accepted invalid limit: {invalid!r}")
        assert lite(input="Hello", max_output_tokens=invalid, store=False).max_output_tokens == 16
    for name, create in (("openai", lambda **kw: client.responses.create(model="test", **kw)), ("litellm", lite)):
        events = list(create(input="exhaust-output", stream=True, max_output_tokens=16, store=False))
        # LiteLLM 1.103.0 relabels the terminal event but preserves response.status.
        expected_type = "response.incomplete" if name == "openai" else "response.completed"
        assert events[-1].type == expected_type
        assert events[-1].response.status == "incomplete"
        assert events[-1].response.output == []
        assert events[-1].response.incomplete_details.reason == "max_output_tokens"
    # This pinned SDK helper only returns response.completed. Applications must
    # consume terminal events directly when incomplete/failed are possible.
    with client.responses.stream(model="test", input="exhaust-output", max_output_tokens=16, store=False) as stream:
        terminal = [e for e in stream if e.type == "response.incomplete"]
        assert len(terminal) == 1
        try:
            stream.get_final_response()
        except RuntimeError as exc:
            assert "response.completed" in str(exc)
        else:
            raise AssertionError("Revisit SDK incomplete handling: helper behavior changed")
    captured = httpx.get(url+"/captured").json()
    boundary = captured[boundary_start:]
    assert boundary[0]["body"]["max_output_tokens"] == 1
    assert boundary[1]["body"]["max_output_tokens"] == 16
    for index in range(2, 10, 2):
        assert boundary[index+1]["body"]["max_output_tokens"] == 16
    usage_start = len(captured)
    chat_messages = [{"role": "user", "content": "Hello"}]
    for include_usage in (True, False):
        options = {"messages": chat_messages, "stream": True,
                   "stream_options": {"include_usage": include_usage}}
        chunks = list(client.chat.completions.create(model="test", **options))
        assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices) == "It is sunny"
        if include_usage:
            assert chunks[-2].choices[0].finish_reason == "stop"
            assert all(chunk.usage is None for chunk in chunks[:-1])
            assert chunks[-1].choices == [] and chunks[-1].usage.total_tokens == 9
        else:
            assert chunks[-1].choices[0].finish_reason == "stop"
            assert all(chunk.usage is None for chunk in chunks)
        adapted = list(litellm.completion(model="openai/test", api_base=url+"/v1", api_key="fixture",
                                          drop_params=False, **options))
        assert "".join(chunk.choices[0].delta.content or "" for chunk in adapted if chunk.choices) == "It is sunny"
        if include_usage:
            # LiteLLM can transform chunks. Check recovery of the actual API total.
            assert adapted[-1].usage.total_tokens == 9
    captured = httpx.get(url+"/captured").json()
    assert [row["body"]["stream_options"]["include_usage"] for row in captured[usage_start:]] == [True, True, False, False]
    print("openai/litellm: Chat SSE include_usage forwarded and SDK usage recovered")
    reasoning = {"effort": "low"}
    response = client.responses.create(model="test", input="Hello", reasoning=reasoning, store=False)
    assert response.reasoning.effort == "low"
    events = list(client.responses.create(
        model="test", input="Hello", reasoning=reasoning, store=False, stream=True,
    ))
    assert events[-1].type == "response.completed"
    assert events[-1].response.reasoning.effort == "low"
    captured = httpx.get(url+"/captured").json()
    assert [row["body"]["reasoning"] for row in captured[-2:]] == [reasoning, reasoning]
    print("openai: Responses reasoning.effort forwarded in JSON and SSE requests")
    # Actual installed LiteLLM -> localhost HTTP -> gateway schema boundary.
    # Generation is deterministic above; these checks prove wire forwarding,
    # not capability validation or real-model thinking. In particular, high and
    # xhigh must reach the gateway unchanged so its model policy can decide.
    chat_start = len(captured)
    chat_cases = [
        {},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
        {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        {"extra_body": {"chat_template_kwargs": {"preserve_thinking": True}}},
        {"extra_body": {"chat_template_kwargs": {"preserve_thinking": False}}},
        {"reasoning_effort": "high", "allowed_openai_params": ["reasoning_effort"]},
        {"reasoning_effort": "xhigh", "allowed_openai_params": ["reasoning_effort"]},
        {"reasoning_effort": "xhigh", "allowed_openai_params": ["reasoning_effort"],
         "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}},
    ]
    for case in chat_cases:
        result = litellm.completion(
            model="openai/qwen-fixture", api_base=url+"/v1", api_key="fixture",
            messages=[{"role": "user", "content": "Hello"}], drop_params=False,
            **case,
        )
        assert result.choices[0].message.content == "It is sunny"
    # Same model_list/litellm_params placement as examples/litellm.config.yaml.
    router = litellm.Router(model_list=[{
        "model_name": "Qwen3-Example",
        "litellm_params": {"model": "openai/qwen-fixture", "api_base": url+"/v1",
                           "api_key": "fixture", "timeout": 300,
                           "allowed_openai_params": ["reasoning_effort"]},
    }])
    for effort in ("high", "xhigh"):
        case = {"reasoning_effort": effort,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
        result = router.completion(model="Qwen3-Example", drop_params=False,
                                   messages=[{"role": "user", "content": "Hello"}], **case)
        assert result.choices[0].message.content == "It is sunny"
        chat_cases.append(case)
    captured = httpx.get(url+"/captured").json()
    wire_chat = captured[chat_start:]
    assert len(wire_chat) == len(chat_cases)
    for case, received in zip(chat_cases, wire_chat, strict=True):
        assert received["path"] == "/v1/chat/completions"
        body = received["body"]
        expected_template = case.get("extra_body", {}).get("chat_template_kwargs")
        if expected_template is None:
            assert "chat_template_kwargs" not in body, body
        else:
            assert body["chat_template_kwargs"] == expected_template, body
            for key, value in expected_template.items():
                assert type(body["chat_template_kwargs"][key]) is bool
                assert body["chat_template_kwargs"][key] is value
        if "reasoning_effort" in case:
            assert body["reasoning_effort"] == case["reasoning_effort"], body
        else:
            assert "reasoning_effort" not in body, body
        assert "allowed_openai_params" not in body
        assert "extra_body" not in body
    print("litellm: actual HTTP chat bodies preserve thinking booleans, effort high/xhigh, and omission")
    # Run real clients through the production DeepSeek prompt/transport/parser.
    # Only the external model engine is replaced in serve().
    deep_start = len(captured)
    weather = {"type": "function", "function": {"name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}}}}
    for name in ("openai", "litellm"):
        for enabled in (True, False):
            for streaming in (False, True):
                for use_tools in (False, True):
                    params = {"messages": [{"role": "user", "content": "Weather?" if use_tools else "Hello"}],
                              "stream": streaming,
                              "extra_body": {"chat_template_kwargs": {"enable_thinking": enabled}}}
                    if use_tools:
                        params["tools"] = [weather]
                    if name == "openai":
                        result = client.chat.completions.create(model="deepseek-fixture", **params)
                    else:
                        result = litellm.completion(model="openai/deepseek-fixture", api_base=url+"/v1",
                            api_key="fixture", drop_params=False, **params)
                    content = ""; reasoning = ""; calls = {}
                    messages = [part.choices[0].delta for part in result] if streaming else [result.choices[0].message]
                    for message in messages:
                        content += message.content or ""
                        reasoning += getattr(message, "reasoning_content", None) or ""
                        for index, call in enumerate(message.tool_calls or []):
                            dst = calls.setdefault(getattr(call, "index", None) or index, {"id": "", "name": "", "arguments": ""})
                            dst["id"] += call.id or ""
                            dst["name"] += call.function.name or ""
                            dst["arguments"] += call.function.arguments or ""
                    assert reasoning == ("Inspect first." if enabled else ""), (name, enabled, streaming, reasoning)
                    assert content == ("" if use_tools else "It is sunny"), (name, content)
                    record = httpx.get(url+"/captured").json()[-1]
                    assert record["backend_prompt"].endswith("<think>" if enabled else "</think>"), record
                    assert record["body"]["chat_template_kwargs"]["enable_thinking"] is enabled
                    assert "chat_template_kwargs" not in record["backend_sampling"]
                    if use_tools:
                        call = calls[0]
                        assert call["id"].startswith("call_") and call["name"] == "weather"
                        assert json.loads(call["arguments"]) == {"city": "Paris"}
                        followup = client.chat.completions.create(model="deepseek-fixture", tools=[weather],
                            messages=[*params["messages"], {"role": "assistant", "content": "", "tool_calls": [{
                                "id": call["id"], "type": "function", "function": {"name": "weather", "arguments": call["arguments"]}}]},
                                {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}])
                        assert followup.choices[0].message.content == "It is sunny"
                        assert "<tool_result>sunny</tool_result>" in httpx.get(url+"/captured").json()[-1]["backend_prompt"]
    captured = httpx.get(url+"/captured").json()
    deep_requests = len(captured) - deep_start
    print("openai/litellm: DeepSeek real gateway on/off, SSE, DSML tool IDs/arguments and replay passed")
    # Real model-independent overflow -> HTTP -> SDK error classification.
    # A generic 400/413 is not sufficient: LiteLLM must identify the context
    # rejection so callers can apply their context fallback policy.
    overflow_input = " ".join(["word"] * 80)
    context_cases = 0
    for route in ("chat", "responses"):
        for streaming in (False, True):
            for name in ("litellm", "openai"):
                params = {"stream": streaming}
                if route == "chat":
                    params.update(messages=[{"role": "user", "content": overflow_input}], max_completion_tokens=64)
                    if name == "openai":
                        create = lambda **kw: client.chat.completions.create(model="context-overflow-fixture", **kw)
                    else:
                        create = lambda **kw: litellm.completion(model="openai/context-overflow-fixture",
                            api_base=url+"/v1", api_key="fixture", num_retries=0, **kw)
                else:
                    params.update(input=overflow_input, max_output_tokens=64, store=False)
                    if name == "openai":
                        create = lambda **kw: client.responses.create(model="context-overflow-fixture", **kw)
                    else:
                        create = lambda **kw: litellm.responses(model="openai/context-overflow-fixture",
                            api_base=url+"/v1", api_key="fixture", num_retries=0, **kw)
                expected_error = openai.BadRequestError if name == "openai" else litellm.ContextWindowExceededError
                try:
                    result = create(**params)
                    if streaming:
                        list(result)
                except expected_error as exc:
                    assert exc.status_code == 400, (name, route, streaming, exc)
                    message = str(exc)
                    assert "maximum context length is 128 tokens" in message, (name, route, message)
                    assert "80" in message and "64" in message, (name, route, message)
                    if name == "openai":
                        assert exc.code == "context_length_exceeded", (route, streaming, exc.body)
                else:
                    raise AssertionError(f"{name}/{route}/stream={streaming}: context overflow was accepted")
                context_cases += 1
    for model, status in (("invalid-request-fixture", 400), ("media-limit-fixture", 413)):
        for route in ("chat", "responses"):
            try:
                if route == "chat":
                    litellm.completion(model="openai/"+model, api_base=url+"/v1", api_key="fixture",
                        messages=[{"role": "user", "content": "hello"}], num_retries=0)
                else:
                    litellm.responses(model="openai/"+model, api_base=url+"/v1", api_key="fixture",
                        input="hello", store=False, num_retries=0)
            except litellm.ContextWindowExceededError as exc:
                raise AssertionError(f"{route}: non-context HTTP {status} was classified as context overflow") from exc
            except (litellm.APIError, litellm.BadRequestError) as exc:
                assert exc.status_code == status, (route, model, exc)
            else:
                raise AssertionError(f"{route}: non-context HTTP {status} was accepted")
    print("openai/litellm: Chat/Responses context overflow, streaming preflight, and non-context errors passed")
    captured = httpx.get(url+"/captured").json()
    print(json.dumps({"openai": version("openai"), "litellm": version("litellm"),
        "requests": len(captured), "routes": ["/v1/responses", "/v1/chat/completions"],
        "deepseek_production_gateway_requests": deep_requests, "chat_thinking_wire_cases": len(chat_cases), "reasoning_effort_forwarding": ["high", "xhigh"], "injected_default_limit": False,
        "litellm_minimum_explicit_limit": 16, "litellm_invalid_integer_limits_normalized": True,
        "sdk_incomplete_requires_terminal_event_handling": True,
        "litellm_incomplete_event_type": "response.completed", "context_overflow_sdk_cases": context_cases}))
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--server-python", default=sys.executable)
    args = parser.parse_args()
    if args.serve:
        serve()
    else:
        process = subprocess.Popen([args.server_python, __file__, "--serve"], stdout=subprocess.PIPE, text=True, cwd=ROOT)
        try:
            port = int(process.stdout.readline().strip())
            check_clients(f"http://127.0.0.1:{port}")
        finally:
            process.terminate()
            process.wait(timeout=10)
