# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextlib import ExitStack
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from jinja2.sandbox import ImmutableSandboxedEnvironment
import numpy as np
from pydantic import ValidationError
import pytest

import gateway.app as app_module
from gateway.admission import AdmissionLease
import gateway.generation as generation
from gateway.context_compression import load_context_compression_settings
from gateway.prompt import RequestTokenizer, build_sampling_parameters, render_chat_prompt
from gateway.schemas import ChatCompletionRequest
import gateway.triton_client as tc


TEMPLATE = """{% for message in messages %}{{ message.role }}:{{ message.content }}
{% if preserve_thinking is not defined or preserve_thinking %}{{ message.reasoning_content|default('') }}{% endif %}
{% endfor %}{% if add_generation_prompt %}assistant:{% if enable_thinking is not defined or enable_thinking %}<think>{% else %}<think></think>{% endif %}{% endif %}"""


class Tokenizer:
    name_or_path = "Qwen/Qwen3.6-27B-FP8"
    chat_template = TEMPLATE

    def __init__(self):
        self.calls = []

    def get_chat_template(self, *args, **kwargs):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        environment = ImmutableSandboxedEnvironment()
        def raise_exception(message):
            raise ValueError(message)
        environment.globals["raise_exception"] = raise_exception
        return environment.from_string(self.chat_template).render(messages=messages, **kwargs)

    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))


def request(**kwargs):
    return ChatCompletionRequest(model="qwen", messages=[{"role": "user", "content": "hello"}], **kwargs)


@pytest.mark.parametrize("key", ["enable_thinking", "preserve_thinking"])
@pytest.mark.parametrize("value", ["false", 0, 1, None, [], {}])
def test_template_controls_reject_non_booleans(key, value):
    with pytest.raises(ValidationError):
        request(chat_template_kwargs={key: value})


def test_unknown_template_kwargs_are_rejected():
    with pytest.raises(ValidationError):
        request(chat_template_kwargs={"enable_thinkng": False})


@pytest.mark.parametrize("value", [0, -2, True, "20", 1.5])
def test_top_k_rejects_invalid_sampling_values(value):
    with pytest.raises(ValidationError):
        request(top_k=value)


@pytest.mark.parametrize("value", [-2.1, 2.1])
def test_presence_penalty_rejects_values_outside_vllm_range(value):
    with pytest.raises(ValidationError):
        request(presence_penalty=value)


def test_qwen_sampling_parameters_are_forwarded_without_template_controls():
    parameters = build_sampling_parameters(request(top_k=20, presence_penalty=1.5,
        chat_template_kwargs={"enable_thinking": False}))
    assert parameters["top_k"] == 20
    assert parameters["presence_penalty"] == 1.5
    assert "enable_thinking" not in parameters
    assert "chat_template_kwargs" not in parameters


@pytest.fixture
def endpoint(tmp_path):
    tokenizer = Tokenizer()
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5"}')
    (tmp_path / "model.json").write_text('{"max_model_len":16384}')
    (tmp_path / "gateway.json").write_text('{"reasoning":{"mode":"separate","parser":"qwen3"}}')
    records = []
    tool_output = '<tool_call>{"name":"weather","arguments":{"city":"London"}}</tool_call>'

    def output(prompt):
        final = tool_output if "Weather?" in prompt else "Final answer."
        return "Inspect first.</think>" + final if prompt.endswith("<think>") else final

    async def infer(model, prompt, sampling, *args, **kwargs):
        records.append((prompt, dict(sampling)))
        await asyncio.sleep(0)
        return output(prompt)

    def inputs(prompt, sampling, *args, **kwargs):
        return [SimpleNamespace(prompt=prompt, sampling=sampling)]

    async def stream(operation, model, inputs, outputs, **kwargs):
        data = inputs[0]
        records.append((data.prompt, dict(data.sampling)))
        for chunk in (output(data.prompt)[:8], output(data.prompt)):
            await asyncio.sleep(0)
            yield SimpleNamespace(as_numpy=lambda name, chunk=chunk:
                np.asarray([chunk.encode()], dtype=np.object_) if name == "text_output" else None)

    async def acquire(*args, **kwargs):
        return AdmissionLease([])

    with ExitStack() as stack:
        for name, replacement in {
            "resolve": lambda *args: tmp_path,
            "validate_route": lambda *args: None,
            "get_backend": lambda *args: "vllm",
            "get_tool_parser": lambda *args: None,
            "get_tokenizer_async": AsyncMock(return_value=(tokenizer, tmp_path)),
        }.items():
            stack.enter_context(patch.object(generation.registry, name, replacement))
        stack.enter_context(patch.object(generation.admission, "acquire", acquire))
        stack.enter_context(patch.object(generation, "call_triton_multimodal", infer))
        stack.enter_context(patch.object(tc, "call_triton_multimodal", infer))
        stack.enter_context(patch.object(tc, "_build_grpc_generate_inputs", inputs))
        stack.enter_context(patch.object(tc, "get_grpc_client", AsyncMock(return_value=object())))
        stack.enter_context(patch.object(tc, "_stream_grpc_results", stream))
        yield SimpleNamespace(tokenizer=tokenizer, records=records, path=tmp_path)


async def post(body):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url="http://test") as client:
        return await client.post("/v1/chat/completions", json=body)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_mixed_concurrent_requests_keep_generation_and_visibility_independent(endpoint):
    bodies = [
        {"model": "qwen", "messages": [{"role": "user", "content": label}],
         "chat_template_kwargs": {"enable_thinking": enabled}, "include_reasoning": visible}
        for label, enabled, visible in [("enabled", True, True), ("disabled", False, True), ("hidden", True, False)]
    ]
    responses = await asyncio.gather(*(post(body) for body in bodies))
    assert [response.status_code for response in responses] == [200, 200, 200]
    messages = [response.json()["choices"][0]["message"] for response in responses]
    assert messages[0]["reasoning_content"] == "Inspect first."
    assert "reasoning_content" not in messages[1]
    assert "reasoning_content" not in messages[2]
    assert all(message["content"] == "Final answer." for message in messages)
    prompts = {label: next(prompt for prompt, _ in endpoint.records if f"user:{label}" in prompt)
               for label in ("enabled", "disabled", "hidden")}
    assert prompts["enabled"].endswith("<think>")
    assert prompts["hidden"].endswith("<think>")
    assert prompts["disabled"].endswith("<think></think>")
    # A later default request must still use the original server policy.
    response = await post({"model": "qwen", "messages": [{"role": "user", "content": "later"}]})
    assert response.json()["reasoning_status"]["mode"] == "separate"
    assert endpoint.records[-1][0].endswith("<think>")
    assert endpoint.tokenizer.chat_template == TEMPLATE


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
async def test_request_controls_reach_normal_sse_and_tool_paths(endpoint, stream, tools, thinking):
    body = {"model": "qwen", "messages": [{"role": "user", "content": "Weather?" if tools else "hello"}],
            "stream": stream, "chat_template_kwargs": {"enable_thinking": thinking}, "include_reasoning": False,
            "top_k": 20, "presence_penalty": 1.5}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]
    response = await post(body)
    assert response.status_code == 200, response.text
    assert endpoint.records[-1][0].endswith("<think>" if thinking else "<think></think>")
    assert endpoint.records[-1][1]["top_k"] == 20
    assert endpoint.records[-1][1]["presence_penalty"] == 1.5
    assert "Inspect first." not in response.text
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
        if tools:
            calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
            assert calls[0]["function"]["name"] == "weather"
        else:
            assert "".join(delta.get("content", "") for delta in deltas) == "Final answer."
    elif tools:
        assert response.json()["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "weather"


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["hidden", "disabled"])
async def test_request_cannot_bypass_server_policy(endpoint, mode):
    (endpoint.path / "gateway.json").write_text(json.dumps({"reasoning": {"mode": mode, "parser": "qwen3"}}))
    response = await post({"model": "qwen", "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": True}, "include_reasoning": True})
    if mode == "disabled":
        assert response.status_code == 400
        assert endpoint.records == []
    else:
        assert response.status_code == 200
        assert "reasoning_content" not in response.json()["choices"][0]["message"]


@pytest.mark.anyio
async def test_explicit_thinking_with_constrained_json_is_rejected_before_inference(endpoint):
    response = await post({"model": "qwen", "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": True}, "response_format": {"type": "json_object"}})
    assert response.status_code == 400
    assert endpoint.records == []


@pytest.mark.anyio
async def test_history_preservation_flag_reaches_the_same_template(endpoint):
    body = {"model": "qwen", "messages": [{"role": "user", "content": "old"},
        {"role": "assistant", "content": "prior answer", "reasoning_content": "PRIOR_REASONING"},
        {"role": "user", "content": "next"}], "chat_template_kwargs": {"preserve_thinking": True}}
    response = await post(body)
    assert response.status_code == 200, response.text
    assert "PRIOR_REASONING" in endpoint.records[-1][0]
    body["chat_template_kwargs"]["preserve_thinking"] = False
    response = await post(body)
    assert response.status_code == 200
    assert "PRIOR_REASONING" not in endpoint.records[-1][0]


@pytest.mark.anyio
async def test_effort_survives_context_truncation_and_is_not_an_engine_sampling_param(endpoint):
    prefix = """{% if reasoning_effort is defined and reasoning_effort not in ('low', 'medium', 'xhigh') %}
{{ raise_exception('unsupported effort') }}{% endif %}effort={{ reasoning_effort|default('xhigh') }}
"""
    endpoint.tokenizer.chat_template = prefix + TEMPLATE
    (endpoint.path / "model.json").write_text('{"max_model_len":100}')
    (endpoint.path / "gateway.json").write_text(json.dumps({"reasoning": {"mode": "hidden", "parser": "qwen3"},
        "context_compression": {"mode": "truncate", "safety_margin_tokens": 0}}))
    body = {"model": "qwen", "messages": [{"role": "user", "content": "OLD " * 90},
        {"role": "assistant", "content": "OLD " * 90}, {"role": "user", "content": "current"}],
        "chat_template_kwargs": {"enable_thinking": True}, "reasoning_effort": "low", "max_tokens": 8}
    response = await post(body)
    assert response.status_code == 200, response.text
    prompt, sampling = endpoint.records[-1]
    assert "effort=low" in prompt
    assert "OLD" not in prompt
    assert prompt.endswith("<think>")
    assert "reasoning_effort" not in sampling
    body["reasoning_effort"] = "high"
    count = len(endpoint.records)
    response = await post(body)
    assert response.status_code == 400
    assert len(endpoint.records) == count
    body["reasoning_effort"] = "xhigh"
    response = await post(body)
    assert response.status_code == 200
    assert "effort=xhigh" in endpoint.records[-1][0]


@pytest.mark.anyio
async def test_auxiliary_context_summary_does_not_reuse_request_overrides(endpoint):
    # The primary request and a summary share the same loaded tokenizer.
    response = await post({"model": "qwen", "messages": [{"role": "user", "content": "hello"}],
        "chat_template_kwargs": {"enable_thinking": True}})
    assert response.status_code == 200
    settings = load_context_compression_settings(endpoint.path)
    summary = await generation._generate_context_summary("qwen", settings, None, "summarize this")
    assert summary.text == "Final answer."
    assert endpoint.records[-1][0].endswith("<think></think>")


def test_explicit_controls_cannot_disappear_in_tokenizer_typeerror_fallback():
    class LegacyTokenizer:
        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):
            return "legacy"
    legacy = LegacyTokenizer()
    assert render_chat_prompt(RequestTokenizer(legacy, {"enable_thinking": False}), []) == "legacy"
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        render_chat_prompt(RequestTokenizer(legacy, {"enable_thinking": False}, strict=True), [])
    assert error.value.status_code == 400


@pytest.mark.anyio
@pytest.mark.parametrize('layout', ['subdirectory', 'parent', 'model_json_external', 'loaded_external'])
async def test_tokenizer_metadata_layout_preserves_actual_qwen_history(endpoint, layout):
    from test_qwen_templates import QwenTemplateTokenizer

    model_path = endpoint.path / 'served' / '1'
    model_path.mkdir(parents=True)
    model_config = {'max_model_len': 16384}
    if layout == 'subdirectory':
        tokenizer_path = model_path / 'tokenizer'
    elif layout == 'parent':
        tokenizer_path = model_path.parent / 'tokenizer'
    else:
        tokenizer_path = endpoint.path / 'external-tokenizer'
        if layout == 'model_json_external':
            model_config['tokenizer'] = str(tokenizer_path)
    tokenizer_path.mkdir(parents=True)
    (tokenizer_path / 'config.json').write_text('{"model_type":"qwen3_5"}')
    (model_path / 'model.json').write_text(json.dumps(model_config))
    (model_path / 'gateway.json').write_text('{"reasoning":{"mode":"separate"}}')
    endpoint.tokenizer.chat_template = QwenTemplateTokenizer('qwen36').chat_template
    endpoint.tokenizer.name_or_path = str(tokenizer_path) if layout == 'loaded_external' else 'Qwen/Tokenizer'
    with patch.object(generation.registry, 'get_tokenizer_async', AsyncMock(return_value=(endpoint.tokenizer, model_path))):
        body = {'model': 'qwen', 'messages': [{'role': 'user', 'content': 'Old'},
            {'role': 'assistant', 'content': 'Prior answer', 'reasoning_content': 'PRIVATE HISTORY'},
            {'role': 'user', 'content': 'Continue'}], 'chat_template_kwargs': {'preserve_thinking': True}}
        response = await post(body)
        assert response.status_code == 200, response.text
        assert '<think>\nPRIVATE HISTORY\n</think>' in endpoint.records[-1][0]
        body['chat_template_kwargs']['preserve_thinking'] = False
        response = await post(body)
        assert response.status_code == 200, response.text
        assert 'PRIVATE HISTORY' not in endpoint.records[-1][0]


@pytest.mark.anyio
async def test_endpoint_validates_actual_named_tool_template_before_inference(endpoint):
    from test_qwen_templates import NamedQwenTemplateTokenizer, TOOLS

    tokenizer = NamedQwenTemplateTokenizer()
    with patch.object(generation.registry, 'get_tokenizer_async', AsyncMock(return_value=(tokenizer, endpoint.path))):
        response = await post({'model': 'qwen', 'messages': [{'role': 'user', 'content': 'Weather?'}],
                               'tools': TOOLS, 'chat_template_kwargs': {'enable_thinking': False}})
    assert response.status_code == 400, response.text
    assert endpoint.records == []
