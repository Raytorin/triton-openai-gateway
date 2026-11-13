# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Check numeric budgets at real generation policy boundaries."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from jinja2 import Template
import pytest

from gateway import generation
from gateway.admission import AdmissionLease
from gateway.schemas import ChatCompletionRequest
from gateway.vllm_media import _summary_sampling, load_vllm_media_settings


class Tokenizer:
    chat_template = ("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% if enable_thinking is not defined or enable_thinking %}<think>{% else %}<think></think>{% endif %}{% endif %}")

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        kwargs.setdefault("add_generation_prompt", True)
        return Template(self.chat_template).render(messages=messages, **kwargs)

    def encode(self, text, **kwargs):
        return [7, 8] if text == '</think>' else [1] * len(text.split())

    def __call__(self, text, **kwargs):
        return type('Encoded', (), {'input_ids': self.encode(text)})()


@pytest.fixture
def model(tmp_path, monkeypatch):
    for name in ('GATEWAY_REASONING_MODE', 'GATEWAY_REASONING_PARSER', 'GATEWAY_REASONING_RESPONSE_FIELD'):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3_8"}')
    (tmp_path / 'model.json').write_text(json.dumps({'max_model_len': 64,
        'reasoning_config': {'reasoning_start_str': '<think>', 'reasoning_end_str': '</think>'}}))
    (tmp_path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': 'separate'},
        'generation': {'reasoning_default_output_tokens': 32},
        'context_compression': {'mode': 'disabled', 'safety_margin_tokens': 2}}))
    return tmp_path


def execute(model, *, stream=False, backend='vllm', prompt='Summarize this.',
            tokenizer=None, history=None, **fields):
    inference = AsyncMock(return_value='brief</think>Summary.')
    request = ChatCompletionRequest(model='test', messages=[
        *(history or []), {'role': 'user', 'content': prompt}],
        stream=stream, **fields)

    async def run():
        with (patch.object(generation.registry, 'resolve', return_value=model),
              patch.object(generation.registry, 'validate_route'),
              patch.object(generation.registry, 'get_backend', return_value=backend),
              patch.object(generation.registry, 'get_tokenizer_async', AsyncMock(return_value=(tokenizer or Tokenizer(), model))),
              patch.object(generation.admission, 'acquire', AsyncMock(return_value=AdmissionLease([]))),
              patch.object(generation, 'call_triton_multimodal', inference),
              patch.object(generation, 'call_triton_python_chat', inference)):
            return await generation.generate(request)
    return asyncio.run(run()), inference


@pytest.mark.parametrize('stream', [False, True])
def test_missing_engine_boundaries_rejected_before_generation(model, stream):
    (model / 'model.json').write_text('{"max_model_len":64}')
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=4, stream=stream)
    assert caught.value.status_code == 400
    assert 'reasoning_config' in caught.value.detail


@pytest.mark.parametrize('stream', [False, True])
def test_disabled_thinking_rejected_before_generation(model, stream):
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=4, stream=stream, chat_template_kwargs={'enable_thinking': False})
    assert caught.value.status_code == 400
    assert 'thinking_token_budget' in caught.value.detail


def test_context_clipping_revalidates_numeric_budget(model):
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=20, prompt='word ' * 40)
    assert caught.value.status_code == 400
    assert 'thinking_token_budget' in caught.value.detail


def test_automatic_chat_limit_drops_history_for_thinking_and_a_visible_token(model):
    path = model / 'gateway.json'
    config = json.loads(path.read_text())
    config['context_compression']['mode'] = 'truncate'
    path.write_text(json.dumps(config))
    result, inference = execute(
        model, thinking_token_budget=12, prompt='word ' * 43,
        history=[{'role': 'user', 'content': 'old request'},
                 {'role': 'assistant', 'content': 'old answer'}],
    )
    used_prompt = inference.call_args.args[1]
    assert 'old request' not in used_prompt
    expected = 64 - 2 - len(Tokenizer().encode(used_prompt))
    assert 15 <= expected < 32
    assert inference.call_args.args[2]['max_tokens'] == expected
    assert result.extensions['context_status']['dropped_messages'] == 2


def test_final_output_reserves_closing_tokens_and_visible_answer(model):
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=30, max_tokens=32)
    assert caught.value.status_code == 400
    assert 'thinking_token_budget' in caught.value.detail


def test_non_vllm_engine_rejects_unsupported_numeric_budget(model):
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=4, backend='python')
    assert caught.value.status_code == 400
    assert 'thinking_token_budget' in caught.value.detail


@pytest.mark.parametrize('backend', ['vllm', 'vllm_multimodal'])
def test_valid_budget_reaches_engine_and_separate_answer_survives(model, backend):
    result, inference = execute(model, thinking_token_budget=4, max_tokens=32, backend=backend)
    assert inference.call_args.args[2]['thinking_token_budget'] == 4
    assert inference.call_args.args[2]['max_tokens'] == 32
    assert result.message['content'] == 'Summary.'


def test_media_summary_does_not_inherit_main_thinking_budget(model):
    settings = load_vllm_media_settings(model)
    original = {'thinking_token_budget': 128, 'max_tokens': 4096, 'temperature': 0.6}
    params = _summary_sampling(original, settings)
    assert 'thinking_token_budget' not in params
    assert original['thinking_token_budget'] == 128
    assert params['max_tokens'] == settings.summary_max_tokens


def test_hybrid_generation_reserves_opening_tokens_after_context_fitting(model):
    class HybridTokenizer(Tokenizer):
        chat_template = ("{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}"
            "{% if add_generation_prompt %}assistant: {% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}")

    # One generated opening token plus two closing tokens leaves no answer room.
    with pytest.raises(HTTPException) as caught:
        execute(model, thinking_token_budget=0, max_tokens=3, tokenizer=HybridTokenizer())
    assert caught.value.status_code == 400
    assert 'thinking_token_budget' in caught.value.detail
