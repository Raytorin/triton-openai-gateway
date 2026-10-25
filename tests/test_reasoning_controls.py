# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Request policy checks; template rendering itself has pinned fixture tests."""

import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from gateway.reasoning import ReasoningSettings, load_reasoning_settings


SWITCH = "{% if enable_thinking is not defined or enable_thinking %}<think>{% else %}<think></think>{% endif %}"
FIXED_ON = "{{ '<think>' }}"
FIXED_OFF = "{{ '<think></think>' }}"
EFFORT = SWITCH + "{% if reasoning_effort in ['low', 'medium', 'xhigh'] %}{{ reasoning_effort }}{% else %}{{ raise_exception('unsupported effort') }}{% endif %}"
PRESERVE = SWITCH + "{% if preserve_thinking %}{{ message.reasoning_content }}{% endif %}"


class TemplateTokenizer:
    """A deterministic renderer boundary for the policy, not a Jinja emulator."""

    def __init__(self, template=SWITCH, *, efforts=(), preserve=False):
        self.chat_template = template
        self.efforts = efforts
        self.preserve = preserve

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        if self.chat_template == FIXED_OFF:
            suffix = '<think></think>'
        elif self.chat_template == FIXED_ON:
            suffix = '<think>'
        elif 'enable_thinking' in self.chat_template:
            suffix = '<think>' if kwargs.get('enable_thinking', True) else '<think></think>'
        else:
            suffix = ''
        if self.efforts:
            effort = kwargs.get('reasoning_effort', 'medium')
            if effort not in self.efforts:
                raise ValueError('unsupported effort')
            suffix = f'{effort}:{suffix}'
        history = ''
        if self.preserve and kwargs.get('preserve_thinking', False):
            history = ''.join(m.get('reasoning_content', '') for m in messages)
        return history + '<|im_start|>assistant\n' + (suffix if add_generation_prompt else '')


@pytest.fixture
def model(tmp_path, monkeypatch):
    for key in ('GATEWAY_REASONING_MODE', 'GATEWAY_REASONING_PARSER', 'GATEWAY_REASONING_RESPONSE_FIELD'):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / 'config.json').write_text('{"model_type":"qwen3"}')
    (tmp_path / 'gateway.json').write_text('{"reasoning":{"mode":"separate"}}')
    return tmp_path


def rejects(model, tokenizer, **kwargs):
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=tokenizer, **kwargs)
    assert caught.value.status_code == 400


def test_off_disables_generation_without_changing_configured_policy(model):
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer(), enable_thinking=False)
    assert settings.mode == 'disabled'
    assert settings.configured_mode == 'separate'
    assert settings.template_kwargs == {'enable_thinking': False}
    assert settings.requires_template_kwargs
    assert settings.qwen_profile


def test_include_reasoning_changes_visibility_only(model):
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer(), include_reasoning=False)
    assert settings.mode == 'hidden'
    assert settings.enable_thinking
    assert settings.template_kwargs['enable_thinking'] is True


def test_server_disabled_cannot_be_overridden_and_env_is_not_stale(model, monkeypatch):
    load_reasoning_settings(model, tokenizer=TemplateTokenizer())
    monkeypatch.setenv('GATEWAY_REASONING_MODE', 'disabled')
    rejects(model, TemplateTokenizer(), enable_thinking=True)
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer())
    assert settings.mode == 'disabled'


def test_hidden_server_policy_cannot_be_revealed(model, monkeypatch):
    monkeypatch.setenv('GATEWAY_REASONING_MODE', 'hidden')
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer(), include_reasoning=True, enable_thinking=True)
    assert settings.mode == 'hidden'
    assert not settings.expose_reasoning


def test_json_constraint_rejects_explicit_on_and_preserves_off(model):
    rejects(model, TemplateTokenizer(), force_disable=True, enable_thinking=True)
    assert load_reasoning_settings(model, tokenizer=TemplateTokenizer(), force_disable=True).template_kwargs == {'enable_thinking': False}


@pytest.mark.parametrize('model_type', ['qwen3', 'qwen3_5', 'qwen3_6', 'qwen3_8'])
def test_metadata_alone_never_proves_template_controls(model, model_type):
    (model / 'config.json').write_text(json.dumps({'model_type': model_type}))
    rejects(model, TemplateTokenizer('plain assistant'), enable_thinking=True)
    rejects(model, TemplateTokenizer('plain assistant'), reasoning_effort='medium')


def test_older_switch_template_rejects_effort_instead_of_ignoring_it(model):
    rejects(model, TemplateTokenizer(), reasoning_effort='medium')


def test_qwen38_efforts_remain_exact_and_high_is_not_xhigh(model):
    (model / 'config.json').write_text('{"model_type":"qwen3_8"}')
    tokenizer = TemplateTokenizer(EFFORT, efforts=('low', 'medium', 'xhigh'))
    for effort in ('low', 'medium', 'xhigh'):
        settings = load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort=effort)
        assert settings.template_kwargs == {'enable_thinking': True, 'reasoning_effort': effort}
    rejects(model, tokenizer, reasoning_effort='high')
    rejects(model, tokenizer, reasoning_effort='max')
    rejects(model, tokenizer, enable_thinking=False, reasoning_effort='low')


def test_custom_template_can_support_high_without_aliasing(model):
    tokenizer = TemplateTokenizer(EFFORT.replace('xhigh', 'high'), efforts=('low', 'medium', 'high'))
    settings = load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort='high')
    assert settings.template_kwargs['reasoning_effort'] == 'high'
    rejects(model, tokenizer, reasoning_effort='xhigh')


def test_preserve_requires_a_template_read_and_stays_request_local(model):
    rejects(model, TemplateTokenizer(), preserve_thinking=False)
    tokenizer = TemplateTokenizer(PRESERVE, preserve=True)
    settings = load_reasoning_settings(model, tokenizer=tokenizer, preserve_thinking=True)
    assert settings.template_kwargs == {'enable_thinking': True, 'preserve_thinking': True}
    copied = settings.template_kwargs
    copied['enable_thinking'] = False
    assert settings.enable_thinking
    assert 'preserve_thinking' not in load_reasoning_settings(model, tokenizer=tokenizer).template_kwargs
    assert tokenizer.__dict__ == {'chat_template': PRESERVE, 'efforts': (), 'preserve': True}


def test_comment_and_literal_variable_names_do_not_enable_controls(model):
    tokenizer = TemplateTokenizer("{# enable_thinking preserve_thinking reasoning_effort #}{{ 'enable_thinking' }}")
    rejects(model, tokenizer, enable_thinking=True)
    rejects(model, tokenizer, preserve_thinking=True)
    rejects(model, tokenizer, reasoning_effort='low')


def test_fixed_thinking_accepts_only_on(model):
    tokenizer = TemplateTokenizer(FIXED_ON)
    assert load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True).enable_thinking
    rejects(model, tokenizer, enable_thinking=False)
    rejects(model, tokenizer, force_disable=True)


def test_fixed_no_thinking_accepts_only_off(model):
    tokenizer = TemplateTokenizer(FIXED_OFF)
    assert not load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False).enable_thinking
    rejects(model, tokenizer, enable_thinking=True)


def test_actual_template_wins_over_disk_and_cache_tracks_replacement(model):
    (model / 'tokenizer_config.json').write_text(json.dumps({'chat_template': SWITCH}))
    tokenizer = TemplateTokenizer(SWITCH)
    assert load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False).mode == 'disabled'
    tokenizer.chat_template = FIXED_ON
    rejects(model, tokenizer, enable_thinking=False)
    assert load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True).mode == 'separate'


def test_legacy_constructor_and_disabled_model_still_work(tmp_path):
    settings = ReasoningSettings('separate', 'hidden', 'qwen3', 'reasoning_content', True)
    assert settings.enable_thinking
    assert settings.template_kwargs == {'enable_thinking': True}
    assert not settings.requires_template_kwargs
    assert load_reasoning_settings(tmp_path).mode == 'disabled'


@pytest.mark.parametrize('kwargs', [{'enable_thinking': 1}, {'preserve_thinking': 'true'}, {'reasoning_effort': 3}])
def test_resolver_rejects_untyped_direct_controls(model, kwargs):
    rejects(model, TemplateTokenizer(), **kwargs)


def test_qwen3_enable_without_opening_prefix_is_a_real_switch(model):
    class HybridTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
            if not add_generation_prompt:
                return ''
            return '<|im_start|>assistant\n' + ('' if kwargs.get('enable_thinking', True) else '<think>\n\n</think>\n\n')

    tokenizer = HybridTokenizer(SWITCH)
    assert load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True).enable_thinking
    assert not load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False).enable_thinking


def test_disabled_server_requires_actual_switch_argument_even_without_override(model, monkeypatch):
    monkeypatch.setenv('GATEWAY_REASONING_MODE', 'disabled')
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer())
    assert settings.requires_template_kwargs
    assert settings.template_kwargs == {'enable_thinking': False}


def test_resolved_switch_keeps_enabled_policy_strict_without_explicit_override(model):
    settings = load_reasoning_settings(model, tokenizer=TemplateTokenizer())
    assert settings.requires_template_kwargs
    assert settings.template_kwargs == {'enable_thinking': True}


def test_non_string_template_render_is_unsupported_instead_of_crashing(model):
    class NonStringTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return {'unexpected': 'tokens'}

    rejects(model, NonStringTokenizer(), enable_thinking=False)


def test_unconditional_opening_prefill_cannot_be_classified_fixed_off(model):
    class UnconditionalTokenizer(TemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return '<think>'

    rejects(model, UnconditionalTokenizer(FIXED_ON), enable_thinking=False)
