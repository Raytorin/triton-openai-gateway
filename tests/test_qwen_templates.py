# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Offline behavioral checks against pinned upstream Qwen Jinja, without weights.

This is a renderer/tokenizer facade, not a model-inference or token-count proof.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from jinja2.sandbox import ImmutableSandboxedEnvironment

from gateway.reasoning import load_reasoning_settings

FIXTURES = Path(__file__).parent / "fixtures" / "qwen"
NAMES = ("qwen3-hybrid", "qwen3-instruct", "qwen3-coder", "qwen3-thinking",
         "qwen35", "qwen36", "qwen36-fp8", "qwen38")
HYBRID = ("qwen3-hybrid", "qwen35", "qwen36", "qwen36-fp8", "qwen38")


def _raise_exception(message):
    raise ValueError(message)


class QwenTemplateTokenizer:
    """Render the real file with the same sandbox/features used by HF Jinja.

    Transformers is intentionally not required: only Jinja2 is a test dependency.
    No vocabulary, token IDs, or model weights are downloaded or simulated.
    """
    def __init__(self, name):
        self.name = name
        self.chat_template = (FIXTURES / name / "chat_template.jinja").read_text()
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        environment.filters["tojson"] = lambda value, **kwargs: json.dumps(
            value, ensure_ascii=False, **kwargs
        )
        environment.globals["raise_exception"] = _raise_exception
        self.template = environment.from_string(self.chat_template)

    def apply_chat_template(self, conversation, *, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        assert tokenize is False, "This fixture proves rendering only"
        return self.template.render(messages=conversation,
                                    add_generation_prompt=add_generation_prompt, **kwargs)


def _model(tmp_path, name, mode="separate"):
    folder = tmp_path / name
    folder.mkdir()
    metadata = json.loads((FIXTURES / name / "metadata.json").read_text())
    (folder / "config.json").write_text(json.dumps({"model_type": metadata["model_type"]}))
    (folder / "gateway.json").write_text(json.dumps({"reasoning": {"mode": mode}}))
    tokenizer = QwenTemplateTokenizer(name)
    (folder / "chat_template.jinja").write_text(tokenizer.chat_template)
    return folder, tokenizer


def _render(tokenizer, settings, messages=None, **extra):
    return tokenizer.apply_chat_template(
        messages or [{"role": "user", "content": "Hello"}],
        **dict(settings.template_kwargs), **extra,
    )


@pytest.mark.parametrize("name", NAMES)
def test_upstream_fixture_has_immutable_provenance_and_integrity(name):
    folder = FIXTURES / name
    metadata = json.loads((folder / "metadata.json").read_text())
    assert len(metadata["revision"]) == 40
    assert set(metadata["revision"]) <= set("0123456789abcdef")
    assert metadata["revision"] in metadata["source_url"]
    assert metadata["license"] == "Apache-2.0"
    for filename, key in [("chat_template.jinja", "template_sha256"), ("LICENSE", "license_sha256")]:
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == metadata[key]


@pytest.mark.parametrize("name", HYBRID)
def test_request_thinking_switch_changes_real_generation_prefix(tmp_path, name):
    model, tokenizer = _model(tmp_path, name)
    enabled = load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True)
    disabled = load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False)
    expected_prefix = "<|im_start|>assistant\n" if name == "qwen3-hybrid" else "<think>\n"
    assert _render(tokenizer, enabled).endswith(expected_prefix)
    assert _render(tokenizer, disabled).endswith("<think>\n\n</think>\n\n")
    assert enabled.enable_thinking is True
    assert disabled.enable_thinking is False


@pytest.mark.parametrize("name", ["qwen3-instruct", "qwen3-coder"])
def test_fixed_instruct_accepts_false_rejects_true(tmp_path, name):
    model, tokenizer = _model(tmp_path, name, mode="disabled")
    settings = load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False)
    assert _render(tokenizer, settings).endswith("<|im_start|>assistant\n")
    assert "<think>" not in _render(tokenizer, settings)
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True)
    assert error.value.status_code == 400


def test_fixed_thinking_accepts_true_rejects_false(tmp_path):
    model, tokenizer = _model(tmp_path, "qwen3-thinking")
    settings = load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=True)
    assert _render(tokenizer, settings).endswith("<think>\n")
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False)
    assert error.value.status_code == 400


def test_qwen38_effort_is_xhigh_not_a_high_alias(tmp_path):
    model, tokenizer = _model(tmp_path, "qwen38")
    settings = load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort="xhigh")
    assert dict(settings.template_kwargs)["reasoning_effort"] == "xhigh"
    assert "Reasoning effort is set to xhigh." in _render(tokenizer, settings)
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort="high")
    assert error.value.status_code == 400


@pytest.mark.parametrize("name", NAMES[:-1])
def test_legacy_qwen_rejects_unsupported_effort(tmp_path, name):
    mode = "disabled" if name in ("qwen3-instruct", "qwen3-coder") else "separate"
    model, tokenizer = _model(tmp_path, name, mode=mode)
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort="high")
    assert error.value.status_code == 400


@pytest.mark.parametrize("name", ["qwen36", "qwen36-fp8", "qwen38"])
def test_preserve_thinking_controls_actual_reasoning_history(tmp_path, name):
    model, tokenizer = _model(tmp_path, name)
    history = [{"role": "user", "content": "First"},
               {"role": "assistant", "content": "Previous answer", "reasoning_content": "PRIVATE HISTORY"},
               {"role": "user", "content": "Continue"}]
    preserve = load_reasoning_settings(model, tokenizer=tokenizer, preserve_thinking=True)
    discard = load_reasoning_settings(model, tokenizer=tokenizer, preserve_thinking=False)
    assert "<think>\nPRIVATE HISTORY\n</think>" in _render(tokenizer, preserve, history)
    assert "PRIVATE HISTORY" not in _render(tokenizer, discard, history)
    assert "Previous answer" in _render(tokenizer, discard, history)


@pytest.mark.parametrize("name", NAMES)
def test_actual_special_tokens_and_tool_roundtrip_framing(name):
    tokenizer = QwenTemplateTokenizer(name)
    tool = {"type": "function", "function": {"name": "weather", "description": "Weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}
    messages = [{"role": "user", "content": "Weather?"},
                {"role": "assistant", "content": "", "tool_calls": [{"type": "function",
                 "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
                {"role": "tool", "content": "sunny"}]
    rendered = tokenizer.apply_chat_template(messages, tools=[tool])
    assert "<|im_start|>user\nWeather?<|im_end|>" in rendered
    assert "<tool_call>" in rendered
    assert "weather" in rendered and "Paris" in rendered
    assert "<tool_response>\nsunny\n</tool_response>" in rendered
    assert rendered.count("<|im_start|>") >= 4


@pytest.mark.parametrize("name", ["qwen3-hybrid", "qwen3-instruct", "qwen3-coder", "qwen3-thinking", "qwen35"])
@pytest.mark.parametrize("preserve", [True, False])
def test_templates_without_preserve_control_reject_explicit_request(tmp_path, name, preserve):
    mode = "disabled" if name in ("qwen3-instruct", "qwen3-coder") else "separate"
    model, tokenizer = _model(tmp_path, name, mode=mode)
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(model, tokenizer=tokenizer, preserve_thinking=preserve)
    assert error.value.status_code == 400


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_qwen38_effort_changes_actual_system_instruction(tmp_path, effort):
    model, tokenizer = _model(tmp_path, "qwen38")
    settings = load_reasoning_settings(model, tokenizer=tokenizer, reasoning_effort=effort)
    rendered = _render(tokenizer, settings)
    if effort == "medium":
        assert "Reasoning effort is set to" not in rendered
    else:
        assert f"Reasoning effort is set to {effort}." in rendered


def test_qwen35_really_drops_reasoning_before_latest_user_turn():
    tokenizer = QwenTemplateTokenizer("qwen35")
    history = [{"role": "user", "content": "First"},
               {"role": "assistant", "content": "Previous answer", "reasoning_content": "PRIVATE HISTORY"},
               {"role": "user", "content": "Continue"}]
    # This upstream version silently ignores an unknown kwarg. Gateway must
    # reject it instead of promising preservation which the template cannot do.
    rendered = tokenizer.apply_chat_template(history, preserve_thinking=True)
    assert "PRIVATE HISTORY" not in rendered
    assert "Previous answer" in rendered


TOOLS = [{'type': 'function', 'function': {'name': 'weather', 'parameters': {'type': 'object'}}}]


class NamedQwenTemplateTokenizer(QwenTemplateTokenizer):
    """Match HF's named default/tool_use selection with genuine pinned Jinja."""

    def __init__(self, default='qwen36', tool_use='qwen3-thinking'):
        self._renderers = {name: QwenTemplateTokenizer(name) for name in (default, tool_use)}
        self.chat_template = {key: self._renderers[name].chat_template
                              for key, name in [('default', default), ('tool_use', tool_use)]}

    def get_chat_template(self, chat_template=None, tools=None):
        if chat_template is not None:
            return self.chat_template.get(chat_template, chat_template)
        return self.chat_template['tool_use' if tools else 'default']

    def apply_chat_template(self, conversation, *, chat_template=None, tools=None,
                            tokenize=False, add_generation_prompt=True, **kwargs):
        source = self.get_chat_template(chat_template, tools)
        renderer = next(item for item in self._renderers.values() if item.chat_template == source)
        return renderer.apply_chat_template(conversation, tools=tools, tokenize=tokenize,
            add_generation_prompt=add_generation_prompt, **kwargs)

    def encode(self, text, add_special_tokens=False):
        return str(text).split()

    def __call__(self, text, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(input_ids=self.encode(text))


@pytest.mark.parametrize('mode, override', [('separate', False), ('disabled', None)])
def test_named_tool_template_cannot_accept_disable_using_default_capabilities(tmp_path, mode, override):
    model, _ = _model(tmp_path, 'qwen36', mode=mode)
    tokenizer = NamedQwenTemplateTokenizer()
    # Prime the same tokenizer's cache with its valid switchable default.
    assert not load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False).enable_thinking
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=tokenizer, tools=TOOLS, enable_thinking=override)
    assert caught.value.status_code == 400


@pytest.mark.parametrize('default, tool_use', [('qwen38', 'qwen36'), ('qwen36', 'qwen38')])
def test_named_templates_validate_effort_in_the_selected_tools_context(tmp_path, default, tool_use):
    model, _ = _model(tmp_path, 'qwen36')
    tokenizer = NamedQwenTemplateTokenizer(default, tool_use)
    for tools, supports_effort in [(None, default == 'qwen38'), (TOOLS, tool_use == 'qwen38')]:
        if not supports_effort:
            with pytest.raises(HTTPException) as caught:
                load_reasoning_settings(model, tokenizer=tokenizer, tools=tools, reasoning_effort='low')
            assert caught.value.status_code == 400
        else:
            settings = load_reasoning_settings(model, tokenizer=tokenizer, tools=tools, reasoning_effort='low')
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hello'}],
                                                     tools=tools, **settings.template_kwargs)
            assert 'Reasoning effort is set to low.' in rendered


def test_same_jinja_capability_cache_includes_tools_content(tmp_path):
    model, _ = _model(tmp_path, 'qwen36')
    tokenizer = QwenTemplateTokenizer('qwen36')
    fixed_source = QwenTemplateTokenizer('qwen3-thinking').chat_template
    source = "{% if tools and tools[0].function.name == 'fixed' %}" + fixed_source + '{% else %}' + tokenizer.chat_template + '{% endif %}'
    tokenizer.chat_template = source
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                extensions=['jinja2.ext.loopcontrols'])
    environment.globals['raise_exception'] = _raise_exception
    tokenizer.template = environment.from_string(source)
    settings = load_reasoning_settings(model, tokenizer=tokenizer, tools=TOOLS, enable_thinking=False)
    assert not settings.enable_thinking
    fixed_tools = [{'type': 'function', 'function': {'name': 'fixed', 'parameters': {'type': 'object'}}}]
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=tokenizer, tools=fixed_tools, enable_thinking=False)
    assert caught.value.status_code == 400


@pytest.mark.parametrize('thinking, effort, preserve', [(False, None, True), (True, 'low', False)])
def test_strict_controls_survive_tools_signature_fallback(tmp_path, thinking, effort, preserve):
    from gateway.prompt import RequestTokenizer, render_chat_prompt

    class NarrowTokenizer(QwenTemplateTokenizer):
        def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True, preserve_thinking=True, reasoning_effort='xhigh'):
            return super().apply_chat_template(conversation, tokenize=tokenize,
                add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking,
                preserve_thinking=preserve_thinking, reasoning_effort=reasoning_effort)

    model, _ = _model(tmp_path, 'qwen38')
    tokenizer = NarrowTokenizer('qwen38')
    # Without tools= in policy resolution this was accepted even before the fix,
    # but RequestTokenizer incorrectly blocked the existing tools translation.
    settings = load_reasoning_settings(model, tokenizer=tokenizer,
        enable_thinking=thinking, preserve_thinking=preserve, reasoning_effort=effort)
    wrapped = RequestTokenizer(tokenizer, settings.template_kwargs, strict=settings.requires_template_kwargs)
    history = [{'role': 'user', 'content': 'Old'},
               {'role': 'assistant', 'content': 'Prior answer', 'reasoning_content': 'PRIVATE HISTORY'},
               {'role': 'user', 'content': 'Continue'}]
    rendered = render_chat_prompt(wrapped, history, TOOLS, enable_thinking=thinking)
    assert '<tools>' in rendered and 'weather' in rendered
    assert rendered.endswith('<think>\n' if thinking else '<think>\n\n</think>\n\n')
    assert ('PRIVATE HISTORY' in rendered) is preserve
    if effort is not None:
        assert f'Reasoning effort is set to {effort}.' in rendered


def test_selected_named_template_is_bound_for_request_retries(tmp_path):
    from gateway.prompt import RequestTokenizer, render_chat_prompt

    model, _ = _model(tmp_path, 'qwen36')
    tokenizer = NamedQwenTemplateTokenizer('qwen36', 'qwen38')
    settings = load_reasoning_settings(model, tokenizer=tokenizer, tools=TOOLS, reasoning_effort='low')
    wrapped = RequestTokenizer(tokenizer, settings.template_kwargs, strict=settings.requires_template_kwargs,
        chat_template=settings.resolved_chat_template, tools=TOOLS)
    # A retry without tools must keep the source and effort that were validated.
    rendered = render_chat_prompt(wrapped, [{'role': 'user', 'content': 'Hello'}],
                                  enable_thinking=settings.enable_thinking)
    assert 'Reasoning effort is set to low.' in rendered
    assert '<tools>' in rendered and 'weather' in rendered


def test_no_tools_signature_probes_the_same_default_used_by_tools_translation(tmp_path):
    from gateway.prompt import RequestTokenizer, render_chat_prompt

    class NarrowNamedTokenizer(NamedQwenTemplateTokenizer):
        def apply_chat_template(self, conversation, *, tokenize=False, add_generation_prompt=True,
                                enable_thinking=True):
            return super().apply_chat_template(conversation, tokenize=tokenize,
                add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)

    model, _ = _model(tmp_path, 'qwen36')
    tokenizer = NarrowNamedTokenizer('qwen36', 'qwen3-thinking')
    settings = load_reasoning_settings(model, tokenizer=tokenizer, tools=TOOLS, enable_thinking=False)
    wrapped = RequestTokenizer(tokenizer, settings.template_kwargs, strict=settings.requires_template_kwargs,
        chat_template=settings.resolved_chat_template, tools=TOOLS)
    rendered = render_chat_prompt(wrapped, [{'role': 'user', 'content': 'Hello'}], TOOLS)
    assert '<tools>' in rendered and 'weather' in rendered
    assert rendered.endswith('<think>\n\n</think>\n\n')


@pytest.mark.parametrize('layout', ['subdirectory', 'parent', 'model_json_external', 'loaded_external'])
def test_tokenizer_metadata_layout_recognizes_fixed_non_thinking(tmp_path, layout):
    model = tmp_path / 'model' / '1'
    model.mkdir(parents=True)
    tokenizer = QwenTemplateTokenizer('qwen3-instruct')
    config = {}
    if layout == 'subdirectory':
        folder = model / 'tokenizer'
    elif layout == 'parent':
        folder = model.parent / 'tokenizer'
    else:
        folder = tmp_path / 'external'
        if layout == 'model_json_external':
            config['tokenizer'] = str(folder)
        else:
            tokenizer.name_or_path = str(folder)
    folder.mkdir(parents=True)
    (folder / 'config.json').write_text('{"model_type":"qwen3"}')
    (model / 'model.json').write_text(json.dumps(config))
    settings = load_reasoning_settings(model, tokenizer=tokenizer, enable_thinking=False)
    assert settings.qwen_profile
    assert not settings.enable_thinking


@pytest.mark.parametrize('name, thinking', [('qwen3-instruct', False), ('qwen3-thinking', True), ('qwen36', False), ('qwen36', True)])
@pytest.mark.parametrize('description', ['Returns a <think> tag', 'Returns a </think> tag'])
def test_tool_description_think_tags_cannot_classify_generation_mode(tmp_path, name, thinking, description):
    model, tokenizer = _model(tmp_path, name, mode='disabled' if name == 'qwen3-instruct' else 'separate')
    tools = [{'type': 'function', 'function': {'name': 'weather', 'description': description,
                                             'parameters': {'type': 'object'}}}]
    settings = load_reasoning_settings(model, tokenizer=tokenizer, tools=tools, enable_thinking=thinking)
    assert settings.enable_thinking is thinking
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hello'}], tools=tools, **settings.template_kwargs)
    expected = '<|im_start|>assistant\n' if name == 'qwen3-instruct' else '<think>\n' if thinking else '<think>\n\n</think>\n\n'
    assert rendered.endswith(expected)
