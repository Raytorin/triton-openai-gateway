# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Native boundaries must match active rendered generation, not source comments."""
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi import HTTPException
from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

from gateway.qwen import qwen_template_capabilities
from gateway.reasoning import load_reasoning_settings, validate_thinking_token_budget_output


FIXTURES = Path(__file__).parent / "fixtures" / "qwen"
COMMENTED = ("{# Legacy example: <think> #}{% for m in messages %}{{m.content}}{% endfor %}"
    "{% if add_generation_prompt %}{% if enable_thinking %}<thinking>{% else %}<thinking></thinking>{% endif %}{% endif %}")


class Tokenizer:
    def __init__(self, source):
        self.chat_template = source
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"])
        env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
        self.template = env.from_string(source)

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)

    def encode(self, text, **kwargs):
        return [17] if text in ("<think>", "</think>", "<thinking>", "</thinking>") else [3] * len(text.split())

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))


def model(path, *, start="<think>", end="</think>"):
    (path / "config.json").write_text('{"model_type":"qwen3_8"}')
    (path / "gateway.json").write_text('{"reasoning":{"mode":"separate","parser":"qwen3"}}')
    (path / "model.json").write_text(json.dumps({"max_model_len":16384,
        "reasoning_config":{"reasoning_start_str":start,"reasoning_end_str":end}}))
    return path


def test_commented_legacy_opening_does_not_prove_active_template_boundary(tmp_path):
    tokenizer = Tokenizer(COMMENTED)
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model(tmp_path), tokenizer=tokenizer, thinking_token_budget=0)
    assert caught.value.status_code == 400


def test_rendered_active_tag_supports_matching_engine_boundary(tmp_path):
    tokenizer = Tokenizer(COMMENTED)
    settings = load_reasoning_settings(model(tmp_path, start="<thinking>", end="</thinking>"),
        tokenizer=tokenizer, thinking_token_budget=0)
    assert settings.thinking_tag == "<thinking>"
    assert settings.prefills_thinking is True


@pytest.mark.parametrize("start,end", [(" <think>", "</think>"), ("<think> ", "</think>"),
    ("<think>", " </think>"), ("<think>", "</think> ")])
def test_padded_native_boundaries_are_rejected(tmp_path, start, end):
    tokenizer = Tokenizer((FIXTURES / "qwen38/chat_template.jinja").read_text())
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model(tmp_path, start=start, end=end), tokenizer=tokenizer, thinking_token_budget=0)
    assert caught.value.status_code == 400


@pytest.mark.parametrize("name,prefilled", [("qwen38", True), ("qwen3-hybrid", False)])
def test_bounded_generation_suffix_exposes_prefill_state(tmp_path, name, prefilled):
    path = model(tmp_path)
    tokenizer = Tokenizer((FIXTURES / name / "chat_template.jinja").read_text())
    _, capabilities = qwen_template_capabilities(path, tokenizer)
    assert capabilities.thinking_tag == "<think>"
    assert capabilities.prefills_thinking is prefilled
    settings = load_reasoning_settings(path, tokenizer=tokenizer, thinking_token_budget=0)
    assert settings.prefills_thinking is prefilled


def test_hybrid_budget_reserves_generated_opening_close_and_final_answer(tmp_path):
    path = model(tmp_path)
    tokenizer = Tokenizer((FIXTURES / "qwen3-hybrid/chat_template.jinja").read_text())
    with pytest.raises(HTTPException) as caught:
        validate_thinking_token_budget_output(path, tokenizer, 0, 2, prefilled_thinking=False)
    assert caught.value.status_code == 400
    validate_thinking_token_budget_output(path, tokenizer, 0, 3, prefilled_thinking=False)


def test_prefilled_budget_does_not_spend_output_on_prompt_opening(tmp_path):
    path = model(tmp_path)
    tokenizer = Tokenizer((FIXTURES / "qwen38/chat_template.jinja").read_text())
    validate_thinking_token_budget_output(path, tokenizer, 0, 2, prefilled_thinking=True)
