# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Numeric thinking budgets are separate from the shared completion limit."""

import json

from fastapi import HTTPException
from pydantic import ValidationError
import pytest

from gateway import reasoning
from gateway.prompt import build_sampling_parameters
from gateway.reasoning import load_reasoning_settings
from gateway.responses import ResponsesRequest, to_chat
from gateway.schemas import ChatCompletionRequest


SWITCH = "{% if enable_thinking is not defined or enable_thinking %}<think>{% else %}<think></think>{% endif %}"


class ThinkingTokenizer:
    chat_template = SWITCH

    def encode(self, text, *, add_special_tokens=False):
        return [17]

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, *, add_generation_prompt=True, **kwargs):
        if not add_generation_prompt:
            return ""
        return "assistant:" + ("<think>" if kwargs.get("enable_thinking", True) else "<think></think>")


def chat(**kwargs):
    return ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "Summarize this."}], **kwargs)


@pytest.mark.parametrize("budget", [-1, True, False, 1.5, "128"])
@pytest.mark.parametrize("protocol", ["chat", "responses"])
def test_budget_requires_a_nonnegative_integer(protocol, budget):
    with pytest.raises(ValidationError):
        if protocol == "chat":
            chat(thinking_token_budget=budget)
        else:
            ResponsesRequest(model="test", input="Summarize this.", thinking_token_budget=budget)


@pytest.mark.parametrize("budget", [None, 0, 128])
@pytest.mark.parametrize("stream", [False, True])
def test_chat_budget_is_declared_and_forwarded_to_native_sampling(budget, stream):
    request = chat(thinking_token_budget=budget, max_completion_tokens=4096, stream=stream)
    assert "thinking_token_budget" in ChatCompletionRequest.model_fields
    assert "thinking_token_budget" not in (request.model_extra or {})
    sampling = build_sampling_parameters(request)
    assert sampling["max_tokens"] == 4096
    if budget is None:
        assert "thinking_token_budget" not in sampling
    else:
        assert sampling["thinking_token_budget"] == budget


@pytest.mark.parametrize("budget", [None, 0, 128])
@pytest.mark.parametrize("stream", [False, True])
def test_responses_budget_survives_conversion_and_native_sampling(budget, stream):
    response = ResponsesRequest(model="test", input="Summarize this.",
        max_output_tokens=4096, thinking_token_budget=budget, stream=stream)
    request = to_chat(response)
    assert request.thinking_token_budget == budget
    assert request.stream is stream
    assert request.include_reasoning is False
    assert build_sampling_parameters(request).get("thinking_token_budget") == budget


@pytest.mark.parametrize("budget", [128, 129])
def test_budget_must_leave_room_within_effective_completion_limit(budget):
    request = chat(thinking_token_budget=budget, max_tokens=4096)
    with pytest.raises(HTTPException) as caught:
        build_sampling_parameters(request, output_token_limit=128)
    assert caught.value.status_code == 400
    assert "thinking_token_budget" in str(caught.value.detail)


def test_numeric_budget_does_not_change_effort_or_other_sampling_controls():
    request = chat(thinking_token_budget=128, reasoning_effort="low", max_tokens=4096, temperature=0.6)
    sampling = build_sampling_parameters(request)
    assert sampling["thinking_token_budget"] == 128
    assert sampling["max_tokens"] == 4096
    assert sampling["temperature"] == 0.6
    assert "reasoning_effort" not in sampling


@pytest.fixture
def model(tmp_path, monkeypatch):
    for key in ("GATEWAY_REASONING_MODE", "GATEWAY_REASONING_PARSER", "GATEWAY_REASONING_RESPONSE_FIELD"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_8"}')
    (tmp_path / "model.json").write_text(json.dumps({"reasoning_config": {"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}}))
    (tmp_path / "gateway.json").write_text('{"reasoning":{"mode":"separate"}}')
    return tmp_path


@pytest.mark.parametrize("budget", [0, 128])
def test_enabled_thinking_accepts_numeric_budget_without_template_keyword(model, budget):
    settings = load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=budget)
    assert settings.enable_thinking
    assert settings.template_kwargs == {"enable_thinking": True}


@pytest.mark.parametrize("budget", [0, 128])
@pytest.mark.parametrize("disabled", [{"enable_thinking": False}, {"force_disable": True}])
def test_disabled_thinking_rejects_budget_instead_of_silently_ignoring_it(model, budget, disabled):
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=budget, **disabled)
    assert caught.value.status_code == 400


def test_hidden_reasoning_still_accepts_budget(model):
    settings = load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), include_reasoning=False, thinking_token_budget=128)
    assert settings.mode == "hidden"
    assert settings.enable_thinking


@pytest.mark.parametrize("budget", [-1, True, False, 1.5, "128"])
def test_direct_resolver_rejects_invalid_budget_values(model, budget):
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=budget)
    assert caught.value.status_code == 400


def test_budget_requires_a_supported_thinking_template(model):
    (model / "gateway.json").write_text(json.dumps({"reasoning": {"mode": "separate", "parser": "qwen3"}}))
    tokenizer = ThinkingTokenizer()
    tokenizer.chat_template = "plain assistant"
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=tokenizer, thinking_token_budget=128)
    assert caught.value.status_code == 400


def test_responses_maps_effort_and_numeric_budget_independently():
    request = ResponsesRequest(model="test", input="text", reasoning={"effort": "low"}, thinking_token_budget=128)
    chat = to_chat(request)
    assert chat.reasoning_effort == "low"
    assert chat.thinking_token_budget == 128


@pytest.mark.parametrize("config", [{}, {"reasoning_start_str": "<think>"},
    {"reasoning_start_str": "", "reasoning_end_str": "</think>"},
    {"reasoning_start_str": "<think>", "reasoning_end_str": 17},
    {"reasoning_start_str": "<think>", "reasoning_end_str": " "}])
def test_budget_requires_explicit_engine_reasoning_boundaries(model, config):
    (model / "model.json").write_text(json.dumps({"reasoning_config": config}))
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=128)
    assert caught.value.status_code == 400
    assert "reasoning_config" in str(caught.value.detail)


def test_budget_rejects_missing_engine_config(model):
    (model / "model.json").unlink()
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=128)
    assert caught.value.status_code == 400


@pytest.mark.parametrize("budget,limit", [(0, 2), (128, 130)])
def test_output_reserve_includes_closing_tag_and_answer(model, budget, limit):
    reasoning.validate_thinking_token_budget_output(model, ThinkingTokenizer(), budget, limit)


@pytest.mark.parametrize("budget,limit", [(0, 1), (128, 129), (128, 128)])
def test_output_limit_must_fit_thinking_close_tag_and_answer(model, budget, limit):
    with pytest.raises(HTTPException) as caught:
        reasoning.validate_thinking_token_budget_output(model, ThinkingTokenizer(), budget, limit)
    assert caught.value.status_code == 400
    assert "final answer" in str(caught.value.detail)


def test_output_reserve_counts_actual_multitoken_closing_tag(model):
    class MultiTokenClose(ThinkingTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            assert text == "</think>" and add_special_tokens is False
            return [17, 18, 19]

    tokenizer = MultiTokenClose()
    reasoning.validate_thinking_token_budget_output(model, tokenizer, 128, 132)
    with pytest.raises(HTTPException):
        reasoning.validate_thinking_token_budget_output(model, tokenizer, 128, 131)


@pytest.mark.parametrize("encoded", [[], None, "17", [True], ["17"]])
def test_output_reserve_rejects_unusable_closing_tokens(model, encoded):
    class BrokenTokenizer(ThinkingTokenizer):
        def encode(self, text, **kwargs):
            return encoded

    with pytest.raises(HTTPException) as caught:
        reasoning.validate_thinking_token_budget_output(model, BrokenTokenizer(), 128, 4096)
    assert caught.value.status_code == 400


def test_omitted_budget_needs_no_engine_config_or_tokenizer(tmp_path):
    reasoning.validate_thinking_token_budget_output(tmp_path, None, None, 1)


@pytest.mark.parametrize("start,end", [("<think>", "WRONG"), ("<think>", "</thinking>"), ("WRONG", "</think>")])
def test_budget_rejects_engine_boundaries_the_gateway_cannot_parse(model, start, end):
    (model / "model.json").write_text(json.dumps({"reasoning_config": {"reasoning_start_str": start, "reasoning_end_str": end}}))
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=128)
    assert caught.value.status_code == 400


def test_budget_rejects_start_tag_absent_from_resolved_template(model):
    (model / "model.json").write_text(json.dumps({"reasoning_config": {"reasoning_start_str": "<thinking>", "reasoning_end_str": "</thinking>"}}))
    with pytest.raises(HTTPException) as caught:
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=128)
    assert caught.value.status_code == 400


def test_budget_rejects_transition_phrase_in_engine_close_marker(model):
    (model / "model.json").write_text(json.dumps({"reasoning_config": {"reasoning_start_str": "<think>", "reasoning_end_str": "Time to finish.\n</think>\n"}}))
    with pytest.raises(HTTPException):
        load_reasoning_settings(model, tokenizer=ThinkingTokenizer(), thinking_token_budget=128)


@pytest.mark.parametrize("budget", [0, 128])
def test_native_budget_preserves_forced_reasoning_markers_for_gateway_parser(budget):
    sampling = build_sampling_parameters(chat(thinking_token_budget=budget, max_tokens=4096))
    assert sampling["skip_special_tokens"] is False


def test_omitted_budget_preserves_existing_native_detokenization_defaults():
    assert "skip_special_tokens" not in build_sampling_parameters(chat(max_tokens=4096))
