# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Automatic output budgets must not bypass configured context compression."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from gateway import generation
from gateway.admission import AdmissionLease
from gateway.context_compression import SummaryGeneration, clear_context_summary_cache
from gateway.schemas import ChatCompletionRequest


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(message.get("content", "")) for message in messages)

    def encode(self, text, **kwargs):
        return text.split()

    def __call__(self, text, **kwargs):
        return type("Encoded", (), {"input_ids": self.encode(text)})()


def run_generation(tmp_path, mode, *, history_tokens=90, trigger_ratio=1.0,
                   explicit_limit=None, context_mode=None, default_output=80, output_cap=32768):
    clear_context_summary_cache()
    (tmp_path / "model.json").write_text('{"max_model_len":256}')
    (tmp_path / "gateway.json").write_text(json.dumps({
        "generation": {"default_output_tokens": default_output, "max_output_tokens": output_cap},
        "context_compression": {
            "mode": mode, "fallback_mode": "disabled", "safety_margin_tokens": 8,
            "trigger_ratio": trigger_ratio, "target_ratio": trigger_ratio,
            "summary_max_tokens": 32, "summary_input_max_tokens": 512,
            "preserve_recent_messages": 0,
        },
    }))
    request = ChatCompletionRequest(model="test", messages=[
        {"role": "user", "content": "old " * history_tokens},
        {"role": "assistant", "content": "history " * history_tokens},
        {"role": "user", "content": "latest " * 20},
    ], max_tokens=explicit_limit)
    inference = AsyncMock(return_value="answer")
    summary = AsyncMock(return_value=SummaryGeneration("compact memory", 180, 2))

    async def run():
        with (patch.object(generation.registry, "resolve", return_value=tmp_path),
              patch.object(generation.registry, "validate_route"),
              patch.object(generation.registry, "get_backend", return_value="vllm"),
              patch.object(generation.registry, "get_tokenizer_async", AsyncMock(return_value=(Tokenizer(), tmp_path))),
              patch.object(generation.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))),
              patch.object(generation, "call_triton_multimodal", inference),
              patch.object(generation, "_generate_context_summary", summary)):
            result = await generation.generate(request, context_mode=context_mode)
        return result, inference, summary
    return asyncio.run(run())


def test_summarize_reserves_full_default_output_before_shortening_budget(tmp_path):
    result, inference, summary = run_generation(tmp_path, "summarize")
    assert result.extensions["context_status"]["action"] == "summarize"
    assert summary.await_count == 1
    assert inference.call_args.args[2]["max_tokens"] == 80
    assert result.headers["X-Output-Token-Limit"] == "80"
    assert "compact memory" in inference.call_args.args[1]
    assert "latest " * 19 in inference.call_args.args[1]


def test_summarize_honors_proactive_trigger_without_explicit_output_limit(tmp_path):
    result, inference, summary = run_generation(tmp_path, "summarize", history_tokens=60, trigger_ratio=0.7)
    assert result.extensions["context_status"]["action"] == "summarize"
    assert summary.await_count == 1
    assert inference.call_args.args[2]["max_tokens"] == 80


def test_truncate_reserves_full_default_output_before_shortening_budget(tmp_path):
    result, inference, summary = run_generation(tmp_path, "truncate")
    assert result.extensions["context_status"]["action"] == "truncate"
    assert inference.call_args.args[2]["max_tokens"] == 80
    assert "old " not in inference.call_args.args[1]
    assert "latest " * 19 in inference.call_args.args[1]
    summary.assert_not_awaited()


def test_disabled_compression_still_keeps_history_and_fits_automatic_output(tmp_path):
    result, inference, summary = run_generation(tmp_path, "disabled")
    assert result.extensions["context_status"]["action"] == "none"
    assert inference.call_args.args[2]["max_tokens"] == 48
    assert "old " * 89 in inference.call_args.args[1]
    summary.assert_not_awaited()


def test_explicit_output_limit_still_reserves_exact_requested_budget(tmp_path):
    result, inference, summary = run_generation(tmp_path, "truncate", explicit_limit=64)
    assert result.extensions["context_status"]["action"] == "truncate"
    assert inference.call_args.args[2]["max_tokens"] == 64
    summary.assert_not_awaited()


def test_responses_auto_output_still_preserves_fitting_history(tmp_path):
    result, inference, summary = run_generation(tmp_path, "summarize", context_mode="truncate")
    assert result.extensions["context_status"]["action"] == "none"
    assert inference.call_args.args[2]["max_tokens"] == 48
    assert "old " * 89 in inference.call_args.args[1]
    summary.assert_not_awaited()


def test_default_larger_than_context_keeps_bounded_output_and_configured_mode(tmp_path):
    result, inference, summary = run_generation(tmp_path, "summarize", default_output=512)
    assert result.extensions["context_status"]["mode"] == "summarize"
    assert result.extensions["context_status"]["action"] == "none"
    assert inference.call_args.args[2]["max_tokens"] == 48
    summary.assert_not_awaited()


def test_default_reserve_respects_model_output_cap(tmp_path):
    result, inference, summary = run_generation(tmp_path, "truncate", output_cap=64)
    assert result.extensions["context_status"]["action"] == "truncate"
    assert inference.call_args.args[2]["max_tokens"] == 64
    summary.assert_not_awaited()
