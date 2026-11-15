# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Context failures remain useful to clients without exposing backend diagnostics."""
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.context_compression import load_context_compression_settings, prepare_conversation_context
from gateway.generation import _prepare_responses_context
from gateway.generation_types import GenerationEvent, GenerationStream
from gateway.prompt import fit_conversation_to_context
from gateway.token_budget import GenerationLimits, choose_budget
from gateway.triton_client import _triton_grpc_error


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(m.get("content", "")) for m in messages)

    def encode(self, text, **kwargs):
        return text.split()


def budget_error():
    with pytest.raises(HTTPException) as caught:
        choose_budget(GenerationLimits(40, 80, 100, "model"), 40,
                      prompt_tokens=70, media_tokens=10, safety_margin=2)
    return caught.value


def assert_context(error, *, limit=100, prompt=70, output=40, media=10, margin=2):
    assert error["code"] == "context_length_exceeded"
    assert error["type"] == "invalid_request_error"
    assert error["param"] is None
    assert f"This model's maximum context length is {limit} tokens" in error["message"]
    details = error["details"]
    assert details["context_limit"] == limit
    assert details["prompt_tokens"] == prompt
    assert str(prompt) in error["message"]
    if output is not None:
        assert details["requested_output_tokens"] == output
        assert details["budget_tokens"] == prompt + output + media + margin
    assert details["reserved_media_tokens"] == media
    assert details["safety_margin_tokens"] == margin


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses", "/responses"])
@pytest.mark.parametrize("stream", [False, True])
def test_preflight_errors_share_http_contract(route, stream):
    body = {"model": "fixture", "stream": stream}
    body.update({"messages": [{"role": "user", "content": "hello"}]}
                if route.endswith("completions") else {"input": "hello"})
    with patch("gateway.app.generate", AsyncMock(side_effect=budget_error())), \
         patch("gateway.responses.generation.generate", AsyncMock(side_effect=budget_error())):
        response = TestClient(app).post(route, json=body)
    assert response.status_code == 400
    assert_context(response.json()["error"])


@pytest.mark.parametrize("mode", ["disabled", "truncate", "summarize"])
def test_context_fitting_reports_real_window_and_reservations(tmp_path, mode):
    settings = load_context_compression_settings(tmp_path)
    from dataclasses import replace
    settings = replace(settings, mode=mode, fallback_mode="disabled", safety_margin_tokens=2)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(prepare_conversation_context(model_name="fixture", tokenizer=Tokenizer(),
            conversation=[{"role": "user", "content": "word " * 70}], tools=None,
            max_model_len=100, max_completion_tokens=40, reserved_media_tokens=10,
            settings=settings, summary_generator=AsyncMock(return_value="summary")))
    error = caught.value
    assert error.status_code == 400
    assert f"maximum context length is 100 tokens" in str(error.detail)
    assert_context(error.error)


def test_responses_fitting_reports_actual_counts(tmp_path):
    from dataclasses import replace
    settings = replace(load_context_compression_settings(tmp_path), mode="disabled", safety_margin_tokens=2)
    with pytest.raises(HTTPException) as caught:
        _prepare_responses_context(Tokenizer(), [{"role": "user", "content": "word " * 70}],
            None, 100, 40, settings, None, False, initial=("word " * 70, 70, 10))
    assert_context(caught.value.error)


def test_truncation_reports_retained_prompt_and_keeps_exact_fit():
    conversation = [{"role": "user", "content": "old " * 30},
                    {"role": "assistant", "content": "old " * 20},
                    {"role": "user", "content": "word " * 70}]
    with pytest.raises(HTTPException) as caught:
        fit_conversation_to_context(Tokenizer(), conversation, None, 100, 40,
                                   reserved_media_tokens=10, safety_margin_tokens=2)
    assert_context(caught.value.error)
    fitted = fit_conversation_to_context(Tokenizer(), [{"role": "user", "content": "word " * 48}],
                                        None, 100, 40, reserved_media_tokens=10, safety_margin_tokens=2)
    assert fitted[2] == 48


@pytest.mark.parametrize("prompt,output", [(100, 1), (1, 100)])
def test_no_room_for_output_or_input_reports_counts(tmp_path, prompt, output):
    from dataclasses import replace
    settings = replace(load_context_compression_settings(tmp_path), mode="disabled", safety_margin_tokens=0)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(prepare_conversation_context(model_name="fixture", tokenizer=Tokenizer(),
            conversation=[{"role": "user", "content": "word " * prompt}], tools=None,
            max_model_len=100, max_completion_tokens=output, reserved_media_tokens=0,
            settings=settings))
    assert_context(caught.value.error, prompt=prompt, output=output, media=0, margin=0)


@pytest.mark.parametrize("message,prompt", [
    ("The decoder prompt (length 120) is longer than the maximum model length of 100. PRIVATE_DIAGNOSTIC", 120),
    ("The decoder prompt (length 100) plus the number of requested output tokens (at least 1) "
     "is longer than the maximum model length of 100. PRIVATE_DIAGNOSTIC", 100),
])
def test_backend_context_numbers_are_safe_and_consistent(message, prompt):
    error = _triton_grpc_error("infer", RuntimeError(message))
    assert error.status_code == 400
    assert_context(error.error, prompt=prompt, output=None, media=0, margin=0)
    assert "requested_output_tokens" not in error.error["details"]
    assert "PRIVATE_DIAGNOSTIC" not in json.dumps(error.error)


@pytest.mark.parametrize("message,status", [
    ("CUDA out of memory", 502),
    ("media items; maximum 4", 413),
    ("The encoder prompt (length 120) is longer than the maximum model length of 100.", 413),
    ("The decoder prompt (length 1.2) is longer than the maximum model length of 100.", 413),
])
def test_non_context_errors_are_not_reclassified(message, status):
    error = _triton_grpc_error("infer", RuntimeError(message))
    assert error.status_code == status
    assert not hasattr(error, "error")


def test_backend_combined_budget_reports_only_verified_token_counts():
    text = ("This model's maximum context length is 100 tokens. However, you requested 40 output tokens "
            "and your prompt contains 70 input tokens, for a total of 110 tokens. PRIVATE_DIAGNOSTIC")
    error = _triton_grpc_error("infer", RuntimeError(text))
    assert error.status_code == 400
    assert_context(error.error, media=0, margin=0)
    assert "PRIVATE_DIAGNOSTIC" not in json.dumps(error.error)


@pytest.mark.parametrize("text", [
    "The decoder prompt (length 20) is longer than the maximum model length of 100.",
    "The decoder prompt (length 100) is longer than the maximum model length of 100.",
    "The decoder prompt (length 120) is longer than the maximum model length of 0.",
    "The decoder prompt (length 9999999999999) is longer than the maximum model length of 100.",
    "This model's maximum context length is 100 tokens. However, you requested 40 output tokens "
    "and your prompt contains at least 70 input tokens, for a total of at least 110 tokens.",
    "This model's maximum context length is 100 tokens. However, you requested 40 output tokens "
    "and your prompt contains 70 input tokens, for a total of 999 tokens.",
    "This model's maximum context length is 100 tokens. However, you requested 40 output tokens "
    "and your prompt contains 70 characters.",
])
def test_unverified_backend_counts_are_not_published(text):
    assert not hasattr(_triton_grpc_error("infer", RuntimeError(text)), "error")


@pytest.mark.parametrize("operation", ["call_triton", "call_triton_python_chat"])
def test_http_backend_context_is_normalized_without_diagnostics(operation):
    import httpx
    import gateway.triton_client as tc
    from types import SimpleNamespace
    text = "The decoder prompt (length 120) is longer than the maximum model length of 100. PRIVATE_DIAGNOSTIC"
    client = SimpleNamespace(post=AsyncMock(return_value=httpx.Response(400, json={"error": text})))
    async def run():
        with patch.object(tc, "get_http_client", return_value=client), \
             patch.object(tc, "_supports_generation_metadata", AsyncMock(return_value=False)):
            with pytest.raises(HTTPException) as caught:
                if operation == "call_triton":
                    await tc.call_triton("fixture", "hello", {})
                else:
                    await tc.call_triton_python_chat("fixture", "hello", [], {})
        return caught.value
    error = asyncio.run(run())
    assert error.status_code == 400
    assert_context(error.error, prompt=120, output=None, media=0, margin=0)
    assert "PRIVATE_DIAGNOSTIC" not in str(error.detail)


@pytest.mark.parametrize("status", [400, 200])
def test_http_stream_backend_context_is_normalized(status):
    import httpx
    import gateway.triton_client as tc
    from types import SimpleNamespace
    text = "The decoder prompt (length 120) is longer than the maximum model length of 100. PRIVATE_DIAGNOSTIC"
    async def run():
        content = json.dumps({"error": text})
        if status == 200:
            content = "data: " + content + "\n\n"
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(status, content=content))) as client:
            with patch.object(tc, "get_http_client", return_value=client), \
                 patch.object(tc, "_supports_generation_metadata", AsyncMock(return_value=False)):
                with pytest.raises(HTTPException) as caught:
                    async for _ in tc.stream_triton_to_openai(SimpleNamespace(model="fixture"), None, "hello", {}):
                        pass
        return caught.value
    error = asyncio.run(run())
    assert error.status_code == 400
    assert_context(error.error, prompt=120, output=None, media=0, margin=0)
    assert "PRIVATE_DIAGNOSTIC" not in str(error.detail)


def test_python_chat_context_failure_is_not_an_assistant_answer():
    import gateway.triton_client as tc
    from types import SimpleNamespace
    async def run():
        with patch.object(tc, "call_triton_python_chat", AsyncMock(side_effect=budget_error())):
            with pytest.raises(HTTPException) as caught:
                async for _ in tc.stream_python_chat_to_openai(SimpleNamespace(model="fixture"), None, "hello", [], {}):
                    pytest.fail("Context error was returned as assistant content")
        return caught.value
    assert_context(asyncio.run(run()).error)


def test_grpc_stream_status_error_keeps_safe_context_contract():
    import gateway.triton_client as tc
    from grpc import StatusCode
    from grpc.aio import AioRpcError
    from types import SimpleNamespace
    text = "The decoder prompt (length 120) is longer than the maximum model length of 100. PRIVATE_DIAGNOSTIC"
    async def source(*args, **kwargs):
        raise AioRpcError(StatusCode.INVALID_ARGUMENT, None, None, details=text)
        yield
    async def run():
        with patch.object(tc, "get_grpc_client", AsyncMock(return_value=object())), \
             patch.object(tc, "_generation_outputs", AsyncMock(return_value=([], False))), \
             patch.object(tc, "_stream_grpc_results", source):
            with pytest.raises(HTTPException) as caught:
                async for _ in tc.stream_triton_multimodal_to_openai(
                        SimpleNamespace(model="fixture"), None, "hello", {}, []):
                    pass
        return caught.value
    error = asyncio.run(run())
    assert error.status_code == 400
    assert_context(error.error, prompt=120, output=None, media=0, margin=0)
    assert "PRIVATE_DIAGNOSTIC" not in str(error.detail)


def test_visual_chunks_still_split_on_normalized_context_errors():
    import gateway.vllm_media as media
    items = [media.VisualItem(label="page 1", data="fixture"), media.VisualItem(label="page 2", data="fixture")]
    async def summarize(_model, _tokenizer, chunk, *_args):
        if len(chunk) > 1:
            raise budget_error()
        return chunk[0].label + " summary"
    with patch.object(media, "_generate_visual_summary", summarize):
        summaries = asyncio.run(media._summarize_visual_chunk_adaptive(
            "fixture", None, items, "summarize", [], {}, None))
    assert [s.text for s in summaries] == ["page 1 summary", "page 2 summary"]


def test_unsplittable_visual_chunk_keeps_media_limit_error():
    import gateway.vllm_media as media
    with patch.object(media, "_generate_visual_summary", AsyncMock(side_effect=budget_error())):
        with pytest.raises(media.VllmMediaProcessingError) as caught:
            asyncio.run(media._summarize_visual_chunk_adaptive(
                "fixture", None, [media.VisualItem(label="page 1", data="fixture")], "summarize", [], {}, None))
    assert caught.value.status_code == 413


def test_intermediate_media_summary_keeps_media_limit_error(tmp_path):
    import gateway.vllm_media as media
    from gateway import triton_client as tc
    with patch.object(tc, "call_triton_multimodal", AsyncMock(side_effect=budget_error())):
        with pytest.raises(media.VllmMediaProcessingError) as caught:
            asyncio.run(media._generate_text_summary("fixture", Tokenizer(), "summarize", [], {},
                media.load_vllm_media_settings(tmp_path)))
    assert caught.value.status_code == 413


@pytest.mark.parametrize("failure", ["fitting", "backend"])
@pytest.mark.parametrize("fallback", ["disabled", "truncate"])
def test_summary_model_context_failure_is_not_the_main_context_limit(tmp_path, failure, fallback):
    from dataclasses import replace
    import gateway.generation as generation
    from gateway.context_compression import clear_context_summary_cache
    clear_context_summary_cache()
    (tmp_path / "model.json").write_text('{"max_model_len":100}')
    settings = replace(load_context_compression_settings(tmp_path), mode="summarize",
        fallback_mode=fallback, summary_model="summary-fixture", summary_max_tokens=8,
        preserve_recent_messages=0, trigger_ratio=0.5, target_ratio=0.4,
        safety_margin_tokens=2, evidence_max_tokens=0)
    async def summary(previous, source):
        if failure == "backend":
            raise budget_error()
        return await generation._generate_context_summary("fixture", settings, previous, source)
    async def run():
        with patch.object(generation.registry, "resolve", return_value=tmp_path), \
             patch.object(generation.registry, "validate_route"), \
             patch.object(generation.registry, "get_tokenizer_async", AsyncMock(return_value=(Tokenizer(), tmp_path))):
            return await prepare_conversation_context(model_name="fixture", tokenizer=Tokenizer(),
                conversation=[{"role": "user", "content": "old " * 300},
                              {"role": "assistant", "content": "old " * 300},
                              {"role": "user", "content": "latest " * 50}],
                tools=None, max_model_len=1000, max_completion_tokens=10, reserved_media_tokens=0,
                settings=settings, summary_generator=summary)
    if fallback == "disabled":
        with pytest.raises(HTTPException) as caught:
            asyncio.run(run())
        assert caught.value.status_code == 502
        assert not hasattr(caught.value, "error")
        assert "maximum context length" not in str(caught.value.detail)
    else:
        result = asyncio.run(run())
        assert result.action == "none_fallback" and result.dropped_messages == 0
        assert result.prompt_tokens == 650


@pytest.mark.parametrize("route", ["/v1/chat/completions", "/v1/responses"])
def test_started_stream_has_context_code_and_releases_source(route):
    closed = []
    async def source():
        try:
            yield GenerationEvent("id", 1, "fixture", {"content": "hello"})
            raise budget_error()
        finally:
            closed.append(True)
    stream = GenerationStream(source(), {})
    body = {"model": "fixture", "stream": True}
    body.update({"messages": [{"role": "user", "content": "hello"}]}
                if route.endswith("completions") else {"input": "hello"})
    with patch("gateway.app.generate", AsyncMock(return_value=stream)), \
         patch("gateway.responses.generation.generate", AsyncMock(return_value=stream)):
        response = TestClient(app).post(route, json=body)
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    if route.endswith("completions"):
        assert_context(events[-1]["error"])
        assert response.text.count("data: [DONE]") == 1
    else:
        assert events[-1]["type"] == "response.failed"
        assert_context(events[-1]["response"]["error"])
        assert events[-2]["type"] == "error"
        assert events[-2]["code"] == "context_length_exceeded"
    assert closed == [True] and stream.closed
