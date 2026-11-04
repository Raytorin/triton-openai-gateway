"""Visual engine facts survive the actual generation transport lifecycle."""
import json
import asyncio
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from gateway import triton_client as tc
from gateway.generation_telemetry import (
    get_generation_telemetry, reset_generation_telemetry, start_generation_telemetry,
    visual_usage_scope,
)
from gateway.multimodal import MediaPayload, MediaPayloads
from gateway.prompt import build_usage
from test_generation_metadata_transport import Client, Result, envelope, run_async, chat_request
from test_reasoning import FakeTokenizer, reasoning_settings, parse_sse


def facts(kind="image", count=3, **extra):
    return {**envelope(**extra), "vision_usage": {"status": "available", "source": "engine",
            "unit": "embedding_positions", "by_source": {kind: count}}}


@pytest.fixture
def telemetry():
    token = start_generation_telemetry("visual-test", "/v1/chat/completions", time.monotonic())
    state = get_generation_telemetry()
    state.configure(model="test-metadata", route="chat")
    try:
        yield state
    finally:
        reset_generation_telemetry(token)


@run_async
async def test_main_visual_transport_records_once_before_usage(telemetry):
    client = Client([Result(b"answer", facts())])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        output = await tc.call_triton_multimodal("test-metadata", "prompt", {}, ["data:image/png;base64,AA=="])
    usage = build_usage(FakeTokenizer(), "prompt", output)
    assert usage["prompt_tokens"] == 42 and usage["total_tokens"] == 43
    assert usage["vision_usage"]["main"] == {
        "status": "available", "source": "engine", "unit": "embedding_positions",
        "by_source": {"image": 3}, "calls": 1}
    before = telemetry.visual_usage.snapshot()
    telemetry.finish(200)
    assert telemetry.finish(200) is None
    assert telemetry.visual_usage.snapshot() == before


@run_async
async def test_stream_empty_final_keeps_cumulative_visual_facts_once(telemetry):
    first = facts(count=3, finished=False, reason=None)
    client = Client([Result(b"a", first), Result(b"", facts())])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        frames = [event async for event in tc.stream_triton_multimodal_to_openai(
            chat_request(), FakeTokenizer(), "prompt", {}, ["data:image/png;base64,AA=="],
            reasoning_settings=reasoning_settings("disabled"))]
    last = parse_sse(frames)[-1]
    assert last["usage"]["vision_usage"]["main"]["by_source"] == {"image": 3}
    assert last["usage"]["vision_usage"]["main"]["status"] == "available"
    assert last["usage"]["vision_usage"]["main"]["calls"] == 1


@run_async
async def test_deepseek_stream_cancellation_retains_partial_observed_usage(telemetry):
    client = Client([Result(b"a", facts(finished=False, reason=None)), Result(b"", facts())])
    media = MediaPayloads(images=[MediaPayload("AA==", "image/png")],
                          videos=[], audios=[], pdfs=[], order=["image"])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        source = tc.stream_triton_text_deltas("test-metadata", "prompt", {}, media=media)
        assert await anext(source) == "a"
        await source.aclose()
    observed = telemetry.visual_usage.snapshot()["main"]
    assert observed["status"] == "partial" and observed["by_source"] == {"image": 3}
    assert observed["calls"] == 1


@run_async
async def test_preprocessing_calls_sum_even_when_main_fails(telemetry):
    client = Client([Result(b"summary", facts("pdf", 6))])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        with visual_usage_scope("preprocessing"):
            for _ in range(2):
                await tc.call_triton_multimodal("test-metadata", "prompt", {}, ["data:image/png;base64,AA=="])
        client.results = [Result(b"partial", facts(finished=False, reason=None))]
        with pytest.raises(Exception):
            await tc.call_triton_multimodal("test-metadata", "prompt", {}, ["data:image/png;base64,AA=="])
    snapshot = telemetry.visual_usage.snapshot()
    assert snapshot["preprocessing"]["status"] == "available"
    assert snapshot["preprocessing"]["by_source"] == {"pdf": 12}
    assert snapshot["preprocessing"]["calls"] == 2
    assert snapshot["main"]["status"] == "partial"


@run_async
async def test_legacy_visual_backend_is_unavailable_not_zero(telemetry):
    client = Client([Result(b"answer")], supported=False)
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        await tc.call_triton_multimodal("test-metadata", "prompt", {}, ["data:image/png;base64,AA=="])
    assert telemetry.visual_usage.snapshot()["main"] == {
        "status": "unavailable", "source": "engine", "unit": "embedding_positions",
        "by_source": {}, "calls": 1}


@run_async
async def test_http_metadata_counts_visual_facts_without_duplicate_call(telemetry):
    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"inputs": [{"name": "return_generation_metadata"}],
                "outputs": [{"name": "generation_metadata"}]})
        return httpx.Response(200, json={"text_output": "answer", "generation_metadata": [json.dumps(facts())]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with patch.object(tc, "get_http_client", return_value=client):
            await tc.call_triton("test-metadata", "prompt", {})
    assert telemetry.visual_usage.snapshot()["main"]["by_source"] == {"image": 3}
    assert telemetry.visual_usage.snapshot()["main"]["calls"] == 1


@run_async
async def test_audio_only_call_does_not_create_visual_usage(telemetry):
    client = Client([Result(b"answer", envelope())])
    media = MediaPayloads(images=[], videos=[], audios=[MediaPayload("AA==", "audio/wav")],
                          pdfs=[], order=["audio"])
    with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
        await tc.call_triton_multimodal("test-metadata", "prompt", {}, [], media=media)
    assert telemetry.visual_usage.snapshot() == {}


@run_async
@pytest.mark.parametrize('cancelled', [False, True])
async def test_transport_failure_after_terminal_metadata_keeps_counts_partial(telemetry, cancelled):
    client = Client([])
    async def wire(*args, **kwargs):
        yield Result(b'answer', facts())
        if cancelled:
            raise asyncio.CancelledError()
        raise RuntimeError('transport failed after final engine packet')
    from fastapi import HTTPException
    with patch.object(tc, 'get_grpc_client', AsyncMock(return_value=client)), patch.object(tc, '_stream_grpc_results', wire):
        with pytest.raises(asyncio.CancelledError if cancelled else HTTPException):
            await tc.call_triton_multimodal('test-metadata', 'prompt', {}, ['data:image/png;base64,AA=='])
    detail = telemetry.visual_usage.snapshot()['main']
    assert detail['status'] == 'partial'
    assert detail['by_source'] == {'image': 3} and detail['calls'] == 1


@run_async
async def test_deepseek_normal_terminal_close_remains_available(telemetry):
    client = Client([Result(b'answer', facts())])
    media = MediaPayloads(images=[MediaPayload('AA==', 'image/png')],
                          videos=[], audios=[], pdfs=[], order=['image'])
    with patch.object(tc, 'get_grpc_client', AsyncMock(return_value=client)):
        source = tc.stream_triton_text_deltas('test-metadata', 'prompt', {}, media=media)
        result = await anext(source)
        assert result.generation_metadata.finished
        await source.aclose()
    assert telemetry.visual_usage.snapshot()['main']['status'] == 'available'
