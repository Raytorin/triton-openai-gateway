"""Visual map workers finish transport cleanup before request accounting closes."""
import asyncio
from contextlib import nullcontext
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import anyio
from fastapi import HTTPException
from tritonclient.utils import deserialize_bytes_tensor

from gateway import triton_client as tc, vllm_media
from gateway.generation_telemetry import (
    get_generation_telemetry, reset_generation_telemetry, start_generation_telemetry,
)
from gateway.metrics import VISUAL_EMBEDDING_POSITIONS
from test_generation_metadata_transport import Result, envelope, run_async
from test_reasoning import FakeTokenizer
from test_visual_summary_wire import VisualRuntime


def image_data(inputs):
    tensor = next(tensor for tensor in inputs if tensor.name() == "image")
    return [json.loads(value)["data"] for value in deserialize_bytes_tensor(tensor._get_content())]


def map_items(model, data, concurrency=2):
    return vllm_media._summarize_visual_items(
        model, FakeTokenizer(), [vllm_media.VisualItem(value, value, source_kind="pdf") for value in data],
        "Describe the document", [], {},
        SimpleNamespace(image_limit=1, media_map_concurrency=concurrency, summary_max_tokens=128),
        chunk_size=99,
    )


def result(text, count=7):
    return Result(text.encode(), {**envelope(), "vision_usage": {
        "status": "available", "source": "engine", "unit": "embedding_positions",
        "by_source": {"pdf": count},
    }})


@run_async
async def test_failed_map_drains_sibling_before_final_visual_counter():
    runtime = VisualRuntime(custom_media=True, metadata=True)
    sibling_started, cleanup_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owned = []

    async def generate(operation, model, inputs, outputs, **kwargs):
        owned.append(asyncio.current_task())
        if image_data(inputs) == ["failed"]:
            await sibling_started.wait()
            raise RuntimeError("chunk failed")
        sibling_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release.wait()
        yield result("Late facts.")

    telemetry_handle = start_generation_telemetry("map-failure", "/v1/chat/completions", time.monotonic())
    telemetry = get_generation_telemetry()
    telemetry.configure(model="visual-map-failure", route="chat")
    counter = VISUAL_EMBEDDING_POSITIONS.labels("visual-map-failure", "chat", "pdf", "preprocessing", "engine")
    before = counter._value.get()
    finished = []

    async def request():
        try:
            return await map_items("visual-map-failure", ["failed", "slow"])
        finally:
            finished.append(telemetry.finish(502))

    unrelated = asyncio.create_task(asyncio.Event().wait())
    request_task = None
    try:
        with (patch.object(tc, "get_grpc_client", AsyncMock(return_value=runtime)),
              patch.object(tc, "_stream_grpc_results", generate)):
            request_task = asyncio.create_task(request())
            await asyncio.wait_for(cleanup_started.wait(), 1)
            assert not request_task.done()
            assert finished == []
            assert not unrelated.done()
            release.set()
            with pytest.raises(HTTPException, match="chunk failed"):
                await request_task
        assert all(task.done() for task in owned)
        assert not unrelated.done()
        assert finished[0]["vision_usage"]["preprocessing"] == {
            "status": "partial", "source": "engine", "unit": "embedding_positions",
            "by_source": {"pdf": 7}, "calls": 2,
        }
        assert counter._value.get() - before == 7
        assert telemetry.finish(502) is None
    finally:
        release.set()
        for task in [*owned, unrelated, request_task]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*owned, unrelated, *([request_task] if request_task else []), return_exceptions=True)
        reset_generation_telemetry(telemetry_handle)


@run_async
@pytest.mark.parametrize("cancellation", ["task", "scope"])
async def test_repeated_caller_cancellation_drains_only_owned_workers(cancellation):
    runtime = VisualRuntime(custom_media=True, metadata=True)
    started, cleanup_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    owned, cancelled = [], []

    async def generate(operation, model, inputs, outputs, **kwargs):
        owned.append(asyncio.current_task())
        if len(owned) == 2:
            started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(image_data(inputs)[0])
            if len(cancelled) == 2:
                cleanup_started.set()
            await release.wait()
            raise
        yield result("Unexpected completion.")

    telemetry_handle = start_generation_telemetry("map-cancel", "/v1/chat/completions", time.monotonic())
    telemetry = get_generation_telemetry()
    telemetry.configure(model="visual-map-cancel", route="chat")
    finished = []
    cancel_scope = anyio.CancelScope() if cancellation == "scope" else None

    async def request():
        with cancel_scope if cancel_scope is not None else nullcontext():
            try:
                return await map_items("visual-map-cancel", ["first", "second", "queued"])
            finally:
                finished.append(telemetry.finish(499))

    unrelated = asyncio.create_task(asyncio.Event().wait())
    request_task = None
    try:
        with (patch.object(tc, "get_grpc_client", AsyncMock(return_value=runtime)),
              patch.object(tc, "_stream_grpc_results", generate)):
            request_task = asyncio.create_task(request())
            await asyncio.wait_for(started.wait(), 1)
            if cancel_scope is not None:
                cancel_scope.cancel()
            else:
                request_task.cancel()
            await asyncio.wait_for(cleanup_started.wait(), 1)
            if cancel_scope is not None:
                cancel_scope.cancel()
            else:
                request_task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not request_task.done()
            assert finished == []
            assert not unrelated.done()
            release.set()
            if cancel_scope is not None:
                await request_task
                assert cancel_scope.cancelled_caught
            else:
                with pytest.raises(asyncio.CancelledError):
                    await request_task
        assert len(owned) == 2
        assert all(task.done() for task in owned)
        assert sorted(cancelled) == ["first", "second"]
        assert finished[0]["vision_usage"]["preprocessing"] == {
            "status": "unavailable", "source": "engine", "unit": "embedding_positions",
            "by_source": {}, "calls": 2,
        }
        assert not unrelated.done()
    finally:
        release.set()
        for task in [*owned, unrelated, request_task]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*owned, unrelated, *([request_task] if request_task else []), return_exceptions=True)
        reset_generation_telemetry(telemetry_handle)


@run_async
async def test_successful_map_keeps_input_order_and_bounds_concurrency():
    runtime = VisualRuntime(custom_media=True, metadata=True)
    releases = {name: asyncio.Event() for name in ("first", "second", "third")}
    started = {name: asyncio.Event() for name in releases}
    active, peak = 0, 0

    async def generate(operation, model, inputs, outputs, **kwargs):
        nonlocal active, peak
        data = image_data(inputs)
        assert len(data) == 1
        name = data[0]
        active += 1
        peak = max(peak, active)
        started[name].set()
        try:
            await releases[name].wait()
            yield result(f"Facts from {name}.")
        finally:
            active -= 1

    with (patch.object(tc, "get_grpc_client", AsyncMock(return_value=runtime)),
          patch.object(tc, "_stream_grpc_results", generate)):
        task = asyncio.create_task(map_items("visual-map-order", list(releases)))
        try:
            await asyncio.wait_for(started["second"].wait(), 1)
            assert not started["third"].is_set()
            releases["second"].set()
            await asyncio.wait_for(started["third"].wait(), 1)
            releases["third"].set()
            releases["first"].set()
            summaries = await task
        finally:
            for release in releases.values():
                release.set()
            await asyncio.gather(task, return_exceptions=True)
    assert peak == 2
    assert [summary.text for summary in summaries] == [
        "Facts from first.", "Facts from second.", "Facts from third.",
    ]
