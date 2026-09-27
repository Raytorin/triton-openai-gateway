# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
import json
import unittest
from unittest.mock import AsyncMock

from gateway.generation_types import generation_stream, chat_events, ManagedStreamingResponse


class GenerationStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_early_close_releases_backend_and_admission(self):
        closed = []
        async def backend():
            try:
                while True:
                    yield 'data: ' + json.dumps({"id": "chatcmpl-a", "created": 1,
                        "model": "test", "choices": [{"delta": {"content": "a"},
                        "finish_reason": None}]}) + '\n\n'
            finally:
                closed.append(True)
        stream = generation_stream(backend(), media_type="text/event-stream", headers={})
        stream.lease = AsyncMock()
        source = chat_events(stream)
        self.assertIn('"content": "a"', await anext(source))
        await source.aclose()
        await stream.aclose()
        self.assertEqual(closed, [True])
        stream.lease.release.assert_awaited_once()

    async def test_asgi_failure_before_iteration_releases_lease(self):
        async def backend():
            yield ''
        stream = generation_stream(backend(), media_type="text/event-stream", headers={})
        stream.lease = AsyncMock()
        response = ManagedStreamingResponse(chat_events(stream), close=stream.aclose)
        async def send(_message):
            raise RuntimeError("socket closed")
        with self.assertRaises(RuntimeError):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, AsyncMock(), send)
        stream.lease.release.assert_awaited_once()

    async def test_responses_disconnect_closes_backend_and_releases_once(self):
        from gateway.generation_types import GenerationEvent, GenerationStream
        from gateway.responses import ResponsesRequest
        from gateway.responses_stream import response_events
        closed = []
        async def backend():
            try:
                yield GenerationEvent("x", 1, "test", {"content": "hello"})
                while True:
                    yield GenerationEvent("x", 1, "test", {"content": "more"})
            finally:
                closed.append(True)
        stream = GenerationStream(backend(), {})
        stream.lease = AsyncMock()
        first = await anext(stream.events)
        source = response_events(ResponsesRequest(model="test", input="hello", stream=True), stream, first)
        await anext(source)
        await source.aclose()
        await stream.aclose()
        self.assertEqual(closed, [True])
        stream.lease.release.assert_awaited_once()

    async def test_native_stream_close_cancels_grpc_iterator(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np
        from gateway import triton_client as tc
        from gateway.multimodal import MediaPayloads
        from gateway.schemas import ChatCompletionRequest
        from gateway.generation_types import backend_events
        class Iterator:
            cancelled = False
            def __aiter__(self):
                return self
            async def __anext__(self):
                return SimpleNamespace(as_numpy=lambda name: np.array([b"x" * 200], dtype=object)), None
            def cancel(self):
                self.cancelled = True
        iterator = Iterator()
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return SimpleNamespace(input_ids=list(text))
        client = SimpleNamespace(stream_infer=lambda *args, **kwargs: iterator)
        request = ChatCompletionRequest(model="test", messages=[{"role":"user", "content":"hi"}])
        with patch.object(tc, "get_grpc_client", AsyncMock(return_value=client)):
            source = backend_events(tc.stream_triton_native_multimodal_to_openai(
                request, Tokenizer(), "prompt", {"max_tokens": 4096}, MediaPayloads([], [], [], [], [])))
            await anext(source)  # role; backend has not started yet
            await anext(source)  # content; backend active
            await source.aclose()
        self.assertTrue(iterator.cancelled)

    async def test_disconnect_before_headers_cancels_inference(self):
        import asyncio
        from fastapi import HTTPException
        from types import SimpleNamespace
        from gateway.generation_types import until_disconnected
        started = asyncio.Event()
        closed = []
        async def operation():
            try:
                started.set()
                await asyncio.Future()
            finally:
                closed.append(True)
        async def receive():
            await started.wait()
            return {"type": "http.disconnect"}
        with self.assertRaises(HTTPException) as exc:
            await until_disconnected(operation(), SimpleNamespace(receive=receive))
        self.assertEqual(exc.exception.status_code, 499)
        self.assertEqual(closed, [True])

    async def test_cancellation_as_preparation_returns_closes_unclaimed_stream(self):
        import asyncio
        from types import SimpleNamespace
        from gateway.generation_types import GenerationEvent, GenerationStream, until_disconnected

        closed = []
        async def backend():
            try:
                yield GenerationEvent("x", 1, "test", {"content": "hello"})
            finally:
                closed.append(True)
        for phase in ("generation", "watcher_cleanup"):
            with self.subTest(phase=phase):
                closed.clear()
                stream = GenerationStream(backend(), {})
                stream.lease = AsyncMock()
                await anext(stream.events)
                watching = asyncio.Event()
                async def operation():
                    await watching.wait()
                    if phase == "generation":
                        owner.cancel()
                    return stream
                async def receive():
                    try:
                        watching.set()
                        await asyncio.Future()
                    finally:
                        if phase == "watcher_cleanup":
                            owner.cancel()
                owner = asyncio.create_task(until_disconnected(operation(), SimpleNamespace(receive=receive)))
                with self.assertRaises(asyncio.CancelledError):
                    await owner
                self.assertEqual(closed, [True])
                self.assertTrue(stream.closed)
                stream.lease.release.assert_awaited_once()

    async def test_successful_preparation_transfers_stream_ownership(self):
        import asyncio
        from types import SimpleNamespace
        from gateway.generation_types import GenerationStream, until_disconnected

        async def backend():
            yield None
        stream = GenerationStream(backend(), {})
        stream.lease = AsyncMock()
        async def operation():
            return stream
        async def receive():
            await asyncio.Future()
        result = await until_disconnected(operation(), SimpleNamespace(receive=receive))
        self.assertIs(result, stream)
        self.assertFalse(stream.closed)
        stream.lease.release.assert_not_awaited()
        await stream.aclose()
        stream.lease.release.assert_awaited_once()
