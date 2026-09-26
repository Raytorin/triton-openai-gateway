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
