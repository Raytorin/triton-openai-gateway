# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import asyncio
from contextvars import ContextVar
import threading
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx
import pytest

from gateway.cpu_work import CpuWorkPool, checkpoint
from gateway import generation
from gateway.app import app
from gateway.schemas import ChatCompletionRequest


async def wait_until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.001)


def test_pool_bounds_queue_and_preserves_request_context():
    async def run():
        pool = CpuWorkPool(workers=1, max_queue=1)
        released = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        request_id = ContextVar('test_request_id', default='missing')
        request_id.set('request-123')
        def work():
            loop.call_soon_threadsafe(started.set)
            assert released.wait(2)
            return request_id.get(), threading.get_ident()
        first = asyncio.create_task(pool.run(work))
        second = None
        try:
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(pool.run(lambda: 'second'))
            await wait_until(lambda: pool.queued == 1)
            with pytest.raises(HTTPException) as exc:
                await pool.run(lambda: 'overflow')
            assert exc.value.status_code == 429
            assert pool.active == 1 and pool.queued == 1
            released.set()
            value, thread = await first
            assert value == 'request-123' and thread != threading.get_ident()
            assert await second == 'second'
        finally:
            released.set()
            await asyncio.gather(*[t for t in (first, second) if t], return_exceptions=True)
        assert pool.active == pool.queued == 0
    asyncio.run(run())


def test_running_cancellation_keeps_capacity_until_native_work_stops():
    async def run():
        pool = CpuWorkPool(workers=1, max_queue=0)
        released = threading.Event()
        started = asyncio.Event()
        loop = asyncio.get_running_loop()
        after_checkpoint = []
        def work():
            loop.call_soon_threadsafe(started.set)
            assert released.wait(2)
            checkpoint()
            after_checkpoint.append(True)
        task = asyncio.create_task(pool.run(work))
        try:
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert pool.active == 1 and not task.done()
            with pytest.raises(HTTPException):
                await pool.run(lambda: None)
        finally:
            released.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert not after_checkpoint
        assert pool.active == pool.queued == 0
        assert await pool.run(lambda: 42) == 42
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [True, False])
def test_queued_cancellation_and_timeout_never_run_work(cancel):
    async def run():
        pool = CpuWorkPool(workers=1, max_queue=1, timeout=0.05)
        released = threading.Event()
        first = asyncio.create_task(pool.run(released.wait, 2))
        calls = []
        second = None
        try:
            await wait_until(lambda: pool.active == 1)
            second = asyncio.create_task(pool.run(lambda: calls.append(True)))
            await wait_until(lambda: pool.queued == 1)
            if cancel:
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
            else:
                with pytest.raises(HTTPException) as exc:
                    await second
                assert exc.value.status_code == 429
            assert not calls and pool.queued == 0 and pool.active == 1
        finally:
            released.set()
            await asyncio.gather(*[t for t in (first, second) if t], return_exceptions=True)
        assert await pool.run(lambda: 'available') == 'available'
    asyncio.run(run())


def test_worker_error_releases_slot():
    async def run():
        pool = CpuWorkPool(workers=1)
        def fail():
            raise ValueError('worker failed')
        with pytest.raises(ValueError, match='worker failed'):
            await pool.run(fail)
        assert pool.active == pool.queued == 0
        assert await pool.run(lambda: 1) == 1
    asyncio.run(run())


class CountingTokenizer:
    def __init__(self):
        self.renders = 0
    def apply_chat_template(self, messages, **kwargs):
        self.renders += 1
        return ' '.join(str(m.get('content', '')) for m in messages)
    def encode(self, text, **kwargs):
        return text.split()
    def __call__(self, text, **kwargs):
        return type('Encoded', (), {'input_ids': self.encode(text)})()


def generation_patches(tmp_path, tokenizer, infer, lease):
    from contextlib import ExitStack
    stack = ExitStack()
    for name, value in [('resolve', tmp_path), ('get_backend', 'vllm')]:
        stack.enter_context(patch.object(generation.registry, name, return_value=value))
    stack.enter_context(patch.object(generation.registry, 'validate_route'))
    stack.enter_context(patch.object(generation.registry, 'get_tokenizer_async', AsyncMock(return_value=(tokenizer, tmp_path))))
    stack.enter_context(patch.object(generation.admission, 'acquire', AsyncMock(return_value=lease)))
    stack.enter_context(patch.object(generation, 'call_triton_multimodal', infer))
    return stack


@pytest.mark.parametrize('context_mode', [None, 'disabled'])
def test_fitting_prompt_is_rendered_once(tmp_path, context_mode):
    (tmp_path / 'model.json').write_text('{"max_model_len":4096}')
    tokenizer = CountingTokenizer()
    request = ChatCompletionRequest(model='test', messages=[{'role':'user','content':'hello'}], max_tokens=32)
    async def run():
        infer, lease = AsyncMock(return_value='ok'), AsyncMock()
        with generation_patches(tmp_path, tokenizer, infer, lease):
            result = await generation.generate(request, context_mode=context_mode)
        assert result.message['content'] == 'ok'
        assert tokenizer.renders == 1
        lease.release.assert_awaited_once()
    asyncio.run(run())


@pytest.mark.parametrize('context_mode', [None, 'truncate'])
def test_generation_cpu_work_keeps_health_responsive_and_cancels_before_inference(tmp_path, context_mode):
    (tmp_path / 'model.json').write_text('{"max_model_len":100}')
    (tmp_path / 'gateway.json').write_text('{"context_compression":{"mode":"truncate"}}')
    async def run():
        loop = asyncio.get_running_loop()
        started, released = asyncio.Event(), threading.Event()
        class SlowTokenizer(CountingTokenizer):
            def encode(self, text, **kwargs):
                loop.call_soon_threadsafe(started.set)
                assert released.wait(2)
                return super().encode(text, **kwargs)
        tokenizer, infer, lease = SlowTokenizer(), AsyncMock(return_value='ok'), AsyncMock()
        request = ChatCompletionRequest(model='test', messages=[
            {'role':'user','content':'history ' * 300}, {'role':'assistant','content':'answer'},
            {'role':'user','content':'hello'},
        ], max_tokens=16)
        with generation_patches(tmp_path, tokenizer, infer, lease):
            task = asyncio.create_task(generation.generate(request, context_mode=context_mode))
            try:
                await asyncio.wait_for(started.wait(), 1)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                    response = await asyncio.wait_for(client.get('/health'), 0.5)
                assert response.status_code == 200 and not task.done()
                infer.assert_not_awaited()
                task.cancel()
                await asyncio.sleep(0)
                lease.release.assert_not_awaited()
            finally:
                released.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
            lease.release.assert_awaited_once()
            infer.assert_not_awaited()
            # No second encode/truncation pass after cancellation.
            assert tokenizer.renders == 1
    asyncio.run(run())


def test_anyio_cancellation_drains_running_worker():
    import anyio
    async def run():
        pool = CpuWorkPool(workers=1)
        released, started = threading.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        def work():
            loop.call_soon_threadsafe(started.set)
            assert released.wait(2)
            checkpoint()
        async with anyio.create_task_group() as group:
            group.start_soon(pool.run, work)
            await started.wait()
            group.cancel_scope.cancel()
            with anyio.CancelScope(shield=True):
                try:
                    await asyncio.sleep(0)
                    assert pool.active == 1
                finally:
                    released.set()
        assert pool.active == pool.queued == 0
    asyncio.run(run())
