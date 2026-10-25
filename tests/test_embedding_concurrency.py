import asyncio
import json
import threading
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import httpx
import pytest

from gateway import app as gateway_app
from gateway.schemas import EmbeddingsRequest


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
async def test_embedding_tokenization_and_vector_serialization_leave_event_loop(tmp_path):
    loop_thread = threading.get_ident()
    work_threads = []

    class Tokenizer:
        def encode(self, text, **kwargs):
            work_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            return [10, 20]

    class Vector(list):
        def __iter__(self):
            work_threads.append(threading.get_ident())
            assert threading.get_ident() != loop_thread
            return super().__iter__()

    with (
        patch.object(gateway_app.registry, 'resolve', return_value=tmp_path),
        patch.object(gateway_app.registry, 'validate_route'),
        patch.object(gateway_app.registry, 'get_backend', return_value='vllm'),
        patch.object(gateway_app.registry, 'get_tokenizer_async', AsyncMock(return_value=(Tokenizer(), tmp_path))),
        patch.object(gateway_app, 'call_triton_embeddings', AsyncMock(return_value=(Vector([0.25, -0.5]), 2))),
    ):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway_app.app), base_url='http://test') as client:
            response = await client.post('/v1/embeddings', json={'model': 'embed', 'input': ['one', 'two']})
    assert response.status_code == 200
    assert response.json() == {
        'object': 'list', 'model': 'embed',
        'data': [{'object': 'embedding', 'index': i, 'embedding': [0.25, -0.5]} for i in (0, 1)],
        'usage': {'prompt_tokens': 4, 'total_tokens': 4},
    }
    assert work_threads and all(t != loop_thread for t in work_threads)


@pytest.mark.anyio
async def test_embedding_batch_failure_cancels_other_inference_calls():
    active = set()
    entered = asyncio.Event()

    async def infer(model, item, dimensions, **kwargs):
        active.add(item)
        try:
            if item == 'fail':
                await entered.wait()
                raise HTTPException(502, 'backend failed')
            entered.set()
            await asyncio.Event().wait()
        finally:
            active.remove(item)

    with (
        patch.object(gateway_app.registry, 'get_backend', return_value='python'),
        patch.object(gateway_app, 'call_triton_embeddings', infer),
    ):
        with pytest.raises(HTTPException, match='backend failed'):
            await gateway_app.create_embeddings.__wrapped__(
                EmbeddingsRequest(model='embed', input=['fail', 'wait'])
            )
    assert not active


@pytest.mark.anyio
async def test_embedding_base64_preserves_order_and_usage():
    async def infer(model, item, dimensions, **kwargs):
        await asyncio.sleep(0 if item == [2] else 0.01)
        return ([1.0, 2.0] if item == [1] else [3.0, 4.0]), 1

    with (
        patch.object(gateway_app.registry, 'get_backend', return_value='python'),
        patch.object(gateway_app, 'call_triton_embeddings', infer),
    ):
        response = await gateway_app.create_embeddings.__wrapped__(
            EmbeddingsRequest(model='embed', input=[[1], [2]], encoding_format='base64')
        )
    payload = json.loads(response.body)
    assert [item['embedding'] for item in payload['data']] == ['AACAPwAAAEA=', 'AABAQAAAgEA=']
    assert [item['index'] for item in payload['data']] == [0, 1]
    assert payload['usage'] == {'prompt_tokens': 2, 'total_tokens': 2}


@pytest.mark.anyio
async def test_batch_cancellation_drains_workers_before_returning():
    from gateway.embeddings import run_embedding_batch
    started = asyncio.Event()
    active = set()
    workers = set()

    async def infer(item):
        active.add(item)
        workers.add(asyncio.current_task())
        if len(active) == 3:
            started.set()
        try:
            await asyncio.Event().wait()
        finally:
            # Cleanup itself is asynchronous, as for a gRPC stream.
            await asyncio.sleep(0)
            active.remove(item)

    task = asyncio.create_task(run_embedding_batch(list(range(1000)), infer, 3))
    await asyncio.wait_for(started.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(workers) == 3
    assert not active
