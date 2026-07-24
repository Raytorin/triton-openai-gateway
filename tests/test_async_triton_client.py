import json
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from gateway.triton_client import _stream_grpc_results, call_triton_embeddings


class _Result:
    def __init__(self, outputs):
        self.outputs = outputs

    def as_numpy(self, name):
        return self.outputs.get(name)


class _ResponseIterator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.cancelled = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.responses)
        except StopIteration:
            raise StopAsyncIteration

    def cancel(self):
        self.cancelled = True


class _Client:
    def __init__(self, responses):
        self.iterator = _ResponseIterator(responses)
        self.request = None

    def stream_infer(self, request_iterator, **kwargs):
        self.request_iterator = request_iterator
        self.kwargs = kwargs
        return self.iterator


class AsyncTritonClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeddings_use_shared_async_stream_client(self):
        result = _Result(
            {
                "text_output": np.asarray(
                    [json.dumps([0.1, 0.2]).encode("utf-8")],
                    dtype=np.object_,
                ),
                "num_input_tokens": np.asarray([3], dtype=np.uint32),
            }
        )
        client = _Client([(result, None)])
        with (
            patch(
                "gateway.triton_client.get_grpc_client",
                new=AsyncMock(return_value=client),
            ),
            patch("gateway.triton_client.asyncio.to_thread") as to_thread,
        ):
            embedding, prompt_tokens = await call_triton_embeddings(
                "embedding-model",
                [1, 2, 3],
                None,
            )

        self.assertEqual(embedding, [0.1, 0.2])
        self.assertEqual(prompt_tokens, 3)
        self.assertFalse(client.iterator.cancelled)
        to_thread.assert_not_called()

    async def test_early_consumer_close_cancels_triton_stream(self):
        result = _Result({"text_output": np.asarray([b"partial"], dtype=np.object_)})
        client = _Client([(result, None), (result, None)])
        with patch(
            "gateway.triton_client.get_grpc_client",
            new=AsyncMock(return_value=client),
        ):
            stream = _stream_grpc_results(
                "generate-stream",
                "chat-model",
                [],
                [],
                streaming=True,
            )
            self.assertIs(await anext(stream), result)
            await stream.aclose()

        self.assertTrue(client.iterator.cancelled)


if __name__ == "__main__":
    unittest.main()
