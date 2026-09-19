import unittest
from unittest.mock import patch

from fastapi import HTTPException
from grpc import StatusCode
from grpc.aio import AioRpcError

from gateway.triton_client import _triton_grpc_error, call_triton_multimodal


class TritonErrorTests(unittest.TestCase):
    def test_backend_failures_remain_bad_gateway(self):
        for detail in (
            "connection refused",
            "CUDA out of memory",
            "model unavailable",
        ):
            with self.subTest(detail=detail):
                error = _triton_grpc_error("generation gRPC infer", RuntimeError(detail))
                self.assertEqual(error.status_code, 502)
                self.assertIn(detail, error.detail)

    def test_unknown_lora_is_a_bad_request(self):
        error = _triton_grpc_error(
            "generation gRPC infer",
            RuntimeError("LoRA unknown is not supported, we currently support ['known']"),
        )
        self.assertEqual(error.status_code, 400)

    def test_context_limit_remains_payload_too_large(self):
        error = _triton_grpc_error(
            "generation gRPC infer", RuntimeError("maximum model length exceeded")
        )
        self.assertEqual(error.status_code, 413)

    def test_task_mismatch_is_a_bad_request(self):
        error = _triton_grpc_error(
            "generation gRPC stream",
            RuntimeError(
                "Model Qwen3-Embedding-4B does not support 'generate' request"
            ),
        )

        self.assertEqual(error.status_code, 400)
        self.assertIn("does not support 'generate'", str(error.detail))


class TritonGrpcGenerationErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_aio_errors_preserve_client_and_backend_statuses(self):
        for detail, expected in (
            ("LoRA unknown is not supported, we currently support ['known']", 400),
            ("connection refused", 502),
        ):
            with self.subTest(detail=detail):
                rpc_error = AioRpcError(StatusCode.INTERNAL, None, None, details=detail)

                async def failing_results(*args, **kwargs):
                    raise rpc_error
                    yield  # Make this an async iterator like _stream_grpc_results.

                with patch("gateway.triton_client._stream_grpc_results", failing_results):
                    with self.assertRaises(HTTPException) as caught:
                        await call_triton_multimodal("chat-model", "hello", {}, [])

                self.assertEqual(caught.exception.status_code, expected)
                self.assertIs(caught.exception.__cause__, rpc_error)


if __name__ == "__main__":
    unittest.main()
