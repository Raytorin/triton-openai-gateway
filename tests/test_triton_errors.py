import unittest

from gateway.triton_client import _triton_grpc_error


class TritonErrorTests(unittest.TestCase):
    def test_task_mismatch_is_a_bad_request(self):
        error = _triton_grpc_error(
            "generation gRPC stream",
            RuntimeError(
                "Model Qwen3-Embedding-4B does not support 'generate' request"
            ),
        )

        self.assertEqual(error.status_code, 400)
        self.assertIn("does not support 'generate'", str(error.detail))


if __name__ == "__main__":
    unittest.main()
