import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from gateway.admission import AdmissionController


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_model_queue_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "admission": {
                            "chat": {
                                "max_inflight": 1,
                                "max_queue": 0,
                                "queue_timeout_seconds": 1,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            controller = AdmissionController()
            with patch.dict(
                "os.environ",
                {
                    "GATEWAY_MAX_INFLIGHT_REQUESTS": "10",
                    "GATEWAY_MAX_QUEUE_SIZE": "10",
                },
                clear=False,
            ):
                lease = await controller.acquire("chat", "model-a", model_path)
                with self.assertRaises(HTTPException) as context:
                    await controller.acquire("chat", "model-a", model_path)
                await lease.release()

        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.headers["Retry-After"], "1")

    async def test_release_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = AdmissionController()
            lease = await controller.acquire("chat", "model-a", Path(directory))
            self.assertGreaterEqual(lease.wait_seconds, 0.0)
            await lease.release()
            await lease.release()


if __name__ == "__main__":
    unittest.main()
