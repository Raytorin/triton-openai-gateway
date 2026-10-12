import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from gateway.admission import AdmissionController, AdmissionPolicy, GatePolicy, load_admission_policy


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_model_waiters_do_not_reserve_global_slots(self):
        controller = AdmissionController()
        policy = AdmissionPolicy(GatePolicy(2, 4, 1), GatePolicy(1, 4, 1))
        with patch("gateway.admission.load_admission_policy", return_value=policy):
            first = await controller.acquire("chat", "busy", Path("/tmp"))
            waiting = asyncio.create_task(controller.acquire("chat", "busy", Path("/tmp")))
            await asyncio.sleep(0.01)
            other = None
            try:
                other = await asyncio.wait_for(controller.acquire("chat", "idle", Path("/tmp")), 0.2)
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                await first.release()
                if other is not None:
                    await other.release()
            self.assertTrue(all(g._inflight == g._queued == 0 for g in controller._gates.values()))

    async def test_global_queue_bounds_waiters_for_busy_models(self):
        controller = AdmissionController()
        policy = AdmissionPolicy(GatePolicy(4, 1, 1), GatePolicy(1, 4, 1))
        with patch("gateway.admission.load_admission_policy", return_value=policy):
            first = await controller.acquire("chat", "busy", Path("/tmp"))
            waiting = asyncio.create_task(controller.acquire("chat", "busy", Path("/tmp")))
            await asyncio.sleep(0.01)
            try:
                with self.assertRaises(HTTPException) as caught:
                    await controller.acquire("chat", "busy", Path("/tmp"))
                self.assertEqual(caught.exception.status_code, 429)
                self.assertIn("queue_full", caught.exception.detail)
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                # Cancellation returns queue capacity; release admits the next request.
                next_request = asyncio.create_task(controller.acquire("chat", "busy", Path("/tmp")))
                await first.release()
                lease = await asyncio.wait_for(next_request, 0.2)
                await lease.release()
            finally:
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                await first.release()
            self.assertTrue(all(g._inflight == g._queued == 0 for g in controller._gates.values()))

    async def test_rerank_is_serialized_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = load_admission_policy(Path(directory), "rerank")

        self.assertEqual(1, policy.model_gate.max_inflight)
        self.assertEqual(64, policy.model_gate.max_queue)

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
