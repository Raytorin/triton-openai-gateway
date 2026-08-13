# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

from gateway.generation_telemetry import GenerationTelemetry


class GenerationTelemetryTests(unittest.TestCase):
    def test_lifecycle_splits_stages_and_records_usage(self):
        telemetry = GenerationTelemetry(
            request_id="request-1",
            path="/v1/chat/completions",
            started_at=0.0,
        )
        telemetry.configure(
            route="chat",
            model="model-a",
            backend="vllm_multimodal",
            transport="grpc",
        )

        with patch(
            "gateway.generation_telemetry.time.monotonic",
            side_effect=[1.0, 2.0, 3.0, 5.0, 6.0],
        ):
            telemetry.admitted(0.25)
            call = telemetry.begin_triton_call("generate-stream", "model-a", "grpc")
            telemetry.mark_first_output()
            telemetry.finish_triton_call(call, "success")
            telemetry.observe_usage(
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                }
            )
            summary = telemetry.finish(200)

        self.assertIsNotNone(summary)
        self.assertEqual(250.0, summary["queue_ms"])
        self.assertEqual(1000.0, summary["preprocessing_ms"])
        self.assertEqual(3000.0, summary["triton_ms"])
        self.assertEqual(1750.0, summary["postprocessing_ms"])
        self.assertEqual(1000.0, summary["ttft_ms"])
        self.assertEqual(2000.0, summary["decode_ms"])
        self.assertEqual(10.0, summary["output_tokens_per_second"])
        self.assertEqual(100, summary["input_tokens"])
        self.assertEqual(20, summary["output_tokens"])
        self.assertEqual(5, summary["reasoning_tokens"])
        self.assertEqual(1, summary["triton_calls"][0]["count"])

    def test_finish_is_idempotent(self):
        telemetry = GenerationTelemetry("request-1", "/v1/chat/completions", 0.0)
        telemetry.configure(route="chat", model="model-a")
        with patch("gateway.generation_telemetry.time.monotonic", return_value=1.0):
            self.assertIsNotNone(telemetry.finish(200))
            self.assertIsNone(telemetry.finish(200))

    def test_error_type_does_not_include_error_message(self):
        telemetry = GenerationTelemetry("request-1", "/v1/chat/completions", 0.0)
        telemetry.configure(route="chat", model="model-a")
        telemetry.fail(ValueError("secret prompt value"))
        with patch("gateway.generation_telemetry.time.monotonic", return_value=1.0):
            summary = telemetry.finish(500)

        self.assertEqual("ValueError", summary["error_type"])
        self.assertEqual("error", summary["status"])
        self.assertNotIn("secret", str(summary))

    def test_stream_error_overrides_successful_http_status(self):
        telemetry = GenerationTelemetry("request-1", "/v1/chat/completions", 0.0)
        telemetry.configure(route="chat", model="model-a")
        telemetry.fail(RuntimeError("stream failed after response headers"))

        with patch("gateway.generation_telemetry.time.monotonic", return_value=1.0):
            summary = telemetry.finish(200)

        self.assertEqual("error", summary["status"])
        self.assertEqual("RuntimeError", summary["error_type"])


if __name__ == "__main__":
    unittest.main()
