# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from unittest.mock import Mock, patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON
from opentelemetry.trace import StatusCode

from gateway.tracing import RequestTrace, configure_tracing, start_request_trace


class TracingTests(unittest.TestCase):
    def test_sampled_span_generates_w3c_context(self):
        provider = TracerProvider(sampler=ALWAYS_ON)
        tracer = provider.get_tracer("test")
        with (
            patch("gateway.tracing.OTEL_ENABLED", True),
            patch("gateway.tracing._tracer", tracer),
        ):
            request_trace = start_request_trace(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                request_id="request-1",
            )

        self.assertTrue(request_trace.managed)
        self.assertRegex(
            request_trace.traceparent,
            re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$"),
        )
        self.assertEqual(1, int(request_trace.traceparent[-2:], 16) & 1)
        self.assertEqual(32, len(request_trace.trace_id))
        request_trace.finish(status_code=200, model="model-a", telemetry=None)
        provider.shutdown()

    def test_unsampled_span_is_not_propagated_to_triton(self):
        provider = TracerProvider(sampler=ALWAYS_OFF)
        tracer = provider.get_tracer("test")
        with (
            patch("gateway.tracing.OTEL_ENABLED", True),
            patch("gateway.tracing._tracer", tracer),
        ):
            request_trace = start_request_trace(
                method="POST",
                path="/v1/chat/completions",
                headers={},
                request_id="request-2",
            )

        self.assertTrue(request_trace.managed)
        self.assertEqual("", request_trace.traceparent)
        self.assertEqual("", request_trace.trace_id)
        request_trace.finish(status_code=200, model="model-a", telemetry=None)
        provider.shutdown()

    def test_client_error_marks_span_as_error(self):
        span = Mock()
        request_trace = RequestTrace(span=span)

        request_trace.finish(status_code=400, model="model-a", telemetry=None)

        status = span.set_status.call_args.args[0]
        self.assertEqual(StatusCode.ERROR, status.status_code)
        span.end.assert_called_once_with()

    def test_stream_lifecycle_error_marks_successful_http_span_as_error(self):
        span = Mock()
        request_trace = RequestTrace(span=span)

        request_trace.finish(
            status_code=200,
            model="model-a",
            telemetry={"status": "error", "error_type": "RuntimeError"},
        )

        status = span.set_status.call_args.args[0]
        self.assertEqual(StatusCode.ERROR, status.status_code)

    def test_exporter_initialization_failure_does_not_break_requests(self):
        with (
            patch("gateway.tracing.OTEL_ENABLED", True),
            patch("gateway.tracing.OTEL_ENDPOINT", "http://collector:4318/v1/traces"),
            patch("gateway.tracing._configuration_attempted", False),
            patch("gateway.tracing._tracer", None),
            patch("gateway.tracing.TracerProvider", side_effect=RuntimeError("boom")),
            patch("gateway.tracing.logger.exception") as log_exception,
        ):
            configure_tracing()
            configure_tracing()

        log_exception.assert_called_once_with(
            "Unable to initialize OpenTelemetry export; gateway tracing is disabled"
        )


if __name__ == "__main__":
    unittest.main()
