import asyncio
import json
import logging
from unittest.mock import patch
import unittest

from gateway.observability import (
    CefFormatter,
    JsonFormatter,
    RequestContextMiddleware,
    get_request_model,
    set_request_model,
)


class ObservabilityTests(unittest.TestCase):
    def test_metrics_scrape_is_not_counted_as_gateway_load(self):
        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"metrics"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_message):
            return None

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": [],
        }
        with (
            patch("gateway.observability.HTTP_REQUESTS_INFLIGHT") as inflight,
            patch("gateway.observability.HTTP_REQUESTS") as requests,
            patch("gateway.observability.HTTP_REQUEST_DURATION") as duration,
            patch("gateway.observability.HTTP_REQUEST_BODY_BYTES") as body_size,
            patch("gateway.observability.log_event") as log_event,
        ):
            asyncio.run(RequestContextMiddleware(app)(scope, receive, send))

        inflight.labels.assert_not_called()
        requests.labels.assert_not_called()
        duration.labels.assert_not_called()
        body_size.labels.assert_not_called()
        log_event.assert_not_called()

    def test_middleware_rejects_chunked_body_after_limit(self):
        async def app(scope, receive, send):
            await receive()
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        chunks = iter(
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"456", "more_body": False},
            ]
        )
        sent = []

        async def receive():
            return next(chunks)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
        with (
            patch("gateway.observability.MAX_REQUEST_BODY_BYTES", 5),
            patch("gateway.observability.log_event"),
        ):
            asyncio.run(RequestContextMiddleware(app)(scope, receive, send))

        self.assertEqual(413, sent[0]["status"])

    def test_middleware_rejects_oversized_content_length_before_app(self):
        app_called = False

        async def app(scope, receive, send):
            nonlocal app_called
            app_called = True

        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"content-length", b"101")],
        }
        with (
            patch("gateway.observability.MAX_REQUEST_BODY_BYTES", 100),
            patch("gateway.observability.log_event"),
        ):
            asyncio.run(RequestContextMiddleware(app)(scope, receive, send))

        self.assertFalse(app_called)
        self.assertEqual(413, sent[0]["status"])

    def test_json_formatter_adds_structured_fields(self):
        record = logging.LogRecord(
            "triton-chat-gateway",
            logging.INFO,
            __file__,
            1,
            "completed",
            (),
            None,
        )
        record.event = "test.completed"
        record.request_id = "request-1"
        record.duration_ms = 12.5

        payload = json.loads(JsonFormatter().format(record))

        self.assertEqual("test.completed", payload["event"])
        self.assertEqual("request-1", payload["request_id"])
        self.assertEqual(12.5, payload["duration_ms"])

    def test_cef_formatter_adds_model_severity_and_escapes_header(self):
        record = logging.LogRecord(
            "triton-chat-gateway",
            logging.ERROR,
            __file__,
            1,
            "failed | retry",
            (),
            None,
        )
        record.event = "request.failed"
        record.request_id = "request-1"
        record.model = "Qwen3-6-35B-A3B-FP8"
        record.error = "invalid=value"

        rendered = CefFormatter().format(record)

        self.assertTrue(rendered.startswith("CEF:0|ML Platform AI|"))
        self.assertIn("|request.failed|failed \\| retry|8|", rendered)
        self.assertIn("request_id=request-1", rendered)
        self.assertIn("model=Qwen3-6-35B-A3B-FP8", rendered)
        self.assertIn("error=invalid\\=value", rendered)

    def test_completed_request_keeps_model_and_uses_warning_for_4xx(self):
        completed = {}

        async def app(scope, receive, send):
            set_request_model("model-A")
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b"not found"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(_message):
            return None

        def capture(_logger, event, _message, **fields):
            if event == "http.request.completed":
                completed.update(fields)
                completed["model"] = get_request_model()

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        }
        with patch("gateway.observability.log_event", side_effect=capture):
            asyncio.run(RequestContextMiddleware(app)(scope, receive, send))

        self.assertEqual(logging.WARNING, completed["level"])
        self.assertEqual("model-A", completed["model"])
        self.assertEqual("", get_request_model())

    def test_middleware_preserves_request_and_trace_context(self):
        observed = {}

        async def app(scope, receive, send):
            from gateway.observability import (
                get_request_id,
                get_trace_id,
                get_traceparent,
            )

            observed["request_id"] = get_request_id()
            observed["trace_id"] = get_trace_id()
            observed["traceparent"] = get_traceparent()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        sent = []

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            sent.append(message)

        traceparent = b"00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [(b"x-request-id", b"request-42"), (b"traceparent", traceparent)],
        }

        with patch("gateway.observability.log_event"):
            asyncio.run(RequestContextMiddleware(app)(scope, receive, send))

        self.assertEqual("request-42", observed["request_id"])
        self.assertEqual("0123456789abcdef0123456789abcdef", observed["trace_id"])
        self.assertEqual(traceparent.decode(), observed["traceparent"])
        response_headers = dict(sent[0]["headers"])
        self.assertEqual(b"request-42", response_headers[b"x-request-id"])


if __name__ == "__main__":
    unittest.main()
