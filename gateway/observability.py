# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import os
import re
import time
import uuid
from typing import Any

from fastapi import HTTPException

from .metrics import (
    HTTP_REQUEST_BODY_BYTES,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS,
    HTTP_REQUESTS_INFLIGHT,
    metric_path,
)


REQUEST_ID_HEADER = b"x-request-id"
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")
CEF_VENDOR = "ML Platform AI"
CEF_PRODUCT = "Triton OpenAI Gateway"
CEF_VERSION = "0.1.0"
MAX_REQUEST_BODY_BYTES = int(
    os.environ.get("GATEWAY_MAX_REQUEST_BODY_BYTES", str(256 * 1024 * 1024))
)
request_id_context: ContextVar[str] = ContextVar("request_id", default="")
trace_id_context: ContextVar[str] = ContextVar("trace_id", default="")
traceparent_context: ContextVar[str] = ContextVar("traceparent", default="")
model_context: ContextVar[str] = ContextVar("model", default="")
_STANDARD_LOG_RECORD_FIELDS = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None) or get_request_id()
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        model = getattr(record, "model", None) or get_request_model()
        if request_id:
            payload["request_id"] = request_id
        if trace_id:
            payload["trace_id"] = trace_id
        if model:
            payload["model"] = model

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key in payload or value is None:
                continue
            payload[key] = _json_safe(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class ContextTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or get_request_id() or "-"
        event = getattr(record, "event", None) or "log"
        model = getattr(record, "model", None) or get_request_model() or "-"
        return (
            f"{datetime.now(timezone.utc).isoformat(timespec='milliseconds')} "
            f"{record.levelname} {record.name} request_id={request_id} "
            f"model={model} event={event} {record.getMessage()}"
        )


class CefFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = str(getattr(record, "event", None) or "log")
        message = record.getMessage()
        extension: dict[str, Any] = {
            "rt": int(record.created * 1000),
            "level": record.levelname,
            "logger": record.name,
        }
        request_id = getattr(record, "request_id", None) or get_request_id()
        trace_id = getattr(record, "trace_id", None) or get_trace_id()
        model = getattr(record, "model", None) or get_request_model()
        if request_id:
            extension["request_id"] = request_id
        if trace_id:
            extension["trace_id"] = trace_id
        if model:
            extension["model"] = model

        for key, value in record.__dict__.items():
            if (
                key in _STANDARD_LOG_RECORD_FIELDS
                or key in {"event", "request_id", "trace_id", "model"}
                or value is None
            ):
                continue
            extension[key] = _json_safe(value)
        if record.exc_info:
            extension["exception"] = self.formatException(record.exc_info)

        cef_extension = " ".join(
            f"{_cef_key(key)}={_cef_extension_value(value)}"
            for key, value in extension.items()
        )
        return (
            f"CEF:0|{_cef_header(CEF_VENDOR)}|{_cef_header(CEF_PRODUCT)}|"
            f"{_cef_header(CEF_VERSION)}|{_cef_header(event)}|"
            f"{_cef_header(message)}|{_cef_severity(record.levelno)}|"
            f"{cef_extension}"
        )


def configure_logging() -> None:
    logger = logging.getLogger("triton-chat-gateway")
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    logger.handlers.clear()

    handler = logging.StreamHandler()
    log_format = os.environ.get("LOG_FORMAT", "json").lower()
    formatter = {
        "cef": CefFormatter,
        "json": JsonFormatter,
        "text": ContextTextFormatter,
    }.get(log_format, CefFormatter)
    handler.setFormatter(formatter())
    logger.addHandler(handler)
    logger.propagate = False


def get_request_id() -> str:
    return request_id_context.get()


def get_trace_id() -> str:
    return trace_id_context.get()


def get_traceparent() -> str:
    return traceparent_context.get()


def get_request_model() -> str:
    return model_context.get()


def set_request_model(model: str) -> None:
    model_context.set(str(model or ""))


def log_event(
    logger: logging.Logger,
    event: str,
    message: str,
    *,
    level: int = logging.INFO,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    extra = {"event": event, **fields}
    if request_id := get_request_id():
        extra["request_id"] = request_id
    if trace_id := get_trace_id():
        extra["trace_id"] = trace_id
    if model := get_request_model():
        extra.setdefault("model", model)
    logger.log(
        level,
        message,
        extra=extra,
        exc_info=exc_info,
    )


class RequestContextMiddleware:
    def __init__(self, app: Any):
        self.app = app
        self.logger = logging.getLogger("triton-chat-gateway")

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        incoming_id = headers.get(REQUEST_ID_HEADER, b"").decode(
            "utf-8", errors="ignore"
        )
        request_id = incoming_id if REQUEST_ID_RE.fullmatch(incoming_id) else uuid.uuid4().hex
        traceparent = _valid_traceparent(headers.get(b"traceparent", b""))
        trace_id = traceparent.split("-")[1] if traceparent else ""
        request_token = request_id_context.set(request_id)
        trace_token = trace_id_context.set(trace_id)
        traceparent_token = traceparent_context.set(traceparent)
        model_token = model_context.set("")
        started_at = time.monotonic()
        status_code = 500
        response_started = False
        response_completed = False
        metrics_completed = False
        body_bytes = 0
        method = str(scope.get("method") or "UNKNOWN")
        path = metric_path(str(scope.get("path") or ""))
        record_http_metrics = path != "/metrics"
        if record_http_metrics:
            HTTP_REQUESTS_INFLIGHT.labels(method, path).inc()

        def complete_request() -> None:
            nonlocal metrics_completed
            if metrics_completed:
                return
            metrics_completed = True
            duration = time.monotonic() - started_at
            if record_http_metrics:
                HTTP_REQUESTS.labels(method, path, str(status_code)).inc()
                HTTP_REQUEST_DURATION.labels(method, path).observe(duration)
                HTTP_REQUEST_BODY_BYTES.labels(method, path).observe(body_bytes)
                HTTP_REQUESTS_INFLIGHT.labels(method, path).dec()
                self._log_completed(scope, status_code, started_at, body_bytes)

        if record_http_metrics:
            log_event(
                self.logger,
                "http.request.started",
                "HTTP request started",
                level=logging.DEBUG,
                method=scope.get("method"),
                path=scope.get("path"),
            )

        content_length = _content_length(headers.get(b"content-length", b""))
        if content_length is not None and content_length > MAX_REQUEST_BODY_BYTES:
            status_code = 413
            body_bytes = content_length
            body = json.dumps(
                {
                    "detail": (
                        f"Request body is {content_length} bytes; maximum is "
                        f"{MAX_REQUEST_BODY_BYTES} bytes"
                    )
                }
            ).encode("utf-8")
            await send(
                {
                    "type": "http.response.start",
                    "status": status_code,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (REQUEST_ID_HEADER, request_id.encode("ascii")),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})
            complete_request()
            traceparent_context.reset(traceparent_token)
            trace_id_context.reset(trace_token)
            model_context.reset(model_token)
            request_id_context.reset(request_token)
            return

        async def receive_with_limit() -> dict[str, Any]:
            nonlocal body_bytes
            message = await receive()
            if message.get("type") == "http.request":
                body_bytes += len(message.get("body", b""))
                if body_bytes > MAX_REQUEST_BODY_BYTES:
                    raise RequestBodyTooLarge(body_bytes)
            return message

        async def send_with_context(message: dict[str, Any]) -> None:
            nonlocal status_code, response_started, response_completed
            if message["type"] == "http.response.start":
                response_started = True
                status_code = int(message["status"])
                response_headers = list(message.get("headers", []))
                if not any(key.lower() == REQUEST_ID_HEADER for key, _ in response_headers):
                    response_headers.append((REQUEST_ID_HEADER, request_id.encode("ascii")))
                message["headers"] = response_headers
            elif (
                message["type"] == "http.response.body"
                and not message.get("more_body", False)
            ):
                response_completed = True
                complete_request()
            await send(message)

        try:
            await self.app(scope, receive_with_limit, send_with_context)
        except RequestBodyTooLarge as exc:
            status_code = 413
            if not response_started:
                body = json.dumps({"detail": exc.detail}).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (REQUEST_ID_HEADER, request_id.encode("ascii")),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                response_completed = True
            complete_request()
        except BaseException as exc:
            if record_http_metrics and not response_completed:
                log_event(
                    self.logger,
                    "http.request.failed",
                    "HTTP request failed",
                    level=logging.ERROR,
                    method=scope.get("method"),
                    path=scope.get("path"),
                    status_code=status_code,
                    duration_ms=round((time.monotonic() - started_at) * 1000, 3),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
            raise
        finally:
            complete_request()
            traceparent_context.reset(traceparent_token)
            trace_id_context.reset(trace_token)
            model_context.reset(model_token)
            request_id_context.reset(request_token)

    def _log_completed(
        self,
        scope: dict[str, Any],
        status_code: int,
        started_at: float,
        body_bytes: int,
    ) -> None:
        level = logging.ERROR if status_code >= 500 else (
            logging.WARNING if status_code >= 400 else logging.INFO
        )
        log_event(
            self.logger,
            "http.request.completed",
            "HTTP request completed",
            level=level,
            method=scope.get("method"),
            path=scope.get("path"),
            status_code=status_code,
            duration_ms=round((time.monotonic() - started_at) * 1000, 3),
            request_body_bytes=body_bytes,
        )


class RequestBodyTooLarge(HTTPException):
    def __init__(self, body_bytes: int):
        super().__init__(
            status_code=413,
            detail=(
                f"Request body exceeds {MAX_REQUEST_BODY_BYTES} bytes "
                f"(received at least {body_bytes} bytes)"
            ),
        )


def _valid_traceparent(value: bytes) -> str:
    try:
        traceparent = value.decode("ascii")
    except UnicodeDecodeError:
        return ""
    parts = traceparent.split("-")
    if (
        len(parts) != 4
        or len(parts[0]) != 2
        or len(parts[1]) != 32
        or len(parts[2]) != 16
        or len(parts[3]) != 2
    ):
        return ""
    try:
        int("".join(parts), 16)
    except ValueError:
        return ""
    if parts[1] == "0" * 32 or parts[2] == "0" * 16:
        return ""
    return traceparent


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    return str(value)


def _cef_header(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def _cef_key(value: Any) -> str:
    key = re.sub(r"[^A-Za-z0-9_.]", "_", str(value))
    return key or "field"


def _cef_extension_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    return (
        rendered.replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _cef_severity(level: int) -> int:
    if level >= logging.CRITICAL:
        return 10
    if level >= logging.ERROR:
        return 8
    if level >= logging.WARNING:
        return 6
    if level >= logging.INFO:
        return 3
    return 1


def _content_length(value: bytes) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
