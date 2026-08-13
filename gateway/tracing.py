# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any, Mapping

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import SpanKind, Status, StatusCode

from . import __version__


logger = logging.getLogger("triton-chat-gateway")
OTEL_ENABLED = os.environ.get("GATEWAY_OTEL_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
}
OTEL_ENDPOINT = os.environ.get(
    "GATEWAY_OTEL_ENDPOINT",
    os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""),
).strip()
OTEL_SERVICE_NAME = os.environ.get(
    "GATEWAY_OTEL_SERVICE_NAME",
    "triton-openai-gateway",
)
DEFAULT_OTEL_SERVICE_VERSION = __version__
OTEL_SERVICE_VERSION = os.environ.get(
    "GATEWAY_OTEL_SERVICE_VERSION", DEFAULT_OTEL_SERVICE_VERSION
)
OTEL_SAMPLE_RATIO = min(
    max(float(os.environ.get("GATEWAY_OTEL_SAMPLE_RATIO", "0.05")), 0.0),
    1.0,
)


@dataclass
class RequestTrace:
    span: Any = None
    context_token: Any = None
    traceparent: str = ""
    trace_id: str = ""
    managed: bool = False
    _finished: bool = False

    def finish(
        self,
        *,
        status_code: int,
        model: str,
        telemetry: Mapping[str, Any] | None,
    ) -> None:
        if self._finished or self.span is None:
            return
        self._finished = True
        self.span.set_attribute("http.response.status_code", status_code)
        if model:
            self.span.set_attribute("gen_ai.request.model", model)
        if telemetry:
            _set_telemetry_attributes(self.span, telemetry)
        if status_code >= 400 or (
            telemetry is not None and telemetry.get("status") == "error"
        ):
            self.span.set_status(Status(StatusCode.ERROR))
        else:
            self.span.set_status(Status(StatusCode.OK))
        self.span.end()
        if self.context_token is not None:
            otel_context.detach(self.context_token)
            self.context_token = None


_provider: TracerProvider | None = None
_tracer: Any = None
_configuration_attempted = False


def configure_tracing() -> None:
    global _configuration_attempted, _provider, _tracer
    if not OTEL_ENABLED or _tracer is not None or _configuration_attempted:
        return
    _configuration_attempted = True
    if not OTEL_ENDPOINT:
        logger.warning(
            "GATEWAY_OTEL_ENABLED=true but no GATEWAY_OTEL_ENDPOINT was configured; "
            "gateway span export is disabled"
        )
        return
    provider = None
    try:
        resource = Resource.create(
            {
                "service.name": OTEL_SERVICE_NAME,
                "service.version": OTEL_SERVICE_VERSION,
                "service.namespace": "triton",
            }
        )
        provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(OTEL_SAMPLE_RATIO)),
        )
        exporter = OTLPSpanExporter(endpoint=OTEL_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        tracer = provider.get_tracer("triton-openai-gateway", OTEL_SERVICE_VERSION)
    except Exception:
        logger.exception(
            "Unable to initialize OpenTelemetry export; gateway tracing is disabled"
        )
        if provider is not None:
            provider.shutdown()
        return
    _provider = provider
    _tracer = tracer


def start_request_trace(
    *,
    method: str,
    path: str,
    headers: Mapping[str, str],
    request_id: str,
) -> RequestTrace:
    if not OTEL_ENABLED or path in {"/health", "/ready", "/metrics"}:
        return RequestTrace()
    configure_tracing()
    if _tracer is None:
        return RequestTrace()
    parent_context = propagate.extract(carrier=dict(headers))
    span = _tracer.start_span(
        f"{method} {path}",
        context=parent_context,
        kind=SpanKind.SERVER,
        attributes={
            "http.request.method": method,
            "url.path": path,
            "triton.gateway.request_id": request_id,
        },
    )
    span_context = trace.set_span_in_context(span)
    token = otel_context.attach(span_context)
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    context = span.get_span_context()
    sampled = bool(context.trace_flags.sampled)
    return RequestTrace(
        span=span,
        context_token=token,
        traceparent=carrier.get("traceparent", "") if sampled else "",
        trace_id=f"{context.trace_id:032x}" if context.is_valid and sampled else "",
        managed=True,
    )


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.shutdown()


def _set_telemetry_attributes(span: Any, telemetry: Mapping[str, Any]) -> None:
    for key in (
        "route",
        "backend",
        "transport",
        "status",
        "error_type",
        "queue_ms",
        "preprocessing_ms",
        "triton_ms",
        "postprocessing_ms",
        "total_ms",
        "ttft_ms",
        "decode_ms",
        "output_tokens_per_second",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
    ):
        value = telemetry.get(key)
        if value is not None:
            span.set_attribute(f"triton.gateway.{key}", value)
