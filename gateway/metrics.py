# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response


HTTP_REQUESTS = Counter(
    "triton_gateway_http_requests_total",
    "HTTP requests handled by the gateway.",
    ("method", "path", "status"),
)
HTTP_REQUEST_DURATION = Histogram(
    "triton_gateway_http_request_duration_seconds",
    "End-to-end gateway HTTP request duration.",
    ("method", "path"),
)
HTTP_REQUESTS_INFLIGHT = Gauge(
    "triton_gateway_http_requests_inflight",
    "HTTP requests currently handled by the gateway.",
    ("method", "path"),
)
HTTP_REQUEST_BODY_BYTES = Histogram(
    "triton_gateway_http_request_body_bytes",
    "Observed HTTP request body size.",
    ("method", "path"),
    buckets=(1024, 16_384, 262_144, 1_048_576, 8_388_608, 33_554_432, 134_217_728, 536_870_912),
)

ADMISSION_INFLIGHT = Gauge(
    "triton_gateway_admission_inflight",
    "Requests holding an admission slot.",
    ("scope", "route", "model"),
)
ADMISSION_QUEUED = Gauge(
    "triton_gateway_admission_queued",
    "Requests waiting for an admission slot.",
    ("scope", "route", "model"),
)
ADMISSION_WAIT = Histogram(
    "triton_gateway_admission_wait_seconds",
    "Time spent waiting for an admission slot.",
    ("scope", "route", "model"),
)
ADMISSION_REJECTED = Counter(
    "triton_gateway_admission_rejected_total",
    "Requests rejected before inference.",
    ("scope", "route", "model", "reason"),
)

TRITON_REQUESTS = Counter(
    "triton_gateway_triton_requests_total",
    "Calls made from the gateway to Triton.",
    ("operation", "model", "transport", "status"),
)
TRITON_REQUEST_DURATION = Histogram(
    "triton_gateway_triton_request_duration_seconds",
    "Duration of calls from the gateway to Triton.",
    ("operation", "model", "transport"),
)
TRITON_REQUESTS_INFLIGHT = Gauge(
    "triton_gateway_triton_requests_inflight",
    "Calls from the gateway currently executing in Triton.",
    ("operation", "model", "transport"),
)
TRITON_STREAMS_ACTIVE = Gauge(
    "triton_gateway_triton_streams_active",
    "Active streaming calls from the gateway to Triton.",
    ("model",),
)
TRITON_STREAM_CANCELLED = Counter(
    "triton_gateway_triton_stream_cancelled_total",
    "Triton streams cancelled by a disconnected or cancelled caller.",
    ("model",),
)

TOKENIZER_LOADS = Counter(
    "triton_gateway_tokenizer_loads_total",
    "Tokenizer load attempts.",
    ("model", "status"),
)
TOKENIZER_LOAD_DURATION = Histogram(
    "triton_gateway_tokenizer_load_duration_seconds",
    "Tokenizer load duration.",
    ("model",),
)
EMBEDDING_BATCH_SIZE = Histogram(
    "triton_gateway_embedding_batch_size",
    "Number of embedding inputs in an OpenAI request.",
    ("model",),
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
MEDIA_PROCESSING = Histogram(
    "triton_gateway_media_processing_duration_seconds",
    "Gateway-side media processing duration.",
    ("model", "kind", "status"),
)
PDF_EMBEDDING_CACHE = Counter(
    "triton_gateway_pdf_embedding_cache_total",
    "In-memory PDF embedding cache lookups.",
    ("model", "result"),
)
RERANK_STRATEGY_SELECTIONS = Counter(
    "triton_gateway_rerank_strategy_selections_total",
    "Resolved rerank selection strategies.",
    ("model", "strategy", "method", "source"),
)
CONTEXT_COMPRESSION_REQUESTS = Counter(
    "triton_gateway_context_compression_total",
    "Context preparation results.",
    ("model", "mode", "action"),
)
CONTEXT_COMPRESSION_DURATION = Histogram(
    "triton_gateway_context_compression_duration_seconds",
    "Duration of context fitting, truncation or summarization.",
    ("model", "mode", "action"),
)
CONTEXT_COMPRESSION_MESSAGES = Histogram(
    "triton_gateway_context_compression_messages",
    "Number of historical messages removed or summarized.",
    ("model", "action"),
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
CONTEXT_SUMMARY_CACHE = Counter(
    "triton_gateway_context_summary_cache_total",
    "Process-local rolling summary cache lookups.",
    ("model", "result"),
)
CONTEXT_SUMMARY_CALLS = Counter(
    "triton_gateway_context_summary_calls_total",
    "Internal context summarization model calls.",
    ("model", "summary_model", "status"),
)
CONTEXT_SUMMARY_TOKENS = Counter(
    "triton_gateway_context_summary_tokens_total",
    "Tokens processed by internal context summarization calls.",
    ("model", "summary_model", "direction"),
)
REASONING_REQUESTS = Counter(
    "triton_gateway_reasoning_requests_total",
    "Chat responses processed by the configured reasoning policy.",
    ("model", "mode", "parser", "result"),
)
REASONING_TOKENS = Counter(
    "triton_gateway_reasoning_tokens_total",
    "Estimated reasoning and final-content tokens returned by chat models.",
    ("model", "mode", "kind"),
)
GENERATION_REQUESTS = Counter(
    "triton_gateway_generation_requests_total",
    "Completed inference requests observed by the gateway telemetry lifecycle.",
    ("route", "model", "status"),
)
GENERATION_STAGE_DURATION = Histogram(
    "triton_gateway_generation_stage_duration_seconds",
    "Request duration split into non-overlapping gateway and Triton stages.",
    ("route", "model", "stage"),
    buckets=(
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    ),
)
GENERATION_TOKENS = Counter(
    "triton_gateway_generation_tokens_total",
    "Input, output and reasoning tokens reported for inference requests.",
    ("route", "model", "kind"),
)
GENERATION_TTFT = Histogram(
    "triton_gateway_generation_time_to_first_token_seconds",
    "Gateway-observed time from the first Triton call to the first streamed output.",
    ("route", "model", "transport"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
GENERATION_OUTPUT_THROUGHPUT = Histogram(
    "triton_gateway_generation_output_tokens_per_second",
    "Output token throughput derived from gateway-observed streaming decode time.",
    ("route", "model"),
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000),
)


_KNOWN_PATHS = {
    "/health",
    "/ready",
    "/metrics",
    "/v1/models",
    "/v1/embeddings",
    "/rerank",
    "/v1/rerank",
    "/v2/rerank",
    "/v1/chat/completions",
}


def metric_path(path: str) -> str:
    return path if path in _KNOWN_PATHS else "other"


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@contextmanager
def triton_call(operation: str, model: str, transport: str) -> Iterator[None]:
    labels = (operation, model, transport)
    started_at = time.monotonic()
    # Keep metric definitions independent from request-local context state.
    from .generation_telemetry import get_generation_telemetry

    telemetry = get_generation_telemetry()
    observation = (
        telemetry.begin_triton_call(operation, model, transport)
        if telemetry is not None
        else None
    )
    TRITON_REQUESTS_INFLIGHT.labels(*labels).inc()
    status = "success"
    try:
        yield
    except (asyncio.CancelledError, GeneratorExit):
        status = "cancelled"
        raise
    except BaseException as exc:
        status = "error"
        if telemetry is not None:
            telemetry.fail(exc)
        raise
    finally:
        if telemetry is not None and observation is not None:
            telemetry.finish_triton_call(observation, status)
        TRITON_REQUESTS_INFLIGHT.labels(*labels).dec()
        TRITON_REQUEST_DURATION.labels(*labels).observe(time.monotonic() - started_at)
        TRITON_REQUESTS.labels(operation, model, transport, status).inc()
