# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import os
import time
from typing import Any

from .metrics import (
    GENERATION_OUTPUT_THROUGHPUT,
    GENERATION_REQUESTS,
    GENERATION_STAGE_DURATION,
    GENERATION_TOKENS,
    GENERATION_TTFT,
)


TELEMETRY_ENABLED = os.environ.get("GATEWAY_GENERATION_TELEMETRY", "true").lower() in {
    "1",
    "true",
    "yes",
}


@dataclass
class TritonCallObservation:
    operation: str
    model: str
    transport: str
    started_at: float


@dataclass
class GenerationTelemetry:
    request_id: str
    path: str
    started_at: float
    route: str = "unknown"
    model: str = "unknown"
    backend: str = "unknown"
    transport: str = "unknown"
    admitted_at: float | None = None
    queue_seconds: float = 0.0
    first_triton_started_at: float | None = None
    last_triton_finished_at: float | None = None
    triton_seconds: float = 0.0
    first_output_at: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    error_type: str = ""
    triton_calls: dict[str, dict[str, Any]] = field(default_factory=dict)
    _finished: bool = False

    def configure(
        self,
        *,
        route: str | None = None,
        model: str | None = None,
        backend: str | None = None,
        transport: str | None = None,
    ) -> None:
        if route:
            self.route = route
        if model:
            self.model = model
        if backend:
            self.backend = backend
        if transport:
            self.transport = transport

    def admitted(self, wait_seconds: float) -> None:
        self.queue_seconds = max(float(wait_seconds), 0.0)
        self.admitted_at = time.monotonic()

    def begin_triton_call(
        self,
        operation: str,
        model: str,
        transport: str,
    ) -> TritonCallObservation:
        started_at = time.monotonic()
        if self.first_triton_started_at is None:
            self.first_triton_started_at = started_at
        if self.transport == "unknown":
            self.transport = transport
        return TritonCallObservation(operation, model, transport, started_at)

    def finish_triton_call(
        self,
        observation: TritonCallObservation,
        status: str,
    ) -> None:
        finished_at = time.monotonic()
        duration = max(finished_at - observation.started_at, 0.0)
        self.triton_seconds += duration
        self.last_triton_finished_at = finished_at
        key = f"{observation.operation}:{observation.model}:{observation.transport}"
        aggregate = self.triton_calls.setdefault(
            key,
            {
                "operation": observation.operation,
                "model": observation.model,
                "transport": observation.transport,
                "count": 0,
                "duration_ms": 0.0,
                "errors": 0,
            },
        )
        aggregate["count"] += 1
        aggregate["duration_ms"] = round(aggregate["duration_ms"] + duration * 1000, 3)
        if status != "success":
            aggregate["errors"] += 1

    def mark_first_output(self) -> None:
        if self.first_output_at is None:
            self.first_output_at = time.monotonic()

    def observe_usage(self, usage: dict[str, Any] | None) -> None:
        if not usage:
            return
        self.input_tokens = max(self.input_tokens, _safe_int(usage.get("prompt_tokens")))
        self.output_tokens = max(
            self.output_tokens,
            _safe_int(usage.get("completion_tokens")),
        )
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            self.reasoning_tokens = max(
                self.reasoning_tokens,
                _safe_int(details.get("reasoning_tokens")),
            )

    def fail(self, error: BaseException | str) -> None:
        self.error_type = type(error).__name__ if isinstance(error, BaseException) else str(error)

    def server_timing(self) -> str:
        now = time.monotonic()
        values: list[tuple[str, float]] = []
        if self.queue_seconds > 0:
            values.append(("queue", self.queue_seconds))
        if self.first_triton_started_at is not None:
            admitted_at = self.admitted_at or self.started_at
            values.append(
                ("preprocess", max(self.first_triton_started_at - admitted_at, 0.0))
            )
        if self.triton_seconds > 0:
            values.append(("triton", self.triton_seconds))
        if self.last_triton_finished_at is not None:
            values.append(("postprocess", max(now - self.last_triton_finished_at, 0.0)))
        return ", ".join(f"{name};dur={seconds * 1000:.3f}" for name, seconds in values)

    def finish(self, status_code: int) -> dict[str, Any] | None:
        if self._finished or not TELEMETRY_ENABLED or self.model == "unknown":
            return None
        self._finished = True
        finished_at = time.monotonic()
        total_seconds = max(finished_at - self.started_at, 0.0)
        admitted_at = self.admitted_at or self.started_at
        if self.first_triton_started_at is None:
            preprocessing_seconds = max(total_seconds - self.queue_seconds, 0.0)
        else:
            preprocessing_seconds = max(
                self.first_triton_started_at - admitted_at,
                0.0,
            )
        postprocessing_seconds = max(
            total_seconds
            - self.queue_seconds
            - preprocessing_seconds
            - self.triton_seconds,
            0.0,
        )
        ttft_seconds = None
        decode_seconds = None
        output_throughput = None
        if self.first_output_at is not None and self.first_triton_started_at is not None:
            ttft_seconds = max(
                self.first_output_at - self.first_triton_started_at,
                0.0,
            )
            decode_end = self.last_triton_finished_at or finished_at
            decode_seconds = max(decode_end - self.first_output_at, 0.0)
            if self.output_tokens > 0 and decode_seconds > 0:
                output_throughput = self.output_tokens / decode_seconds

        status = "error" if self.error_type or status_code >= 400 else "success"
        labels = (self.route, self.model)
        GENERATION_REQUESTS.labels(*labels, status).inc()
        for stage, seconds in (
            ("queue", self.queue_seconds),
            ("preprocessing", preprocessing_seconds),
            ("triton", self.triton_seconds),
            ("postprocessing", postprocessing_seconds),
            ("total", total_seconds),
        ):
            GENERATION_STAGE_DURATION.labels(*labels, stage).observe(seconds)
        for kind, count in (
            ("input", self.input_tokens),
            ("output", self.output_tokens),
            ("reasoning", self.reasoning_tokens),
        ):
            if count > 0:
                GENERATION_TOKENS.labels(*labels, kind).inc(count)
        if ttft_seconds is not None:
            GENERATION_TTFT.labels(*labels, self.transport).observe(ttft_seconds)
        if output_throughput is not None:
            GENERATION_OUTPUT_THROUGHPUT.labels(*labels).observe(output_throughput)

        return {
            "route": self.route,
            "backend": self.backend,
            "transport": self.transport,
            "status": status,
            "status_code": status_code,
            "error_type": self.error_type or None,
            "queue_ms": _milliseconds(self.queue_seconds),
            "preprocessing_ms": _milliseconds(preprocessing_seconds),
            "triton_ms": _milliseconds(self.triton_seconds),
            "postprocessing_ms": _milliseconds(postprocessing_seconds),
            "total_ms": _milliseconds(total_seconds),
            "ttft_ms": _milliseconds(ttft_seconds),
            "decode_ms": _milliseconds(decode_seconds),
            "output_tokens_per_second": (
                round(output_throughput, 3) if output_throughput is not None else None
            ),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "triton_calls": list(self.triton_calls.values()),
        }


_telemetry_context: ContextVar[GenerationTelemetry | None] = ContextVar(
    "generation_telemetry",
    default=None,
)


def start_generation_telemetry(
    request_id: str,
    path: str,
    started_at: float,
) -> Token:
    return _telemetry_context.set(
        GenerationTelemetry(request_id=request_id, path=path, started_at=started_at)
    )


def reset_generation_telemetry(token: Token) -> None:
    _telemetry_context.reset(token)


def get_generation_telemetry() -> GenerationTelemetry | None:
    return _telemetry_context.get()


def configure_generation_telemetry(**fields: str) -> None:
    if telemetry := get_generation_telemetry():
        telemetry.configure(**fields)


def observe_generation_usage(usage: dict[str, Any] | None) -> None:
    if telemetry := get_generation_telemetry():
        telemetry.observe_usage(usage)


def mark_first_generation_output() -> None:
    if telemetry := get_generation_telemetry():
        telemetry.mark_first_output()


def _safe_int(value: Any) -> int:
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError):
        return 0


def _milliseconds(value: float | None) -> float | None:
    return round(value * 1000, 3) if value is not None else None
