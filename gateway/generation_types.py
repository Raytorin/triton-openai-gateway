# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Protocol-neutral generation outcomes and the legacy backend stream bridge."""
from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
import json
import anyio
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from .admission import AdmissionLease
from .context_errors import ContextLengthExceeded


@dataclass
class GenerationResult:
    id: str
    created: int
    model: str
    message: dict[str, Any]
    finish_reason: str
    usage: dict[str, Any]
    extensions: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_chat(cls, payload: dict[str, Any]) -> GenerationResult:
        choice = payload["choices"][0]
        return cls(payload["id"], payload["created"], payload["model"],
                   choice["message"], choice["finish_reason"], payload["usage"],
                   {k: v for k, v in payload.items()
                    if k not in {"id", "created", "object", "model", "choices", "usage", "_generation_headers"}},
                   payload.get("_generation_headers", {}))

    def to_chat(self) -> dict[str, Any]:
        return {"id": self.id, "object": "chat.completion", "created": self.created,
                "model": self.model, "choices": [{"index": 0, "message": self.message,
                "finish_reason": self.finish_reason}], "usage": self.usage, **self.extensions}


@dataclass
class GenerationEvent:
    id: str
    created: int
    model: str
    delta: dict[str, Any]
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None


async def backend_events(source: AsyncIterator[str]) -> AsyncIterator[GenerationEvent]:
    """Adapt existing backend serializers at one boundary; no HTTP loopback."""
    try:
        async for frame in source:
            for line in frame.splitlines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                payload = json.loads(data)
                if "error" in payload:
                    raise HTTPException(502, payload["error"].get("message", "Generation failed"))
                choices = payload.get("choices") or []
                choice = choices[0] if choices else {}
                yield GenerationEvent(payload["id"], payload["created"], payload["model"],
                                      choice.get("delta", {}), choice.get("finish_reason"),
                                      payload.get("usage"))
    finally:
        if close := getattr(source, "aclose", None):
            await close()


def stream_error_payload(exc: Exception) -> dict[str, Any]:
    """Build a public stream error without forwarding backend diagnostics."""
    if isinstance(exc, ContextLengthExceeded):
        return exc.error
    client_error = isinstance(exc, HTTPException) and exc.status_code < 500
    kind = "invalid_request_error" if client_error else "server_error"
    return {"message": "Generation request failed" if client_error else "Generation failed",
            "type": kind, "param": None, "code": kind}


@dataclass
class GenerationStream:
    events: AsyncIterator[GenerationEvent]
    headers: dict[str, str]
    lease: AdmissionLease | None = None
    closed: bool = False

    async def aclose(self) -> None:
        if self.closed:
            return
        self.closed = True
        with anyio.CancelScope(shield=True):
            try:
                if close := getattr(self.events, "aclose", None):
                    await close()
            finally:
                if self.lease is not None:
                    await self.lease.release()


def generation_stream(source: AsyncIterator[str], *, media_type: str,
                      headers: dict[str, str]) -> GenerationStream:
    return GenerationStream(backend_events(source), headers)


async def chat_events(stream: GenerationStream, *, include_usage: bool | None = None) -> AsyncIterator[str]:
    def chunk(event: GenerationEvent) -> dict[str, Any]:
        payload = {"id": event.id, "object": "chat.completion.chunk",
                   "created": event.created, "model": event.model,
                   "choices": [{"index": 0, "delta": event.delta,
                                "finish_reason": event.finish_reason}]}
        if include_usage is True:
            payload["usage"] = None
        elif include_usage is None and event.usage is not None:
            payload["usage"] = event.usage
        return payload

    terminal = None
    try:
        async for event in stream.events:
            if terminal is not None:
                raise HTTPException(502, "Backend stream continued after its completion status")
            if event.finish_reason is not None:
                # Wait for clean exhaustion so a late backend error cannot look successful.
                terminal = event
                continue
            yield f"data: {json.dumps(chunk(event), ensure_ascii=False)}\n\n"
        if terminal is None:
            raise HTTPException(502, "Backend stream ended without a completion status")
        payload = chunk(terminal)
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        if include_usage is True and terminal.usage is not None:
            usage = {**payload, "choices": [], "usage": terminal.usage}
            yield f"data: {json.dumps(usage, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        await stream.aclose()


class ManagedStreamingResponse(StreamingResponse):
    """Release an admitted stream even if ASGI fails before iteration starts."""
    def __init__(self, *args, close: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self._close = close

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._close()


async def until_disconnected(operation, request=None):
    """Cancel preparation/inference even before response headers can be sent."""
    if request is None:
        return await operation
    import asyncio
    async def watch():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return
    task = asyncio.create_task(operation)
    watcher = asyncio.create_task(watch())
    try:
        try:
            done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return await task
            raise HTTPException(499, "Client disconnected")
        finally:
            for pending in (task, watcher):
                if not pending.done():
                    pending.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, watcher, return_exceptions=True)
    except BaseException:
        # Cancellation can win as generation returns or while the disconnect
        # watcher is closing. The caller never received ownership of the stream.
        with anyio.CancelScope(shield=True):
            if task.done() and not task.cancelled() and task.exception() is None:
                abandoned = task.result()
                if isinstance(abandoned, GenerationStream):
                    await abandoned.aclose()
        raise
