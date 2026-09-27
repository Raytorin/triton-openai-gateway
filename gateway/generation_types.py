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


async def chat_events(stream: GenerationStream) -> AsyncIterator[str]:
    try:
        async for event in stream.events:
            payload = {"id": event.id, "object": "chat.completion.chunk",
                       "created": event.created, "model": event.model,
                       "choices": [{"index": 0, "delta": event.delta,
                                    "finish_reason": event.finish_reason}]}
            if event.usage is not None:
                payload["usage"] = event.usage
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
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
