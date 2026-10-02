# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Responses SSE lifecycle, independent of the Chat wire protocol."""
from __future__ import annotations

from collections.abc import AsyncIterator
import json
import time

from fastapi import HTTPException

from .generation_telemetry import get_generation_telemetry
from .generation_types import GenerationEvent, GenerationStream
from .responses import ResponsesRequest, new_id, response_object, text_part
from .settings import logger


async def response_events(request: ResponsesRequest, stream: GenerationStream,
                          first: GenerationEvent | None) -> AsyncIterator[str]:
    response_id, created = new_id("resp"), int(time.time())
    output: list[dict] = []
    tool_indices: dict[int, int] = {}
    message_index: int | None = None
    sequence = 0
    usage = None
    finish = None

    def emit(kind, **fields):
        nonlocal sequence
        payload = {"type": kind, "sequence_number": sequence, **fields}
        sequence += 1
        return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def snapshot(status, error=None):
        return response_object(request, response_id=response_id, created=created,
            status=status, output=output, usage=usage, headers=stream.headers, error=error)

    async def events():
        if first is not None:
            yield first
        async for event in stream.events:
            yield event

    try:
        yield emit("response.created", response=snapshot("in_progress"))
        yield emit("response.in_progress", response=snapshot("in_progress"))
        async for event in events():
            if event.usage is not None:
                usage = event.usage
            if event.finish_reason is not None:
                finish = event.finish_reason
            if text := event.delta.get("content"):
                if message_index is None:
                    message_index = len(output)
                    item = {"id": new_id("msg"), "type": "message", "role": "assistant",
                            "status": "in_progress", "content": []}
                    output.append(item)
                    yield emit("response.output_item.added", output_index=message_index, item=item)
                    item["content"].append(text_part(""))
                    yield emit("response.content_part.added", item_id=item["id"], output_index=message_index,
                               content_index=0, part=item["content"][0])
                item = output[message_index]
                item["content"][0]["text"] += text
                yield emit("response.output_text.delta", item_id=item["id"], output_index=message_index,
                           content_index=0, delta=text, logprobs=[])
            for call in event.delta.get("tool_calls") or []:
                backend_index = call.get("index", 0)
                function = call.get("function", {})
                if backend_index not in tool_indices:
                    # Backend parsers provide name/call ID before argument deltas.
                    if not call.get("id") or not function.get("name"):
                        raise HTTPException(502, "Backend tool stream is missing the function name or call ID")
                    index = len(output)
                    tool_indices[backend_index] = index
                    item = {"id": new_id("fc"), "type": "function_call", "status": "in_progress",
                            "call_id": call["id"], "name": function["name"], "arguments": ""}
                    output.append(item)
                    yield emit("response.output_item.added", output_index=index, item=item)
                index = tool_indices[backend_index]
                item = output[index]
                if delta := function.get("arguments"):
                    item["arguments"] += delta
                    yield emit("response.function_call_arguments.delta", item_id=item["id"], output_index=index, delta=delta)
        if finish is None:
            raise HTTPException(502, "Backend stream ended without a completion status")
        status = "incomplete" if finish == "length" else "completed"
        for index, item in enumerate(output):
            item["status"] = status
            if item["type"] == "message":
                part = item["content"][0]
                yield emit("response.output_text.done", item_id=item["id"], output_index=index,
                           content_index=0, text=part["text"], logprobs=[])
                yield emit("response.content_part.done", item_id=item["id"], output_index=index,
                           content_index=0, part=part)
            else:
                yield emit("response.function_call_arguments.done", item_id=item["id"], output_index=index,
                           name=item["name"], arguments=item["arguments"])
            yield emit("response.output_item.done", output_index=index, item=item)
        yield emit(f"response.{status}", response=snapshot(status))
    except Exception as exc:
        if telemetry := get_generation_telemetry():
            telemetry.fail(exc)
        logger.exception("Responses stream failed")
        message = str(exc.detail) if isinstance(exc, HTTPException) else "Generation failed"
        error = {"code": "server_error", "message": message}
        for item in output:
            item["status"] = "incomplete"
        yield emit("error", **error, param=None)
        yield emit("response.failed", response=snapshot("failed", error=error))
    finally:
        await stream.aclose()
