# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any

from .observability import log_event


DEBUG_LOG_PAYLOADS = os.environ.get("DEBUG_LOG_PAYLOADS", "false").lower() in {
    "1",
    "true",
    "yes",
}
GATEWAY_DEBUG = os.environ.get("GATEWAY_DEBUG", "false").lower() in {
    "1",
    "true",
    "yes",
}
DEBUG_PREVIEW_CHARS = max(int(os.environ.get("DEBUG_PREVIEW_CHARS", "2000")), 0)
logger = logging.getLogger("triton-chat-gateway")


def debug_enabled(request: Any) -> bool:
    return GATEWAY_DEBUG or bool(getattr(request, "debug", False))


def conversation_diagnostics(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics = []
    for index, message in enumerate(conversation):
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        item: dict[str, Any] = {
            "index": index,
            "role": message.get("role"),
            "content_type": type(content).__name__,
            "content_chars": len(content) if isinstance(content, str) else 0,
            "content_parts": len(content) if isinstance(content, list) else 0,
            "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        }
        if message.get("tool_call_id"):
            item["tool_call_id"] = str(message["tool_call_id"])
        if message.get("name"):
            item["name"] = str(message["name"])
        if isinstance(tool_calls, list):
            item["tool_call_names"] = [
                str((call.get("function") or {}).get("name"))
                for call in tool_calls
                if isinstance(call, dict) and (call.get("function") or {}).get("name")
            ]
        diagnostics.append(item)
    return diagnostics


def log_chat_request_debug(
    request: Any,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_parser: str | None,
) -> None:
    if not debug_enabled(request):
        return
    log_event(
        logger,
        "chat.debug.request",
        "Chat request debug details",
        model=request.model,
        stream=bool(request.stream),
        tool_choice=request.tool_choice,
        selected_tool_names=[
            str((tool.get("function") or {}).get("name"))
            for tool in tools or []
            if (tool.get("function") or {}).get("name")
        ],
        tool_parser=tool_parser or "fallback",
        messages=conversation_diagnostics(conversation),
    )


def log_chat_prompt_debug(
    request: Any,
    prompt: str,
    *,
    prompt_tokens: int,
    reserved_media_tokens: int,
) -> None:
    if not debug_enabled(request):
        return
    fields: dict[str, Any] = {
        "model": request.model,
        "prompt_chars": len(prompt),
        "prompt_tokens": prompt_tokens,
        "reserved_media_tokens": reserved_media_tokens,
    }
    if DEBUG_LOG_PAYLOADS and DEBUG_PREVIEW_CHARS:
        fields["prompt_preview"] = prompt[:DEBUG_PREVIEW_CHARS]
    log_event(logger, "chat.debug.prompt", "Rendered chat prompt details", **fields)


def log_chat_response_debug(
    request: Any,
    generated_text: str,
    remaining_text: str,
    tool_calls: list[dict[str, Any]],
    finish_reason: str,
) -> None:
    if not debug_enabled(request):
        return
    fields: dict[str, Any] = {
        "model": request.model,
        "generated_chars": len(generated_text),
        "content_chars": len(remaining_text),
        "tool_call_count": len(tool_calls),
        "tool_call_names": [
            str((call.get("function") or {}).get("name"))
            for call in tool_calls
            if (call.get("function") or {}).get("name")
        ],
        "finish_reason": finish_reason,
    }
    if DEBUG_LOG_PAYLOADS and DEBUG_PREVIEW_CHARS:
        fields["generated_preview"] = generated_text[:DEBUG_PREVIEW_CHARS]
    log_event(logger, "chat.debug.response", "Chat response debug details", **fields)
