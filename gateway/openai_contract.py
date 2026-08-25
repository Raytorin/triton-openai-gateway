# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from .schemas import ResponseFormat


def structured_outputs_parameter(
    response_format: ResponseFormat | None,
) -> str | None:
    """Translate OpenAI response_format into Triton vLLM parameters."""
    if response_format is None or response_format.type == "text":
        return None

    if response_format.type == "json_object":
        structured_outputs: dict[str, Any] = {"json_object": True}
    else:
        if response_format.json_schema is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "response_format.json_schema is required when type is "
                    "json_schema"
                ),
            )
        structured_outputs = {
            "json": response_format.json_schema.json_schema,
        }

    # Triton 26.07 parses this nested object from a serialized JSON value.
    return json.dumps(structured_outputs, ensure_ascii=False, separators=(",", ":"))


def normalize_system_messages(
    conversation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge system messages into one leading message for strict templates."""
    system_messages: list[tuple[int, dict[str, Any]]] = [
        (index, message)
        for index, message in enumerate(conversation)
        if message.get("role") == "system"
    ]
    if not system_messages:
        return conversation, 0, 0

    if len(system_messages) == 1 and system_messages[0][0] == 0:
        return conversation, 1, 0

    contents = [
        _system_content_as_text(message.get("content"))
        for _, message in system_messages
    ]
    merged = "\n\n".join(content for content in contents if content).strip()
    remaining = [
        dict(message)
        for message in conversation
        if message.get("role") != "system"
    ]
    normalized = [{"role": "system", "content": merged}, *remaining]
    moved = sum(index != 0 for index, _ in system_messages)
    return normalized, len(system_messages), moved


def _system_content_as_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            raise HTTPException(
                status_code=400,
                detail="System messages may contain only text content",
            )
        return "\n".join(parts).strip()

    raise HTTPException(
        status_code=400,
        detail="System messages may contain only text content",
    )
