# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .metrics import REASONING_REQUESTS, REASONING_TOKENS


_MODES = {"disabled", "hidden", "separate"}
_PARSERS = {"auto", "none", "qwen3", "think_tags"}
_RESPONSE_FIELDS = {"reasoning", "reasoning_content"}
_OPEN_TAGS = ("<think>", "<thinking>")
_CLOSE_TAGS = ("</think>", "</thinking>")
_TOOL_MARKERS = ("<tool_call>", "<function=")


@dataclass(frozen=True)
class ReasoningSettings:
    configured_mode: str
    mode: str
    parser: str
    response_field: str
    supported: bool

    @property
    def enable_thinking(self) -> bool:
        return self.mode != "disabled"

    @property
    def expose_reasoning(self) -> bool:
        return self.mode == "separate"

    def response_status(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "supported": self.supported,
            "parser": self.parser if self.supported else None,
            "response_field": (
                self.response_field if self.expose_reasoning else None
            ),
        }

    def response_headers(self) -> dict[str, str]:
        return {
            "X-Reasoning-Mode": self.mode,
            "X-Reasoning-Supported": str(self.supported).lower(),
            "X-Reasoning-Parser": self.parser if self.supported else "none",
        }


@dataclass(frozen=True)
class ReasoningResult:
    reasoning: str
    content: str
    detected: bool
    incomplete: bool = False


DISABLED_REASONING_SETTINGS = ReasoningSettings(
    configured_mode="disabled",
    mode="disabled",
    parser="none",
    response_field="reasoning_content",
    supported=False,
)


def load_reasoning_settings(
    model_path: Path,
    *,
    include_reasoning: bool | None = None,
    force_disable: bool = False,
) -> ReasoningSettings:
    config_path = model_path / "gateway.json"
    modified_ns = config_path.stat().st_mtime_ns if config_path.is_file() else 0
    configured = _read_reasoning_settings(
        str(config_path),
        modified_ns,
        str(model_path),
    )

    mode = configured.configured_mode
    # OpenAI structured output constrains the visible assistant response. A
    # thinking prompt would put the constrained JSON inside an unfinished
    # reasoning block, leaving message.content empty after post-processing.
    if force_disable:
        mode = "disabled"
    # A caller may hide reasoning allowed by server policy, but cannot make a
    # hidden or disabled chain-of-thought visible.
    elif include_reasoning is False and mode == "separate":
        mode = "hidden"

    if mode != "disabled" and not configured.supported:
        raise HTTPException(
            status_code=500,
            detail=(
                "Reasoning is enabled for this model, but no supported reasoning "
                "parser was detected. Set reasoning.parser explicitly in gateway.json "
                "or use reasoning.mode='disabled'."
            ),
        )

    return ReasoningSettings(
        configured_mode=configured.configured_mode,
        mode=mode,
        parser=configured.parser,
        response_field=configured.response_field,
        supported=configured.supported,
    )


@lru_cache(maxsize=256)
def _read_reasoning_settings(
    config_path: str,
    modified_ns: int,
    model_path: str,
) -> ReasoningSettings:
    del modified_ns
    payload = _read_gateway_json(Path(config_path))
    configured = payload.get("reasoning", {})
    if not isinstance(configured, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json reasoning must be an object",
        )

    mode = _configured_choice(
        configured,
        "mode",
        "GATEWAY_REASONING_MODE",
        "disabled",
        _MODES,
    )
    requested_parser = _configured_choice(
        configured,
        "parser",
        "GATEWAY_REASONING_PARSER",
        "auto",
        _PARSERS,
    )
    response_field = _configured_choice(
        configured,
        "response_field",
        "GATEWAY_REASONING_RESPONSE_FIELD",
        "reasoning_content",
        _RESPONSE_FIELDS,
    )
    parser = (
        _detect_reasoning_parser(Path(model_path))
        if requested_parser == "auto"
        else requested_parser
    )
    supported = parser != "none"

    return ReasoningSettings(
        configured_mode=mode,
        mode=mode,
        parser=parser,
        response_field=response_field,
        supported=supported,
    )


def split_reasoning_output(
    text: str,
    settings: ReasoningSettings,
) -> ReasoningResult:
    if settings.parser == "none":
        return ReasoningResult("", text, False)

    open_index, open_tag = _find_first(text, _OPEN_TAGS)
    close_index, close_tag = _find_first(text, _CLOSE_TAGS)
    has_tags = open_index != -1 or close_index != -1

    if not settings.enable_thinking:
        reasoning, content, incomplete = _strip_reasoning_blocks(text)
        return ReasoningResult(
            reasoning=reasoning,
            content=_remove_service_tags(content),
            detected=has_tags or bool(reasoning),
            incomplete=incomplete,
        )

    if close_index != -1 and (open_index == -1 or close_index < open_index):
        reasoning = text[:close_index]
        content = text[close_index + len(close_tag) :]
        return ReasoningResult(
            _clean_reasoning(reasoning),
            _remove_service_tags(content).lstrip(),
            True,
        )

    if open_index != -1:
        prefix = text[:open_index]
        reasoning_start = open_index + len(open_tag)
        close_index, close_tag = _find_first(
            text,
            _CLOSE_TAGS,
            reasoning_start,
        )
        if close_index != -1:
            reasoning = text[reasoning_start:close_index]
            content = prefix + text[close_index + len(close_tag) :]
            return ReasoningResult(
                _clean_reasoning(reasoning),
                _remove_service_tags(content).lstrip(),
                True,
            )

        tool_index = _find_tool_marker(text, reasoning_start)
        if tool_index != -1:
            reasoning = text[reasoning_start:tool_index]
            content = prefix + text[tool_index:]
            return ReasoningResult(
                _clean_reasoning(reasoning),
                _remove_service_tags(content).lstrip(),
                True,
            )

        return ReasoningResult(
            _clean_reasoning(text[reasoning_start:]),
            _remove_service_tags(prefix).rstrip(),
            True,
            incomplete=True,
        )

    tool_index = _find_tool_marker(text)
    if tool_index != -1:
        return ReasoningResult(
            _clean_reasoning(text[:tool_index]),
            _remove_service_tags(text[tool_index:]).lstrip(),
            bool(text[:tool_index].strip()),
        )

    # Qwen3.5 commonly places <think> in the generation prompt, so generated
    # output starts with reasoning and contains only the closing tag.
    return ReasoningResult(
        _clean_reasoning(text),
        "",
        bool(text),
        incomplete=True,
    )


def reasoning_message_fields(
    result: ReasoningResult,
    settings: ReasoningSettings,
) -> dict[str, str | None]:
    if not settings.expose_reasoning:
        return {}
    return {
        settings.response_field: result.reasoning if result.reasoning else None,
    }


def reasoning_token_count(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except (AttributeError, TypeError):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(text))


def observe_reasoning_result(
    model_name: str,
    tokenizer: Any,
    settings: ReasoningSettings,
    result: ReasoningResult,
) -> tuple[int, int]:
    reasoning_tokens = reasoning_token_count(tokenizer, result.reasoning)
    content_tokens = reasoning_token_count(tokenizer, result.content)
    outcome = (
        "incomplete"
        if result.incomplete
        else "detected"
        if result.detected
        else "not_present"
    )
    REASONING_REQUESTS.labels(
        model_name,
        settings.mode,
        settings.parser,
        outcome,
    ).inc()
    REASONING_TOKENS.labels(model_name, settings.mode, "reasoning").inc(
        reasoning_tokens
    )
    REASONING_TOKENS.labels(model_name, settings.mode, "content").inc(
        content_tokens
    )
    return reasoning_tokens, content_tokens


def _strip_reasoning_blocks(text: str) -> tuple[str, str, bool]:
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    cursor = 0
    incomplete = False

    while cursor < len(text):
        open_index, open_tag = _find_first(text, _OPEN_TAGS, cursor)
        close_index, close_tag = _find_first(text, _CLOSE_TAGS, cursor)

        if close_index != -1 and (open_index == -1 or close_index < open_index):
            reasoning_parts.append(text[cursor:close_index])
            cursor = close_index + len(close_tag)
            continue

        if open_index == -1:
            content_parts.append(text[cursor:])
            break

        content_parts.append(text[cursor:open_index])
        reasoning_start = open_index + len(open_tag)
        close_index, close_tag = _find_first(
            text,
            _CLOSE_TAGS,
            reasoning_start,
        )
        if close_index == -1:
            tool_index = _find_tool_marker(text, reasoning_start)
            if tool_index == -1:
                reasoning_parts.append(text[reasoning_start:])
                incomplete = True
                break
            reasoning_parts.append(text[reasoning_start:tool_index])
            content_parts.append(text[tool_index:])
            break

        reasoning_parts.append(text[reasoning_start:close_index])
        cursor = close_index + len(close_tag)

    return (
        "\n".join(part.strip() for part in reasoning_parts if part.strip()),
        "".join(content_parts),
        incomplete,
    )


def _detect_reasoning_parser(model_path: Path) -> str:
    metadata: list[str] = [model_path.name.lower()]
    templates: list[str] = []
    for filename in ("config.json", "tokenizer_config.json"):
        path = model_path / filename
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        serialized = json.dumps(payload, ensure_ascii=False).lower()
        metadata.append(serialized)
        if filename == "tokenizer_config.json":
            templates.append(serialized)

    chat_template_path = model_path / "chat_template.jinja"
    if chat_template_path.is_file():
        try:
            templates.append(chat_template_path.read_text(encoding="utf-8").lower())
        except (OSError, UnicodeDecodeError):
            pass

    combined_metadata = "\n".join(metadata)
    combined_templates = "\n".join(templates)
    has_thinking_template = (
        "enable_thinking" in combined_templates
        or any(tag in combined_templates for tag in _OPEN_TAGS)
    )
    if "qwen3" in combined_metadata and has_thinking_template:
        return "qwen3"
    if any(tag in combined_templates for tag in _OPEN_TAGS):
        return "think_tags"
    return "none"


def _read_gateway_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to parse gateway.json: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json root must be an object",
        )
    return payload


def _configured_choice(
    configured: dict[str, Any],
    key: str,
    environment_key: str,
    default: str,
    allowed: set[str],
) -> str:
    value = str(
        os.environ.get(environment_key, configured.get(key, default))
    ).strip().lower()
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise HTTPException(
            status_code=500,
            detail=f"reasoning.{key} must be one of: {choices}",
        )
    return value


def _find_first(
    text: str,
    candidates: tuple[str, ...],
    start: int = 0,
) -> tuple[int, str]:
    index = -1
    found = ""
    for candidate in candidates:
        candidate_index = text.find(candidate, start)
        if candidate_index != -1 and (index == -1 or candidate_index < index):
            index = candidate_index
            found = candidate
    return index, found


def _find_tool_marker(text: str, start: int = 0) -> int:
    indexes = [
        text.find(marker, start)
        for marker in _TOOL_MARKERS
        if text.find(marker, start) != -1
    ]
    return min(indexes) if indexes else -1


def _remove_service_tags(text: str) -> str:
    result = text
    for tag in (*_OPEN_TAGS, *_CLOSE_TAGS):
        result = result.replace(tag, "")
    return result


def _clean_reasoning(text: str) -> str:
    return _remove_service_tags(text).strip()
