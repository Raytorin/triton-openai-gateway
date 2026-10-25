# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .cpu_work import checkpoint
from .metrics import REASONING_REQUESTS, REASONING_TOKENS
from .qwen import qwen_template_capabilities


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
    qwen_profile: bool = False
    deepseek_profile: bool = False
    requires_template_kwargs: bool = False
    resolved_chat_template: str | None = None
    _template_options: tuple[tuple[str, Any], ...] = ()

    @property
    def template_kwargs(self) -> dict[str, Any]:
        return {"enable_thinking": self.enable_thinking, **dict(self._template_options)}

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
    tokenizer: Any = None,
    enable_thinking: bool | None = None,
    preserve_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> ReasoningSettings:
    for name, value in (("enable_thinking", enable_thinking), ("preserve_thinking", preserve_thinking)):
        if value is not None and type(value) is not bool:
            raise HTTPException(status_code=400, detail=f"{name} must be a boolean")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise HTTPException(status_code=400, detail="reasoning_effort must be a string")
    from .deepseek import is_deepseek_profile, template_capabilities
    deepseek_profile = is_deepseek_profile(model_path, tokenizer)
    if deepseek_profile:
        qwen_profile = False
        capabilities = template_capabilities(model_path, tokenizer, tools=tools)
    else:
        qwen_profile, capabilities = qwen_template_capabilities(model_path, tokenizer, tools=tools)
    config_path = model_path / "gateway.json"
    configured = _read_reasoning_settings(
        str(config_path),
        str(model_path),
        template=capabilities.template,
        qwen_profile=qwen_profile,
        has_think_tags=capabilities.has_think_tags,
    )

    mode = configured.configured_mode
    if enable_thinking is True and (mode == "disabled" or force_disable):
        raise HTTPException(status_code=400, detail="enable_thinking=true is prohibited by server policy or constrained JSON output")
    if enable_thinking is not None:
        supported_values = {"switch": (False, True), "on": (True,), "off": (False,)}
        if enable_thinking not in supported_values.get(capabilities.thinking, ()):
            raise HTTPException(status_code=400, detail="The resolved chat template does not support this enable_thinking value")
    if force_disable or enable_thinking is False or capabilities.thinking == "off":
        mode = "disabled"
    elif include_reasoning is False and mode == "separate":
        mode = "hidden"
    if mode == "disabled" and capabilities.thinking == "on":
        raise HTTPException(status_code=400, detail="The resolved fixed-thinking template cannot disable thinking")
    if preserve_thinking is not None and not capabilities.preserves_thinking:
        raise HTTPException(status_code=400, detail="preserve_thinking is not supported by the resolved chat template")
    if reasoning_effort is not None:
        if mode == "disabled":
            raise HTTPException(status_code=400, detail="reasoning_effort requires thinking to be enabled")
        if reasoning_effort not in capabilities.efforts:
            raise HTTPException(status_code=400, detail="reasoning_effort is not supported by the resolved chat template")

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
        qwen_profile=qwen_profile,
        deepseek_profile=deepseek_profile,
        resolved_chat_template=capabilities.template,
        requires_template_kwargs=(
            enable_thinking is not None or preserve_thinking is not None
            or reasoning_effort is not None
            or capabilities.thinking == "switch"
        ),
        _template_options=tuple((name, value) for name, value in (
            ("preserve_thinking", preserve_thinking), ("reasoning_effort", reasoning_effort)
        ) if value is not None),
    )


def _read_reasoning_settings(
    config_path: str,
    model_path: str,
    *,
    template: str | None = None,
    qwen_profile: bool = False,
    has_think_tags: bool = False,
) -> ReasoningSettings:
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
    if requested_parser != "auto":
        parser = requested_parser
    elif template is not None and has_think_tags:
        parser = "qwen3" if qwen_profile else "think_tags"
    elif template is not None:
        parser = "none"
    else:
        parser = _detect_reasoning_parser(Path(model_path))
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
    checkpoint()
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
