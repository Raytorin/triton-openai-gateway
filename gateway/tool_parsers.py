# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import re
import uuid
from pathlib import Path
from typing import Any


QWEN3_CODER_PARSER = "qwen3_coder"
KNOWN_PARSER_FILES = {
    "qwen3coder_tool_parser.py": QWEN3_CODER_PARSER,
    "qwen3_coder_tool_parser.py": QWEN3_CODER_PARSER,
}

TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
FUNCTION_RE = re.compile(
    r"<function=([^>\r\n]+)>(.*?)(?:</function>|(?=<function=)|$)",
    re.DOTALL,
)
PARAMETER_RE = re.compile(
    r"<parameter=([^>\r\n]+)>(.*?)"
    r"(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


def detect_model_tool_parser(model_path: Path) -> str | None:
    """Select a trusted built-in parser using files shipped with the model."""
    search_roots = (
        model_path,
        model_path / "tokenizer",
        model_path.parent / "tokenizer",
    )
    for root in search_roots:
        for filename, parser_name in KNOWN_PARSER_FILES.items():
            if (root / filename).is_file():
                return parser_name
    return None


def normalize_tool_calls_for_template(value: Any) -> Any:
    """Convert OpenAI JSON argument strings to mappings used by chat templates."""
    if not isinstance(value, list):
        return value

    normalized_calls: list[Any] = []
    for tool_call in value:
        if not isinstance(tool_call, dict):
            normalized_calls.append(tool_call)
            continue

        normalized_call = dict(tool_call)
        function = normalized_call.get("function")
        if isinstance(function, dict):
            normalized_function = dict(function)
            arguments = normalized_function.get("arguments")
            if isinstance(arguments, str):
                try:
                    decoded_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    decoded_arguments = None
                if isinstance(decoded_arguments, dict):
                    normalized_function["arguments"] = decoded_arguments
            normalized_call["function"] = normalized_function
        normalized_calls.append(normalized_call)
    return normalized_calls


def extract_tool_calls(
    generated_text: str,
    tools: list[dict[str, Any]] | None = None,
    parser_name: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Convert supported model tool formats to OpenAI-compatible tool calls.

    The model-specific parser is tried first when detected. Both parsers are
    always retained as fallbacks so models without a bundled parser file still
    work when they emit one of the known formats.
    """
    parser_order = (
        (_parse_qwen3_coder_payload, _parse_json_payload)
        if parser_name == QWEN3_CODER_PARSER
        else (_parse_json_payload, _parse_qwen3_coder_payload)
    )

    tool_calls: list[dict[str, Any]] = []
    parsed_spans: list[tuple[int, int]] = []

    for match in TOOL_BLOCK_RE.finditer(generated_text):
        parsed = _parse_with_fallbacks(match.group(1), tools, parser_order)
        if parsed:
            tool_calls.extend(parsed)
            parsed_spans.append(match.span())

    if tool_calls:
        return tool_calls, _remove_spans(generated_text, parsed_spans)

    # Qwen3-Coder may omit the outer <tool_call> wrapper on a truncated output.
    qwen_calls, qwen_spans = _parse_qwen3_coder_functions(generated_text, tools)
    if qwen_calls:
        remaining_text = _remove_spans(generated_text, qwen_spans)
        remaining_text = re.sub(r"</?tool_call>\s*", "", remaining_text).strip()
        return qwen_calls, remaining_text

    stripped = generated_text.strip()
    json_call = _parse_json_payload(stripped, tools)
    if json_call:
        return json_call, ""

    return [], generated_text


def _parse_with_fallbacks(
    payload: str,
    tools: list[dict[str, Any]] | None,
    parsers,
) -> list[dict[str, Any]]:
    for parser in parsers:
        parsed = parser(payload, tools)
        if parsed:
            return parsed
    return []


def _parse_json_payload(
    payload: str,
    _tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    try:
        value = json.loads(payload.strip())
    except (json.JSONDecodeError, TypeError):
        return []

    values = value if isinstance(value, list) else [value]
    calls = []
    for item in values:
        if not isinstance(item, dict):
            return []
        call = _openai_tool_call_from_mapping(item)
        if call is None:
            return []
        calls.append(call)
    return calls


def _parse_qwen3_coder_payload(
    payload: str,
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    calls, _ = _parse_qwen3_coder_functions(payload, tools)
    return calls


def _parse_qwen3_coder_functions(
    text: str,
    tools: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    schemas = _tool_parameter_schemas(tools)
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    for match in FUNCTION_RE.finditer(text):
        function_name = match.group(1).strip()
        if not function_name:
            continue

        parameters: dict[str, Any] = {}
        parameter_schema = schemas.get(function_name, {})
        for parameter in PARAMETER_RE.finditer(match.group(2)):
            parameter_name = parameter.group(1).strip()
            if not parameter_name:
                continue
            parameter_value = parameter.group(2).strip("\r\n")
            parameters[parameter_name] = _convert_parameter_value(
                parameter_value,
                parameter_schema.get(parameter_name),
            )

        calls.append(_build_openai_tool_call(function_name, parameters))
        spans.append(match.span())

    return calls, spans


def _tool_parameter_schemas(
    tools: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties", parameters)
        if isinstance(properties, dict):
            schemas[name] = properties
    return schemas


def _convert_parameter_value(value: str, schema: Any) -> Any:
    if value.lower() == "null":
        return None
    if not isinstance(schema, dict):
        return value

    parameter_type = str(schema.get("type", "string")).strip().lower()
    if parameter_type in {"string", "str", "text", "varchar", "char", "enum"}:
        return value
    if parameter_type.startswith(("int", "uint", "long", "short", "unsigned")):
        try:
            return int(value)
        except ValueError:
            return value
    if parameter_type.startswith(("num", "float")):
        try:
            number = float(value)
        except ValueError:
            return value
        return int(number) if number.is_integer() else number
    if parameter_type in {"boolean", "bool", "binary"}:
        return value.lower() == "true"
    if parameter_type in {"object", "array", "arr"} or parameter_type.startswith(
        ("dict", "list")
    ):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _openai_tool_call_from_mapping(value: dict[str, Any]) -> dict[str, Any] | None:
    function = value.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", {})
    else:
        name = value.get("name")
        arguments = value.get("arguments", {})

    if not isinstance(name, str) or not name:
        return None
    return _build_openai_tool_call(name, arguments)


def _build_openai_tool_call(name: str, arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        arguments_string = arguments
    else:
        arguments_string = json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments_string,
        },
    }


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    parts: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts).strip()
