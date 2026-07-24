# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from fastapi import HTTPException

from .multimodal import normalize_message_content
from .schemas import ChatCompletionRequest, ChatMessage
from .settings import TRITON_DEFAULT_STOP_SEQUENCE, logger
from .tool_parsers import normalize_tool_calls_for_template


CLIENT_ERROR_PREFIXES = (
    "Error fetching response:",
    "Не удалось обработать запрос",
    "Не удалось обработать запрос через Triton",
)


def build_conversation(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    conversation: list[dict[str, Any]] = []
    for message in messages:
        content = normalize_message_content(message.content)
        if message.role == "assistant" and _is_client_error_content(content):
            continue

        message_dict: dict[str, Any] = {
            "role": message.role,
            "content": content,
        }
        message_extra = getattr(message, "model_extra", None) or {}
        for key in ("tool_calls", "tool_call_id", "name"):
            if key in message_extra:
                value = message_extra[key]
                message_dict[key] = (
                    normalize_tool_calls_for_template(value)
                    if key == "tool_calls"
                    else value
                )
        conversation.append(message_dict)
    return conversation


def _is_client_error_content(content: Any) -> bool:
    if not isinstance(content, str):
        return False

    stripped = content.strip()
    return any(stripped.startswith(prefix) for prefix in CLIENT_ERROR_PREFIXES)


def _is_tool_choice_none(tool_choice: Any) -> bool:
    return isinstance(tool_choice, str) and tool_choice == "none"


def selected_tools(
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> list[dict[str, Any]] | None:
    if not tools or _is_tool_choice_none(tool_choice):
        return None

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function_name = (tool_choice.get("function") or {}).get("name")
        if not function_name:
            raise HTTPException(status_code=400, detail="tool_choice.function.name is required")

        selected = [
            tool
            for tool in tools
            if (tool.get("function") or {}).get("name") == function_name
        ]
        if not selected:
            raise HTTPException(
                status_code=400,
                detail=f"tool_choice function '{function_name}' is not present in tools",
            )
        return selected

    return tools


def tool_choice_instruction(
    tool_choice: Any,
    *,
    has_tool_result: bool = False,
) -> str | None:
    base_instruction = (
        "When tools are provided and a user request can be satisfied by one of them, "
        "call the relevant tool instead of answering directly. "
        "Do not explain your reasoning. Do not describe the tool. "
        "If no provided tool is appropriate, answer normally without a tool call. "
        "When calling a tool, return only the tool call in the required tool-call format."
    )

    if tool_choice is None or (isinstance(tool_choice, str) and tool_choice == "auto"):
        if has_tool_result:
            return (
                "Tool results are present in the conversation. Use those results to answer "
                "the user directly and do not return an empty response. Call another tool "
                "only when the provided results are insufficient."
            )
        return base_instruction

    if isinstance(tool_choice, str) and tool_choice == "required":
        return (
            "You must call one of the provided tools. Do not answer directly. "
            "Do not explain your reasoning. Return only the tool call in the "
            "required tool-call format."
        )

    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function_name = (tool_choice.get("function") or {}).get("name")
        if function_name:
            return (
                f"You must call the function named `{function_name}`. "
                "Do not answer directly. Do not explain your reasoning. "
                "Return only the tool call in the required tool-call format."
            )

    return None


def has_tool_result(conversation: list[dict[str, Any]]) -> bool:
    return any(message.get("role") in {"tool", "function"} for message in conversation)


def add_system_instruction(
    conversation: list[dict[str, Any]],
    instruction: str | None,
) -> list[dict[str, Any]]:
    if not instruction:
        return conversation

    updated = [dict(message) for message in conversation]
    if updated and updated[0].get("role") == "system":
        content = updated[0].get("content", "")
        if isinstance(content, list):
            updated[0]["content"] = [
                *content,
                {"type": "text", "text": f"\n\n{instruction}"},
            ]
        else:
            updated[0]["content"] = f"{content}\n\n{instruction}".strip()
        return updated

    return [{"role": "system", "content": instruction}, *updated]


def _tool_prompt_fallback_conversation(
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tools_json = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
    instruction = (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "For each function call, return a JSON object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )
    return add_system_instruction(conversation, instruction)


def render_chat_prompt(
    tokenizer,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> str:
    attempts: list[dict[str, Any]] = [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
            **({"tools": tools} if tools else {}),
        },
        {
            "tokenize": False,
            "add_generation_prompt": True,
            **({"tools": tools} if tools else {}),
        },
    ]

    for kwargs in attempts:
        try:
            return tokenizer.apply_chat_template(conversation, **kwargs)
        except TypeError:
            continue

    if not tools:
        return tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Some tokenizers do not expose a tools= argument even when the model can
    # follow the standard Qwen tool-call XML convention.
    return tokenizer.apply_chat_template(
        _tool_prompt_fallback_conversation(conversation, tools),
        tokenize=False,
        add_generation_prompt=True,
    )


def fit_conversation_to_context(
    tokenizer,
    conversation: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_model_len: int,
    max_completion_tokens: int,
    reserved_media_tokens: int = 0,
    safety_margin_tokens: int = 64,
) -> tuple[list[dict[str, Any]], str, int, int]:
    prompt_limit = (
        int(max_model_len)
        - max(int(max_completion_tokens), 1)
        - max(int(reserved_media_tokens), 0)
        - max(int(safety_margin_tokens), 0)
    )
    if prompt_limit <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                "Requested output and media reserve leave no room for the input prompt: "
                f"max_model_len={max_model_len}, max_tokens={max_completion_tokens}, "
                f"media_reserve={reserved_media_tokens}."
            ),
        )

    fitted = [dict(message) for message in conversation]
    dropped_messages = 0
    while True:
        prompt = render_chat_prompt(tokenizer, fitted, tools)
        prompt_tokens = _prompt_token_count(tokenizer, prompt)
        if prompt_tokens <= prompt_limit:
            return fitted, prompt, prompt_tokens, dropped_messages

        removable = _oldest_removable_turn(fitted)
        if not removable:
            raise HTTPException(
                status_code=400,
                detail=(
                    "The system prompt and latest user message do not fit the model "
                    f"context: prompt_tokens={prompt_tokens}, allowed_prompt_tokens="
                    f"{prompt_limit}, max_model_len={max_model_len}."
                ),
            )
        removable_set = set(removable)
        fitted = [
            message for index, message in enumerate(fitted) if index not in removable_set
        ]
        dropped_messages += len(removable)


def _prompt_token_count(tokenizer, prompt: str) -> int:
    try:
        return len(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(prompt))


def _oldest_removable_turn(conversation: list[dict[str, Any]]) -> list[int]:
    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return []

    candidates = [
        index
        for index in range(latest_user_index)
        if conversation[index].get("role") != "system"
    ]
    if not candidates:
        return []

    start = candidates[0]
    if conversation[start].get("role") != "user":
        return [start]

    end = latest_user_index
    for index in range(start + 1, latest_user_index):
        if conversation[index].get("role") == "user":
            end = index
            break
    return [
        index
        for index in range(start, end)
        if conversation[index].get("role") != "system"
    ]


def build_sampling_parameters(request: ChatCompletionRequest) -> dict[str, Any]:
    max_tokens = request.max_completion_tokens or request.max_tokens or 256
    sampling: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(request.temperature if request.temperature is not None else 0.2),
    }

    if request.top_p is not None:
        sampling["top_p"] = float(request.top_p)

    stop = normalize_triton_stop_sequence(request.stop)
    if stop:
        sampling["stop"] = stop

    if request.repetition_penalty is not None:
        sampling["repetition_penalty"] = float(request.repetition_penalty)

    return sampling


def normalize_triton_stop_sequence(stop: str | list[str] | None) -> str | None:
    # Triton vLLM HTTP generate endpoint accepts only scalar parameter values.
    # Keep richer stop handling in gateway post-processing, not in Triton payload.
    if isinstance(stop, str):
        return stop

    if isinstance(stop, list):
        for item in stop:
            if item:
                logger.warning(
                    "Triton HTTP generate accepts one stop string; using first stop sequence"
                )
                return str(item)

    return TRITON_DEFAULT_STOP_SEQUENCE


def build_usage(tokenizer, prompt: str, generated_text: str) -> dict[str, int]:
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    completion_tokens = len(tokenizer(generated_text, add_special_tokens=False).input_ids)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
