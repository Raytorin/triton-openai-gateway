# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Stateless Responses protocol adapter. Generation remains shared with Chat."""
from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from . import generation
from .generation_types import GenerationResult, GenerationStream
from .schemas import ChatCompletionRequest, ChatMessage, ResponseFormat


class ResponsesRequest(BaseModel):
    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, strict=True, gt=0)
    temperature: float | None = 0.2
    top_p: float | None = None
    stream: bool = False
    store: bool | None = False
    background: bool | None = False
    previous_response_id: str | None = None
    conversation: Any | None = None
    truncation: Literal["disabled", "auto"] | None = "disabled"
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    text: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, str] | None = None
    lora_name: str | None = None

    model_config = {"extra": "forbid"}


def reject(message: str) -> None:
    raise HTTPException(400, message)


def only_keys(value: dict, allowed: set[str], label: str) -> None:
    if unknown := value.keys() - allowed:
        reject(f"Unsupported {label}: {', '.join(sorted(unknown))}")


def required_string(value: dict, name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result:
        reject(f"{name} must be a nonempty string")
    return result


def input_messages(request: ResponsesRequest) -> list[ChatMessage]:
    messages = []
    if request.instructions is not None:
        messages.append(ChatMessage(role="system", content=request.instructions))
    if isinstance(request.input, str):
        return messages + [ChatMessage(role="user", content=request.input)]
    calls: set[str] = set()
    pending: set[str] = set()
    for item in request.input:
        kind = item.get("type", "message")
        if kind == "function_call":
            only_keys(item, {"type", "id", "call_id", "name", "arguments", "status"}, kind)
            call_id = required_string(item, "call_id")
            if call_id in calls:
                reject(f"Duplicate function call_id: {call_id}")
            name = required_string(item, "name")
            arguments = item.get("arguments")
            if not isinstance(arguments, str):
                reject("function_call.arguments must be a string")
            calls.add(call_id)
            pending.add(call_id)
            call = {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
            if messages and messages[-1].role == "assistant" and getattr(messages[-1], "tool_calls", None):
                messages[-1].tool_calls.append(call)
            else:
                messages.append(ChatMessage(role="assistant", content=None, tool_calls=[call]))
        elif kind == "function_call_output":
            only_keys(item, {"type", "id", "call_id", "output", "status"}, kind)
            call_id = required_string(item, "call_id")
            if call_id not in pending:
                reject("function_call_output requires a preceding unmatched function_call with the same call_id")
            if not isinstance(item.get("output"), str):
                reject("Only string function_call_output.output is supported")
            pending.remove(call_id)
            messages.append(ChatMessage(role="tool", content=item["output"], tool_call_id=call_id))
        elif kind == "message":
            only_keys(item, {"type", "id", "role", "content", "status"}, kind)
            if pending:
                reject("Function calls must have outputs before the next message")
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant"}:
                reject("Unsupported input message role")
            content = item.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        reject("Input content parts must be objects")
                    ptype = part.get("type")
                    if ptype in {"input_text", "output_text"}:
                        only_keys(part, {"type", "text", "annotations", "logprobs"}, ptype)
                        if part.get("annotations") or part.get("logprobs"):
                            reject("Annotated input and logprobs are not supported")
                        if not isinstance(part.get("text"), str):
                            reject("Text content must be a string")
                        parts.append({"type": "text", "text": part["text"]})
                    elif ptype == "input_image":
                        only_keys(part, {"type", "image_url", "detail"}, ptype)
                        url = required_string(part, "image_url")
                        if not url.startswith(("https://", "http://", "data:image/")):
                            reject("input_image requires an HTTP(S) URL or image data URL")
                        detail = part.get("detail", "auto")
                        if detail not in {"auto", "high", "low"}:
                            reject("Unsupported image detail")
                        if role != "user":
                            reject("input_image is supported in user messages only")
                        parts.append({"type": "image_url", "image_url": {"url": url, "detail": detail}})
                    else:
                        reject(f"Unsupported Responses input content type: {ptype}")
                content = parts
            elif not isinstance(content, str):
                reject("Message content must be a string or list of content parts")
            messages.append(ChatMessage(role="system" if role == "developer" else role, content=content))
        else:
            reject(f"Unsupported Responses input item: {kind}")
    if pending:
        reject("All function calls in stateless input must include function_call_output")
    if not request.input:
        reject("input must contain at least one item")
    return messages


def to_chat(request: ResponsesRequest) -> ChatCompletionRequest:
    for name in ("store", "background", "previous_response_id", "conversation"):
        value = getattr(request, name)
        if (value is True if name in {"store", "background"} else value is not None):
            reject(f"{name} is unavailable: Responses currently supports stateless generation only; send history in input and use store=false")
    if request.reasoning:
        reject("Responses reasoning parameters are not supported; thinking is configured per model")
    if request.parallel_tool_calls is False:
        reject("parallel_tool_calls=false is not supported by the current backend")
    tools = []
    tool_names: set[str] = set()
    for tool in request.tools or []:
        only_keys(tool, {"type", "name", "description", "parameters", "strict"}, "tool parameter")
        if tool.get("type") != "function":
            reject("Only client-executed function tools are supported")
        name = required_string(tool, "name")
        if name in tool_names:
            reject("Function names must be unique")
        tool_names.add(name)
        if tool.get("parameters") is not None and not isinstance(tool["parameters"], dict):
            reject("Function parameters must be a JSON Schema object")
        if tool.get("description") is not None and not isinstance(tool["description"], str):
            reject("Function description must be a string")
        if tool.get("strict") is not None and type(tool["strict"]) is not bool:
            reject("Function strict must be a boolean")
        if tool.get("strict") is True:
            reject("Strict function argument validation is not supported; use text.format for constrained JSON output")
        tools.append({"type": "function", "function": {k: v for k, v in tool.items() if k != "type"}})
    choice = request.tool_choice
    if isinstance(choice, dict):
        only_keys(choice, {"type", "name"}, "tool_choice")
        if choice.get("type") != "function":
            reject("Only function tool_choice is supported")
        choice = {"type": "function", "function": {"name": required_string(choice, "name")}}
        if choice["function"]["name"] not in tool_names:
            reject("Selected function must be present in tools")
    elif choice is not None and choice not in ("auto", "none", "required"):
        reject("Unsupported tool_choice")
    if choice in ("required",) and not tools:
        reject("tool_choice=required requires tools")
    response_format = None
    if request.text is not None:
        only_keys(request.text, {"format"}, "text parameter")
        fmt = request.text.get("format", {"type": "text"})
        if not isinstance(fmt, dict):
            reject("text.format must be an object")
        only_keys(fmt, {"type", "name", "description", "schema", "strict"}, "text.format")
        if fmt.get("type") == "json_schema":
            response_format = ResponseFormat.model_validate({"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}})
        elif fmt.get("type") in {"text", "json_object"}:
            only_keys(fmt, {"type"}, "text.format")
            response_format = ResponseFormat(type=fmt["type"])
        else:
            reject("Unsupported text.format.type")
    return ChatCompletionRequest(model=request.model, messages=input_messages(request),
        max_completion_tokens=request.max_output_tokens, temperature=request.temperature,
        top_p=request.top_p, stream=request.stream, tools=tools or None,
        tool_choice=choice, response_format=response_format, lora_name=request.lora_name,
        include_reasoning=False)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def usage_object(usage: dict[str, Any]) -> dict[str, Any]:
    return {"input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "input_tokens_details": {"cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)},
        "output_tokens_details": {"reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)}}


def response_object(request: ResponsesRequest, *, response_id: str, created: int,
                    status: str, output: list, usage: dict | None = None,
                    headers: dict | None = None, error: dict | None = None) -> dict:
    effective_limit = (headers or {}).get("X-Output-Token-Limit")
    return {"id": response_id, "object": "response", "created_at": created,
        "status": status, "error": error,
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "model": request.model, "output": output, "usage": usage_object(usage) if usage is not None else None,
        "instructions": request.instructions, "max_output_tokens": int(effective_limit) if effective_limit else request.max_output_tokens,
        "temperature": request.temperature, "top_p": request.top_p, "tools": request.tools or [],
        "tool_choice": request.tool_choice or "auto", "parallel_tool_calls": True,
        "text": request.text or {"format": {"type": "text"}}, "reasoning": {"effort": None, "summary": None},
        "truncation": request.truncation or "disabled", "metadata": request.metadata or {},
        "store": False, "background": False, "previous_response_id": None}


def text_part(text: str) -> dict:
    return {"type": "output_text", "text": text, "annotations": [], "logprobs": []}


def serialize_result(request: ResponsesRequest, result: GenerationResult) -> dict:
    status = "incomplete" if result.finish_reason == "length" else "completed"
    output = []
    if content := result.message.get("content"):
        output.append({"id": new_id("msg"), "type": "message", "role": "assistant",
                       "status": status, "content": [text_part(content)]})
    for tool in result.message.get("tool_calls") or []:
        output.append({"id": new_id("fc"), "type": "function_call", "status": status,
            "call_id": tool["id"], "name": tool["function"]["name"], "arguments": tool["function"]["arguments"]})
    return response_object(request, response_id=new_id("resp"), created=result.created,
        status=status, output=output, usage=result.usage, headers=result.headers)


router = APIRouter()


@router.post("/responses")
@router.post("/v1/responses")
async def create_response(request: ResponsesRequest, response: Response):
    from pydantic import ValidationError
    try:
        chat = to_chat(request)
    except ValidationError as exc:
        reject(str(exc))
    except (TypeError, ValueError):
        reject("Malformed Responses input or parameter type")
    if request.stream:
        reject("Responses streaming arrives in the next stage; use stream=false")
    result = await generation.generate(chat, context_mode="truncate" if request.truncation == "auto" else "disabled")
    response.headers.update(result.headers)
    return serialize_result(request, result)
