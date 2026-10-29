# SPDX-License-Identifier: Apache-2.0
"""Engine facts shared by generation transports without changing text callers."""
from dataclasses import dataclass, replace
import json
import logging
from typing import Any

from fastapi import HTTPException


class GenerationMetadataError(HTTPException):
    def __init__(self, detail: str):
        super().__init__(502, detail)


@dataclass(frozen=True)
class GenerationMetadata:
    finished: bool = False
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    first_token_id: int | None = None
    last_token_id: int | None = None


class GenerationOutput(str):
    def __new__(cls, text: str, metadata: GenerationMetadata | None = None):
        value = super().__new__(cls, text)
        value.generation_metadata = metadata
        return value


def metadata_of(value: Any) -> GenerationMetadata | None:
    metadata = getattr(value, "generation_metadata", None)
    return metadata if isinstance(metadata, GenerationMetadata) else None


def _invalid() -> None:
    raise GenerationMetadataError("Invalid generation metadata from Triton")


def _count(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        _invalid()
    return value


def decode_generation_metadata(payload: Any) -> GenerationMetadata:
    try:
        data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    except (ValueError, UnicodeError):
        _invalid()
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        _invalid()
    if type(data.get("finished")) is not bool:
        _invalid()
    outputs = data.get("outputs")
    # The gateway exposes one choice. Never join token counts from unrelated choices.
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        _invalid()
    output = outputs[0]
    # Version 1 always carries the current cumulative count, including final packets.
    if type(output.get("completion_tokens")) is not int:
        _invalid()
    if type(output.get("index")) is not int or output["index"] != 0:
        _invalid()
    reason = output.get("finish_reason")
    stop = output.get("stop_reason")
    if reason is not None and not isinstance(reason, str):
        _invalid()
    if stop is not None and (type(stop) not in (str, int) or isinstance(stop, int) and stop < 0):
        _invalid()
    return GenerationMetadata(
        finished=data["finished"], finish_reason=reason, stop_reason=stop,
        prompt_tokens=_count(data.get("prompt_tokens")),
        completion_tokens=_count(output.get("completion_tokens")),
        first_token_id=_count(output.get("first_token_id")),
        last_token_id=_count(output.get("last_token_id")),
    )


class GenerationMetadataState:
    def __init__(self):
        self.metadata: GenerationMetadata | None = None

    def update_json(self, payload: Any) -> None:
        current = decode_generation_metadata(payload)
        previous = self.metadata
        if previous is not None:
            if previous.finished and current != previous:
                _invalid()
            if (previous.prompt_tokens is not None and current.prompt_tokens is not None
                    and previous.prompt_tokens != current.prompt_tokens):
                _invalid()
            if (previous.completion_tokens is not None and current.completion_tokens is not None
                    and current.completion_tokens < previous.completion_tokens):
                _invalid()
            current = replace(current,
                prompt_tokens=current.prompt_tokens if current.prompt_tokens is not None else previous.prompt_tokens,
                first_token_id=current.first_token_id if current.first_token_id is not None else previous.first_token_id,
            )
        self.metadata = current


def resolve_finish_reason(metadata: GenerationMetadata | None, *,
                          token_limit_reached: bool = False, tool_calls: bool = False,
                          stopped_locally: bool = False) -> str:
    if metadata is not None and metadata.finished:
        if metadata.finish_reason == "length":
            return "length"
        if metadata.finish_reason == "stop":
            return "tool_calls" if tool_calls else "stop"
        if metadata.finish_reason in {"abort", "error"}:
            raise GenerationMetadataError("Triton generation was aborted or failed")
        raise GenerationMetadataError("Triton returned an unsupported generation finish reason")
    if stopped_locally:
        return "tool_calls" if tool_calls else "stop"
    if metadata is not None:
        raise GenerationMetadataError("Triton generation ended without terminal metadata")
    if token_limit_reached:
        return "length"
    return "tool_calls" if tool_calls else "stop"


def apply_engine_usage(usage: dict[str, Any], metadata: GenerationMetadata | None) -> dict[str, Any]:
    if metadata is None:
        return usage
    result = dict(usage)
    if metadata.prompt_tokens is not None:
        result["prompt_tokens"] = metadata.prompt_tokens
    if metadata.completion_tokens is not None:
        result["completion_tokens"] = metadata.completion_tokens
        if "completion_tokens_details" in result:
            details = dict(result["completion_tokens_details"])
            # Reasoning segmentation is still estimated from text, unlike engine totals.
            if "reasoning_tokens" in details:
                details["reasoning_tokens"] = min(details["reasoning_tokens"], metadata.completion_tokens)
            result["completion_tokens_details"] = details
    result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def log_completion(model_name: str, text: str, metadata: GenerationMetadata | None) -> None:
    if text:
        return
    from .observability import log_event
    stop = metadata.stop_reason if metadata is not None else None
    log_event(logging.getLogger("triton-chat-gateway"), "generation.empty_output",
        "Generation completed without decoded text", level=logging.WARNING,
        model=model_name, metadata_available=metadata is not None,
        engine_finished=metadata.finished if metadata else None,
        finish_reason=metadata.finish_reason if metadata and metadata.finish_reason in {"stop", "length", "abort", "error"} else None,
        prompt_tokens=metadata.prompt_tokens if metadata else None,
        completion_tokens=metadata.completion_tokens if metadata else None,
        first_token_id=metadata.first_token_id if metadata else None,
        last_token_id=metadata.last_token_id if metadata else None,
        stop_reason_type="token_id" if type(stop) is int else "string" if isinstance(stop, str) else None,
        stop_reason_token_id=stop if type(stop) is int else None,
        stop_reason_length=len(stop) if isinstance(stop, str) else None,
    )
