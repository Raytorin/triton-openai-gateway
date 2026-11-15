# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Public context errors built from counts, never from backend diagnostics."""
from __future__ import annotations

import re

from fastapi import HTTPException


class ContextLengthExceeded(HTTPException):
    def __init__(self, context_limit: int, prompt_tokens: int | None,
                 requested_output_tokens: int | None = None, *,
                 reserved_media_tokens: int = 0, safety_margin_tokens: int = 0):
        details = {"context_limit": context_limit,
                   "reserved_media_tokens": reserved_media_tokens,
                   "safety_margin_tokens": safety_margin_tokens}
        parts = []
        if prompt_tokens is not None:
            details["prompt_tokens"] = prompt_tokens
            parts.append(f"{prompt_tokens} input tokens")
        if requested_output_tokens is not None:
            details["requested_output_tokens"] = requested_output_tokens
            parts.append(f"an output budget of {requested_output_tokens} tokens")
        if reserved_media_tokens:
            parts.append(f"{reserved_media_tokens} estimated media reserve tokens")
        if safety_margin_tokens:
            parts.append(f"{safety_margin_tokens} safety margin tokens")
        message = f"This model's maximum context length is {context_limit} tokens."
        if parts:
            message += " Your request has " + ", ".join(parts) + "."
        if prompt_tokens is not None and requested_output_tokens is not None:
            budget = prompt_tokens + requested_output_tokens + reserved_media_tokens + safety_margin_tokens
            details["budget_tokens"] = budget
            message += f" The total request budget is {budget} tokens."
        elif prompt_tokens is not None and prompt_tokens >= context_limit:
            message += " The input leaves no room for output."
        message += " Please reduce the input or output budget."
        self.error = {"message": message, "type": "invalid_request_error",
                      "param": None, "code": "context_length_exceeded", "details": details}
        super().__init__(status_code=400, detail=message)


# vLLM decoder admission messages. Encoder-cache and character limits are
# different resources; unknown wording deliberately retains its existing path.
_DECODER_CONTEXT = re.compile(
    r"The decoder prompt \(length ([0-9]{1,12})\)"
    r"( plus the number of requested output tokens \(at least 1\))? "
    r"is longer than the maximum model length of ([0-9]{1,12})\."
)
_COMBINED_CONTEXT = re.compile(
    r"This model's maximum context length is ([0-9]{1,12}) tokens\. "
    r"However, you requested ([0-9]{1,12}) output tokens and your prompt contains "
    r"([0-9]{1,12}) input tokens, for a total of ([0-9]{1,12}) tokens\."
)


def backend_context_length_error(detail: str) -> ContextLengthExceeded | None:
    matches = list(_DECODER_CONTEXT.finditer(detail))
    combined = list(_COMBINED_CONTEXT.finditer(detail))
    if len(matches) + len(combined) != 1:
        return None
    if combined:
        limit, output, prompt, total = map(int, combined[0].groups())
        if limit > 0 and output > 0 and total == prompt + output and total > limit:
            return ContextLengthExceeded(limit, prompt, output)
        return None
    raw_prompt, minimum_output, raw_limit = matches[0].groups()
    prompt, limit = int(raw_prompt), int(raw_limit)
    if limit <= 0 or prompt < limit or (prompt == limit and minimum_output is None):
        return None
    return ContextLengthExceeded(limit, prompt)
