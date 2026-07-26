# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    # OpenAI permits assistant tool-call messages without a content field.
    content: Any | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    temperature: float | None = 0.2
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    repetition_penalty: float | None = None
    stream: bool = False
    debug: bool = False
    include_reasoning: bool | None = None

    model_config = {"extra": "allow"}


class EmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    dimensions: int | None = None
    encoding_format: str | None = "float"

    model_config = {"extra": "allow"}


class RerankSelectionRequest(BaseModel):
    strategy: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class RerankRequest(BaseModel):
    model: str
    query: str
    documents: list[Any]
    top_n: int | None = None
    return_documents: bool | None = False
    max_length: int | None = None
    batch_size: int | None = None
    normalize: bool | None = True
    selection: RerankSelectionRequest | str | None = None
    custom_top: RerankSelectionRequest | str | None = None

    model_config = {"extra": "allow"}
