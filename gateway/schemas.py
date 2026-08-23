# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    role: str
    # OpenAI permits assistant tool-call messages without a content field.
    content: Any | None = None

    model_config = {"extra": "allow"}


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: str | None = None
    json_schema: dict[str, Any] = Field(alias="schema")
    strict: bool | None = None

    model_config = {"extra": "forbid", "populate_by_name": True}


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaResponseFormat | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_json_schema(self):
        if self.type == "json_schema" and self.json_schema is None:
            raise ValueError(
                "response_format.json_schema is required when type is json_schema"
            )
        return self


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
    seed: int | None = Field(
        default=None,
        ge=-(2**63),
        le=2**63 - 1,
    )
    response_format: ResponseFormat | None = None
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
