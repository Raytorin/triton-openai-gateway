# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import array
import base64

from fastapi import HTTPException

from .schemas import EmbeddingsRequest


def build_embedding_inputs(request: EmbeddingsRequest) -> list[str | list[int]]:
    model_input = request.input

    if isinstance(model_input, str):
        return [model_input]

    if isinstance(model_input, list):
        if not model_input:
            raise HTTPException(status_code=400, detail="Embedding input must not be empty")

        if isinstance(model_input[0], str):
            return [str(item) for item in model_input]

        if isinstance(model_input[0], int):
            return [model_input]

        if isinstance(model_input[0], list):
            return model_input

    raise HTTPException(status_code=400, detail="Unsupported embeddings input format")


def tokenize_embedding_inputs(tokenizer, model_inputs: list[str | list[int]]) -> list[list[int]]:
    tokenized_inputs: list[list[int]] = []

    for model_input in model_inputs:
        if isinstance(model_input, list):
            tokenized_inputs.append(model_input)
            continue

        token_ids = tokenizer.encode(
            model_input,
            add_special_tokens=True,
        )
        tokenized_inputs.append([int(token_id) for token_id in token_ids])

    return tokenized_inputs


def encode_embedding(embedding: list[float], encoding_format: str) -> list[float] | str:
    if encoding_format == "float":
        return embedding
    if encoding_format == "base64":
        return base64.b64encode(array.array("f", embedding).tobytes()).decode("utf-8")

    raise HTTPException(status_code=400, detail=f"Unsupported encoding_format: {encoding_format}")
