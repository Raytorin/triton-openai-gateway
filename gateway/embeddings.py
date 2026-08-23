# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import array
import base64
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .schemas import EmbeddingsRequest, HybridEmbeddingsRequest


@dataclass(frozen=True)
class HybridEmbeddingSettings:
    enabled: bool = False
    output_types: frozenset[str] = frozenset()
    max_batch_size: int = 32
    default_sparse_top_k: int | None = None
    max_sparse_top_k: int = 8192


_HYBRID_REQUEST_FIELDS = frozenset(
    {
        "output_type",
        "output_types",
        "return_sparse",
        "sparse_format",
        "sparse_top_k",
    }
)


def reject_hybrid_options_on_dense_endpoint(request: EmbeddingsRequest) -> None:
    extras = request.model_extra or {}
    hybrid_fields = set(extras) & _HYBRID_REQUEST_FIELDS
    nested_extra_body = extras.get("extra_body")
    if isinstance(nested_extra_body, dict):
        hybrid_fields.update(set(nested_extra_body) & _HYBRID_REQUEST_FIELDS)
    if not hybrid_fields:
        return

    raise HTTPException(
        status_code=400,
        detail=(
            "Sparse embedding options are not supported on /v1/embeddings: "
            + ", ".join(sorted(hybrid_fields))
            + ". Use /v1/hybrid_embeddings through the configured LiteLLM "
            "pass-through route."
        ),
    )


def load_hybrid_embedding_settings(model_path: Path) -> HybridEmbeddingSettings:
    config_path = model_path / "gateway.json"
    modified_ns = config_path.stat().st_mtime_ns if config_path.is_file() else 0
    return _read_hybrid_embedding_settings(str(config_path), modified_ns)


@lru_cache(maxsize=256)
def _read_hybrid_embedding_settings(
    config_path: str,
    modified_ns: int,
) -> HybridEmbeddingSettings:
    del modified_ns
    path = Path(config_path)
    if not path.is_file():
        return HybridEmbeddingSettings()

    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read hybrid embeddings configuration: {exc}",
        ) from exc

    embeddings = root.get("embeddings") or {}
    if not isinstance(embeddings, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json embeddings must be an object",
        )
    hybrid = embeddings.get("hybrid") or {}
    if not isinstance(hybrid, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json embeddings.hybrid must be an object",
        )
    if not hybrid:
        return HybridEmbeddingSettings()

    enabled = _config_bool(hybrid, "enabled", False)
    raw_output_types = hybrid.get("output_types", ["dense", "sparse"])
    if not isinstance(raw_output_types, list) or not raw_output_types:
        raise HTTPException(
            status_code=500,
            detail="embeddings.hybrid.output_types must be a non-empty array",
        )
    output_types = frozenset(str(item).strip().lower() for item in raw_output_types)
    unknown = output_types - {"dense", "sparse"}
    if unknown:
        raise HTTPException(
            status_code=500,
            detail=(
                "embeddings.hybrid.output_types contains unsupported values: "
                + ", ".join(sorted(unknown))
            ),
        )

    max_batch_size = _config_positive_int(hybrid, "max_batch_size", 32)
    max_sparse_top_k = _config_positive_int(hybrid, "max_sparse_top_k", 8192)
    default_sparse_top_k = _config_optional_positive_int(
        hybrid,
        "default_sparse_top_k",
    )
    if (
        default_sparse_top_k is not None
        and default_sparse_top_k > max_sparse_top_k
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                "embeddings.hybrid.default_sparse_top_k must not exceed "
                "max_sparse_top_k"
            ),
        )

    return HybridEmbeddingSettings(
        enabled=enabled,
        output_types=output_types,
        max_batch_size=max_batch_size,
        default_sparse_top_k=default_sparse_top_k,
        max_sparse_top_k=max_sparse_top_k,
    )


def validate_hybrid_embedding_request(
    request: HybridEmbeddingsRequest,
    settings: HybridEmbeddingSettings,
    input_count: int,
) -> int | None:
    if not settings.enabled:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' is not configured for "
                "/v1/hybrid_embeddings"
            ),
        )

    requested = set(request.output_types)
    unsupported = requested - settings.output_types
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' does not support hybrid embedding output: "
                + ", ".join(sorted(unsupported))
            ),
        )
    if input_count > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Hybrid embedding batch size {input_count} exceeds configured limit "
                f"{settings.max_batch_size}"
            ),
        )
    if request.dimensions is not None and "dense" not in requested:
        raise HTTPException(
            status_code=400,
            detail="dimensions requires dense output",
        )
    if request.encoding_format not in {None, "float", "base64"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported encoding_format: {request.encoding_format}",
        )
    if request.sparse_top_k is not None and "sparse" not in requested:
        raise HTTPException(
            status_code=400,
            detail="sparse_top_k requires sparse output",
        )

    sparse_top_k = request.sparse_top_k
    if sparse_top_k is None and "sparse" in requested:
        sparse_top_k = settings.default_sparse_top_k
    if sparse_top_k is not None and sparse_top_k > settings.max_sparse_top_k:
        raise HTTPException(
            status_code=400,
            detail=(
                f"sparse_top_k={sparse_top_k} exceeds configured limit "
                f"{settings.max_sparse_top_k}"
            ),
        )
    return sparse_top_k


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise HTTPException(
            status_code=500,
            detail=f"embeddings.hybrid.{key} must be a boolean",
        )
    return value


def _config_positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = _config_optional_positive_int(config, key)
    return default if value is None else value


def _config_optional_positive_int(
    config: dict[str, Any],
    key: str,
) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        value = None
    try:
        parsed = int(value) if value is not None else 0
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"embeddings.hybrid.{key} must be a positive integer",
        )
    return parsed


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
