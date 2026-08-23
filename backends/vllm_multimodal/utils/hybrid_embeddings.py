# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


_BGE_M3_ARCHITECTURE = "BgeM3EmbeddingModel"
_OUTPUT_TYPES = frozenset({"dense", "sparse"})


@dataclass(frozen=True)
class PoolingModelMetadata:
    architecture: str | None = None
    hidden_size: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    additional_special_token_ids: frozenset[int] = frozenset()

    @property
    def supports_bge_m3_sparse(self) -> bool:
        return self.architecture == _BGE_M3_ARCHITECTURE

    @property
    def special_token_ids(self) -> frozenset[int]:
        configured = frozenset(
            token_id
            for token_id in (
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.unk_token_id,
            )
            if token_id is not None
        )
        return configured | self.additional_special_token_ids


@dataclass(frozen=True)
class EmbeddingOutputSpec:
    output_types: tuple[str, ...]
    task: str
    dimensions: int | None
    sparse_top_k: int | None
    explicit_output_types: bool

    @property
    def includes_dense(self) -> bool:
        return "dense" in self.output_types

    @property
    def includes_sparse(self) -> bool:
        return "sparse" in self.output_types


def load_pooling_model_metadata(
    engine_config: dict[str, Any],
    *,
    model_dir: str | Path | None = None,
) -> PoolingModelMetadata:
    """Read only the HF fields required to decode BGE-M3 pooling output."""

    config: dict[str, Any] = {}
    model_ref = engine_config.get("model")
    if isinstance(model_ref, str) and model_ref:
        model_path = Path(model_ref)
        if not model_path.is_absolute() and model_dir is not None:
            model_path = Path(model_dir) / model_path
        config_path = model_path / "config.json"
        if config_path.is_file():
            try:
                parsed = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Unable to read pooling model metadata from {config_path}: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Pooling model config must be an object: {config_path}")
            config.update(parsed)

    overrides = engine_config.get("hf_overrides")
    if isinstance(overrides, dict):
        config.update(overrides)

    architectures = config.get("architectures")
    architecture = None
    if isinstance(architectures, list) and architectures:
        architecture = str(architectures[0])
    elif isinstance(architectures, str):
        architecture = architectures

    # Metadata is only interpreted for the native BGE-M3 sparse contract. This
    # avoids imposing BGE-specific token-id assumptions on generation models.
    if architecture != _BGE_M3_ARCHITECTURE:
        return PoolingModelMetadata(architecture=architecture)

    additional_special_token_ids = _load_special_token_ids(
        model_path if isinstance(model_ref, str) and model_ref else None
    )
    return PoolingModelMetadata(
        architecture=architecture,
        hidden_size=_optional_positive_int(config.get("hidden_size"), "hidden_size"),
        bos_token_id=_optional_token_id(config.get("bos_token_id"), "bos_token_id"),
        eos_token_id=_optional_token_id(config.get("eos_token_id"), "eos_token_id"),
        pad_token_id=_optional_token_id(config.get("pad_token_id"), "pad_token_id"),
        unk_token_id=_optional_token_id(config.get("unk_token_id"), "unk_token_id"),
        additional_special_token_ids=additional_special_token_ids,
    )


def parse_embedding_output_spec(payload: dict[str, Any]) -> EmbeddingOutputSpec:
    pooling_params = payload.get("pooling_params") or {}
    if not isinstance(pooling_params, dict):
        raise ValueError("embedding_request.pooling_params must be an object")

    dimensions = _parse_dimensions(pooling_params.get("dimensions"))
    explicit_output_types = "output_types" in payload
    raw_output_types = payload.get("output_types", ["dense"])
    if not isinstance(raw_output_types, list) or not raw_output_types:
        raise ValueError("embedding_request.output_types must be a non-empty array")

    output_types: list[str] = []
    for value in raw_output_types:
        if not isinstance(value, str):
            raise ValueError("embedding_request.output_types must contain strings")
        normalized = value.strip().lower()
        if normalized not in output_types:
            output_types.append(normalized)

    unknown = set(output_types) - _OUTPUT_TYPES
    if unknown:
        raise ValueError(
            "unsupported embedding output types: " + ", ".join(sorted(unknown))
        )

    if payload.get("sparse_format", "indices_values") != "indices_values":
        raise ValueError("only sparse_format=indices_values is supported")

    sparse_top_k = _optional_positive_int(payload.get("sparse_top_k"), "sparse_top_k")
    if dimensions is not None and "dense" not in output_types:
        raise ValueError("dimensions requires dense output")
    if sparse_top_k is not None and "sparse" not in output_types:
        raise ValueError("sparse_top_k requires sparse output")

    output_set = set(output_types)
    if output_set == {"dense"}:
        task = "embed"
    elif output_set == {"sparse"}:
        task = "token_classify"
    elif output_set == {"dense", "sparse"}:
        task = "embed&token_classify"
    else:  # Defensive guard; the validation above should make this unreachable.
        raise ValueError("unable to resolve embedding pooling task")

    return EmbeddingOutputSpec(
        output_types=tuple(output_types),
        task=task,
        dimensions=dimensions,
        sparse_top_k=sparse_top_k,
        explicit_output_types=explicit_output_types,
    )


def serialize_pooling_output(
    data: Sequence[float],
    prompt_token_ids: Sequence[int],
    spec: EmbeddingOutputSpec,
    metadata: PoolingModelMetadata,
) -> list[float] | dict[str, Any]:
    values = [float(value) for value in data]
    result: dict[str, Any] = {}

    if spec.includes_sparse and not metadata.supports_bge_m3_sparse:
        architecture = metadata.architecture or "unknown"
        raise ValueError(
            "sparse embeddings require vLLM architecture "
            f"{_BGE_M3_ARCHITECTURE}; resolved architecture is {architecture}"
        )

    sparse_token_ids: list[int] = []
    if spec.includes_sparse:
        sparse_token_ids = _sparse_token_ids(prompt_token_ids, metadata)

    if spec.includes_dense and spec.includes_sparse:
        dense_size = len(values) - len(sparse_token_ids)
        expected_dense_size = spec.dimensions or metadata.hidden_size
        if dense_size <= 0:
            raise ValueError("BGE-M3 hybrid pooling output does not contain dense data")
        if expected_dense_size is not None and dense_size != expected_dense_size:
            raise ValueError(
                "BGE-M3 hybrid pooling output has an unexpected dense size: "
                f"expected {expected_dense_size}, received {dense_size}"
            )
        result["dense"] = values[:dense_size]
        sparse_values = values[dense_size:]
    elif spec.includes_dense:
        if spec.dimensions is not None and len(values) != spec.dimensions:
            raise ValueError(
                "dense pooling output has an unexpected size: "
                f"expected {spec.dimensions}, received {len(values)}"
            )
        result["dense"] = values
        sparse_values = []
    else:
        sparse_values = values

    if spec.includes_sparse:
        result["sparse"] = build_sparse_embedding(
            sparse_token_ids,
            sparse_values,
            metadata.special_token_ids,
            spec.sparse_top_k,
        )

    if not spec.explicit_output_types:
        return result["dense"]
    return result


def build_sparse_embedding(
    token_ids: Sequence[int],
    weights: Sequence[float],
    special_token_ids: frozenset[int],
    sparse_top_k: int | None,
) -> dict[str, list[int] | list[float]]:
    if len(token_ids) != len(weights):
        raise ValueError(
            "BGE-M3 sparse pooling output length does not match prompt tokens: "
            f"{len(weights)} weights for {len(token_ids)} tokens"
        )

    best_weights: dict[int, float] = {}
    for raw_token_id, raw_weight in zip(token_ids, weights):
        token_id = int(raw_token_id)
        weight = float(raw_weight)
        if token_id in special_token_ids or weight <= 0 or not math.isfinite(weight):
            continue
        best_weights[token_id] = max(best_weights.get(token_id, 0.0), weight)

    pairs = list(best_weights.items())
    if sparse_top_k is not None and len(pairs) > sparse_top_k:
        pairs = heapq.nlargest(sparse_top_k, pairs, key=lambda pair: pair[1])
    pairs.sort(key=lambda pair: pair[0])
    return {
        "indices": [token_id for token_id, _ in pairs],
        "values": [weight for _, weight in pairs],
    }


def _sparse_token_ids(
    prompt_token_ids: Sequence[int],
    metadata: PoolingModelMetadata,
) -> list[int]:
    token_ids = [int(token_id) for token_id in prompt_token_ids]
    if token_ids and token_ids[0] == metadata.bos_token_id:
        token_ids = token_ids[1:]
    if token_ids and token_ids[-1] == metadata.eos_token_id:
        token_ids = token_ids[:-1]
    return token_ids


def _parse_dimensions(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        if len(value) != 1:
            raise ValueError("pooling_params.dimensions must contain one value")
        value = value[0]
    return _optional_positive_int(value, "pooling_params.dimensions")


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _optional_token_id(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _load_special_token_ids(model_path: Path | None) -> frozenset[int]:
    if model_path is None:
        return frozenset()
    tokenizer_path = model_path / "tokenizer.json"
    if not tokenizer_path.is_file():
        return frozenset()
    try:
        tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Unable to read tokenizer metadata from {tokenizer_path}: {exc}"
        ) from exc
    if not isinstance(tokenizer, dict):
        raise ValueError(f"Tokenizer config must be an object: {tokenizer_path}")

    token_ids: set[int] = set()
    for token in tokenizer.get("added_tokens") or []:
        if not isinstance(token, dict) or token.get("special") is not True:
            continue
        token_id = _optional_token_id(token.get("id"), "tokenizer special token id")
        if token_id is not None:
            token_ids.add(token_id)
    return frozenset(token_ids)
