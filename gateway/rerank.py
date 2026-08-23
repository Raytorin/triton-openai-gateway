# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Iterator

from fastapi import HTTPException

from .rerank_strategies import (
    RerankStrategy,
    apply_rerank_strategy,
)
from .schemas import RerankRequest


@dataclass(frozen=True)
class RerankExecutionSettings:
    max_documents_per_request: int = 256
    default_batch_size: int = 4
    max_batch_size: int = 8
    max_batch_tokens: int = 8192
    default_max_length: int = 512
    max_length: int = 8192


@dataclass(frozen=True)
class RerankExecutionPlan:
    batch_size: int
    max_length: int
    batch_count: int


def build_rerank_documents(request: RerankRequest) -> list[str]:
    if not request.documents:
        raise HTTPException(status_code=400, detail="Rerank documents must not be empty")

    documents: list[str] = []
    for document in request.documents:
        if isinstance(document, str):
            documents.append(document)
            continue

        if isinstance(document, dict):
            text = document.get("text")
            if isinstance(text, str):
                documents.append(text)
                continue

            content = document.get("content")
            if isinstance(content, str):
                documents.append(content)
                continue

        documents.append(str(document))

    return documents


def plan_rerank_execution(
    request: RerankRequest,
    documents: list[str],
    model_path: Path,
) -> RerankExecutionPlan:
    settings = load_rerank_execution_settings(model_path)
    if len(documents) > settings.max_documents_per_request:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Rerank request contains {len(documents)} documents; "
                f"configured limit is {settings.max_documents_per_request}"
            ),
        )

    max_length = (
        request.max_length
        if request.max_length is not None
        else settings.default_max_length
    )
    if max_length <= 0:
        raise HTTPException(status_code=400, detail="max_length must be greater than 0")
    if max_length > settings.max_length:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_length={max_length} exceeds configured rerank limit "
                f"{settings.max_length}"
            ),
        )

    requested_batch_size = (
        request.batch_size
        if request.batch_size is not None
        else settings.default_batch_size
    )
    if requested_batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be greater than 0")
    if requested_batch_size > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"batch_size={requested_batch_size} exceeds configured rerank limit "
                f"{settings.max_batch_size}"
            ),
        )

    token_limited_batch_size = settings.max_batch_tokens // max_length
    if token_limited_batch_size <= 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_length={max_length} exceeds configured rerank batch token "
                f"budget {settings.max_batch_tokens}"
            ),
        )
    batch_size = min(requested_batch_size, token_limited_batch_size)
    batch_count = (len(documents) + batch_size - 1) // batch_size
    return RerankExecutionPlan(
        batch_size=batch_size,
        max_length=max_length,
        batch_count=batch_count,
    )


def iter_rerank_batches(
    documents: list[str],
    plan: RerankExecutionPlan,
) -> Iterator[list[str]]:
    for start in range(0, len(documents), plan.batch_size):
        yield documents[start : start + plan.batch_size]


def load_rerank_execution_settings(model_path: Path) -> RerankExecutionSettings:
    config_path = model_path / "gateway.json"
    modified_ns = config_path.stat().st_mtime_ns if config_path.is_file() else 0
    return _read_rerank_execution_settings(str(config_path), modified_ns)


@lru_cache(maxsize=256)
def _read_rerank_execution_settings(
    config_path: str,
    modified_ns: int,
) -> RerankExecutionSettings:
    del modified_ns
    path = Path(config_path)
    if not path.is_file():
        return RerankExecutionSettings()

    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read rerank gateway.json: {exc}",
        ) from exc
    if not isinstance(root, dict):
        raise HTTPException(status_code=500, detail="gateway.json must contain a JSON object")

    rerank_config = root.get("rerank") or {}
    if not isinstance(rerank_config, dict):
        raise HTTPException(status_code=500, detail="gateway.json rerank must be an object")
    execution = rerank_config.get("execution") or {}
    if not isinstance(execution, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json rerank.execution must be an object",
        )

    defaults = RerankExecutionSettings()
    settings = RerankExecutionSettings(
        max_documents_per_request=_positive_config_int(
            execution,
            "max_documents_per_request",
            defaults.max_documents_per_request,
        ),
        default_batch_size=_positive_config_int(
            execution,
            "default_batch_size",
            defaults.default_batch_size,
        ),
        max_batch_size=_positive_config_int(
            execution,
            "max_batch_size",
            defaults.max_batch_size,
        ),
        max_batch_tokens=_positive_config_int(
            execution,
            "max_batch_tokens",
            defaults.max_batch_tokens,
        ),
        default_max_length=_positive_config_int(
            execution,
            "default_max_length",
            defaults.default_max_length,
        ),
        max_length=_positive_config_int(
            execution,
            "max_length",
            defaults.max_length,
        ),
    )
    if settings.default_batch_size > settings.max_batch_size:
        raise HTTPException(
            status_code=500,
            detail=(
                "gateway.json rerank.execution.default_batch_size must not "
                "exceed max_batch_size"
            ),
        )
    if settings.default_max_length > settings.max_length:
        raise HTTPException(
            status_code=500,
            detail=(
                "gateway.json rerank.execution.default_max_length must not "
                "exceed max_length"
            ),
        )
    return settings


def _positive_config_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HTTPException(
            status_code=500,
            detail=f"gateway.json rerank.execution.{key} must be a positive integer",
        )
    return value


def build_rerank_response(
    request: RerankRequest,
    documents: list[str],
    scores: list[float],
    strategy: RerankStrategy | None = None,
    execution: RerankExecutionPlan | None = None,
) -> dict[str, Any]:
    if len(scores) != len(documents):
        raise HTTPException(
            status_code=502,
            detail=(
                "Rerank backend returned an unexpected number of scores: "
                f"{len(scores)} for {len(documents)} documents"
            ),
        )

    ranked = sorted(
        (
            {
                "index": index,
                "relevance_score": float(score),
                "document": {"text": documents[index]},
            }
            for index, score in enumerate(scores)
        ),
        key=lambda item: item["relevance_score"],
        reverse=True,
    )

    if strategy is not None:
        ranked = apply_rerank_strategy(strategy, ranked, request.documents)
    elif request.top_n is not None:
        if request.top_n <= 0:
            raise HTTPException(status_code=400, detail="top_n must be greater than 0")
        ranked = ranked[: request.top_n]

    if not request.return_documents:
        for item in ranked:
            item.pop("document", None)

    meta: dict[str, Any] = {
        "api_version": {
            "version": "1",
        }
    }
    if strategy is not None:
        meta["selection"] = {
            "strategy": strategy.name,
            "method": strategy.method,
            "source": strategy.source,
            "version": strategy.version,
        }
    if execution is not None:
        meta["execution"] = {
            "batch_size": execution.batch_size,
            "batch_count": execution.batch_count,
            "max_length": execution.max_length,
        }

    return {
        "id": "rerank",
        "results": ranked,
        "meta": meta,
    }
