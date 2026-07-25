# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from fastapi import HTTPException

from .schemas import RerankRequest, RerankSelectionRequest
from .settings import logger


_STRATEGY_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SQL_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


RankedItem = dict[str, Any]
StrategyHandler = Callable[
    [list[RankedItem], list[Any], Mapping[str, Any]],
    list[RankedItem],
]


@dataclass(frozen=True)
class RerankMethod:
    name: str
    handler: StrategyHandler
    allowed_parameters: frozenset[str]
    required_parameters: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RerankStrategy:
    name: str
    method: str
    parameters: dict[str, Any]
    allowed_request_parameters: frozenset[str]
    source: str
    version: str


class RerankMethodRegistry:
    def __init__(self) -> None:
        self._methods: dict[str, RerankMethod] = {}

    def register(
        self,
        name: str,
        handler: StrategyHandler,
        *,
        allowed_parameters: set[str] | frozenset[str],
        required_parameters: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        normalized = _validated_name(name, "method", status_code=500)
        if normalized in self._methods:
            raise ValueError(f"Rerank method '{normalized}' is already registered")
        allowed = frozenset(allowed_parameters)
        required = frozenset(required_parameters)
        if not required.issubset(allowed):
            raise ValueError(
                f"Required parameters for rerank method '{normalized}' must be allowed"
            )
        self._methods[normalized] = RerankMethod(
            name=normalized,
            handler=handler,
            allowed_parameters=allowed,
            required_parameters=required,
        )

    def get(self, name: str) -> RerankMethod:
        normalized = _validated_name(name, "method", status_code=500)
        method = self._methods.get(normalized)
        if method is None:
            available = ", ".join(sorted(self._methods))
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Rerank strategy references unknown method '{normalized}'. "
                    f"Registered methods: {available}"
                ),
            )
        return method

    def names(self) -> frozenset[str]:
        return frozenset(self._methods)


METHODS = RerankMethodRegistry()


def resolve_rerank_strategy(
    request: RerankRequest,
    model_path: Path,
) -> RerankStrategy:
    configured = _load_rerank_config(model_path)
    catalog = _builtin_strategies()
    _merge_configured_strategies(
        catalog,
        configured.get("strategies"),
        source="gateway_json",
    )
    _merge_database_strategies(
        catalog,
        configured.get("database"),
        request.model,
        model_path,
    )

    requested = _requested_selection(request)
    default_name = configured.get("default_strategy", "top_n")
    if not isinstance(default_name, str):
        raise HTTPException(
            status_code=500,
            detail="gateway.json rerank.default_strategy must be a string",
        )

    strategy_name = requested.strategy if requested is not None else default_name
    strategy_name = _validated_name(strategy_name, "strategy", status_code=400)
    strategy = catalog.get(strategy_name)
    if strategy is None:
        available = ", ".join(sorted(catalog))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown rerank strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            ),
        )

    method = METHODS.get(strategy.method)
    parameters = dict(strategy.parameters)
    request_parameters = requested.parameters if requested is not None else {}
    denied = sorted(
        key
        for key in request_parameters
        if key not in strategy.allowed_request_parameters
    )
    if denied:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Rerank strategy '{strategy.name}' does not allow request overrides: "
                f"{', '.join(denied)}"
            ),
        )
    parameters.update(request_parameters)

    # top_n is part of the established rerank API and always remains a final cap.
    if request.top_n is not None:
        parameters["top_n"] = request.top_n

    _validate_method_parameters(method, parameters)
    return RerankStrategy(
        name=strategy.name,
        method=strategy.method,
        parameters=parameters,
        allowed_request_parameters=strategy.allowed_request_parameters,
        source=strategy.source,
        version=strategy.version,
    )


def apply_rerank_strategy(
    strategy: RerankStrategy,
    ranked: list[RankedItem],
    source_documents: list[Any],
) -> list[RankedItem]:
    method = METHODS.get(strategy.method)
    return method.handler(list(ranked), source_documents, strategy.parameters)


def _requested_selection(request: RerankRequest) -> RerankSelectionRequest | None:
    if request.selection is not None and request.custom_top is not None:
        raise HTTPException(
            status_code=400,
            detail="Use either selection or custom_top, not both",
        )
    raw = request.selection if request.selection is not None else request.custom_top
    if raw is None:
        return None
    if isinstance(raw, str):
        return RerankSelectionRequest(strategy=raw)
    return raw


def _builtin_strategies() -> dict[str, RerankStrategy]:
    return {
        name: RerankStrategy(
            name=name,
            method=name,
            parameters={},
            allowed_request_parameters=METHODS.get(name).allowed_parameters,
            source="builtin",
            version="1",
        )
        for name in METHODS.names()
    }


def _merge_configured_strategies(
    catalog: dict[str, RerankStrategy],
    raw_strategies: Any,
    *,
    source: str,
) -> None:
    if raw_strategies is None:
        return
    if not isinstance(raw_strategies, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json rerank.strategies must be an object",
        )

    for raw_name, raw_definition in raw_strategies.items():
        name = _validated_name(raw_name, "strategy", status_code=500)
        if name in METHODS.names():
            raise HTTPException(
                status_code=500,
                detail=f"Configured rerank strategy cannot replace builtin '{name}'",
            )
        catalog[name] = _parse_strategy_definition(
            name,
            raw_definition,
            source=source,
        )


def _parse_strategy_definition(
    name: str,
    raw_definition: Any,
    *,
    source: str,
) -> RerankStrategy:
    if not isinstance(raw_definition, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Rerank strategy '{name}' must be an object",
        )

    method_name = raw_definition.get("method")
    if not isinstance(method_name, str):
        raise HTTPException(
            status_code=500,
            detail=f"Rerank strategy '{name}' must define a string method",
        )
    method = METHODS.get(method_name)

    parameters = raw_definition.get("parameters", {})
    if not isinstance(parameters, dict):
        raise HTTPException(
            status_code=500,
            detail=f"Rerank strategy '{name}' parameters must be an object",
        )

    allowed = raw_definition.get("allow_request_parameters", [])
    if not isinstance(allowed, list) or not all(
        isinstance(item, str) for item in allowed
    ):
        raise HTTPException(
            status_code=500,
            detail=(
                f"Rerank strategy '{name}' allow_request_parameters "
                "must be an array of strings"
            ),
        )
    unknown_allowed = sorted(set(allowed) - method.allowed_parameters)
    if unknown_allowed:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Rerank strategy '{name}' allows unsupported parameters: "
                f"{', '.join(unknown_allowed)}"
            ),
        )
    missing_from_config = sorted(
        method.required_parameters - set(parameters) - set(allowed)
    )
    if missing_from_config:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Rerank strategy '{name}' must configure or allow request "
                f"parameters: {', '.join(missing_from_config)}"
            ),
        )

    _validate_method_parameters(
        method,
        parameters,
        require_all=False,
        status_code=500,
    )
    return RerankStrategy(
        name=name,
        method=method.name,
        parameters=dict(parameters),
        allowed_request_parameters=frozenset(allowed),
        source=source,
        version=str(raw_definition.get("version") or "1"),
    )


def _load_rerank_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "gateway.json"
    modified_ns = config_path.stat().st_mtime_ns if config_path.is_file() else 0
    return _read_rerank_config(str(config_path), modified_ns)


@lru_cache(maxsize=256)
def _read_rerank_config(config_path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to read rerank configuration from {path}: {exc}",
        ) from exc
    configured = payload.get("rerank", {}) if isinstance(payload, dict) else {}
    if not isinstance(configured, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json rerank section must be an object",
        )
    return configured


def _merge_database_strategies(
    catalog: dict[str, RerankStrategy],
    raw_database: Any,
    model_name: str,
    model_path: Path,
) -> None:
    if raw_database is None:
        return
    if not isinstance(raw_database, dict):
        raise HTTPException(
            status_code=500,
            detail="gateway.json rerank.database must be an object",
        )
    if raw_database.get("enabled", True) is False:
        return

    driver = str(raw_database.get("driver") or "sqlite").strip().lower()
    if driver != "sqlite":
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unsupported rerank strategy database driver '{driver}'. "
                "Release 6 currently supports sqlite."
            ),
        )

    raw_path = raw_database.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise HTTPException(
            status_code=500,
            detail="rerank.database.path must be a non-empty string",
        )
    database_path = Path(raw_path.strip())
    if not database_path.is_absolute():
        database_path = model_path / database_path

    table = str(raw_database.get("table") or "rerank_strategies")
    if not _SQL_IDENTIFIER_RE.fullmatch(table):
        raise HTTPException(
            status_code=500,
            detail="rerank.database.table is not a valid SQL identifier",
        )

    required = bool(raw_database.get("required", False))
    refresh_seconds = _positive_float(
        raw_database.get("refresh_seconds", 30),
        "rerank.database.refresh_seconds",
    )
    if not database_path.is_file():
        _database_failure(
            required,
            f"Rerank strategy database does not exist: {database_path}",
        )
        return

    modified_ns = database_path.stat().st_mtime_ns
    wal_path = Path(f"{database_path}-wal")
    wal_modified_ns = wal_path.stat().st_mtime_ns if wal_path.is_file() else 0
    refresh_bucket = int(time.monotonic() // refresh_seconds)
    try:
        rows = _read_sqlite_strategies(
            str(database_path),
            table,
            model_name,
            modified_ns,
            wal_modified_ns,
            refresh_bucket,
        )
    except (OSError, sqlite3.Error, ValueError) as exc:
        _database_failure(
            required,
            f"Unable to load rerank strategies from {database_path}: {exc}",
        )
        return

    for row in rows:
        try:
            name = _validated_name(
                row["name"],
                "strategy",
                status_code=500,
            )
            if name in METHODS.names():
                _database_failure(
                    required,
                    f"Database strategy cannot replace builtin '{name}'",
                )
                continue
            catalog[name] = _parse_strategy_definition(
                name,
                {
                    "method": row["method"],
                    "parameters": json.loads(row["parameters_json"] or "{}"),
                    "allow_request_parameters": json.loads(
                        row["allowed_request_parameters_json"] or "[]"
                    ),
                    "version": row["version"],
                },
                source="database",
            )
        except (json.JSONDecodeError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            row_name = str(row.get("name") or "<invalid>")
            _database_failure(
                required,
                f"Invalid database rerank strategy '{row_name}': {detail}",
            )


@lru_cache(maxsize=512)
def _read_sqlite_strategies(
    database_path: str,
    table: str,
    model_name: str,
    modified_ns: int,
    wal_modified_ns: int,
    refresh_bucket: int,
) -> tuple[dict[str, Any], ...]:
    del modified_ns, wal_modified_ns, refresh_bucket
    uri = f"{Path(database_path).resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=2.0) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT
                name,
                method,
                parameters_json,
                allowed_request_parameters_json,
                version,
                model
            FROM {table}
            WHERE enabled = 1 AND (model = ? OR model = '*')
            ORDER BY CASE WHEN model = '*' THEN 0 ELSE 1 END
            """,
            (model_name,),
        ).fetchall()
    return tuple(dict(row) for row in rows)


def _database_failure(required: bool, detail: str) -> None:
    if required:
        raise HTTPException(status_code=503, detail=detail)
    logger.warning(
        detail,
        extra={
            "event": "rerank.strategy_database_unavailable",
        },
    )


def _validate_method_parameters(
    method: RerankMethod,
    parameters: Mapping[str, Any],
    *,
    require_all: bool = True,
    status_code: int = 400,
) -> None:
    unknown = sorted(set(parameters) - method.allowed_parameters)
    if unknown:
        raise HTTPException(
            status_code=status_code,
            detail=(
                f"Rerank method '{method.name}' received unsupported parameters: "
                f"{', '.join(unknown)}"
            ),
        )
    missing = sorted(method.required_parameters - set(parameters))
    if require_all and missing:
        raise HTTPException(
            status_code=status_code,
            detail=(
                f"Rerank method '{method.name}' requires parameters: "
                f"{', '.join(missing)}"
            ),
        )

    if "top_n" in parameters:
        top_n = parameters["top_n"]
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
            raise HTTPException(
                status_code=status_code,
                detail="top_n must be a positive integer",
            )
    if "score_threshold" in parameters:
        threshold = parameters["score_threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise HTTPException(
                status_code=status_code,
                detail="score_threshold must be a finite number",
            )
    if "max_similarity" in parameters:
        similarity = parameters["max_similarity"]
        if (
            isinstance(similarity, bool)
            or not isinstance(similarity, (int, float))
            or not 0 <= float(similarity) <= 1
        ):
            raise HTTPException(
                status_code=status_code,
                detail="max_similarity must be a number between 0 and 1",
            )
    if "filters" in parameters:
        filters = parameters["filters"]
        if not isinstance(filters, dict) or not all(
            isinstance(key, str) and key for key in filters
        ):
            raise HTTPException(
                status_code=status_code,
                detail="filters must be an object with non-empty string keys",
            )


def _top_n(
    ranked: list[RankedItem],
    _documents: list[Any],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    return _limit(ranked, parameters)


def _score_threshold(
    ranked: list[RankedItem],
    _documents: list[Any],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    threshold = float(parameters["score_threshold"])
    filtered = [
        item for item in ranked if float(item["relevance_score"]) >= threshold
    ]
    return _limit(filtered, parameters)


def _top_n_and_threshold(
    ranked: list[RankedItem],
    documents: list[Any],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    return _score_threshold(ranked, documents, parameters)


def _metadata_filter(
    ranked: list[RankedItem],
    documents: list[Any],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    filters = parameters["filters"]
    filtered = [
        item
        for item in ranked
        if _matches_filters(documents[int(item["index"])], filters)
    ]
    if "score_threshold" in parameters:
        threshold = float(parameters["score_threshold"])
        filtered = [
            item
            for item in filtered
            if float(item["relevance_score"]) >= threshold
        ]
    return _limit(filtered, parameters)


def _diversity(
    ranked: list[RankedItem],
    _documents: list[Any],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    if "score_threshold" in parameters:
        threshold = float(parameters["score_threshold"])
        ranked = [
            item
            for item in ranked
            if float(item["relevance_score"]) >= threshold
        ]

    max_similarity = float(parameters.get("max_similarity", 0.85))
    selected: list[RankedItem] = []
    selected_tokens: list[set[str]] = []
    top_n = parameters.get("top_n")
    for item in ranked:
        text = str((item.get("document") or {}).get("text") or "")
        tokens = set(_TOKEN_RE.findall(text.casefold()))
        if any(
            _jaccard_similarity(tokens, previous) > max_similarity
            for previous in selected_tokens
        ):
            continue
        selected.append(item)
        selected_tokens.append(tokens)
        if top_n is not None and len(selected) >= int(top_n):
            break
    return selected


def _limit(
    ranked: list[RankedItem],
    parameters: Mapping[str, Any],
) -> list[RankedItem]:
    top_n = parameters.get("top_n")
    return ranked if top_n is None else ranked[: int(top_n)]


def _matches_filters(document: Any, filters: Mapping[str, Any]) -> bool:
    if not isinstance(document, dict):
        return False
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return all(
        _matches_expected(_nested_value(metadata, key), expected)
        for key, expected in filters.items()
    )


def _nested_value(metadata: Mapping[str, Any], key: str) -> Any:
    current: Any = metadata
    for part in str(key).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _matches_expected(actual: Any, expected: Any) -> bool:
    expected_values = expected if isinstance(expected, list) else [expected]
    actual_values = actual if isinstance(actual, list) else [actual]
    return any(value in expected_values for value in actual_values)


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _positive_float(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise HTTPException(
            status_code=500,
            detail=f"{field} must be a positive number",
        )
    return float(value)


def _validated_name(value: Any, kind: str, *, status_code: int) -> str:
    if not isinstance(value, str) or not _STRATEGY_NAME_RE.fullmatch(value):
        raise HTTPException(
            status_code=status_code,
            detail=(
                f"Invalid rerank {kind} name. Use 1-128 letters, numbers, "
                "dots, underscores or hyphens."
            ),
        )
    return value


METHODS.register(
    "top_n",
    _top_n,
    allowed_parameters={"top_n"},
)
METHODS.register(
    "score_threshold",
    _score_threshold,
    allowed_parameters={"score_threshold", "top_n"},
    required_parameters={"score_threshold"},
)
METHODS.register(
    "top_n_and_threshold",
    _top_n_and_threshold,
    allowed_parameters={"score_threshold", "top_n"},
    required_parameters={"score_threshold", "top_n"},
)
METHODS.register(
    "metadata_filter",
    _metadata_filter,
    allowed_parameters={"filters", "score_threshold", "top_n"},
    required_parameters={"filters"},
)
METHODS.register(
    "diversity",
    _diversity,
    allowed_parameters={"max_similarity", "score_threshold", "top_n"},
)
