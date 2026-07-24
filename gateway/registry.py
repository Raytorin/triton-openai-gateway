# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from functools import lru_cache
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any

from fastapi import HTTPException
from .metrics import TOKENIZER_LOAD_DURATION, TOKENIZER_LOADS
from .settings import MODELS_ACTIVE_DIR, TRUST_REMOTE_CODE, logger
from .tool_parsers import detect_model_tool_parser


@lru_cache(maxsize=256)
def _read_backend(config_path: str, modified_ns: int) -> str | None:
    # modified_ns is part of the cache key, so replacing config.pbtxt
    # invalidates the cached backend without explicit cache management.
    del modified_ns
    text = Path(config_path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'^\s*backend\s*:\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        match = re.search(r"^\s*backend\s*:\s*([^\s\"]+)", text, re.MULTILINE)
    return match.group(1).strip().lower() if match else None


@lru_cache(maxsize=256)
def _read_model_capabilities(
    model_json_path: str,
    modified_ns: int,
    backend: str,
) -> frozenset[str] | None:
    # modified_ns and backend are cache keys. This keeps capability detection
    # current when the watcher replaces either the model config or backend.
    del modified_ns
    try:
        data = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    runner = str(data.get("runner") or "").strip().lower()
    task = str(data.get("task") or "").strip().lower()
    convert = str(data.get("convert") or "").strip().lower()

    if convert in {"embed", "embedding"} or task in {"embed", "embedding"}:
        return frozenset({"embeddings"})
    if runner == "pooling":
        # The gateway currently exposes vLLM pooling through /v1/embeddings.
        return frozenset({"embeddings"})
    if backend in {"vllm", "vllm_multimodal"}:
        return frozenset({"chat"})
    return None


class ModelRegistry:
    def __init__(self):
        self._cache: dict[str, tuple[str, Any]] = {}
        self._lock = Lock()

    def list_models(self) -> list[str]:
        if not MODELS_ACTIVE_DIR.exists():
            return []
        return sorted(entry.name for entry in MODELS_ACTIVE_DIR.iterdir() if entry.exists())

    def resolve(self, model_name: str) -> Path:
        link_path = MODELS_ACTIVE_DIR / model_name
        if not link_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found in {MODELS_ACTIVE_DIR}",
            )

        resolved = link_path.resolve()
        if not resolved.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Resolved path for model '{model_name}' does not exist: {resolved}",
            )
        return resolved

    def get_backend(self, model_name: str) -> str | None:
        model_path = self.resolve(model_name)
        config_path = model_path.parent / "config.pbtxt"
        if not config_path.is_file():
            return None

        return _read_backend(str(config_path), config_path.stat().st_mtime_ns)

    def get_capabilities(
        self,
        model_name: str,
        model_path: Path | None = None,
    ) -> frozenset[str] | None:
        model_path = model_path or self.resolve(model_name)
        backend = self.get_backend(model_name)
        model_json = model_path / "model.json"
        if not model_json.is_file() or backend is None:
            return None
        return _read_model_capabilities(
            str(model_json),
            model_json.stat().st_mtime_ns,
            backend,
        )

    def validate_route(
        self,
        model_name: str,
        route: str,
        model_path: Path | None = None,
    ) -> None:
        required_capability = "chat" if route in {"chat", "media"} else route
        capabilities = self.get_capabilities(model_name, model_path)
        if capabilities is None or required_capability in capabilities:
            return

        if capabilities == frozenset({"embeddings"}) and required_capability == "chat":
            detail = (
                f"Model '{model_name}' is configured for embeddings and cannot be used "
                "with /v1/chat/completions. In a RAG pipeline, call /v1/embeddings "
                "with this model for retrieval, then call /v1/chat/completions with "
                "a generative model for the final answer."
            )
        elif capabilities == frozenset({"chat"}) and required_capability == "embeddings":
            detail = (
                f"Model '{model_name}' is configured for text generation and cannot be "
                "used with /v1/embeddings. Select an embedding model for this request."
            )
        else:
            supported = ", ".join(sorted(capabilities))
            detail = (
                f"Model '{model_name}' does not support the '{required_capability}' "
                f"operation; supported operations: {supported}."
            )
        raise HTTPException(status_code=400, detail=detail)

    def get_tool_parser(self, model_name: str) -> str | None:
        return detect_model_tool_parser(self.resolve(model_name))

    def get_tokenizer(self, model_name: str):
        model_path = self.resolve(model_name)
        tokenizer_path = self.resolve_tokenizer_path(model_path)
        tokenizer_path_str = str(tokenizer_path)

        with self._lock:
            cached = self._cache.get(model_name)
            if cached and cached[0] == tokenizer_path_str:
                return cached[1], model_path

            # Transformers 5 imports PyTorch eagerly and noticeably delays
            # Uvicorn startup. Import it only when a tokenizer must be loaded.
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer for model '%s' from %s", model_name, tokenizer_path)
            started_at = time.monotonic()
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path_str,
                    trust_remote_code=TRUST_REMOTE_CODE,
                )
            except BaseException:
                TOKENIZER_LOADS.labels(model_name, "error").inc()
                raise
            else:
                TOKENIZER_LOADS.labels(model_name, "success").inc()
            finally:
                TOKENIZER_LOAD_DURATION.labels(model_name).observe(
                    time.monotonic() - started_at
                )
            self._cache[model_name] = (tokenizer_path_str, tokenizer)
            return tokenizer, model_path

    async def get_tokenizer_async(self, model_name: str):
        # Loading a Hugging Face tokenizer performs filesystem work and may
        # import model-specific Python. Keep it outside the ASGI event loop.
        return await asyncio.to_thread(self.get_tokenizer, model_name)

    async def preload_tokenizers(self) -> None:
        models = self.list_models()
        if not models:
            return
        results = await asyncio.gather(
            *(self.get_tokenizer_async(model_name) for model_name in models),
            return_exceptions=True,
        )
        for model_name, result in zip(models, results):
            if isinstance(result, BaseException):
                logger.warning(
                    "Tokenizer preload skipped for model '%s': %s",
                    model_name,
                    result,
                )

    def resolve_tokenizer_path(self, model_path: Path) -> Path:
        model_json = model_path / "model.json"
        if model_json.is_file():
            try:
                data = json.loads(model_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = None

            if isinstance(data, dict):
                tokenizer = data.get("tokenizer")
                if isinstance(tokenizer, str) and tokenizer.strip():
                    tokenizer_path = Path(tokenizer.strip())
                    if tokenizer_path.exists():
                        return tokenizer_path

        for candidate in (model_path / "tokenizer", model_path.parent / "tokenizer"):
            if candidate.exists():
                return candidate

        return model_path
