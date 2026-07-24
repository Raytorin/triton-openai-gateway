# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path
from urllib.parse import urlparse


from .observability import configure_logging


configure_logging()
logger = logging.getLogger("triton-chat-gateway")

TRITON_BASE_URL = os.environ.get("TRITON_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _default_triton_grpc_url() -> str:
    parsed = urlparse(TRITON_BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    if port == 8000:
        port = 8001
    return f"{host}:{port}"


def _normalize_grpc_url(value: str) -> str:
    value = value.rstrip("/")
    if "://" not in value:
        return value

    parsed = urlparse(value)
    if parsed.netloc:
        return parsed.netloc
    return value


TRITON_GRPC_URL = _normalize_grpc_url(
    os.environ.get("TRITON_GRPC_URL", _default_triton_grpc_url())
)
MODELS_ACTIVE_DIR = Path(os.environ.get("MODELS_ACTIVE_DIR", "/tmp/models-active"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "600"))
EMBEDDING_MAX_CONCURRENCY = max(
    int(os.environ.get("GATEWAY_EMBEDDING_MAX_CONCURRENCY", "8")),
    1,
)
TOKENIZER_PRELOAD = os.environ.get("TOKENIZER_PRELOAD", "true").lower() in {
    "1",
    "true",
    "yes",
}
TRUST_REMOTE_CODE = os.environ.get("TOKENIZER_TRUST_REMOTE_CODE", "true").lower() in {
    "1",
    "true",
    "yes",
}
TRITON_DEFAULT_STOP_SEQUENCE = "<|im_end|>"
STREAM_HOLDBACK_CHARS = int(os.environ.get("STREAM_HOLDBACK_CHARS", "48"))
LOG_PROMPT_PREVIEW = os.environ.get("LOG_PROMPT_PREVIEW", "false").lower() in {
    "1",
    "true",
    "yes",
}
