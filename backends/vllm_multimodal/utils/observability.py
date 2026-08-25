# Copyright 2026 Raytorin and Triton OpenAI Gateway contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
import re
import time
from typing import Any


CEF_VENDOR = "ML Platform AI"
CEF_PRODUCT = "Triton vLLM Multimodal Backend"
BACKEND_VERSION = "0.3.0"
CEF_VERSION = BACKEND_VERSION


def get_triton_request_id(request: Any) -> str:
    try:
        value = request.request_id()
    except (AttributeError, TypeError, RuntimeError):
        return ""
    return str(value or "")


def log_event(logger: Any, event: str, level: str = "info", **fields: Any) -> None:
    level = level.lower()
    payload = {
        "component": "vllm_multimodal_backend",
        "event": event,
        "timestamp_unix_ms": int(time.time() * 1000),
        **{key: value for key, value in fields.items() if value is not None},
    }
    log_format = os.environ.get("LOG_FORMAT", "json").lower()
    if log_format == "json":
        message = json.dumps(
            {"level": level.upper(), **payload},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    elif log_format == "text":
        message = " ".join(
            f"{key}={_text_value(value)}" for key, value in payload.items()
        )
    else:
        extension = " ".join(
            f"{_cef_key(key)}={_cef_value(value)}" for key, value in payload.items()
        )
        message = (
            f"CEF:0|{_cef_header(CEF_VENDOR)}|{_cef_header(CEF_PRODUCT)}|"
            f"{CEF_VERSION}|{_cef_header(event)}|{_cef_header(event)}|"
            f"{_cef_severity(level)}|{extension}"
        )
    method = {
        "debug": "log_verbose",
        "warn": "log_warn",
        "warning": "log_warn",
        "error": "log_error",
        "critical": "log_error",
    }.get(level, "log_info")
    getattr(logger, method)(message)


def _text_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value).replace("\n", "\\n")


def _cef_header(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n")


def _cef_key(value: Any) -> str:
    key = re.sub(r"[^A-Za-z0-9_.]", "_", str(value))
    return key or "field"


def _cef_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    return (
        rendered
        .replace("\\", "\\\\")
        .replace("=", "\\=")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def _cef_severity(level: str) -> int:
    return {
        "debug": 1,
        "info": 3,
        "warn": 6,
        "warning": 6,
        "error": 8,
        "critical": 10,
    }.get(level, 3)
