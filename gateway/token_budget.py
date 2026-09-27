# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Resolve explicit/default output budgets against server and runtime limits."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from fastapi import HTTPException


@dataclass(frozen=True)
class GenerationLimits:
    default: int
    cap: int
    context_window: int
    source: str


@dataclass(frozen=True)
class TokenBudget:
    requested: int | None
    effective: int
    default: int
    cap: int
    context_window: int
    prompt_tokens: int
    media_tokens: int
    safety_margin: int
    source: str

    def headers(self) -> dict[str, str]:
        return {"X-Output-Token-Limit": str(self.effective),
                "X-Output-Token-Limit-Source": self.source,
                "X-Token-Usage-Source": "estimated"}


def requested_output_limit(request: Any) -> int | None:
    modern, legacy = request.max_completion_tokens, request.max_tokens
    if modern is not None and legacy is not None and modern != legacy:
        raise HTTPException(400, "max_tokens and max_completion_tokens must agree")
    return modern if modern is not None else legacy


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise HTTPException(500, f"Invalid generation setting {name}: expected a positive integer")
    return value


def _env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError:
        raise HTTPException(500, f"Invalid generation setting {name}") from None
    return _positive(value, name)


def _json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        raise HTTPException(500, f"Invalid model configuration: {path.name}") from None
    if not isinstance(data, dict):
        raise HTTPException(500, f"Invalid model configuration: {path.name}")
    return data


def load_generation_limits(model_path: Path, *, thinking: bool) -> GenerationLimits:
    config = _json(model_path / "gateway.json").get("generation", {})
    if not isinstance(config, dict):
        raise HTTPException(500, "generation must be an object")
    global_cap = _env("GATEWAY_MAX_OUTPUT_TOKENS", 32768)
    cap = _positive(config.get("max_output_tokens", global_cap), "max_output_tokens")
    if cap > global_cap:
        raise HTTPException(500, "Model max_output_tokens exceeds GATEWAY_MAX_OUTPUT_TOKENS")
    regular = _positive(config.get("default_output_tokens",
        _env("GATEWAY_DEFAULT_OUTPUT_TOKENS", 4096)), "default_output_tokens")
    reasoning = _positive(config.get("reasoning_default_output_tokens",
        _env("GATEWAY_REASONING_DEFAULT_OUTPUT_TOKENS", 8192)), "reasoning_default_output_tokens")
    # A gateway override is an explicit conservative bound, never an expansion
    # of a known runtime limit. Tokenizer sentinel values are deliberately unused.
    runtime = _json(model_path / "model.json").get("max_model_len")
    override = config.get("context_window")
    if type(runtime) is int and runtime > 0:
        context = runtime
        if override is not None:
            context = min(context, _positive(override, "context_window"))
    elif override is not None:
        context = _positive(override, "context_window")
    else:
        raise HTTPException(500, "Unknown runtime context window: set model.json max_model_len "
                            "or generation.context_window in gateway.json")
    field = "reasoning_default_output_tokens" if thinking else "default_output_tokens"
    return GenerationLimits(reasoning if thinking else regular, cap, context,
                            "model" if field in config else "global")


def choose_budget(limits: GenerationLimits, requested: int | None, *,
                  prompt_tokens: int, media_tokens: int, safety_margin: int) -> TokenBudget:
    if requested is not None and requested > limits.cap:
        raise HTTPException(400, f"Requested output limit {requested} exceeds maximum {limits.cap}")
    available = limits.context_window - prompt_tokens - media_tokens - safety_margin
    desired = requested if requested is not None else limits.default
    effective = desired if requested is not None else min(desired, limits.cap, available)
    if effective <= 0 or effective > available:
        raise HTTPException(400, "Input and requested output exceed the model context window")
    return TokenBudget(requested, effective, limits.default, limits.cap,
                       limits.context_window, prompt_tokens, media_tokens, safety_margin,
                       "request" if requested is not None else limits.source)
