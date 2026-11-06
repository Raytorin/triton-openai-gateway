# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Validate the boundary to vLLM's native thinking-budget sampler.

This module is also copied into the stock Triton vLLM backend by the image
build. Keep it independent of gateway modules and eager native imports.
"""
from functools import wraps


def normalize_engine_reasoning_config(engine_args):
    """Construct native config before EngineArgs accesses its attributes."""
    configured = getattr(engine_args, "reasoning_config", None)
    if isinstance(configured, dict):
        from vllm.config import ReasoningConfig
        engine_args.reasoning_config = ReasoningConfig(**configured)


def validate_sampling_budget(parameters, supported_fields):
    """Check raw JSON before the legacy sampler parser can coerce its value."""
    budget = parameters.get("thinking_token_budget")
    if budget is None:
        return
    if type(budget) is not int or budget < 0:
        raise ValueError("thinking_token_budget must be a nonnegative integer")
    if "thinking_token_budget" not in supported_fields:
        raise ValueError("This vLLM runtime does not support thinking_token_budget")


def _prompt_is_thinking(prompt, reasoning, start_ids, end_ids):
    """Mirror native last-boundary detection without copying a long prompt."""
    if isinstance(prompt, dict):
        token_ids = prompt.get("prompt_token_ids")
        if isinstance(token_ids, (list, tuple)):
            # One reverse pass; normally the active opening is at the end.
            def matches(index, sequence):
                return (index + len(sequence) <= len(token_ids)
                        and all(token_ids[index + offset] == token
                                for offset, token in enumerate(sequence)))
            for index in range(len(token_ids) - 1, -1, -1):
                token_id = token_ids[index]
                if token_id == end_ids[0] and matches(index, end_ids):
                    return False
                if token_id == start_ids[0] and matches(index, start_ids):
                    return True
            return False
        prompt = prompt.get("prompt")
    if isinstance(prompt, str):
        start = getattr(reasoning, "reasoning_start_str", None)
        end = getattr(reasoning, "reasoning_end_str", None)
        if isinstance(start, str) and start and isinstance(end, str) and end:
            return prompt.rfind(start) > prompt.rfind(end)
    # Unknown formats cannot establish a prefilled opening; reserve its cost.
    return False


def validate_thinking_budget(params, engine, *, prompt=None):
    """Never submit a budget the running engine cannot enforce."""
    budget = getattr(params, "thinking_token_budget", None)
    if budget is None:
        return
    if type(budget) is not int or budget < 0:
        raise ValueError("thinking_token_budget must be a nonnegative integer")
    native_config = getattr(engine, "vllm_config", None)
    reasoning = getattr(native_config, "reasoning_config", None)
    start_ids = getattr(reasoning, "reasoning_start_token_ids", None)
    end_ids = getattr(reasoning, "reasoning_end_token_ids", None)
    if not getattr(reasoning, "enabled", False) or not start_ids or not end_ids:
        raise ValueError(
            "thinking_token_budget requires an initialized native reasoning_config "
            "with reasoning_start_str and reasoning_end_str in model.json")
    opening_tokens = (0 if _prompt_is_thinking(prompt, reasoning, start_ids, end_ids)
                      else len(start_ids))
    maximum = getattr(params, "max_tokens", None)
    if maximum is not None and budget + opening_tokens + len(end_ids) >= maximum:
        raise ValueError(
            "thinking_token_budget and reasoning boundary tokens must leave "
            "room for a final answer within max_tokens")


def guard_engine_thinking_budget(engine):
    """Guard one owned engine instance, preserving its identity and lifecycle."""
    original = getattr(engine, "generate", None)
    if not callable(original) or getattr(original, "_triton_thinking_budget_guard", False):
        return engine

    @wraps(original)
    def generate(prompt, sampling_params, *args, **kwargs):
        validate_thinking_budget(sampling_params, engine, prompt=prompt)
        return original(prompt, sampling_params, *args, **kwargs)

    generate._triton_thinking_budget_guard = True
    engine.generate = generate
    return engine
