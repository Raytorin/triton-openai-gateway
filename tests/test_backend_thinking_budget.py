"""CPU contracts around the native vLLM thinking-budget boundary."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "backends/vllm_multimodal/utils/thinking_budget.py"


def helper():
    spec = importlib.util.spec_from_file_location("backend_thinking_budget_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engine(*, enabled=True, start=(10,), end=(11, 12)):
    calls = []
    reason = SimpleNamespace(enabled=enabled, reasoning_start_token_ids=list(start),
        reasoning_end_token_ids=list(end))
    native = SimpleNamespace(vllm_config=SimpleNamespace(reasoning_config=reason))
    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return iter(["native answer"])
    native.generate = generate
    return native, calls


def test_model_json_reasoning_dictionary_is_native_config_before_creation(monkeypatch):
    class ReasoningConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(ReasoningConfig=ReasoningConfig))
    config = {"reasoning_start_str":"<think>", "reasoning_end_str":"</think>"}
    args = SimpleNamespace(reasoning_config=config)
    helper().normalize_engine_reasoning_config(args)
    assert isinstance(args.reasoning_config, ReasoningConfig)
    assert args.reasoning_config.reasoning_start_str == "<think>"
    assert args.reasoning_config.reasoning_end_str == "</think>"
    assert config == {"reasoning_start_str":"<think>", "reasoning_end_str":"</think>"}


def test_absent_or_native_config_is_not_replaced():
    module = helper()
    for config in [None, object()]:
        args = SimpleNamespace(reasoning_config=config)
        module.normalize_engine_reasoning_config(args)
        assert args.reasoning_config is config
    module.normalize_engine_reasoning_config(SimpleNamespace())


@pytest.mark.parametrize("budget", [0, 128])
def test_valid_budget_reaches_native_generate_unchanged(budget):
    native, calls = engine()
    guarded = helper().guard_engine_thinking_budget(native)
    params = SimpleNamespace(thinking_token_budget=budget, max_tokens=4096)
    output = guarded.generate("prompt", params, "request-id", lora_request=None)
    assert guarded is native
    assert list(output) == ["native answer"]
    assert calls == [(("prompt", params, "request-id"), {"lora_request":None})]
    assert params.thinking_token_budget == budget


@pytest.mark.parametrize("change", ["none", "disabled", "no_start", "no_end"])
def test_budget_cannot_silently_run_without_native_reasoning_boundaries(change):
    native, calls = engine(enabled=change != "disabled", start=() if change == "no_start" else (10,),
                           end=() if change == "no_end" else (11,))
    if change == "none":
        native.vllm_config.reasoning_config = None
    guarded = helper().guard_engine_thinking_budget(native)
    with pytest.raises(ValueError, match="reasoning_config"):
        guarded.generate("prompt", SimpleNamespace(thinking_token_budget=128, max_tokens=4096), "request")
    assert calls == []


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "128"])
def test_invalid_direct_triton_budget_is_rejected_before_native_generate(budget):
    native, calls = engine()
    with pytest.raises(ValueError, match="nonnegative integer"):
        helper().guard_engine_thinking_budget(native).generate(
            "prompt", SimpleNamespace(thinking_token_budget=budget, max_tokens=4096), "request")
    assert calls == []


@pytest.mark.parametrize("output_limit", [128, 129, 130, 131])
def test_total_budget_reserves_native_closing_tokens_and_final_output(output_limit):
    native, calls = engine(end=(11, 12))
    with pytest.raises(ValueError, match="final answer"):
        helper().guard_engine_thinking_budget(native).generate(
            "prompt", SimpleNamespace(thinking_token_budget=128, max_tokens=output_limit), "request")
    assert calls == []


def test_smallest_total_budget_with_final_output_room_is_accepted():
    native, calls = engine(end=(11, 12))
    result = helper().guard_engine_thinking_budget(native).generate(
        "prompt", SimpleNamespace(thinking_token_budget=128, max_tokens=132), "request")
    assert list(result) == ["native answer"]
    assert len(calls) == 1


def test_requests_without_budget_preserve_unconfigured_native_generation():
    native, calls = engine(enabled=False, start=(), end=())
    params = SimpleNamespace(max_tokens=16)
    guarded = helper().guard_engine_thinking_budget(native)
    helper().guard_engine_thinking_budget(native)
    assert list(guarded.generate("prompt", params, "request")) == ["native answer"]
    assert calls == [(("prompt", params, "request"), {})]


def test_incomplete_runtime_does_not_break_engine_lifecycle():
    native = SimpleNamespace(shutdown=lambda: None)
    assert helper().guard_engine_thinking_budget(native) is native


def sampling_utility(monkeypatch, *, supports_budget=True):
    class SamplingParams:
        __annotations__ = {"max_tokens": int, **({"thinking_token_budget": int | None} if supports_budget else {})}
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    modules = {
        "vllm.engine.arg_utils": SimpleNamespace(AsyncEngineArgs=object),
        "vllm.engine.protocol": SimpleNamespace(EngineClient=object),
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParams, StructuredOutputsParams=object),
        "vllm.usage.usage_lib": SimpleNamespace(UsageContext=SimpleNamespace(OPENAI_API_SERVER="openai")),
        "vllm.v1.metrics.loggers": SimpleNamespace(StatLoggerFactory=object),
        "utils.thinking_budget": helper(),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("native_budget_sampling_utility", SOURCE.with_name("vllm_backend_utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TritonSamplingParams


@pytest.mark.parametrize("budget", [True, False, -1, 1.5, "128"])
def test_raw_triton_budget_is_not_coerced_into_a_supported_integer(monkeypatch, budget):
    import json
    messages = []
    params = sampling_utility(monkeypatch).from_dict(json.dumps({"thinking_token_budget":budget}),
                                                   SimpleNamespace(log_error=messages.append))
    assert params is None
    assert "nonnegative integer" in messages[0]


def test_old_sampling_runtime_does_not_silently_accept_unknown_budget(monkeypatch):
    messages = []
    params = sampling_utility(monkeypatch, supports_budget=False).from_dict(
        '{"thinking_token_budget":128}', SimpleNamespace(log_error=messages.append))
    assert params is None
    assert "does not support thinking_token_budget" in messages[0]


@pytest.mark.parametrize("budget", [0, 128])
def test_raw_triton_budget_reaches_sampling_params(monkeypatch, budget):
    import json
    params = sampling_utility(monkeypatch).from_dict(json.dumps({"thinking_token_budget":budget}),
        SimpleNamespace(log_error=lambda message: pytest.fail(message)))
    assert params.thinking_token_budget == budget
    assert type(params.thinking_token_budget) is int


def test_shared_factory_normalizes_config_before_native_engine_creation(monkeypatch):
    import asyncio
    from test_deepseek_lifecycle import shared_engine_factory
    native, calls = engine()
    native.reset_mm_cache = lambda: None
    async def reset():
        pass
    native.reset_mm_cache = reset
    native.shutdown = lambda: None
    class ReasoningConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    monkeypatch.setitem(sys.modules, "vllm.config", SimpleNamespace(ReasoningConfig=ReasoningConfig))
    monkeypatch.setitem(sys.modules, "utils.thinking_budget", helper())
    args = SimpleNamespace(reasoning_config={"reasoning_start_str":"<think>", "reasoning_end_str":"</think>"},
        enable_log_requests=False, aggregate_engine_logging=False, disable_log_stats=False)
    def create(**kwargs):
        assert isinstance(args.reasoning_config, ReasoningConfig)
        return native.vllm_config
    args.create_engine_config = create
    factory = shared_engine_factory(monkeypatch, native)
    async def use():
        async with factory(args, SimpleNamespace()) as client:
            assert client is native
    asyncio.run(use())


def test_shared_factory_installs_guard_before_publishing_engine(monkeypatch):
    import asyncio
    from test_deepseek_lifecycle import shared_engine_factory
    native, calls = engine(enabled=False, start=(), end=())
    async def reset():
        pass
    native.reset_mm_cache = reset
    native.shutdown = lambda: None
    monkeypatch.setitem(sys.modules, "utils.thinking_budget", helper())
    args = SimpleNamespace(create_engine_config=lambda **kwargs:native.vllm_config,
        enable_log_requests=False, aggregate_engine_logging=False, disable_log_stats=False)
    factory = shared_engine_factory(monkeypatch, native)
    async def use():
        async with factory(args, SimpleNamespace()) as client:
            with pytest.raises(ValueError, match="reasoning_config"):
                client.generate("prompt", SimpleNamespace(thinking_token_budget=128, max_tokens=4096), "request")
    asyncio.run(use())
    assert calls == []


@pytest.mark.parametrize("prompt", [
    "user question", "<think>old</think>assistant", {"prompt":"user question"},
    {"prompt_token_ids":[9, 10, 11, 9]}, {"type":"multimodal", "prompt_token_ids":[9]}, object(),
])
def test_hybrid_prompt_reserves_generated_opening_before_closing_and_answer(prompt):
    native, calls = engine(end=(11,))
    native.vllm_config.reasoning_config.reasoning_start_str = "<think>"
    native.vllm_config.reasoning_config.reasoning_end_str = "</think>"
    with pytest.raises(ValueError, match="final answer"):
        helper().guard_engine_thinking_budget(native).generate(
            prompt, SimpleNamespace(thinking_token_budget=0, max_tokens=2), "request")
    assert calls == []


@pytest.mark.parametrize("prompt", [
    "assistant<think>", "assistant<think>\n", "<think>old</think>assistant<think>\n",
    {"prompt":"assistant<think>\n"}, {"prompt_token_ids":[9,10]},
    {"type":"multimodal", "prompt_token_ids":[10,11,9,10,9]},
])
def test_prefilled_native_reasoning_does_not_charge_opening_tokens_again(prompt):
    native, calls = engine(end=(11,))
    native.vllm_config.reasoning_config.reasoning_start_str = "<think>"
    native.vllm_config.reasoning_config.reasoning_end_str = "</think>"
    response = helper().guard_engine_thinking_budget(native).generate(
        prompt, SimpleNamespace(thinking_token_budget=0, max_tokens=2), "request")
    assert list(response) == ["native answer"]
    assert len(calls) == 1


def test_native_multitoken_opening_reserve_is_exact_when_prompt_not_prefilled():
    native, calls = engine(start=(7,8), end=(11,))
    guarded = helper().guard_engine_thinking_budget(native)
    with pytest.raises(ValueError, match="final answer"):
        guarded.generate({"prompt_token_ids":[9]}, SimpleNamespace(thinking_token_budget=0,max_tokens=3), "request")
    assert list(guarded.generate({"prompt_token_ids":[9]},
        SimpleNamespace(thinking_token_budget=0,max_tokens=4), "request")) == ["native answer"]
    assert len(calls) == 1
