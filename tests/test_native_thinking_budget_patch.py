# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""The build-time native backend patch must be safe and effective."""
import asyncio
import importlib.util
from pathlib import Path
import sys
import types

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "docker/patch-native-thinking-budget.py"
# Factory anchors verified against the pinned Triton 26.10 base image.
SOURCE = """import json
from contextlib import asynccontextmanager

class TritonSamplingParams(SamplingParams):
    @staticmethod
    def from_dict(params_dict_str, logger):
        try:
            params_dict = json.loads(params_dict_str)
            vllm_params_dict = SamplingParams.__annotations__
            for key, value in params_dict.items():
                if vllm_params_dict.get(key) is int:
                    params_dict[key] = int(value)
            return TritonSamplingParams(**params_dict)
        except Exception as error:
            logger.log_error(str(error))
            return None

@asynccontextmanager
async def build_async_engine_client_from_engine_args(
    engine_args, logger, *, usage_context=None,
    disable_frontend_multiprocessing=False, stat_loggers=None,
):
    # Create the EngineConfig
    vllm_config = engine_args.create_engine_config(usage_context=usage_context)
    if disable_frontend_multiprocessing:
        logger.log_warn("frontend multiprocessing disabled")
    from vllm.v1.engine.async_llm import AsyncLLM
    async_llm = None
    try:
        async_llm = AsyncLLM.from_vllm_config(
            vllm_config=vllm_config,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=engine_args.enable_log_requests,
            aggregate_engine_logging=engine_args.aggregate_engine_logging,
            disable_log_stats=engine_args.disable_log_stats,
        )
        assert async_llm is not None
        await async_llm.reset_mm_cache()

        yield async_llm
    finally:
        if async_llm:
            async_llm.shutdown()
"""


class SamplingParams:
    __annotations__ = {"thinking_token_budget": int}

    def __init__(self, **values):
        self.__dict__.update(values)


def installer():
    assert INSTALLER.is_file(), "Native backend installer has not been implemented"
    spec = importlib.util.spec_from_file_location("native_thinking_installer", INSTALLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_factory_normalizes_before_config_and_guards_before_yield(monkeypatch):
    patched = installer().patch_source(SOURCE)
    events = []
    engine = types.SimpleNamespace()
    async def reset_mm_cache():
        events.append("reset")
    engine.reset_mm_cache = reset_mm_cache
    engine.shutdown = lambda: events.append("shutdown")
    def normalize(args):
        events.append("normalize")
        args.reasoning_config = "typed-config"
    def guard(value):
        assert value is engine
        events.append("guard")
        engine.guarded = True
        return value
    helper = types.ModuleType("utils.thinking_budget")
    helper.normalize_engine_reasoning_config = normalize
    helper.guard_engine_thinking_budget = guard
    monkeypatch.setitem(sys.modules, "utils.thinking_budget", helper)
    runtime = types.ModuleType("vllm.v1.engine.async_llm")
    def construct(**kwargs):
        assert kwargs["vllm_config"] == "validated-config"
        events.append("construct")
        return engine
    runtime.AsyncLLM = types.SimpleNamespace(from_vllm_config=construct)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", runtime)
    def config(**kwargs):
        assert args.reasoning_config == "typed-config"
        events.append("config")
        return "validated-config"
    args = types.SimpleNamespace(reasoning_config={}, create_engine_config=config,
        enable_log_requests=False, aggregate_engine_logging=False, disable_log_stats=False)
    namespace = {"SamplingParams": SamplingParams}
    exec(compile(patched, "native_backend.py", "exec"), namespace)
    async def run():
        async with namespace["build_async_engine_client_from_engine_args"](args, None) as value:
            assert value is engine and value.guarded
            events.append("yield")
    asyncio.run(run())
    assert events == ["normalize", "config", "construct", "reset", "guard", "yield", "shutdown"]


def test_installer_is_idempotent():
    module = installer()
    patched = module.patch_source(SOURCE)
    assert module.patch_source(patched) == patched


@pytest.mark.parametrize("source", [
    SOURCE.replace("engine_args.create_engine_config(usage_context=usage_context)", "engine_args.create_engine_config()"),
    SOURCE.replace("        yield async_llm", "        yield another_engine"),
    SOURCE + SOURCE,
    SOURCE.replace("build_async_engine_client_from_engine_args", "another_factory"),
    SOURCE.replace("    # Create the EngineConfig", "    normalize_engine_reasoning_config(engine_args)"),
    SOURCE.replace("SamplingParams.__annotations__", "SamplingParams.fields"),
    SOURCE.replace("def from_dict(params_dict_str, logger):", "def other_parser(params_dict_str, logger):"),
], ids=["config-signature", "wrong-engine", "duplicate-factory", "unknown-factory", "partial-patch", "unknown-sampler-anchor", "unknown-sampler-method"])
def test_unrecognized_or_partial_factory_fails_closed(source):
    with pytest.raises(ValueError):
        installer().patch_source(source)


def test_corrupted_installed_patch_fails_closed():
    module = installer()
    patched = module.patch_source(SOURCE)
    with pytest.raises(ValueError):
        module.patch_source(patched.replace("        guard_engine_thinking_budget(async_llm)\n", ""))


def test_file_installer_checks_helper_and_preserves_source_on_failure(tmp_path):
    module = installer()
    path = tmp_path / "vllm_backend_utils.py"
    path.write_text(SOURCE)
    with pytest.raises(FileNotFoundError):
        module.patch_native_backend(path)
    assert path.read_text() == SOURCE
    (tmp_path / "thinking_budget.py").write_text("# helper installed by Docker COPY\n")
    path.write_text("# unknown native backend\n")
    with pytest.raises(ValueError):
        module.patch_native_backend(path)
    assert path.read_text() == "# unknown native backend\n"


def test_file_installer_updates_once_and_preserves_permissions(tmp_path):
    module = installer()
    path = tmp_path / "vllm_backend_utils.py"
    path.write_text(SOURCE)
    path.chmod(0o640)
    (tmp_path / "thinking_budget.py").write_text("# helper installed by Docker COPY\n")
    assert module.patch_native_backend(path) is True
    assert path.stat().st_mode & 0o777 == 0o640
    assert module.patch_native_backend(path) is False


@pytest.mark.parametrize("budget", [True, False, "128", 1.5, -1], ids=["true", "false", "string", "float", "negative"])
def test_native_sampler_rejects_invalid_raw_budget_before_legacy_coercion(monkeypatch, budget):
    patched = installer().patch_source(SOURCE)
    helper_path = INSTALLER.parents[1] / "backends/vllm_multimodal/utils/thinking_budget.py"
    spec = importlib.util.spec_from_file_location("utils.thinking_budget", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setitem(sys.modules, "utils.thinking_budget", helper)
    namespace = {"SamplingParams": SamplingParams}
    exec(compile(patched, "native_backend.py", "exec"), namespace)
    logger = types.SimpleNamespace(log_error=lambda text: None)
    result = namespace["TritonSamplingParams"].from_dict(
        __import__("json").dumps({"thinking_token_budget": budget}), logger)
    assert result is None
