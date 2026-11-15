# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Install thinking-budget compatibility in the stock Triton vLLM factory.

The anchors were verified against the pinned Triton 26.10 / vLLM 0.31.0
image. Unknown, ambiguous or partially patched factories fail the build.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
import stat
import tempfile


_MARKER = "# triton-openai-gateway: native-thinking-budget-v1"
_CONFIG = "    vllm_config = engine_args.create_engine_config(usage_context=usage_context)\n"
_YIELD = "        yield async_llm\n"
_SAMPLER = "            vllm_params_dict = SamplingParams.__annotations__\n"
_PATCHED_SAMPLER = (_SAMPLER
    + "            from utils.thinking_budget import validate_sampling_budget\n"
    + "            validate_sampling_budget(params_dict, vllm_params_dict)\n")
_PATCHED_CONFIG = (
    "    " + _MARKER + "\n"
    "    from utils.thinking_budget import (\n"
    "        guard_engine_thinking_budget,\n"
    "        normalize_engine_reasoning_config,\n"
    "    )\n\n"
    "    normalize_engine_reasoning_config(engine_args)\n"
    + _CONFIG
)
_PATCHED_YIELD = "        guard_engine_thinking_budget(async_llm)\n" + _YIELD
_FACTORY = "build_async_engine_client_from_engine_args"
_DEFAULT_PATH = Path("/opt/tritonserver/backends/vllm/utils/vllm_backend_utils.py")


def patch_source(source: str) -> str:
    """Patch only the verified factory shape; accept our complete patch twice."""
    if _MARKER in source:
        if source.count(_PATCHED_CONFIG) != 1 or source.count(_PATCHED_YIELD) != 1 or source.count(_PATCHED_SAMPLER) != 1:
            raise ValueError("Incomplete native thinking-budget patch")
        original = source.replace(_PATCHED_CONFIG, _CONFIG).replace(_PATCHED_YIELD, _YIELD).replace(_PATCHED_SAMPLER, _SAMPLER)
        if _MARKER in original or patch_source(original) != source:
            raise ValueError("Unexpected native thinking-budget patch")
        return source
    if any(name in source for name in (
        "normalize_engine_reasoning_config", "guard_engine_thinking_budget", "validate_sampling_budget",
    )):
        raise ValueError("Native backend has an unrecognized partial patch")
    if source.count(_CONFIG) != 1 or source.count(_YIELD) != 1 or source.count(_SAMPLER) != 1:
        raise ValueError("Native vLLM engine factory anchors are missing or ambiguous")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError("Native backend source is invalid") from error
    factories = [node for node in tree.body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name == _FACTORY]
    if len(factories) != 1:
        raise ValueError("Expected exactly one native async engine factory")
    factory = factories[0]
    samplers = [node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "TritonSamplingParams"]
    parsers = [node for sampler in samplers for node in sampler.body
               if isinstance(node, ast.FunctionDef) and node.name == "from_dict"]
    sampler_line = source[:source.index(_SAMPLER)].count("\n") + 1
    if len(samplers) != 1 or len(parsers) != 1 or not (
        parsers[0].lineno < sampler_line <= parsers[0].end_lineno
    ):
        raise ValueError("Native sampling parser anchor is outside the expected method")
    config_line = source[:source.index(_CONFIG)].count("\n") + 1
    yield_line = source[:source.index(_YIELD)].count("\n") + 1
    if not (factory.lineno < config_line < yield_line <= factory.end_lineno):
        raise ValueError("Native backend anchors are outside the expected factory")
    if not any(node.lineno == config_line for node in factory.body):
        raise ValueError("Native engine config creation changed scope")
    if not any(isinstance(node, ast.Yield) and node.lineno == yield_line
               and isinstance(node.value, ast.Name) and node.value.id == "async_llm"
               for node in ast.walk(factory)):
        raise ValueError("Native factory no longer yields the expected engine")
    patched = source.replace(_CONFIG, _PATCHED_CONFIG).replace(_YIELD, _PATCHED_YIELD).replace(_SAMPLER, _PATCHED_SAMPLER)
    compile(patched, "native_vllm_backend.py", "exec")
    return patched


def patch_native_backend(path: Path) -> bool:
    """Install atomically, retaining file permissions and the original on error."""
    if not (path.parent / "thinking_budget.py").is_file():
        raise FileNotFoundError("Install the native thinking-budget helper before patching")
    original = path.read_text(encoding="utf-8")
    patched = patch_source(original)
    if patched == original:
        return False
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as out:
            temporary = Path(out.name)
            out.write(patched)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=_DEFAULT_PATH)
    args = parser.parse_args()
    changed = patch_native_backend(args.path)
    print("Native vLLM thinking-budget patch " + ("installed" if changed else "already installed"))


if __name__ == "__main__":
    main()
