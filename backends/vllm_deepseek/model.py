# SPDX-License-Identifier: Apache-2.0
"""Opt-in Python backend: isolate the process before importing shared vLLM code."""
import importlib.util
from pathlib import Path
import sys

# Import by absolute path: a model repository must not shadow the bootstrap.
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('deepseek_runtime', _here / 'runtime.py')
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)
_shared = _here.parent / 'vllm_multimodal'
_python_backend = _here.parent / 'python'
RUNTIME_IDENTITY = _runtime.bootstrap(trusted_paths=(_here, _shared, _python_backend))
RUNTIME_IDENTITY['modules'] = _runtime.validate_module_origins(_runtime.ENV_ROOT)

# The shared helpers use utils.* imports; each backend runs in its own stub.
_spec = importlib.util.spec_from_file_location('deepseek_shared_model', _shared / 'model.py')
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
_spec = importlib.util.spec_from_file_location('deepseek_runtime_policy', _here / 'policy.py')
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
# Imports above can dlopen native libraries. Check the actual resulting maps,
# not only the clean pre-import process and find_spec module locations.
RUNTIME_IDENTITY['native_maps'] = _runtime.validate_native_maps()
RUNTIME_IDENTITY['modules'] = _runtime.validate_module_origins(_runtime.ENV_ROOT)


class TritonPythonModel(_policy.GenerationOnlyModel, _module.TritonPythonModel):
    pass
