# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""CPU boundary helper invoking the unchanged production response serializer."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

_BACKEND = Path(__file__).resolve().parents[1] / 'backends' / 'vllm_multimodal'


class _Tensor:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def as_numpy(self):
        return self.data


class _Response:
    def __init__(self, *, output_tensors):
        self.output_tensors = output_tensors

    def as_numpy(self, name):
        return next((tensor.as_numpy() for tensor in self.output_tensors if tensor.name == name), None)


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


@contextmanager
def production_text_wire():
    """Load the whole request module; replace only unavailable runtime imports.

    Production create_response receives cumulative engine outputs and owns the
    state that converts them to Triton deltas. No serializer code is copied.
    """
    modules = {
        'triton_python_backend_utils': _module('triton_python_backend_utils',
            Tensor=_Tensor, InferenceResponse=_Response),
        'vllm.inputs': _module('vllm.inputs', TokensPrompt=object),
        'vllm.lora.request': _module('vllm.lora.request', LoRARequest=object),
        'vllm.outputs': _module('vllm.outputs', PoolingRequestOutput=object, RequestOutput=object),
        'vllm.pooling_params': _module('vllm.pooling_params', PoolingParams=object),
        'vllm.utils': _module('vllm.utils', random_uuid=lambda: 'test-engine-id'),
        'utils.vllm_backend_utils': _module('utils.vllm_backend_utils', TritonSamplingParams=object),
    }
    with patch.dict(sys.modules, modules), patch.object(sys, 'path', [str(_BACKEND), *sys.path]):
        spec = importlib.util.spec_from_file_location('deepseek_wire_request', _BACKEND / 'utils' / 'request.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        request = object.__new__(module.GenerateRequest)
        request.output_dtype = np.object_
        request.additional_outputs = {name: False for name in (
            'return_finish_reason', 'return_cumulative_logprob', 'return_logprobs',
            'return_num_input_tokens', 'return_num_output_tokens')}
    # Restore module/path patches before a caller can await/yield. The loaded
    # request class retains its own module globals/native tensor doubles.
    state = {}

    def serialize(cumulative_text):
        output = SimpleNamespace(prompt='unused', outputs=[SimpleNamespace(text=cumulative_text)])
        return request.create_response(output, state, prepend_input=False)

    yield serialize
