# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production backend serializer and completion log on CPU.

Only unavailable Triton/vLLM boundaries and unrelated media/engine setup are
substituted. Serialization, request execution, config registration and logging
come from the complete production modules.
"""
import asyncio
from contextlib import ExitStack
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


BACKEND = Path(__file__).resolve().parents[1] / "backends" / "vllm_multimodal"


class Tensor:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def as_numpy(self):
        return self.values


class InferenceResponse:
    def __init__(self, *, output_tensors):
        self.outputs = {tensor.name: tensor.as_numpy() for tensor in output_tensors}


class TritonRequest:
    def __init__(self, **inputs):
        self.inputs = {name: Tensor(name, np.asarray([value])) for name, value in inputs.items()}

    def request_id(self):
        return "metadata-request"

    def parameters(self):
        return "{}"


class Logger:
    def __init__(self):
        self.events = []

    def log_info(self, message):
        self.events.append(json.loads(message))

    log_warn = log_info
    log_error = log_info


class SamplingParams:
    @classmethod
    def from_dict(cls, _parameters, _logger):
        return SimpleNamespace(lora_name=None)


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    modules = {
        "utils": _module("utils", __path__=[str(BACKEND / "utils")]),
        "triton_python_backend_utils": _module(
            "triton_python_backend_utils", Tensor=Tensor, InferenceResponse=InferenceResponse,
            get_input_tensor_by_name=lambda request, name: request.inputs.get(name),
        ),
        "vllm.inputs": _module("vllm.inputs", TokensPrompt=object),
        "vllm.lora.request": _module("vllm.lora.request", LoRARequest=object),
        "vllm.outputs": _module("vllm.outputs", PoolingRequestOutput=object, RequestOutput=object),
        "vllm.pooling_params": _module("vllm.pooling_params", PoolingParams=object),
        "vllm.utils": _module("vllm.utils", random_uuid=lambda: "engine-id"),
        "vllm.engine.arg_utils": _module("vllm.engine.arg_utils", AsyncEngineArgs=object),
        "utils.vllm_backend_utils": _module(
            "utils.vllm_backend_utils", TritonSamplingParams=SamplingParams,
            build_async_engine_client_from_engine_args=object,
        ),
        "utils.media": _module(
            "utils.media", build_multimodal_prompt=lambda prompt, **kwargs: prompt,
            parse_media_parameters=lambda value: {},
        ),
        "utils.metrics": _module(
            "utils.metrics", RequestTokenAccumulator=object, RequestTokenMetrics=object,
            VllmStatLoggerFactory=object,
        ),
    }
    # Restore imports before callers await anything; module globals keep their
    # own boundary substitutes without altering the rest of the pytest suite.
    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, modules))
        request_module = _load("utils.request", BACKEND / "utils" / "request.py")
        model_module = _load("metadata_test_model", BACKEND / "model.py")
        policy_module = _load("metadata_test_policy", BACKEND.parent / "vllm_deepseek" / "policy.py")
    return SimpleNamespace(request=request_module, model=model_module, policy=policy_module)


def completion(text="", token_ids=(), *, index=0, finish_reason=None, stop_reason=None):
    return SimpleNamespace(
        index=index, text=text, token_ids=token_ids, finish_reason=finish_reason,
        stop_reason=stop_reason, cumulative_logprob=None, logprobs=None,
    )


def engine_output(*outputs, finished=False, prompt_ids=(1, 2)):
    return SimpleNamespace(
        request_id="engine-id", prompt="PRIVATE PROMPT", prompt_token_ids=prompt_ids,
        outputs=list(outputs), finished=finished,
    )


def generation_request(backend, *, metadata=True, legacy_tokens=True, logger=None, snapshots=()):
    async def generate(*args, **kwargs):
        for snapshot in snapshots:
            yield snapshot

    request = backend.request.GenerateRequest(
        TritonRequest(text_input=b"PRIVATE PROMPT", stream=True, exclude_input_in_output=True,
                      return_generation_metadata=metadata,
                      return_num_output_tokens=legacy_tokens),
        generate, np.object_, logger or Logger(), model_name="fixture-model",
    )
    request.additional_outputs = {
        "return_finish_reason": False, "return_cumulative_logprob": False,
        "return_logprobs": False, "return_num_input_tokens": False,
        "return_num_output_tokens": legacy_tokens, "return_generation_metadata": metadata,
    }
    return request


def metadata(response):
    tensor = response.outputs.get("generation_metadata")
    assert tensor is not None, "opted-in generation_metadata must reach the Triton response"
    assert tensor.shape == (1,)
    return json.loads(tensor[0])


def test_metadata_is_cumulative_while_legacy_counts_and_terminal_text_stay_delta(backend):
    request = generation_request(backend)
    snapshots = [
        engine_output(completion("A", [10])),
        engine_output(completion("ABC", [10, 11, 12])),
        engine_output(completion("ABC", [10, 11, 12], finish_reason="length"), finished=True),
    ]
    state = {}
    responses = [request.create_response(snapshot, state, False) for snapshot in snapshots]
    assert [response.outputs["text_output"].tolist() for response in responses] == [[b"A"], [b"BC"], [b""]]
    assert [response.outputs["num_output_tokens"].tolist() for response in responses] == [[1], [2], [0]]
    envelopes = [metadata(response) for response in responses]
    assert [item["outputs"][0]["completion_tokens"] for item in envelopes] == [1, 3, 3]
    assert envelopes[0] == {
        "version": 1, "finished": False, "prompt_tokens": 2,
        "outputs": [{"index": 0, "finish_reason": None, "stop_reason": None,
                     "completion_tokens": 1, "first_token_id": 10, "last_token_id": 10}],
    }
    assert envelopes[-1] == {
        "version": 1, "finished": True, "prompt_tokens": 2,
        "outputs": [{"index": 0, "finish_reason": "length", "stop_reason": None,
                     "completion_tokens": 3, "first_token_id": 10, "last_token_id": 12}],
    }


def test_metadata_preserves_eos_stop_string_empty_tokens_and_multiple_choices(backend):
    request = generation_request(backend)
    response = request.create_response(engine_output(
        completion("", [2], index=4, finish_reason="stop", stop_reason=2),
        completion("done", [31, 32], index=7, finish_reason="stop", stop_reason="PRIVATE STOP"),
        completion("", [], index=9, finish_reason="length"),
        finished=True, prompt_ids=None,
    ), {}, False)
    assert metadata(response) == {
        "version": 1, "finished": True, "prompt_tokens": None,
        "outputs": [
            {"index": 4, "finish_reason": "stop", "stop_reason": 2,
             "completion_tokens": 1, "first_token_id": 2, "last_token_id": 2},
            {"index": 7, "finish_reason": "stop", "stop_reason": "PRIVATE STOP",
             "completion_tokens": 2, "first_token_id": 31, "last_token_id": 32},
            {"index": 9, "finish_reason": "length", "stop_reason": None,
             "completion_tokens": 0, "first_token_id": None, "last_token_id": None},
        ],
    }


@pytest.mark.parametrize("include_flag", [False, True])
def test_legacy_requests_do_not_gain_metadata_output(backend, include_flag):
    request = generation_request(backend, metadata=False)
    if not include_flag:
        del request.additional_outputs["return_generation_metadata"]
    response = request.create_response(engine_output(completion("x", [3])), {}, False)
    assert set(response.outputs) == {"text_output", "num_output_tokens"}


def test_metadata_flag_is_read_from_triton_request(backend):
    request = generation_request(backend)
    assert request._get_input_tensors()[-1].get("return_generation_metadata") is True


@pytest.mark.parametrize("with_deepseek_policy", [False, True])
def test_metadata_registration_is_optional_and_idempotent_for_shared_backends(backend, with_deepseek_policy):
    class Config:
        def __init__(self):
            self.value = {"input": [], "output": []}

        def as_dict(self):
            return self.value

        def add_input(self, value):
            self.value["input"].append(value)

        def add_output(self, value):
            self.value["output"].append(value)

        def set_model_transaction_policy(self, value):
            pass

        def set_max_batch_size(self, value):
            pass

    backend_class = backend.model.TritonPythonModel
    if with_deepseek_policy:
        backend_class = type("DeepSeekBackend", (backend.policy.GenerationOnlyModel, backend_class), {})
    config = Config()
    backend_class.auto_complete_config(config)
    backend_class.auto_complete_config(config)
    assert [item for item in config.value["input"] if item["name"] == "return_generation_metadata"] == [
        {"name": "return_generation_metadata", "data_type": "TYPE_BOOL", "dims": [1], "optional": True}]
    assert [item for item in config.value["output"] if item["name"] == "generation_metadata"] == [
        {"name": "generation_metadata", "data_type": "TYPE_STRING", "dims": [1]}]


@pytest.mark.parametrize("prior_text, expected_level, empty", [("", "WARNING", True), ("PRIVATE ANSWER", "INFO", False)])
def test_completion_diagnostic_is_once_and_uses_whole_generation_without_text(backend, prior_text, expected_level, empty):
    logger = Logger()
    request = generation_request(backend, logger=logger, snapshots=[
        engine_output(completion(prior_text, [10])),
        engine_output(completion("", [10, 11], finish_reason="stop", stop_reason="PRIVATE STOP"), finished=True),
    ])

    async def collect():
        return [snapshot async for snapshot in request.execute()]

    asyncio.run(collect())
    completed = [event for event in logger.events if event["event"] == "generation.completed"]
    assert len(completed) == 1
    event = completed[0]
    assert event["level"] == expected_level
    assert event["model"] == "fixture-model" and event["request_id"] == "metadata-request"
    assert event["engine_request_id"] == "metadata-request:engine-id"
    assert event["prompt_tokens"] == 2
    assert event["outputs"] == [{
        "index": 0, "finish_reason": "stop", "completion_tokens": 2,
        "first_token_id": 10, "last_token_id": 11, "empty_text": empty,
        "stop_reason_type": "str", "stop_reason_length": 12,
    }]
    logged = json.dumps(logger.events)
    assert "PRIVATE PROMPT" not in logged and "PRIVATE ANSWER" not in logged and "PRIVATE STOP" not in logged


def test_eos_completion_diagnostic_keeps_numeric_stop_id(backend):
    logger = Logger()
    request = generation_request(backend, logger=logger, snapshots=[
        engine_output(completion("", [2], finish_reason="stop", stop_reason=2), finished=True),
    ])

    async def collect():
        return [snapshot async for snapshot in request.execute()]

    asyncio.run(collect())
    completed = [event for event in logger.events if event["event"] == "generation.completed"]
    assert len(completed) == 1
    assert completed[0]["level"] == "WARNING"
    assert completed[0]["outputs"][0]["stop_reason"] == 2
