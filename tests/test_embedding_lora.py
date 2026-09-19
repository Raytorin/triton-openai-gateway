"""Exercise the real request adapter with lightweight Triton/vLLM doubles."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
from fastapi import HTTPException
from tritonclient.utils import deserialize_bytes_tensor

from gateway import app as gateway_app
from gateway.schemas import EmbeddingsRequest
from gateway.triton_client import _build_grpc_embedding_inputs, call_triton_embeddings

BACKEND = Path(__file__).resolve().parents[1] / "backends" / "vllm_multimodal"


class Tensor:
    def __init__(self, name, data):
        self.name = name
        self.data = data

    def as_numpy(self):
        return self.data


class LoRARequest:
    def __init__(self, lora_name, lora_int_id, lora_path):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


def _module(name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def backend():
    # Import production modules, stubbing only unavailable runtime dependencies.
    # Restore sys.modules afterward so test order cannot change other tests.
    class TritonError:
        def __init__(self, message):
            self.message = message

    modules = {
        "triton_python_backend_utils": _module(
            "triton_python_backend_utils", Tensor=Tensor, TritonError=TritonError,
            InferenceResponse=SimpleNamespace, TRITONSERVER_RESPONSE_COMPLETE_FINAL=1,
            get_input_tensor_by_name=lambda request, name: request.tensors.get(name),
        ),
        "vllm.inputs": _module("vllm.inputs", TokensPrompt=lambda **kw: kw),
        "vllm.lora.request": _module("vllm.lora.request", LoRARequest=LoRARequest),
        "vllm.outputs": _module(
            "vllm.outputs", PoolingRequestOutput=object, RequestOutput=object,
        ),
        "vllm.pooling_params": _module("vllm.pooling_params", PoolingParams=SimpleNamespace),
        "vllm.utils": _module("vllm.utils", random_uuid=lambda: "engine-id"),
        "vllm.engine.arg_utils": _module("vllm.engine.arg_utils", AsyncEngineArgs=object),
        "utils.vllm_backend_utils": _module(
            "utils.vllm_backend_utils", TritonSamplingParams=object,
            build_async_engine_client_from_engine_args=Mock(),
        ),
        "utils.metrics": _module(
            "utils.metrics", RequestTokenAccumulator=lambda: SimpleNamespace(
                observe=Mock(), prompt_tokens=2, generation_tokens=0,
            ),
            RequestTokenMetrics=Mock(), VllmStatLoggerFactory=Mock(),
        ),
    }
    with patch.dict(sys.modules, modules), patch.object(sys, "path", [str(BACKEND), *sys.path]):
        request_module = _load("lora_test_request", BACKEND / "utils" / "request.py")
        sys.modules["utils.request"] = request_module
        model_module = _load("lora_test_model", BACKEND / "model.py")
        yield SimpleNamespace(request=request_module, model=model_module)


def _request(lora_name):
    tensors = {}
    for item in _build_grpc_embedding_inputs([10, 20], None, lora_name=lora_name):
        if item.datatype() == "BYTES":
            value = deserialize_bytes_tensor(item._raw_content)
        else:
            value = np.frombuffer(item._raw_content, dtype=np.bool_)
        tensors[item.name()] = Tensor(item.name(), value)
    return SimpleNamespace(tensors=tensors, parameters=lambda: "{}")


@pytest.mark.anyio
@pytest.mark.parametrize("lora_name,expected_id", [(None, None), ("adapter-a", 1), ("adapter-b", 2)])
async def test_gateway_payload_reaches_encode_with_selected_adapter(backend, lora_name, expected_id):
    captured = {}

    async def encode(prompt, pooling_params, request_id, *, lora_request=None):
        captured.update(prompt=prompt, lora_request=lora_request, task=pooling_params.task)
        data = Mock()
        data.detach.return_value.float.return_value.cpu.return_value.reshape.return_value = np.asarray([0.1, 0.2])
        yield SimpleNamespace(outputs=SimpleNamespace(data=data), prompt_token_ids=[10, 20])

    request = _request(lora_name)
    sender = Mock()
    sender.is_cancelled.return_value = False
    request.get_response_sender = lambda: sender
    model = backend.model.TritonPythonModel()
    model.enable_lora = True
    model.lora_repository = {"adapter-a": "/models/a", "adapter-b": "/models/b"}
    model.supported_loras = list(model.lora_repository)
    model.supported_tasks = {"embed"}
    model.args = {"model_name": "embedding-model"}
    model.logger = Mock()
    assert model._verify_loras(request) is request
    assert model._validate_request_task_name(request) == "embed"
    model._llm_engine = SimpleNamespace(encode=encode)
    model.output_dtype = np.object_
    model.pooling_model_metadata = None
    model._ongoing_request_count = 0
    model._request_token_metrics = None
    await model._infer(request)
    sender.send.assert_called_once()
    response = sender.send.call_args.args[0]
    assert response.output_tensors[0].as_numpy()[0] == "[0.1, 0.2]"
    assert model._ongoing_request_count == 0
    assert captured["prompt"] == {"prompt_token_ids": [10, 20]}
    assert captured["task"] == "embed"
    if expected_id is None:
        assert captured["lora_request"] is None
    else:
        assert captured["lora_request"].lora_int_id == expected_id
        assert captured["lora_request"].lora_path == model.lora_repository[lora_name]
        # GenerateRequest uses the same name and integer ID for this repository.
        assert captured["lora_request"].lora_name == str(expected_id)


@pytest.mark.parametrize("enabled,name,message", [
    (False, "adapter-a", "LoRA feature is not enabled"),
    (True, "unknown", "is not supported, we currently support"),
])
def test_embedding_lora_is_validated_before_execution(backend, enabled, name, message):
    model = backend.model.TritonPythonModel()
    model.enable_lora = enabled
    model.supported_loras = ["adapter-a"]
    model.args = {"model_name": "embedding-model"}
    model.logger = Mock()
    model.respond_error = Mock()
    assert model._verify_loras(_request(name)) is None
    assert message in model.respond_error.call_args.args[1]


@pytest.mark.anyio
@pytest.mark.parametrize("lora_name", [None, "adapter-a"])
async def test_dense_endpoint_forwards_lora_for_every_batch_item(lora_name):
    request = EmbeddingsRequest(model="embed", input=[[10], [20]], lora_name=lora_name)
    infer = AsyncMock(return_value=([0.1, 0.2], 1))
    with (
        patch.object(gateway_app.registry, "get_backend", return_value="vllm_multimodal"),
        patch.object(gateway_app.registry, "get_tokenizer_async", AsyncMock(return_value=(Mock(), None))),
        patch.object(gateway_app, "call_triton_embeddings", infer),
    ):
        response = await gateway_app.create_embeddings.__wrapped__(request)
    assert response["usage"]["prompt_tokens"] == 2
    assert [item["index"] for item in response["data"]] == [0, 1]
    assert [call.args[1] for call in infer.await_args_list] == [[10], [20]]
    assert all(call.kwargs == {"lora_name": lora_name} for call in infer.await_args_list)


@pytest.mark.anyio
@pytest.mark.parametrize("backend_name", ["vllm", "python", None])
async def test_unsupported_backend_cannot_silently_ignore_lora(backend_name):
    with patch.object(gateway_app.registry, "get_backend", return_value=backend_name):
        with pytest.raises(HTTPException) as caught:
            await gateway_app.create_embeddings.__wrapped__(
                EmbeddingsRequest(model="embed", input="hello", lora_name="adapter-a")
            )
    assert caught.value.status_code == 400
    assert "vllm_multimodal" in caught.value.detail


@pytest.mark.anyio
@pytest.mark.parametrize("message,status", [
    ("LoRA feature is not enabled.", 400),
    ("LoRA missing is not supported, we currently support ['a']", 400),
    ("connection refused", 502),
])
async def test_embedding_backend_errors_keep_correct_http_status(message, status):
    from grpc import StatusCode
    from grpc.aio import AioRpcError

    async def results(*args, **kwargs):
        raise AioRpcError(StatusCode.INTERNAL, None, None, details=message)
        yield

    with patch("gateway.triton_client._stream_grpc_results", results):
        with pytest.raises(HTTPException) as caught:
            await call_triton_embeddings("embed", [10], None, lora_name="adapter-a")
    assert caught.value.status_code == status


@pytest.mark.parametrize("source", ["sampling_parameters", "parameters"])
def test_chat_lora_validation_still_uses_generation_parameters(backend, source):
    request = SimpleNamespace(tensors={}, parameters=lambda: '{}')
    payload = '{"lora_name":"unknown"}'
    if source == "sampling_parameters":
        request.tensors[source] = Tensor(source, np.asarray([payload.encode()]))
    else:
        request.parameters = lambda: payload
    model = backend.model.TritonPythonModel()
    model.enable_lora = True
    model.supported_loras = ["adapter-a"]
    model.args = {"model_name": "chat-model"}
    model.logger = Mock()
    model.respond_error = Mock()
    assert model._verify_loras(request) is None
    assert "LoRA unknown is not supported" in model.respond_error.call_args.args[1]
