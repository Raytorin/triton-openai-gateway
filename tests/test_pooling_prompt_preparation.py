# SPDX-License-Identifier: Apache-2.0
"""Keep pooling prompts on the same native-rendering boundary as generation."""
import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_generation_backend_metadata import (  # reuse Triton-only import boundaries
    Logger, TritonRequest, backend, completion, engine_output, generation_request,
)


@pytest.fixture
def pooling_backend(backend, monkeypatch):
    monkeypatch.setattr(backend.request, "TokensPrompt", lambda **kwargs: kwargs)
    monkeypatch.setattr(backend.request, "PoolingParams", SimpleNamespace)
    monkeypatch.setattr(backend.request, "LoRARequest", lambda name, ident, path: SimpleNamespace(
        lora_name=name, lora_int_id=ident, lora_path=path,
    ))
    return backend


def pooling_request(backend, value, encode, *, output_types=None):
    payload = {"input": value, "lora_name": "adapter", "pooling_params": {}}
    if output_types is not None:
        payload["output_types"] = output_types
    return backend.request.EmbedRequest(
        TritonRequest(embedding_request=json.dumps(payload).encode()),
        encode, np.object_, Logger(),
        lora_repository={"adapter": "/models/adapter"}, supported_loras=["adapter"],
    )


async def collect(request):
    return [response async for response in request.execute()]


@pytest.mark.parametrize("value", ["hello world", [10, 20]])
@pytest.mark.parametrize("output_types,task", [
    (None, "embed"), (["sparse"], "token_classify"),
    (["dense", "sparse"], "embed&token_classify"),
])
def test_pooling_prepares_once_before_encode_and_keeps_task_and_lora(
    pooling_backend, value, output_types, task,
):
    backend = pooling_backend
    prepared = {"type": "token", "prompt_token_ids": [10, 20]}
    calls = []

    async def prepare(built):
        calls.append(built)
        expected = value if isinstance(value, str) else {"prompt_token_ids": value}
        assert built.prompt == expected
        assert built.origins == {}
        return SimpleNamespace(prompt=prepared, vision_usage=None)

    async def encode(prompt, params, _request_id, *, lora_request):
        assert prompt is prepared
        assert params.task == task
        assert params.use_activation is True
        assert (lora_request.lora_name, lora_request.lora_int_id, lora_request.lora_path) == (
            "1", 1, "/models/adapter",
        )
        yield "encoded"

    request = pooling_request(backend, value, encode, output_types=output_types)
    # Assignment also catches a callback silently ignored by execution.
    request.prompt_preparer = prepare
    assert asyncio.run(collect(request)) == ["encoded"]
    assert len(calls) == 1


@pytest.mark.parametrize("value", ["hello world", [10, 20]])
def test_legacy_pooling_without_preparer_keeps_original_input(pooling_backend, value):
    async def encode(prompt, _params, _request_id, **_kwargs):
        expected = value if isinstance(value, str) else {"prompt_token_ids": value}
        assert prompt == expected
        yield "encoded"

    request = pooling_request(pooling_backend, value, encode)
    assert asyncio.run(collect(request)) == ["encoded"]


def test_pooling_renderer_failure_never_retries_raw_input(pooling_backend):
    async def prepare(_built):
        raise ValueError("native render failed")

    async def encode(*_args, **_kwargs):
        pytest.fail("encode must not receive raw input after renderer failure")
        yield

    request = pooling_request(pooling_backend, "hello", encode)
    request.prompt_preparer = prepare
    with pytest.raises(ValueError, match="native render failed"):
        asyncio.run(collect(request))


def test_plain_text_generation_uses_preparer_once_without_visual_usage(backend):
    request = generation_request(backend)
    prepared = {"type": "token", "prompt_token_ids": [10]}
    calls = []

    async def prepare(built):
        assert isinstance(built.prompt, str)
        assert built.origins == {}
        calls.append(built)
        return SimpleNamespace(prompt=prepared, vision_usage=None)

    async def generate(prompt, *_args, **_kwargs):
        assert prompt is prepared
        yield engine_output(completion("answer", [20], finish_reason="stop"), finished=True)

    request.prompt_preparer = prepare
    request.executor_callback = generate
    assert len(asyncio.run(collect(request))) == 1
    assert len(calls) == 1
    assert request.vision_usage is None


def test_model_pooling_dispatch_supplies_native_preparer(pooling_backend, monkeypatch):
    backend = pooling_backend
    prepared = {"type": "token", "prompt_token_ids": [10, 20]}
    calls = []

    class Values:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def reshape(self, _shape): return self
        def tolist(self): return [0.25, 0.75]

    class Accumulator:
        prompt_tokens = generation_tokens = 0

        def observe(self, output):
            self.prompt_tokens = len(output.prompt_token_ids)

    async def prepare(built):
        calls.append(built)
        return SimpleNamespace(prompt=prepared, vision_usage=None)

    async def encode(prompt, _params, request_id, **_kwargs):
        assert prompt is prepared
        yield SimpleNamespace(request_id=request_id, prompt_token_ids=[10, 20],
                              outputs=SimpleNamespace(data=Values()), finished=True)

    model = backend.model.TritonPythonModel.__new__(backend.model.TritonPythonModel)
    model.args = {"model_name": "pooling-fixture"}
    model.logger = Logger()
    model.output_dtype = np.object_
    model._ongoing_request_count = 0
    model._request_token_metrics = None
    model._llm_engine = SimpleNamespace(encode=encode)
    model.supported_tasks = ["embed"]
    model.pooling_model_metadata = backend.request.PoolingModelMetadata()
    model.enable_lora = False
    responses = []
    sender = SimpleNamespace(is_cancelled=lambda: False,
                             send=lambda response, **kwargs: responses.append(response))
    request = TritonRequest(embedding_request=json.dumps({"input": "hello"}).encode())
    request.get_response_sender = lambda: sender
    monkeypatch.setattr(backend.model, "RequestTokenAccumulator", Accumulator)
    monkeypatch.setattr(backend.model, "make_prompt_preparer", lambda engine: prepare)
    monkeypatch.setattr(backend.model.pb_utils, "TRITONSERVER_RESPONSE_COMPLETE_FINAL", 1,
                        raising=False)
    asyncio.run(model._infer(request))
    assert len(calls) == len(responses) == 1
    assert json.loads(responses[0].outputs["text_output"][0]) == [0.25, 0.75]
    assert model._ongoing_request_count == 0
