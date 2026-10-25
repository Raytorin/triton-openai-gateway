# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""CPU API contracts for the real vLLM runtime in the project Triton image.

Run directly with ``python3 tests/runtime_vllm_smoke.py``. Missing runtime
dependencies are errors, not skips. The GPU engine and the APIs injected by a
running Triton server are the only doubles; this does not run GPU inference.
"""

import asyncio
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import vllm
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import StructuredOutputsParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.stats import (
    FinishedRequestStats,
    IterationStats,
    KVCacheEvictionEvent,
    MultiModalCacheStats,
    PrefixCacheStats,
    SchedulerStats,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backends" / "vllm_multimodal"))
sys.path.insert(0, "/opt/tritonserver/backends/python")

import triton_python_backend_utils as pb_utils  # noqa: E402
from utils.hybrid_embeddings import PoolingModelMetadata  # noqa: E402
from utils.metrics import VllmStatLoggerFactory  # noqa: E402
from utils.vllm_backend_utils import (  # noqa: E402
    TritonSamplingParams,
    build_async_engine_client_from_engine_args,
)


class Logger:
    def __init__(self):
        self.errors = []

    def log_error(self, message):
        self.errors.append(message)

    def log_warn(self, _message):
        pass

    def log_info(self, _message):
        pass


class Tensor:
    """The tensor interface normally supplied by the Triton Python stub."""

    def __init__(self, name, array):
        self._name = name
        self._array = array

    def name(self):
        return self._name

    def as_numpy(self):
        return self._array


class TritonRequest:
    def __init__(self, inputs):
        self._inputs = [Tensor(name, np.asarray([value])) for name, value in inputs.items()]

    def inputs(self):
        return self._inputs

    def request_id(self):
        return "runtime-smoke"


class InferenceResponse:
    def __init__(self, *, output_tensors):
        self.output_tensors = {tensor.name(): tensor.as_numpy() for tensor in output_tensors}


class MetricFamily:
    """Capture only the metric writer boundary owned by the Triton server."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"

    def __init__(self, *, name, description, kind):
        self.name = name
        self.kind = kind
        self.samples = []

    def Metric(self, *, labels, buckets=None):
        return self

    def increment(self, value):
        self.samples.append(value)

    def observe(self, value):
        self.samples.append(value)

    def set(self, value):
        self.samples.append(value)


class RuntimeVllmContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A local HF config is enough for real config validation; no weights,
        # tokenizer download, device detection, or engine process is needed.
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        model_path = Path(cls.directory.name)
        (model_path / "config.json").write_text(json.dumps({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "vocab_size": 128,
            "max_position_embeddings": 64,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "torch_dtype": "float32",
        }))
        cls.config = VllmConfig(
            model_config=ModelConfig(
                model=str(model_path),
                tokenizer=str(model_path),
                dtype="float32",
                max_model_len=64,
                enforce_eager=True,
                skip_tokenizer_init=True,
            ),
            device_config=DeviceConfig(device="cpu"),
        )

    def test_sampling_numeric_strings_preserve_lora_selection(self):
        # Catches annotation/type changes that break Triton's string parameters.
        logger = Logger()
        params = TritonSamplingParams.from_dict(json.dumps({
            "max_tokens": "32", "temperature": "0.7", "seed": "7",
            "lora_name": "adapter",
        }), logger)
        self.assertIsNotNone(params, logger.errors)
        self.assertEqual(32, params.max_tokens)
        self.assertEqual(0.7, params.temperature)
        self.assertEqual(7, params.seed)
        self.assertEqual("adapter", params.lora_name)
        self.assertEqual([], logger.errors)

    def test_structured_outputs_cross_the_sampling_boundary(self):
        # Catches lost JSON conversion or removed StructuredOutputsParams fields.
        for payload in ({"json_object": True}, {"json": {
            "type": "object", "properties": {"answer": {"type": "string"}},
            "required": ["answer"], "additionalProperties": False,
        }}):
            with self.subTest(payload=payload):
                logger = Logger()
                params = TritonSamplingParams.from_dict(json.dumps({
                    "structured_outputs": json.dumps(payload),
                }), logger)
                self.assertIsNotNone(params, logger.errors)
                self.assertIsInstance(params.structured_outputs, StructuredOutputsParams)
                for name, expected in payload.items():
                    self.assertEqual(expected, getattr(params.structured_outputs, name))

    def test_generation_dispatches_real_sampling_and_lora_types(self):
        # Catches a changed generate signature or lost generation LoRA selection.
        from utils.request import GenerateRequest

        dispatched = []

        async def generate(prompt, sampling_params, request_id, **kwargs):
            inspect.signature(AsyncLLM.generate).bind(
                None, prompt, sampling_params, request_id, **kwargs,
            )
            dispatched.append((prompt, sampling_params, kwargs["lora_request"]))
            yield RequestOutput(
                request_id=request_id, prompt="Hello", prompt_token_ids=[10],
                prompt_logprobs=None,
                outputs=[CompletionOutput(
                    index=0, text="world", token_ids=[20], cumulative_logprob=None,
                    logprobs=None, finish_reason="stop",
                )], finished=True,
            )

        request = GenerateRequest(
            TritonRequest({
                "text_input": b"Hello",
                "sampling_parameters": json.dumps({
                    "max_tokens": "1", "lora_name": "adapter",
                }).encode(),
                "exclude_input_in_output": True,
            }), generate, np.object_, Logger(),
            lora_repository={"adapter": "/models/adapter"},
            supported_loras=["adapter"],
        )

        async def collect():
            return [response async for response in request.execute()]

        responses = asyncio.run(collect())
        self.assertEqual(1, len(responses))
        prompt, params, lora = dispatched[0]
        self.assertEqual("Hello", prompt)
        self.assertEqual(1, params.max_tokens)
        self.assertIsInstance(params, TritonSamplingParams)
        self.assertIsInstance(lora, LoRARequest)
        self.assertEqual(("1", 1, "/models/adapter"),
                         (lora.lora_name, lora.lora_int_id, lora.lora_path))
        with patch.object(pb_utils, "Tensor", Tensor, create=True), \
                patch.object(pb_utils, "InferenceResponse", InferenceResponse, create=True):
            wire = request.create_response(responses[0], {}, False).output_tensors
        self.assertEqual(b"world", wire["text_output"][0])

    def test_embedding_pooling_and_lora_dispatch_return_wire_payloads(self):
        # Catches task/LoRA forwarding and changes to real pooling output shapes.
        from utils.request import EmbedRequest

        metadata = PoolingModelMetadata(
            architecture="BgeM3EmbeddingModel", hidden_size=2,
            bos_token_id=0, eos_token_id=2, pad_token_id=1, unk_token_id=3,
        )
        fixtures = [
            (None, "embed", [0.25, 0.75], [0.25, 0.75]),
            (["sparse"], "token_classify", [0.5, 0.75],
             {"sparse": {"indices": [10, 20], "values": [0.5, 0.75]}}),
            (["dense", "sparse"], "embed&token_classify", [0.25, 0.75, 0.5, 0.75],
             {"dense": [0.25, 0.75],
              "sparse": {"indices": [10, 20], "values": [0.5, 0.75]}}),
        ]
        for output_types, task, raw, expected in fixtures:
            with self.subTest(task=task):
                payload = {"input": [0, 10, 20, 2], "pooling_params": {},
                           "lora_name": "adapter"}
                if task != "token_classify":
                    payload["pooling_params"]["dimensions"] = [2]
                if output_types is not None:
                    payload["output_types"] = output_types
                dispatched = []

                async def encode(prompt, pooling_params, request_id, **kwargs):
                    inspect.signature(AsyncLLM.encode).bind(
                        None, prompt, pooling_params, request_id, **kwargs,
                    )
                    dispatched.append((prompt, pooling_params, kwargs["lora_request"]))
                    yield PoolingRequestOutput(
                        request_id=request_id,
                        outputs=PoolingOutput(torch.tensor(raw)),
                        prompt_token_ids=[0, 10, 20, 2],
                        num_cached_tokens=0, finished=True,
                    )

                request = EmbedRequest(
                    TritonRequest({
                        "embedding_request": json.dumps(payload).encode(),
                        "return_num_input_tokens": True,
                    }), encode, np.object_, Logger(),
                    pooling_model_metadata=metadata,
                    lora_repository={"adapter": "/models/adapter"},
                    supported_loras=["adapter"],
                )

                async def collect():
                    return [response async for response in request.execute()]

                responses = asyncio.run(collect())
                self.assertEqual(1, len(responses))
                prompt, pooling_params, lora = dispatched[0]
                self.assertEqual([0, 10, 20, 2], prompt["prompt_token_ids"])
                self.assertIsInstance(pooling_params, PoolingParams)
                self.assertEqual(task, pooling_params.task)
                self.assertTrue(pooling_params.use_activation)
                self.assertEqual(None if task == "token_classify" else 2,
                                 pooling_params.dimensions)
                self.assertIsInstance(lora, LoRARequest)
                self.assertEqual(("1", 1, "/models/adapter"),
                                 (lora.lora_name, lora.lora_int_id, lora.lora_path))
                with patch.object(pb_utils, "Tensor", Tensor, create=True), \
                        patch.object(pb_utils, "InferenceResponse", InferenceResponse, create=True):
                    wire = request.create_response(responses[0]).output_tensors
                self.assertEqual(expected, json.loads(wire["text_output"][0]))
                self.assertEqual(4, wire["num_input_tokens"].item())

    def test_engine_builder_releases_engine_on_success_and_failure(self):
        # Catches unsupported factory kwargs, missing cache reset and leaked engines.
        real_factory_signature = inspect.signature(AsyncLLM.from_vllm_config)
        for failure in (None, "factory", "reset", "body"):
            with self.subTest(failure=failure):
                events = []
                engine_args = AsyncEngineArgs(
                    model=self.directory.name, enable_log_requests=True,
                    aggregate_engine_logging=True, disable_log_stats=False,
                )
                stat_loggers = [VllmStatLoggerFactory({"model": "runtime-smoke"}, Logger())]

                class Engine:
                    async def reset_mm_cache(self):
                        events.append("reset")
                        if failure == "reset":
                            raise RuntimeError("reset failed")

                    def shutdown(self):
                        events.append("shutdown")

                engine = Engine()

                def create_config(*, usage_context):
                    self.assertEqual(UsageContext.OPENAI_API_SERVER, usage_context)
                    return self.config

                def launch(**kwargs):
                    real_factory_signature.bind(**kwargs)
                    self.assertIs(self.config, kwargs["vllm_config"])
                    self.assertIs(stat_loggers, kwargs["stat_loggers"])
                    self.assertEqual(UsageContext.OPENAI_API_SERVER, kwargs["usage_context"])
                    self.assertTrue(kwargs["enable_log_requests"])
                    self.assertTrue(kwargs["aggregate_engine_logging"])
                    self.assertFalse(kwargs["disable_log_stats"])
                    events.append("launch")
                    if failure == "factory":
                        raise RuntimeError("factory failed")
                    return engine

                async def run():
                    async with build_async_engine_client_from_engine_args(
                        engine_args, Logger(), stat_loggers=stat_loggers,
                    ) as client:
                        self.assertIs(engine, client)
                        events.append("body")
                        if failure == "body":
                            raise RuntimeError("body failed")

                # Device autodetection is part of GPU startup. Keep the real
                # args/config types and factory signature, replacing that startup.
                with patch.object(engine_args, "create_engine_config", create_config), \
                        patch.object(AsyncLLM, "from_vllm_config", launch):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                            asyncio.run(run())
                    else:
                        asyncio.run(run())
                self.assertEqual({
                    None: ["launch", "reset", "body", "shutdown"],
                    "factory": ["launch"],
                    "reset": ["launch", "reset", "shutdown"],
                    "body": ["launch", "reset", "body", "shutdown"],
                }[failure], events)

    def test_real_stats_reach_the_triton_metric_writer(self):
        # Catches renamed stats fields and silent loss of token/cache/latency data.
        families = {}

        def metric_family(**kwargs):
            family = MetricFamily(**kwargs)
            families[family.name] = family
            return family

        for name in ("COUNTER", "GAUGE", "HISTOGRAM"):
            setattr(metric_family, name, getattr(MetricFamily, name))
        scheduler = SchedulerStats(
            num_running_reqs=3, num_waiting_reqs=2, kv_cache_usage=0.75,
            prefix_cache_stats=PrefixCacheStats(queries=100, hits=80),
            kv_cache_eviction_events=[KVCacheEvictionEvent(1.0, 0.5, ())],
        )
        iteration = IterationStats()
        iteration.prompt_token_stats.computed = 6
        iteration.prompt_token_stats.total = 10
        iteration.num_generation_tokens = 2
        iteration.num_preempted_reqs = 1
        iteration.time_to_first_tokens_iter = [0.125]
        iteration.inter_token_latencies_iter = [0.25]
        iteration.n_params_iter = [1]
        iteration.finished_requests.append(FinishedRequestStats(
            finish_reason=FinishReason.STOP, e2e_latency=1.5,
            num_prompt_tokens=10, num_generation_tokens=2,
            queued_time=0.125, prefill_time=0.25, decode_time=0.5,
            inference_time=0.75, num_cached_tokens=4,
        ))
        with patch.object(pb_utils, "MetricFamily", metric_family, create=True):
            factory = VllmStatLoggerFactory({"model": "runtime-smoke"}, Logger())
            logger = factory(self.config, 0)
            try:
                logger.record(scheduler, iteration, MultiModalCacheStats(queries=4, hits=3))
            finally:
                # Joins the real adapter thread, so all writes have completed.
                factory.finalize()
        expected = {
            "num_requests_running": [3], "num_requests_waiting": [2],
            "kv_cache_usage_ratio": [0.75], "prefix_cache_queries_total": [100],
            "prefix_cache_hits_total": [80], "kv_cache_evictions_total": [1],
            "mm_cache_queries_total": [4], "mm_cache_hits_total": [3],
            "prompt_tokens_total": [10], "generation_tokens_total": [2],
            "preemptions_total": [1], "time_to_first_token_seconds": [0.125],
            "time_per_output_token_seconds": [0.25],
            "e2e_request_latency_seconds": [1.5], "request_prompt_tokens": [10],
            "request_generation_tokens": [2], "request_params_n": [1],
            "request_queue_time_seconds": [0.125], "request_prefill_time_seconds": [0.25],
            "request_decode_time_seconds": [0.5], "request_inference_time_seconds": [0.75],
            "request_cached_prompt_tokens": [4],
        }
        for name, samples in expected.items():
            self.assertEqual(samples, families[f"vllm:{name}"].samples, name)


if __name__ == "__main__":
    print(f"CPU vLLM API contracts: vLLM {vllm.__version__}", flush=True)
    unittest.main(verbosity=2)
