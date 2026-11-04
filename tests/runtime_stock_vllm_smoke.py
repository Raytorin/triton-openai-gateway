# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the stock Triton 26.10 / NVIDIA vLLM 0.31 backend.

Run directly in the actual image. ``--backend PATH`` selects an
alternate stock-backend copy for qualification. Missing dependencies fail;
only GPU engine execution and Triton's injected response/metric APIs are doubled.
With a read-only root, point ``VLLM_CACHE_ROOT`` at writable scratch space.
"""
import argparse
import asyncio
import importlib.util
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
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, PoolingOutput, PoolingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.renderers.hf import HfRenderer
from vllm.sampling_params import StructuredOutputsParams
from vllm.tokenizers.hf import get_cached_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import FinishReason
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.metrics.stats import FinishedRequestStats, IterationStats

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--backend', type=Path, default=Path('/opt/tritonserver/backends/vllm'))
options, test_arguments = parser.parse_known_args()
sys.argv[1:] = test_arguments
sys.path[:0] = [str(options.backend.resolve()), '/opt/tritonserver/backends/python']

import triton_python_backend_utils as pb_utils  # noqa: E402
from utils.metrics import VllmStatLoggerFactory  # noqa: E402
from utils.request import EmbedRequest, GenerateRequest  # noqa: E402
from utils.vllm_backend_utils import (  # noqa: E402
    TritonSamplingParams,
    build_async_engine_client_from_engine_args,
)

# Import the real facade too: auto-completion must work before engine creation.
spec = importlib.util.spec_from_file_location('stock_vllm_model', options.backend/'model.py')
stock_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stock_model)


class Logger:
    def __init__(self):
        self.errors = []

    def log_error(self, message):
        self.errors.append(message)

    def log_warn(self, message):
        pass


class Tensor:
    def __init__(self, name, array):
        self._name, self._array = name, array

    def name(self):
        return self._name

    def as_numpy(self):
        return self._array


class Request:
    def __init__(self, values):
        self._inputs = [Tensor(name, np.asarray([value])) for name, value in values.items()]

    def inputs(self):
        return self._inputs

    def request_id(self):
        return 'stock-runtime-smoke'


class Response:
    def __init__(self, *, output_tensors):
        self.outputs = {tensor.name(): tensor.as_numpy() for tensor in output_tensors}


class MetricFamily:
    COUNTER, GAUGE, HISTOGRAM = 'counter', 'gauge', 'histogram'

    def __init__(self, *, name, description, kind):
        self.name, self.samples = name, []

    def Metric(self, *, labels, buckets=None):
        self.buckets = buckets
        return self

    def increment(self, value):
        self.samples.append(value)

    def observe(self, value):
        self.samples.append(value)


async def collect(request):
    return [response async for response in request.execute()]


class StockVllmContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(directory.cleanup)
        cls.model_path = Path(directory.name)
        (cls.model_path/'config.json').write_text(json.dumps({
            'architectures': ['LlamaForCausalLM'], 'model_type': 'llama',
            'hidden_size': 32, 'intermediate_size': 64, 'num_hidden_layers': 1,
            'num_attention_heads': 2, 'num_key_value_heads': 2, 'vocab_size': 128,
            'max_position_embeddings': 64, 'bos_token_id': 1, 'eos_token_id': 2,
            'torch_dtype': 'float32',
        }))
        cls.config = VllmConfig(model_config=ModelConfig(
            model=str(cls.model_path), tokenizer=str(cls.model_path), dtype='float32',
            max_model_len=64, enforce_eager=True, skip_tokenizer_init=True,
        ), device_config=DeviceConfig(device='cpu'))
        tokenizer = Tokenizer(models.WordLevel({
            '<bos>': 0, '<unk>': 1, '<eos>': 2, 'Hello': 10, 'world': 20,
        }, unk_token='<unk>'))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        cls.renderer = HfRenderer(cls.config, get_cached_tokenizer(PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, bos_token='<bos>', eos_token='<eos>',
            unk_token='<unk>',
        )))
        cls.addClassCleanup(cls.renderer.shutdown)

    def test_numeric_sampling_strings_preserve_lora_selection(self):
        logger = Logger()
        params = TritonSamplingParams.from_dict(json.dumps({
            'max_tokens': '32', 'temperature': '0.7', 'seed': '7', 'lora_name': 'adapter',
        }), logger)
        self.assertIsNotNone(params, logger.errors)
        self.assertEqual((32, 0.7, 7, 'adapter'),
                         (params.max_tokens, params.temperature, params.seed, params.lora_name))
        self.assertEqual([], logger.errors)

    def test_structured_json_and_schema_keep_real_parameter_types(self):
        for payload in ({'json_object': True}, {'json': {
            'type': 'object', 'properties': {'answer': {'type': 'string'}},
            'required': ['answer'], 'additionalProperties': False,
        }}):
            with self.subTest(payload=payload):
                logger = Logger()
                params = TritonSamplingParams.from_dict(json.dumps({
                    'structured_outputs': json.dumps(payload),
                }), logger)
                self.assertIsNotNone(params, logger.errors)
                self.assertIsInstance(params.structured_outputs, StructuredOutputsParams)
                for name, value in payload.items():
                    self.assertEqual(value, getattr(params.structured_outputs, name))

    def test_generation_lora_dispatch_and_original_text_wire_output(self):
        dispatched = []
        render_inputs = []
        async def render(prompts):
            render_inputs.append(prompts)
            return await self.renderer.render_cmpl_async(prompts)
        async def generate(prompt, params, request_id, **kwargs):
            inspect.signature(AsyncLLM.generate).bind(None, prompt, params, request_id, **kwargs)
            self.assertEqual('token', prompt['type'])
            self.assertEqual([10], prompt['prompt_token_ids'])
            dispatched.append((prompt, params, kwargs['lora_request']))
            yield RequestOutput(request_id=request_id, prompt='Hello', prompt_token_ids=[10],
                prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='world',
                    token_ids=[20], cumulative_logprob=None, logprobs=None, finish_reason='stop')],
                finished=True)
        request = GenerateRequest(Request({'text_input': b'Hello',
            'sampling_parameters': json.dumps({'max_tokens': '1', 'lora_name': 'adapter'}).encode(),
            'exclude_input_in_output': True,
        }), generate, render, np.object_, Logger(), lora_repository={'adapter': '/models/adapter'},
            supported_loras=['adapter'])
        responses = asyncio.run(collect(request))
        self.assertEqual(1, len(responses))
        prompt, params, lora = dispatched[0]
        self.assertEqual([[{'prompt': 'Hello'}]], render_inputs)
        self.assertEqual('Hello', prompt['prompt'])
        self.assertIsInstance(params, TritonSamplingParams)
        self.assertEqual(1, params.max_tokens)
        self.assertIsInstance(lora, LoRARequest)
        self.assertEqual(('1', 1, '/models/adapter'),
                         (lora.lora_name, lora.lora_int_id, lora.lora_path))
        with patch.object(pb_utils, 'Tensor', Tensor, create=True), \
                patch.object(pb_utils, 'InferenceResponse', Response, create=True):
            wire = request.create_response(responses[0], {}, False).outputs
        self.assertEqual(b'world', wire['text_output'][0])

    def test_dense_embedding_preserves_original_pooling_and_wire_output(self):
        for value, expected_tokens in (([0, 10, 20, 2], [0, 10, 20, 2]),
                                       ('Hello world', [10, 20])):
            with self.subTest(value=value):
                self._assert_embedding_case(value, expected_tokens)

    def _assert_embedding_case(self, value, expected_tokens):
        dispatched = []
        render_inputs = []
        async def render(prompts):
            render_inputs.append(prompts)
            return await self.renderer.render_cmpl_async(prompts)
        async def encode(prompt, params, request_id):
            inspect.signature(AsyncLLM.encode).bind(None, prompt, params, request_id)
            # Signature binding also accepts a raw prompt. The real renderer's
            # EngineInput contract is what protects encode on vLLM >=0.30.
            self.assertEqual('token', prompt['type'])
            dispatched.append((prompt, params))
            yield PoolingRequestOutput(request_id=request_id,
                outputs=PoolingOutput(torch.tensor([0.25, 0.75])),
                prompt_token_ids=prompt['prompt_token_ids'], num_cached_tokens=0, finished=True)
        request = EmbedRequest(Request({'embedding_request': json.dumps({
            'input': value, 'pooling_params': {'dimensions': [2]},
        }).encode(), 'return_num_input_tokens': True, 'return_num_output_tokens': True}),
            encode, render, np.object_, Logger())
        responses = asyncio.run(collect(request))
        self.assertEqual(1, len(responses))
        prompt, params = dispatched[0]
        raw = {'prompt': value} if isinstance(value, str) else {'prompt_token_ids': value}
        self.assertEqual([[raw]], render_inputs)
        self.assertEqual(expected_tokens, prompt['prompt_token_ids'])
        self.assertIsInstance(params, PoolingParams)
        self.assertEqual(('embed', 2), (params.task, params.dimensions))
        with patch.object(pb_utils, 'Tensor', Tensor, create=True), \
                patch.object(pb_utils, 'InferenceResponse', Response, create=True):
            wire = request.create_response(responses[0]).outputs
        self.assertEqual([0.25, 0.75], json.loads(wire['text_output'][0]))
        self.assertEqual(len(expected_tokens), wire['num_input_tokens'].item())
        self.assertEqual(0, wire['num_output_tokens'].item())

    def test_renderer_failure_never_dispatches_or_retries_a_raw_prompt(self):
        calls = []
        async def render(prompts):
            calls.append('render')
            await self.renderer.render_cmpl_async(prompts)
            raise RuntimeError('renderer failed')
        def execute(*args, **kwargs):
            self.fail('Engine must not receive a raw prompt after renderer failure')
        for request in (
            GenerateRequest(Request({'text_input': b'Hello',
                'sampling_parameters': b'{"max_tokens": 1}',
            }), execute, render, np.object_, Logger()),
            EmbedRequest(Request({'embedding_request': b'{"input": "Hello"}'}),
                         execute, render, np.object_, Logger()),
        ):
            with self.subTest(request=type(request).__name__):
                before = len(calls)
                with self.assertRaisesRegex(RuntimeError, 'renderer failed'):
                    asyncio.run(collect(request))
                self.assertEqual(before + 1, len(calls))

    def test_engine_factory_contract_and_cleanup_on_success_and_failure(self):
        signature = inspect.signature(AsyncLLM.from_vllm_config)
        for failure in (None, 'factory', 'reset', 'body'):
            with self.subTest(failure=failure):
                events = []
                args = AsyncEngineArgs(model=str(self.model_path), enable_log_requests=True,
                    aggregate_engine_logging=True, disable_log_stats=False)
                loggers = [VllmStatLoggerFactory({'model': 'stock-runtime-smoke'}, Logger())]
                class Engine:
                    async def reset_mm_cache(self):
                        events.append('reset')
                        if failure == 'reset':
                            raise RuntimeError('reset failed')
                    def shutdown(self):
                        events.append('shutdown')
                engine = Engine()
                def create_config(*, usage_context):
                    self.assertEqual(UsageContext.OPENAI_API_SERVER, usage_context)
                    return self.config
                def launch(**kwargs):
                    signature.bind(**kwargs)
                    self.assertIs(self.config, kwargs['vllm_config'])
                    self.assertIs(loggers, kwargs['stat_loggers'])
                    self.assertEqual(UsageContext.OPENAI_API_SERVER, kwargs['usage_context'])
                    self.assertTrue(kwargs['enable_log_requests'])
                    self.assertTrue(kwargs['aggregate_engine_logging'])
                    self.assertFalse(kwargs['disable_log_stats'])
                    events.append('launch')
                    if failure == 'factory':
                        raise RuntimeError('factory failed')
                    return engine
                async def run():
                    async with build_async_engine_client_from_engine_args(args, Logger(),
                            stat_loggers=loggers) as client:
                        self.assertIs(engine, client)
                        events.append('body')
                        if failure == 'body':
                            raise RuntimeError('body failed')
                with patch.object(args, 'create_engine_config', create_config), \
                        patch.object(AsyncLLM, 'from_vllm_config', launch):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, f'{failure} failed'):
                            asyncio.run(run())
                    else:
                        asyncio.run(run())
                self.assertEqual({None: ['launch', 'reset', 'body', 'shutdown'],
                    'factory': ['launch'], 'reset': ['launch', 'reset', 'shutdown'],
                    'body': ['launch', 'reset', 'body', 'shutdown']}[failure], events)

    def test_original_eight_metric_families_keep_values_and_token_buckets(self):
        families = {}
        def metric_family(**kwargs):
            family = MetricFamily(**kwargs)
            families[family.name] = family
            return family
        for name in ('COUNTER', 'GAUGE', 'HISTOGRAM'):
            setattr(metric_family, name, getattr(MetricFamily, name))
        iteration = IterationStats()
        iteration.prompt_token_stats.computed = 6
        iteration.prompt_token_stats.total = 10
        iteration.num_generation_tokens = 2
        iteration.time_to_first_tokens_iter = [0.125]
        iteration.inter_token_latencies_iter = [0.25]
        iteration.n_params_iter = [1]
        iteration.finished_requests.append(FinishedRequestStats(finish_reason=FinishReason.STOP,
            e2e_latency=1.5, num_prompt_tokens=10, num_generation_tokens=2,
            queued_time=0.125, prefill_time=0.25, decode_time=0.5, inference_time=0.75,
            num_cached_tokens=4))
        with patch.object(pb_utils, 'MetricFamily', metric_family, create=True):
            factory = VllmStatLoggerFactory({'model': 'stock-runtime-smoke'}, Logger())
            try:
                factory(self.config, 0).record(None, iteration)
            finally:
                factory.finalize()
        self.assertEqual({
            'vllm:prompt_tokens_total': [10], 'vllm:generation_tokens_total': [2],
            'vllm:time_to_first_token_seconds': [0.125],
            'vllm:time_per_output_token_seconds': [0.25],
            'vllm:e2e_request_latency_seconds': [1.5], 'vllm:request_prompt_tokens': [10],
            'vllm:request_generation_tokens': [2], 'vllm:request_params_n': [1],
        }, {name: family.samples for name, family in families.items()})
        for name in ('vllm:request_prompt_tokens', 'vllm:request_generation_tokens'):
            self.assertEqual([1, 2, 5, 10, 20, 50], families[name].buckets)


if __name__ == '__main__':
    print(f'Stock CPU API contracts: vLLM {vllm.__version__}', flush=True)
    unittest.main(verbosity=2)
