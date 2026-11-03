"""Processor facts are counted once, independently of text usage and reservations."""
import asyncio
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / 'backends/vllm_multimodal/utils/vision_usage.py'
spec = importlib.util.spec_from_file_location('visual_usage_backend', MODULE_PATH)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


@dataclass
class Placeholder:
    offset: int = 1
    length: int = 5
    is_embed: object = (True, False, True, False, True)

    def get_num_embeds(self):
        return sum(self.is_embed) if self.is_embed is not None else self.length


def prepared(ranges=None):
    return {'type': 'multimodal', 'prompt_token_ids': list(range(8)),
            'mm_placeholders': {'image': ranges if ranges is not None else [Placeholder()]}}


def test_masked_positions_are_actual_embeddings_not_span_length():
    usage = backend.extract_vision_usage(prepared(), {'image': ('pdf',)})
    assert usage['by_source'] == {'pdf': 3}
    assert usage['unit'] == 'embedding_positions'


@pytest.mark.parametrize('ranges,origins', [
    ([Placeholder(offset=5)], {'image': ('image',)}),
    ([Placeholder(is_embed=(True,))], {'image': ('image',)}),
    ([Placeholder()], {'image': ('image', 'pdf')}),
    ([Placeholder()], {'image': ('unknown',)}),
    ([Placeholder(), Placeholder(offset=2)], {'image': ('image', 'pdf')}),
])
def test_invalid_facts_are_unavailable(ranges, origins):
    usage = backend.extract_vision_usage(prepared(ranges), origins)
    assert usage['status'] == 'unavailable'
    assert usage['by_source'] == {}


@pytest.fixture(autouse=True)
def native_api(monkeypatch):
    module = ModuleType('vllm.renderers.inputs.preprocess')
    module.parse_model_prompt = lambda config, prompt: prompt
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_preparer_returns_native_identity_and_renders_once():
    result = prepared()
    seen = []
    async def render(prompts, tok_params):
        seen.append((prompts, tok_params))
        return [result]
    engine = SimpleNamespace(model_config=object(), renderer=SimpleNamespace(render_cmpl_async=render, default_cmpl_tok_params=object()))
    built = backend.BuiltMediaPrompt({'prompt': 'x'}, {'image': ('image',)})
    ready = asyncio.run(backend.make_prompt_preparer(engine)(built))
    assert ready.prompt is result
    assert ready.vision_usage['by_source'] == {'image': 3}
    assert len(seen) == 1


def test_unsupported_runtime_retains_raw_identity():
    built = backend.BuiltMediaPrompt({'prompt': 'x'}, {'image': ('image',)})
    assert backend.make_prompt_preparer(object()) is None


def test_renderer_failure_is_not_retried_raw():
    async def render(*args, **kwargs):
        raise RuntimeError('renderer failed')
    engine = SimpleNamespace(model_config=object(), renderer=SimpleNamespace(render_cmpl_async=render, default_cmpl_tok_params=object()))
    with pytest.raises(RuntimeError, match='renderer failed'):
        asyncio.run(backend.make_prompt_preparer(engine)(backend.BuiltMediaPrompt('x', {'image': ('image',)})))


def metadata_payload(vision, finished=False):
    return {'version': 1, 'finished': finished, 'prompt_tokens': 8,
            'vision_usage': vision,
            'outputs': [{'index': 0, 'completion_tokens': 1,
                         'finish_reason': 'stop' if finished else None}]}


def actual(kind='image', count=3):
    return {'status': 'available', 'source': 'engine',
            'unit': 'embedding_positions', 'by_source': {kind: count}}


def test_wire_usage_is_immutable_and_does_not_add_prompt_tokens():
    from gateway.generation_metadata import GenerationMetadataState, apply_engine_usage, GenerationMetadataError
    state = GenerationMetadataState()
    state.update_json(metadata_payload(actual()))
    usage = apply_engine_usage({'prompt_tokens': 8, 'completion_tokens': 1, 'total_tokens': 9}, state.metadata)
    assert usage['vision_usage']['main']['by_source'] == {'image': 3}
    assert usage['total_tokens'] == 9
    with pytest.raises(GenerationMetadataError):
        state.update_json(metadata_payload(actual(count=4), True))


@pytest.mark.parametrize('vision', [actual(count=True), actual(count=-1), actual('url'),
    {**actual(), 'status': 'unavailable'}, {**actual(), 'unit': 'tokens'},
    {**actual(), 'status': []}, {**actual(), 'status': {}}])
def test_invalid_wire_visual_usage_is_rejected(vision):
    from gateway.generation_metadata import decode_generation_metadata, GenerationMetadataError
    with pytest.raises(GenerationMetadataError):
        decode_generation_metadata(metadata_payload(vision))


def test_accumulator_sums_calls_and_keeps_partial_on_cancellation():
    from gateway.generation_metadata import decode_generation_metadata
    from gateway.generation_telemetry import VisualUsageAccumulator
    accumulator = VisualUsageAccumulator()
    first = accumulator.begin('preprocessing')
    second = accumulator.begin('preprocessing')
    main = accumulator.begin('main')
    facts = decode_generation_metadata(metadata_payload(actual('pdf', 3), True))
    for _ in range(3):
        accumulator.observe(first, facts)
    accumulator.end(first, 'success')
    accumulator.end(first, 'success')
    accumulator.observe(second, facts)
    accumulator.end(second, 'success')
    accumulator.observe(main, decode_generation_metadata(metadata_payload(actual('image', 7))))
    accumulator.end(main, 'cancelled')
    result = accumulator.snapshot()
    assert result['preprocessing']['by_source'] == {'pdf': 6}
    assert result['preprocessing']['status'] == 'available'
    assert result['main']['by_source'] == {'image': 7}
    assert result['main']['status'] == 'partial'


def test_responses_carries_visual_detail_unchanged():
    from gateway.responses import usage_object
    details = {'main': actual(), 'preprocessing': {'status': 'unavailable', 'by_source': {}}}
    result = usage_object({'prompt_tokens': 8, 'completion_tokens': 1, 'total_tokens': 9, 'vision_usage': details})
    assert result['vision_usage'] == details
    assert result['total_tokens'] == 9


def test_extraction_failure_keeps_prepared_input():
    result = prepared([Placeholder(is_embed=(True,))])
    async def render(*args):
        return [result]
    engine = SimpleNamespace(model_config=object(), renderer=SimpleNamespace(
        render_cmpl_async=render, default_cmpl_tok_params=object()))
    ready = asyncio.run(backend.make_prompt_preparer(engine)(backend.BuiltMediaPrompt('x', {'image': ('image',)})))
    assert ready.prompt is result
    assert ready.vision_usage['status'] == 'unavailable'


def test_actual_and_reservation_metrics_are_separate_and_finish_idempotent():
    import time
    from gateway.generation_metadata import decode_generation_metadata
    from gateway.generation_telemetry import GenerationTelemetry
    from gateway.metrics import VISUAL_EMBEDDING_POSITIONS, VISUAL_CONTEXT_RESERVED, VISUAL_USAGE_STATUS
    telemetry = GenerationTelemetry('private-id', '/private', time.monotonic(), model='fixture', route='chat')
    actual_metric = VISUAL_EMBEDDING_POSITIONS.labels('fixture', 'chat', 'pdf', 'preprocessing', 'engine')
    reserve_metric = VISUAL_CONTEXT_RESERVED.labels('fixture', 'chat', 'main', 'estimated')
    status_metric = VISUAL_USAGE_STATUS.labels('fixture', 'chat', 'main', 'unavailable')
    before = (actual_metric._value.get(), reserve_metric._value.get(), status_metric._value.get())
    call = telemetry.visual_usage.begin('preprocessing')
    metadata = decode_generation_metadata(metadata_payload(actual('pdf', 3), True))
    telemetry.visual_usage.observe(call, metadata)
    telemetry.visual_usage.observe(call, metadata)
    telemetry.visual_usage.end(call, 'success')
    main = telemetry.visual_usage.begin('main')
    telemetry.visual_usage.end(main, 'cancelled')
    telemetry.visual_usage.reserve('main', 100)
    telemetry.fail('cancelled')
    summary = telemetry.finish(499)
    assert telemetry.finish(499) is None
    assert actual_metric._value.get() - before[0] == 3
    assert reserve_metric._value.get() - before[1] == 100
    assert status_metric._value.get() - before[2] == 1
    assert summary['vision_usage']['preprocessing']['by_source'] == {'pdf': 3}


def test_partial_available_and_legacy_calls_do_not_invent_zero_actual():
    from gateway.generation_metadata import decode_generation_metadata
    from gateway.generation_telemetry import VisualUsageAccumulator
    accumulator = VisualUsageAccumulator()
    first = accumulator.begin('preprocessing')
    second = accumulator.begin('preprocessing')
    accumulator.observe(first, decode_generation_metadata(metadata_payload(actual('video', 6), True)))
    accumulator.end(first, 'success')
    accumulator.end(second, 'success')
    assert accumulator.snapshot()['preprocessing'] == {
        'status': 'partial', 'source': 'engine', 'unit': 'embedding_positions',
        'by_source': {'video': 6}, 'calls': 2,
    }


def test_auxiliary_visual_summary_preserves_origin_and_scope(monkeypatch):
    import time
    from gateway import vllm_media
    from gateway.generation_telemetry import (
        start_generation_telemetry, reset_generation_telemetry, get_generation_telemetry,
        begin_visual_call, finish_visual_call, observe_visual_metadata,
    )
    from gateway.generation_metadata import decode_generation_metadata
    monkeypatch.setattr('gateway.prompt.render_chat_prompt', lambda *args: 'prompt')
    monkeypatch.setattr(vllm_media, '_summary_sampling', lambda *args: {})
    async def generate(model, prompt, sampling, images, *, media):
        assert [image.source_kind for image in media.images] == ['pdf', 'video']
        call = begin_visual_call()
        observe_visual_metadata(call, decode_generation_metadata(metadata_payload(actual('pdf', 3), True)))
        finish_visual_call(call, 'success')
        return 'summary'
    monkeypatch.setattr('gateway.triton_client.call_triton_multimodal', generate)
    token = start_generation_telemetry('id', '/chat', time.monotonic())
    try:
        result = asyncio.run(vllm_media._generate_visual_summary('m', object(), [
            vllm_media.VisualItem('private label', 'a', source_kind='pdf'),
            vllm_media.VisualItem('unrelated label', 'b', source_kind='video'),
        ], 'question', [], {}, object()))
        assert result == 'summary'
        assert set(get_generation_telemetry().visual_usage.snapshot()) == {'preprocessing'}
    finally:
        reset_generation_telemetry(token)


def test_encoder_decoder_multimodal_discriminator_is_not_decoder_usage():
    engine_input = {**prepared(), 'encoder_prompt_token_ids': [10, 11, 12, 13, 14, 15]}
    assert backend.extract_vision_usage(engine_input, {'image': ('image',)})['status'] == 'unavailable'
def test_visual_count_is_bounded_after_carrying_forward_prompt_count():
    from gateway.generation_metadata import GenerationMetadataState, GenerationMetadataError
    from test_generation_metadata_transport import envelope
    state = GenerationMetadataState()
    first = envelope(finished=False, reason=None)
    first['prompt_tokens'] = 2
    state.update_json(first)
    second = envelope()
    second.pop('prompt_tokens')
    second['vision_usage'] = {'status': 'available', 'source': 'engine',
        'unit': 'embedding_positions', 'by_source': {'image': 3}}
    with pytest.raises(GenerationMetadataError):
        state.update_json(second)
