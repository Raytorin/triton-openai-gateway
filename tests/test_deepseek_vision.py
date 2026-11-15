"""Opt-in pixels cross the real gateway/Triton input boundary; no GPU needed."""
import base64
from copy import deepcopy
from io import BytesIO
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from PIL import Image
import pytest

import gateway.triton_client as tc
from gateway.multimodal import extract_media_payloads, scope_media_history
from test_deepseek_protocol import endpoint, anyio_backend, post, body, TOOLS, generation

BUILD_INPUTS = tc._build_grpc_generate_inputs


def image(color):
    target = BytesIO()
    Image.new('RGB', (8, 6), color).save(target, format='PNG')
    return {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,' + base64.b64encode(target.getvalue()).decode()}}


def enable(endpoint, limit=2):
    original_encode = endpoint.tokenizer.encode
    endpoint.tokenizer.encode = lambda text, **kwargs: [
        token for part in re.split('(<｜deepseek_image｜>)', text)
        for token in ([129264] if part == '<｜deepseek_image｜>' else original_encode(part, **kwargs))]
    (endpoint.path / 'config.json').write_text(json.dumps({
        'model_type': 'deepseek_v41', 'architectures': ['DeepseekV41ForCausalLM'],
        'image_token_id': 129264, 'vision_config': {
            'model_type': 'deepseek_v41_vision', 'max_image_tokens': 1024,
            'patch_size': 14, 'downsample_ratio': 3}}))
    (endpoint.path / 'model.json').write_text(json.dumps({
        'model': str(endpoint.path), 'language_model_only': False,
        'enable_mm_embeds': False, 'limit_mm_per_prompt': {'image': limit},
        'max_model_len': 32768}))


@pytest.mark.anyio
@pytest.mark.parametrize('route', ['/v1/chat/completions', '/v1/responses'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
async def test_opt_in_pixels_reach_real_triton_inputs(endpoint, route, stream, thinking):
    enable(endpoint)
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'separate' if thinking else 'disabled'}}))
    picture = image('red')
    if route.endswith('responses'):
        request = {'model': 'deepseek', 'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': picture['image_url']['url']}]}]}
    else:
        request = body(messages=[{'role': 'user', 'content': [picture]}])
    captured = []
    def inputs(*args, **kwargs):
        tensors = BUILD_INPUTS(*args, **kwargs)
        captured.extend(tensors)
        # Keep protocol fixture's deterministic fake engine, using actual tensors above.
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post({**request, 'stream': stream}, route)
    assert response.status_code == 200, response.text
    if stream:
        assert 'event: error' not in response.text and '"error": {' not in response.text
    else:
        assert not response.json().get('error')
    tensors = {item.name(): item for item in captured}
    from tritonclient.utils import deserialize_bytes_tensor
    pixels = deserialize_bytes_tensor(tensors['image']._get_content())[0]
    encoded = json.loads(pixels)
    with Image.open(BytesIO(base64.b64decode(encoded['data']))) as decoded:
        assert decoded.size == (8, 6)
        assert decoded.convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    assert '<｜deepseek_image｜>' in endpoint.records[0][0]


@pytest.mark.anyio
@pytest.mark.parametrize('route', ['/v1/chat/completions', '/v1/responses'])
@pytest.mark.parametrize('stream', [False, True])
async def test_visual_usage_and_reservation_cross_public_api(endpoint, monkeypatch, route, stream):
    from test_generation_metadata_transport import Result, envelope
    # Metadata discovery is the only external boundary besides generation.
    client = SimpleNamespace(get_model_metadata=AsyncMock(return_value={
        'inputs': [{'name': 'return_generation_metadata'}],
        'outputs': [{'name': 'generation_metadata'}]}))
    monkeypatch.setattr(tc, 'get_grpc_client', AsyncMock(return_value=client))
    enable(endpoint)
    (endpoint.path / 'gateway.json').write_text('{"reasoning":{"mode":"disabled"}}')
    async def wire(*args, **kwargs):
        data = {**envelope(), 'vision_usage': {'status': 'available', 'source': 'engine',
            'unit': 'embedding_positions', 'by_source': {'image': 3}}}
        yield Result(b'answer', data)
    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    picture = image('red')
    if route.endswith('responses'):
        request = {'model': 'deepseek', 'input': [{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': picture['image_url']['url']}]}]}
    else:
        request = body(messages=[{'role': 'user', 'content': [picture]}])
    response = await post({**request, 'stream': stream}, route)
    assert response.status_code == 200, response.text
    if stream:
        events = [json.loads(line[5:]) for line in response.text.splitlines()
                  if line.startswith('data:') and line[5:].strip() != '[DONE]']
        usage = (events[-1]['response']['usage'] if route.endswith('responses') else events[-1]['usage'])
    else:
        usage = response.json()['usage']
    assert usage['vision_usage']['main'] == {'status': 'available', 'source': 'engine',
        'unit': 'embedding_positions', 'by_source': {'image': 3}, 'calls': 1}
    assert usage['vision_context_reservation'] == {'source': 'estimated',
        'unit': 'context_positions', 'by_scope': {'main': 1023}}
    assert usage.get('prompt_tokens', usage.get('input_tokens')) == 42
    assert usage['total_tokens'] == 43


@pytest.mark.anyio
@pytest.mark.parametrize('content', [
    [image('red'), image('blue'), image('green')],
    [{'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,eA=='}}],
    [{'type': 'input_audio', 'input_audio': {'data': 'eA==', 'format': 'wav'}}],
    [{'type': 'image_url', 'image_url': {'url': 'data:application/pdf;base64,eA=='}}],
])
async def test_bad_modality_or_image_count_rejected_before_engine(endpoint, content):
    enable(endpoint)
    response = await post(body(messages=[{'role': 'user', 'content': content}]))
    assert response.status_code == 400, response.text
    assert not endpoint.records


def test_latest_user_turn_keeps_tool_images_without_mutating_input():
    conversation = [{'role': 'user', 'content': [image('blue')]},
        {'role': 'assistant', 'content': 'old'},
        {'role': 'user', 'content': 'inspect tool output'},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [image('red')]}]
    original = deepcopy(conversation)
    scoped, removed = scope_media_history(conversation, keep_active_tool_media=True)
    assert removed == 1
    assert [p.data for p in extract_media_payloads(scoped).images] == [
        image('red')['image_url']['url'].split(',', 1)[1]]
    assert conversation == original


def test_default_media_scoping_preserves_legacy_non_deepseek_tool_behavior():
    conversation = [{'role': 'user', 'content': 'inspect tool output'},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [image('red')]}]
    scoped, removed = scope_media_history(conversation)
    assert removed == 1
    assert not extract_media_payloads(scoped).has_any


def test_canonical_tool_order_aligns_interleaved_user_images_without_mutation():
    from gateway.deepseek_vision import canonicalize_conversation
    conversation = [{'role': 'assistant', 'tool_calls': [
        {'id': 'a'}, {'id': 'b'}]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': [image('blue')]},
        {'role': 'user', 'content': [image('green')]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [image('red')]}]
    original = deepcopy(conversation)
    canonical = canonicalize_conversation(conversation)
    assert [p.data for p in extract_media_payloads(canonical).images] == [
        image(color)['image_url']['url'].split(',', 1)[1] for color in ('red', 'green', 'blue')]
    assert conversation == original


def test_deepseek_reservation_counts_full_grid_span_minus_exact_placeholder():
    from gateway.deepseek_vision import VisionPolicy, additional_image_tokens
    # Native max_image_tokens already includes row/start/end delimiters.
    class Tokenizer:
        def encode(self, value, **kwargs):
            return [129264] * value.count('<｜deepseek_image｜>') + [7]
    policy = VisionPolicy(image_limit=3, max_image_tokens=1024)
    media = extract_media_payloads([{'role': 'user', 'content': [image('red'), image('blue')]}])
    assert additional_image_tokens(policy, Tokenizer(), 'x<｜deepseek_image｜><｜deepseek_image｜>', media) == 2046


def test_backend_opt_in_accepts_images_but_rejects_processor_overrides(monkeypatch):
    from test_deepseek_runtime import load_policy
    policy = load_policy(monkeypatch)()
    policy._vision_image_limit = 2
    tensor = lambda data: SimpleNamespace(as_numpy=lambda: np.asarray(data, dtype=np.object_))
    assert policy._validate_request_task_name({'image': tensor([b'a', b'b']),
        'media_parameters': tensor([b'{"media_order":["image","image"]}'])}) == 'generate'
    for params in ({'mm_processor_kwargs': {'max_image_tokens': 9999}}, {'image_max_pixels': 200}):
        with pytest.raises(ValueError, match='media_parameters'):
            policy._validate_request_task_name({'image': tensor([b'a']),
                'media_parameters': tensor([json.dumps(params).encode()])})


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('history', ['all', 'latest'])
async def test_tool_result_pixels_follow_call_order_in_actual_wire_inputs(endpoint, stream, history):
    enable(endpoint, 3)
    (endpoint.path / 'gateway.json').write_text(json.dumps({'multimodal': {
        'media_history_mode': history, 'focus_current_media': False}}))
    messages = [{'role': 'user', 'content': 'inspect'},
        {'role': 'assistant', 'tool_calls': [
            {'id': name, 'type': 'function', 'function': {'name': 'typed', 'arguments': '{}'}}
            for name in ('a', 'b')]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': [image('blue')]},
        {'role': 'user', 'content': [image('green')]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [image('red')]}]
    original = deepcopy(messages)
    captured = []
    def inputs(*args, **kwargs):
        captured.extend(BUILD_INPUTS(*args, **kwargs))
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(body(messages=messages, tools=TOOLS, stream=stream))
    assert response.status_code == 200, response.text
    from tritonclient.utils import deserialize_bytes_tensor
    tensor = next(item for item in captured if item.name() == 'image')
    actual = []
    for encoded in deserialize_bytes_tensor(tensor._get_content()):
        with Image.open(BytesIO(base64.b64decode(json.loads(encoded)['data']))) as picture:
            actual.append(picture.convert('RGB').getpixel((0, 0)))
    want = [(255, 0, 0), (0, 128, 0), (0, 0, 255)] if history == 'all' else [(255, 0, 0), (0, 128, 0)]
    assert actual == want
    assert endpoint.records[0][0].count('<｜deepseek_image｜>') == len(want)
    assert messages == original


@pytest.mark.anyio
@pytest.mark.parametrize('focus', [False, True])
@pytest.mark.parametrize('stream', [False, True])
async def test_image_limit_counts_latest_retained_turn(endpoint, focus, stream):
    enable(endpoint, limit=1)
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'}, 'multimodal': {
            'media_history_mode': 'latest', 'focus_current_media': focus}}))
    messages = [{'role': 'user', 'content': [image('blue')]},
        {'role': 'assistant', 'content': 'previous answer'},
        {'role': 'user', 'content': [image('red')]}]
    captured = []
    def inputs(*args, **kwargs):
        captured.extend(BUILD_INPUTS(*args, **kwargs))
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(body(messages=messages, stream=stream))
    assert response.status_code == 200, response.text
    assert '"error":' not in response.text
    from tritonclient.utils import deserialize_bytes_tensor
    tensor = next(item for item in captured if item.name() == 'image')
    pixels = deserialize_bytes_tensor(tensor._get_content())
    assert len(pixels) == 1
    with Image.open(BytesIO(base64.b64decode(json.loads(pixels[0])['data']))) as retained:
        assert retained.convert('RGB').getpixel((0, 0)) == (255, 0, 0)
    assert endpoint.records[0][0].count('<｜deepseek_image｜>') == 1


@pytest.mark.anyio
async def test_image_limit_still_rejects_retained_overflow(endpoint):
    enable(endpoint, limit=1)
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'}, 'multimodal': {
            'media_history_mode': 'all', 'focus_current_media': False}}))
    messages = [{'role': 'user', 'content': [image('blue')]},
        {'role': 'assistant', 'content': 'previous answer'},
        {'role': 'user', 'content': [image('red')]}]
    response = await post(body(messages=messages))
    assert response.status_code == 400
    assert 'maximum is 1' in response.json()['detail']
    assert not endpoint.records


@pytest.mark.anyio
async def test_active_image_overflow_rejected_before_summary_engine(endpoint):
    enable(endpoint, limit=1)
    config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({**config, 'max_model_len': 4000}))
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'},
        'multimodal': {'media_history_mode': 'latest', 'focus_current_media': False},
        'context_compression': {'mode': 'summarize', 'preserve_recent_messages': 0,
            'summary_max_tokens': 32, 'fallback_mode': 'disabled', 'cache_size': 0}}))
    response = await post(body(messages=[
        {'role': 'user', 'content': 'previous text ' * 3000},
        {'role': 'assistant', 'content': 'previous answer'},
        {'role': 'user', 'content': [image('red'), image('blue')]}], max_tokens=200))
    assert response.status_code == 400, response.text
    assert 'maximum is 1' in response.json()['detail']
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize('route', ['/v1/chat/completions', '/v1/responses'])
@pytest.mark.parametrize('image_limit', [1, 2])
async def test_truncation_reserves_only_retained_deepseek_images(endpoint, route, image_limit):
    enable(endpoint, limit=image_limit)
    config = json.loads((endpoint.path / 'model.json').read_text())
    config['max_model_len'] = 1600
    (endpoint.path / 'model.json').write_text(json.dumps(config))
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'}, 'multimodal': {
            'media_history_mode': 'all', 'focus_current_media': False},
        'context_compression': {'mode': 'truncate'}}))
    messages = [{'role': 'user', 'content': [image('blue')]},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': [image('red')]}]
    if route.endswith('responses'):
        messages = [dict(message, content=[{'type': 'input_image',
            'image_url': part['image_url']['url']} for part in message['content']])
            if isinstance(message['content'], list) else message for message in messages]
        request = {'model': 'deepseek', 'input': messages, 'max_output_tokens': 200,
                   'truncation': 'auto'}
    else:
        request = body(messages=messages, max_tokens=200)
    captured = []
    def inputs(*args, **kwargs):
        captured.extend(BUILD_INPUTS(*args, **kwargs))
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(request, route)
    assert response.status_code == 200, response.text
    assert endpoint.records[0][0].count('<｜deepseek_image｜>') == 1
    assert response.headers['x-output-token-limit'] == '200'


@pytest.mark.anyio
async def test_chat_shortens_automatic_output_for_protected_deepseek_image(endpoint):
    enable(endpoint, limit=1)
    model_config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({
        **model_config, 'max_model_len': 1300,
    }))
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'},
        'generation': {'default_output_tokens': 200},
        'context_compression': {'mode': 'truncate'},
    }))

    def inputs(prompt, sampling, pictures, stream, **kwargs):
        assert len(kwargs["media"].images) == 1
        return [SimpleNamespace(prompt=prompt, sampling=sampling, stream=stream)]

    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(body(messages=[{'role': 'user', 'content': [image('red')]}]))
    assert response.status_code == 200, response.text
    prompt, sampling, _ = endpoint.records[0]
    available = 1300 - 64 - 1023 - len(endpoint.tokenizer.encode(prompt))
    assert 0 < available < 200
    assert sampling['max_tokens'] == available
    assert response.headers['x-output-token-limit'] == str(available)

@pytest.mark.anyio
async def test_chat_can_reclaim_historical_image_reserve_for_full_output(endpoint):
    enable(endpoint, limit=3)
    model_config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({
        **model_config, 'max_model_len': 2800,
    }))
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'},
        'generation': {'default_output_tokens': 1200},
        'multimodal': {'media_history_mode': 'all', 'focus_current_media': False},
        'context_compression': {'mode': 'truncate'},
    }))
    messages = [
        {'role': 'user', 'content': [image('blue')]},
        {'role': 'assistant', 'content': 'previous answer'},
        {'role': 'user', 'content': [image('red')]},
    ]

    def inputs(prompt, sampling, pictures, stream, **kwargs):
        return [SimpleNamespace(prompt=prompt, sampling=sampling, stream=stream)]

    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(body(messages=messages))
    assert response.status_code == 200, response.text
    prompt, sampling, _ = endpoint.records[0]
    assert prompt.count('<｜deepseek_image｜>') == 1
    assert sampling['max_tokens'] == 1200
    assert response.headers['x-output-token-limit'] == '1200'


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('history', ['all', 'latest'])
async def test_truncation_selects_original_turns_before_tool_image_ordering(endpoint, stream, history):
    enable(endpoint, 3)
    config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({**config, 'max_model_len': 2600}))
    (endpoint.path / 'gateway.json').write_text(json.dumps({
        'reasoning': {'mode': 'disabled'}, 'multimodal': {
            'media_history_mode': history, 'focus_current_media': False},
        'context_compression': {'mode': 'truncate'}}))
    messages = [{'role': 'user', 'content': 'HISTORICAL ' * 6000},
        {'role': 'assistant', 'tool_calls': [
            {'id': name, 'type': 'function', 'function': {'name': 'typed', 'arguments': '{}'}}
            for name in ('a', 'b')]},
        {'role': 'tool', 'tool_call_id': 'b', 'content': [image('blue')]},
        {'role': 'user', 'content': [image('green')]},
        {'role': 'tool', 'tool_call_id': 'a', 'content': [image('red')]}]
    original = deepcopy(messages)
    captured = []
    def inputs(*args, **kwargs):
        captured.extend(BUILD_INPUTS(*args, **kwargs))
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(body(messages=messages, tools=TOOLS, max_tokens=200, stream=stream))
    assert response.status_code == 200, response.text
    from tritonclient.utils import deserialize_bytes_tensor
    tensor = next(item for item in captured if item.name() == 'image')
    actual = []
    for encoded in deserialize_bytes_tensor(tensor._get_content()):
        with Image.open(BytesIO(base64.b64decode(json.loads(encoded)['data']))) as picture:
            actual.append(picture.convert('RGB').getpixel((0, 0)))
    assert actual == [(0, 128, 0), (255, 0, 0)]
    assert endpoint.records[0][0].count('<｜deepseek_image｜>') == 2
    assert 'HISTORICAL' not in endpoint.records[0][0]
    assert messages == original


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_responses_auto_keeps_current_images_after_supported_function_results(endpoint, stream):
    enable(endpoint, 3)
    config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({**config, 'max_model_len': 2600}))
    (endpoint.path / 'gateway.json').write_text('{"reasoning":{"mode":"disabled"}}')
    def content(color):
        return {'type': 'input_image', 'image_url': image(color)['image_url']['url']}
    request = {'model': 'deepseek', 'max_output_tokens': 200, 'truncation': 'auto', 'stream': stream,
        'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'HISTORICAL ' * 6000}, content('blue')]},
            {'type': 'function_call', 'call_id': 'a', 'name': 'typed', 'arguments': '{}'},
            {'type': 'function_call', 'call_id': 'b', 'name': 'typed', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'b', 'output': 'historical B'},
            {'type': 'function_call_output', 'call_id': 'a', 'output': 'historical A'},
            {'role': 'user', 'content': [content('green'), content('red')]}]}
    captured = []
    def inputs(*args, **kwargs):
        captured.extend(BUILD_INPUTS(*args, **kwargs))
        return [SimpleNamespace(prompt=args[0], sampling=args[1], stream=kwargs['stream'])]
    with patch.object(tc, '_build_grpc_generate_inputs', inputs):
        response = await post(request, '/v1/responses')
    assert response.status_code == 200, response.text
    from tritonclient.utils import deserialize_bytes_tensor
    tensor = next(item for item in captured if item.name() == 'image')
    actual = []
    for encoded in deserialize_bytes_tensor(tensor._get_content()):
        with Image.open(BytesIO(base64.b64decode(json.loads(encoded)['data']))) as picture:
            actual.append(picture.convert('RGB').getpixel((0, 0)))
    assert actual == [(0, 128, 0), (255, 0, 0)]
    assert endpoint.records[0][0].count('<｜deepseek_image｜>') == 2
    assert 'HISTORICAL' not in endpoint.records[0][0]


@pytest.mark.parametrize('change', [
    {'language_model_only': False}, {'language_model_only': False, 'limit_mm_per_prompt': {'image': 0}},
    {'language_model_only': False, 'limit_mm_per_prompt': {'image': True}},
    {'language_model_only': False, 'limit_mm_per_prompt': {'image': 2, 'video': 1}},
    {'language_model_only': False, 'limit_mm_per_prompt': {'image': 2}, 'enable_mm_embeds': True},
    {'language_model_only': False, 'limit_mm_per_prompt': {'image': 2}, 'mm_processor_kwargs': {'max_image_tokens': 2}},
])
def test_backend_rejects_invalid_vision_opt_in(monkeypatch, change):
    from test_deepseek_runtime import load_policy
    cls = load_policy(monkeypatch)
    helper = cls._preflight_model.__globals__['_checkpoint']
    with pytest.raises(ValueError):
        helper.validate_vision_policy(change, {'model_type': 'deepseek_v41',
            'architectures': ['DeepseekV41ForCausalLM'], 'image_token_id': 129264,
            'vision_config': {'model_type': 'deepseek_v41_vision', 'max_image_tokens': 1024}})


@pytest.mark.anyio
async def test_wrong_tokenizer_image_ids_fail_closed_before_engine(endpoint):
    enable(endpoint)
    endpoint.tokenizer.encode = lambda text, **kwargs: [7] * max(1, len(text.split()))
    response = await post(body(messages=[{'role': 'user', 'content': [image('red')]}]))
    assert response.status_code == 500, response.text
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize('change', [
    {'language_model_only': 'false'}, {'limit_mm_per_prompt': {'image': 0}},
    {'enable_mm_embeds': True}, {'mm_processor_kwargs': {'max_image_tokens': 100000}},
])
async def test_invalid_gateway_vision_policy_fails_before_tokenizer(endpoint, change):
    enable(endpoint)
    config = json.loads((endpoint.path / 'model.json').read_text())
    (endpoint.path / 'model.json').write_text(json.dumps({**config, **change}))
    with patch.object(generation.registry, 'get_tokenizer_async',
            AsyncMock(side_effect=AssertionError('invalid policy reached tokenizer'))):
        response = await post(body(messages=[{'role': 'user', 'content': [image('red')]}]))
    assert response.status_code == 500, response.text
    assert not endpoint.records


def test_pinned_native_tokenizer_image_ids_and_reserve():
    snapshot = os.environ.get('DEEPSEEK_TOKENIZER_SNAPSHOT')
    if not snapshot:
        pytest.skip('set DEEPSEEK_TOKENIZER_SNAPSHOT for offline native image token parity')
    from transformers import TokenizersBackend
    from gateway.deepseek_vision import VisionPolicy, additional_image_tokens
    tokenizer = TokenizersBackend.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    conversation = [{'role': 'user', 'content': [image('red'), image('blue')]}]
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert ids.count(129264) == 2
    assert tokenizer.encode('<｜deepseek_image｜>', add_special_tokens=False) == [129264]
    assert additional_image_tokens(VisionPolicy(2, 1024), tokenizer, prompt,
        extract_media_payloads(conversation)) == 2046


def test_backend_vision_policy_accepts_only_verified_architecture(monkeypatch):
    from test_deepseek_runtime import load_policy
    helper = load_policy(monkeypatch)._preflight_model.__globals__['_checkpoint']
    engine = {'language_model_only': False, 'limit_mm_per_prompt': {'image': 2},
              'enable_mm_embeds': False}
    metadata = {'model_type': 'deepseek_v41', 'architectures': ['DeepseekV41ForCausalLM'],
        'image_token_id': 129264, 'vision_config': {
            'model_type': 'deepseek_v41_vision', 'max_image_tokens': 1024}}
    assert helper.validate_vision_policy(engine, metadata) == 2
    assert helper.validate_vision_policy({'language_model_only': True}, {}) == 0
    for key, value in [('model_type', 'other'), ('architectures', ['Foreign']),
                       ('image_token_id', 9), ('vision_config', {})]:
        with pytest.raises(ValueError):
            helper.validate_vision_policy(engine, {**metadata, key: value})


@pytest.mark.parametrize('count', [3, 0])
def test_backend_vision_limit_cannot_be_bypassed(monkeypatch, count):
    from test_deepseek_runtime import load_policy
    policy = load_policy(monkeypatch)()
    policy._vision_image_limit = 2 if count else 0
    request = {'image': SimpleNamespace(as_numpy=lambda: np.asarray([b'pixel'] * max(count, 1)))}
    with pytest.raises(ValueError):
        policy._validate_request_task_name(request)
