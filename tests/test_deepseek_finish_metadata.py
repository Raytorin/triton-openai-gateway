# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Finish semantics through the pinned DeepSeek recipe and HTTP adapters."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from test_deepseek_protocol import body, chunks, endpoint, post


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def metadata_engine(monkeypatch):
    import gateway.triton_client as tc
    # Discovery and generation share the same remote boundary. A new client
    # identity keeps this advertised schema independent of other test models.
    client = SimpleNamespace(get_model_metadata=AsyncMock(return_value={
        'inputs': [{'name': 'return_generation_metadata', 'datatype': 'BOOL', 'shape': [1]}],
        'outputs': [{'name': 'generation_metadata', 'datatype': 'BYTES', 'shape': [1]}],
    }))
    monkeypatch.setattr(tc, 'get_grpc_client', AsyncMock(return_value=client))


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_empty_prefilled_reasoning_is_not_a_token_limit(endpoint, stream):
    endpoint.output = ''
    response = await post(body(stream=stream))
    assert response.status_code == 200, response.text
    if stream:
        reasons = [choice['finish_reason'] for event in chunks(response)
                   for choice in event.get('choices', []) if choice.get('finish_reason')]
        assert reasons == ['stop']
    else:
        assert response.json()['choices'][0]['finish_reason'] == 'stop'
        assert response.json()['usage']['completion_tokens'] == 0


def engine_response(text, *, finished=False, reason=None, prompt_tokens=29,
                    completion_tokens=3, stop_reason=None):
    """The remote engine's versioned output, independent of recipe parsing."""
    from types import SimpleNamespace
    import json
    import numpy as np

    envelope = {'version': 1, 'finished': finished, 'prompt_tokens': prompt_tokens,
                'outputs': [{'index': 0, 'finish_reason': reason, 'stop_reason': stop_reason,
                             'completion_tokens': completion_tokens,
                             'first_token_id': 17, 'last_token_id': 23}]}
    arrays = {'text_output': np.asarray([text.encode()], dtype=np.object_),
              'generation_metadata': np.asarray([json.dumps(envelope).encode()], dtype=np.object_)}
    return SimpleNamespace(as_numpy=lambda name: arrays.get(name))


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('raw,reason,budget', [
    ('reasoning remains open', 'stop', 1),
    ('short', 'length', 8192),
])
async def test_engine_finish_reason_and_counts_override_retokenization(endpoint, metadata_engine, monkeypatch, stream, raw, reason, budget):
    import gateway.triton_client as tc

    async def wire(*args, **kwargs):
        yield engine_response(raw, finished=True, reason=reason)

    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    response = await post(body(stream=stream, max_tokens=budget,
                               stream_options={'include_usage': True} if stream else None))
    assert response.status_code == 200, response.text
    if stream:
        events = chunks(response)
        reasons = [choice['finish_reason'] for event in events
                   for choice in event.get('choices', []) if choice.get('finish_reason')]
        assert reasons == [reason]
        usage = next(event['usage'] for event in events if event.get('usage'))
    else:
        result = response.json()
        assert result['choices'][0]['finish_reason'] == reason
        usage = result['usage']
    assert usage['prompt_tokens'] == 29
    assert usage['completion_tokens'] == 3
    assert usage['total_tokens'] == 32


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('engine_reason', ['stop', 'length'])
async def test_parser_eos_drains_metadata_only_final_before_reporting_usage(endpoint, metadata_engine, monkeypatch, stream, engine_reason):
    import gateway.triton_client as tc

    async def wire(*args, **kwargs):
        yield engine_response('Answer<｜end▁of▁sentence｜>', completion_tokens=4)
        yield engine_response('ignored after parser EOS', completion_tokens=5)
        yield engine_response('', finished=True, reason=engine_reason, completion_tokens=7, stop_reason=1)

    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    response = await post(body(stream=stream, chat_template_kwargs={'enable_thinking': False},
                               stream_options={'include_usage': True} if stream else None))
    assert response.status_code == 200, response.text
    if stream:
        events = chunks(response)
        content = ''.join(choice['delta'].get('content') or '' for event in events
                          for choice in event.get('choices', []))
        assert [choice['finish_reason'] for event in events for choice in event.get('choices', [])
                if choice.get('finish_reason')] == [engine_reason]
        usage = next(event['usage'] for event in events if event.get('usage'))
    else:
        result = response.json()
        content = result['choices'][0]['message']['content']
        assert result['choices'][0]['finish_reason'] == engine_reason
        usage = result['usage']
    assert content == 'Answer'
    assert usage['prompt_tokens'] == 29
    assert usage['completion_tokens'] == 7


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('raw,reason', [
    ('Answer<｜end▁of▁sentence｜>', 'abort'),
    ('Answer<｜end▁of▁sentence｜>', 'unexpected'),
    ('<｜DSML｜ calls><｜DSML｜ invoke name="typed">', 'stop'),
])
async def test_engine_metadata_does_not_turn_failed_or_malformed_output_into_success(endpoint, metadata_engine, monkeypatch, stream, raw, reason):
    import gateway.triton_client as tc
    from test_deepseek_protocol import TOOLS

    async def wire(*args, **kwargs):
        yield engine_response(raw, finished=True, reason=reason)

    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    response = await post(body(stream=stream, max_tokens=1, tools=TOOLS, chat_template_kwargs={'enable_thinking': False}))
    if stream:
        events = chunks(response)
        assert any('error' in event for event in events), response.text
        assert not any(choice.get('finish_reason') for event in events
                       for choice in event.get('choices', []))
    else:
        assert response.status_code == 502, response.text


@pytest.mark.anyio
@pytest.mark.parametrize('engine_reason,wanted', [('stop', 'tool_calls'), ('length', 'length')])
async def test_valid_dsml_calls_keep_engine_length_and_map_normal_stop(endpoint, metadata_engine, monkeypatch, engine_reason, wanted):
    import gateway.triton_client as tc
    from test_deepseek_protocol import ARGUMENTS, DSML, TOOLS
    import json

    async def wire(*args, **kwargs):
        yield engine_response(DSML, finished=True, reason=engine_reason)

    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    response = await post(body(tools=TOOLS, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 200, response.text
    choice = response.json()['choices'][0]
    assert choice['finish_reason'] == wanted
    assert json.loads(choice['message']['tool_calls'][0]['function']['arguments']) == ARGUMENTS
    assert response.json()['usage']['completion_tokens'] == 3


@pytest.mark.anyio
@pytest.mark.parametrize('engine_reason,status', [('stop', 'completed'), ('length', 'incomplete')])
async def test_responses_uses_engine_finish_and_token_counts(endpoint, metadata_engine, monkeypatch, engine_reason, status):
    import gateway.triton_client as tc

    async def wire(*args, **kwargs):
        yield engine_response('private reasoning', finished=True, reason=engine_reason)

    monkeypatch.setattr(tc, '_stream_grpc_results', wire)
    response = await post({'model': 'deepseek', 'input': 'hello', 'max_output_tokens': 100}, '/v1/responses')
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['status'] == status
    assert result['incomplete_details'] == ({'reason': 'max_output_tokens'} if status == 'incomplete' else None)
    assert result['usage']['input_tokens'] == 29
    assert result['usage']['output_tokens'] == 3
