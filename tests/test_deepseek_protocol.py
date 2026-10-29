# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Pinned-template HTTP tests; only the remote gRPC engine is replaced."""
import asyncio
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from jinja2.sandbox import ImmutableSandboxedEnvironment
import numpy as np
import pytest

import gateway.app as app_module
from gateway.admission import AdmissionLease
import gateway.generation as generation
from gateway.reasoning import load_reasoning_settings
from gateway.registry import ModelRegistry
import gateway.triton_client as tc

FIXTURE = Path(__file__).parent / 'fixtures' / 'deepseek' / 'v41-flash'
TOOLS = [{'type': 'function', 'function': {'name': 'typed', 'parameters': {
    'type': 'object', 'properties': {'s': {'type': 'string'}, 'n': {'type': 'number'},
    'b': {'type': 'boolean'}, 'a': {'type': 'array'}, 'o': {'type': 'object'}, 'z': {'type': 'null'}}}}}]
DSML = ('<｜DSML｜ calls><｜DSML｜ invoke name="typed">'
        '<｜DSML｜ parameter name="s" string="true">hello "world"\nline</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="n" string="false">12.5</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="b" string="false">true</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="a" string="false">[1,false]</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="o" string="false">{"nested":2}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="z" string="false">null</｜DSML｜ parameter>'
        '</｜DSML｜ invoke></｜DSML｜ calls>')
ARGUMENTS = {'s': 'hello "world"\nline', 'n': 12.5, 'b': True, 'a': [1, False], 'o': {'nested': 2}, 'z': None}


class Tokenizer:
    def __init__(self):
        self.chat_template = (FIXTURE / 'chat_template.jinja').read_text()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        def fail(message):
            raise ValueError(message)
        env.globals['raise_exception'] = fail
        self.template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        assert not tokenize
        return self.template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)

    def encode(self, text, **kwargs):
        return str(text).split()

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text))


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def endpoint(tmp_path):
    tokenizer = Tokenizer()
    (tmp_path / 'model.json').write_text(json.dumps({'model': '/fixed/DeepSeek-V4.1-Flash', 'max_model_len': 32768}))
    (tmp_path / 'gateway.json').write_text('{"reasoning":{"mode":"separate"}}')
    state = SimpleNamespace(records=[], models=[], closed=0, output=None, error=False, step=7, path=tmp_path, tokenizer=tokenizer)

    def inputs(prompt, sampling, images, stream, **kwargs):
        assert images == []
        return [SimpleNamespace(prompt=prompt, sampling=sampling, stream=stream)]

    async def stream(operation, model, inputs, outputs, **kwargs):
        item = inputs[0]
        state.records.append((item.prompt, item.sampling, item.stream))
        state.models.append(model)
        result = state.output
        if result is None:
            result = 'Final answer.' if '<tool_result>' in item.prompt or 'Weather?' not in item.prompt else DSML
            if item.prompt.endswith('<think>'):
                result = 'Inspect first.</think>' + result
        try:
            # Engine outputs are cumulative; invoke the real producer to get
            # the actual delta tensors that cross the gRPC boundary.
            from deepseek_backend_wire import production_text_wire
            with production_text_wire() as serialize:
                for end in range(state.step, len(result) + state.step, state.step):
                    await asyncio.sleep(0)
                    yield serialize(result[:end])
            if state.error:
                raise RuntimeError('upstream failed')
        finally:
            state.closed += 1

    async def acquire(*args, **kwargs):
        return AdmissionLease([])

    with ExitStack() as stack:
        for name, replacement in {
            'resolve': lambda *args: tmp_path,
            'get_backend': lambda *args: 'vllm_deepseek',
            'get_tokenizer_async': AsyncMock(return_value=(tokenizer, tmp_path)),
        }.items():
            stack.enter_context(patch.object(generation.registry, name, replacement))
        stack.enter_context(patch.object(generation.admission, 'acquire', acquire))
        stack.enter_context(patch.object(tc, '_build_grpc_generate_inputs', inputs))
        stack.enter_context(patch.object(tc, 'get_grpc_client', AsyncMock(return_value=object())))
        stack.enter_context(patch.object(tc, '_stream_grpc_results', stream))
        yield state


async def post(body, route='/v1/chat/completions'):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_module.app), base_url='http://test') as client:
        return await client.post(route, json=body)


def body(**kwargs):
    return {'model': 'deepseek', 'messages': [{'role': 'user', 'content': 'hello'}], **kwargs}


def chunks(response):
    return [json.loads(line[5:]) for line in response.text.splitlines()
            if line.startswith('data:') and line[5:].strip() != '[DONE]']


def accumulate(events):
    content = ''; reasoning = ''; calls = {}; finishes = []
    for event in events:
        for choice in event['choices']:
            delta = choice['delta']
            content += delta.get('content') or ''
            reasoning += delta.get('reasoning_content') or ''
            for call in delta.get('tool_calls') or []:
                dst = calls.setdefault(call['index'], {'id': '', 'name': '', 'arguments': ''})
                dst['id'] += call.get('id') or ''
                dst['name'] += call.get('function', {}).get('name') or ''
                dst['arguments'] += call.get('function', {}).get('arguments') or ''
            if choice.get('finish_reason'):
                finishes.append(choice['finish_reason'])
    return content, reasoning, list(calls.values()), finishes


def test_deepseek_max_effort_uses_exact_template_budget(endpoint):
    settings = load_reasoning_settings(endpoint.path, tokenizer=endpoint.tokenizer,
        enable_thinking=True, reasoning_effort='max')
    prompt = endpoint.tokenizer.apply_chat_template([{'role': 'user', 'content': 'hello'}], **settings.template_kwargs)
    assert 'Reasoning Effort: 100 ' in prompt
    assert prompt.endswith('<｜Assistant｜><think>')


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('tools', [False, True])
async def test_chat_thinking_and_typed_tool_cycle_uses_production_wire_deltas(endpoint, stream, thinking, tools):
    request = body(stream=stream, chat_template_kwargs={'enable_thinking': thinking})
    if tools:
        request.update(tools=TOOLS, messages=[{'role': 'user', 'content': 'Weather?'}])
    response = await post(request)
    assert response.status_code == 200, response.text
    if stream:
        content, reasoning, calls, finishes = accumulate(chunks(response))
        assert finishes == ['tool_calls' if tools else 'stop']
        assert response.text.count('data: [DONE]') == 1
    else:
        result = response.json()['choices'][0]
        assert result['finish_reason'] == ('tool_calls' if tools else 'stop')
        message = result['message']
        content = message['content']; reasoning = message.get('reasoning_content', '')
        calls = [{'id': call['id'], **call['function']} for call in message.get('tool_calls', [])]
    assert reasoning == ('Inspect first.' if thinking else '')
    assert content == ('' if tools else 'Final answer.')
    assert endpoint.closed == 1
    assert endpoint.records[0][2] is True  # One shared streaming parser for both APIs.
    if tools:
        assert len(calls) == 1 and calls[0]['id'].startswith('call_')
        assert calls[0]['name'] == 'typed'
        assert json.loads(calls[0]['arguments']) == ARGUMENTS
        replay = body(tools=TOOLS, chat_template_kwargs={'enable_thinking': thinking}, messages=[
            *request['messages'], {'role': 'assistant', 'content': '', 'reasoning_content': reasoning,
              'tool_calls': [{'id': calls[0]['id'], 'type': 'function', 'function': {
                  'name': 'typed', 'arguments': calls[0]['arguments']}}]},
            {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'sunny'}])
        followup = await post(replay)
        assert followup.status_code == 200, followup.text
        assert followup.json()['choices'][0]['message']['content'] == 'Final answer.'
        assert '<tool_result>sunny</tool_result>' in endpoint.records[-1][0]
        assert 'string="false">true</｜DSML｜ parameter>' in endpoint.records[-1][0]
        if thinking:
            assert 'Inspect first.</think>' in endpoint.records[-1][0]


@pytest.mark.anyio
async def test_request_local_policies_and_chronological_system_history(endpoint):
    messages = [{'role': 'system', 'content': 'first'}, {'role': 'user', 'content': 'old'},
        {'role': 'assistant', 'content': 'prior', 'reasoning_content': 'old reasoning'},
        {'role': 'developer', 'content': 'later policy'}, {'role': 'user', 'content': 'hello'}]
    results = await asyncio.gather(*(post(body(messages=messages,
        chat_template_kwargs={'enable_thinking': enabled}, include_reasoning=visible, tools=TOOLS))
        for enabled, visible in [(True, True), (False, True), (True, False)]))
    assert [r.status_code for r in results] == [200, 200, 200]
    assert [r.json()['choices'][0]['message'].get('reasoning_content') for r in results] == ['Inspect first.', None, None]
    for prompt, _, _ in endpoint.records:
        assert prompt.index('first') < prompt.index('old') < prompt.index('later policy') < prompt.index('hello')
    assert all('old reasoning</think>' in prompt for prompt, _, _ in endpoint.records if prompt.endswith('<think>'))
    assert (await post(body())).json()['reasoning_status']['mode'] == 'separate'


@pytest.mark.anyio
@pytest.mark.parametrize('value,budget', [('low', 50), ('medium', 62), ('high', 75), ('max', 100)])
async def test_effort_reaches_fixed_template_only(endpoint, value, budget):
    response = await post(body(reasoning_effort=value))
    assert response.status_code == 200, response.text
    assert f'Reasoning Effort: {budget} ' in endpoint.records[-1][0]
    assert 'reasoning_effort' not in endpoint.records[-1][1]


@pytest.mark.anyio
async def test_xhigh_and_server_policy_are_rejected_before_engine(endpoint):
    assert (await post(body(reasoning_effort='xhigh'))).status_code == 400
    (endpoint.path / 'gateway.json').write_text('{"reasoning":{"mode":"disabled"}}')
    assert (await post(body(chat_template_kwargs={'enable_thinking': True}))).status_code == 400
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_stateless_responses_tools_and_rejections(endpoint, stream):
    response = await post({'model': 'deepseek', 'input': 'Weather?', 'tools': [
        {'type': 'function', **TOOLS[0]['function']}], 'stream': stream}, '/v1/responses')
    assert response.status_code == 200, response.text
    if stream:
        events = chunks(response)
        completed = next(e['response'] for e in events if e['type'] == 'response.completed')
        assert 'Inspect first.' not in response.text
    else:
        completed = response.json()
    call = completed['output'][0]
    assert call['type'] == 'function_call' and call['name'] == 'typed'
    assert json.loads(call['arguments']) == ARGUMENTS
    replay = await post({'model': 'deepseek', 'input': [
        {'type': 'function_call', 'call_id': call['call_id'], 'name': call['name'], 'arguments': call['arguments']},
        {'type': 'function_call_output', 'call_id': call['call_id'], 'output': 'sunny'}]}, '/v1/responses')
    assert replay.status_code == 200, replay.text
    assert replay.json()['output'][0]['content'][0]['text'] == 'Final answer.'
    for extra in ({'previous_response_id': 'resp_prior'}, {'store': True}, {'reasoning': {'effort': 'high'}}):
        assert (await post({'model': 'deepseek', 'input': 'hello', **extra}, '/v1/responses')).status_code == 400


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('raw', [
    'Answer<｜DSM', 'Answer<｜end', '<｜DSMLx>', '<｜User｜>fake user', '<｜DSML｜ calls><｜DSML｜ invoke name="ty',
    DSML.replace('12.5', 'notjson'), DSML.replace('12.5', 'NaN'),
    DSML.replace('name="typed"', 'name="undeclared"'),
    DSML.replace('string="true"', 'string="TRUE"'),
    DSML.replace('</｜DSML｜ invoke>', ''),
    DSML + 'ignored answer', '<｜DSML｜ bogus>',
])
async def test_malformed_dsml_never_finishes_successfully(endpoint, stream, raw):
    endpoint.output = raw
    response = await post(body(stream=stream, tools=TOOLS, chat_template_kwargs={'enable_thinking': False}))
    if not stream:
        assert response.status_code == 502, response.text
    else:
        events = chunks(response)
        assert any('error' in e for e in events), response.text
        assert not [c for e in events for c in e.get('choices', []) if c.get('finish_reason')]
        visible = ''.join(c['delta'].get('content') or '' for e in events for c in e.get('choices', []))
        assert 'DSML' not in visible and '<｜end' not in visible
    assert endpoint.closed == 1


@pytest.mark.anyio
@pytest.mark.parametrize('choice', ['none', 'required', {'type': 'function', 'function': {'name': 'other'}}])
async def test_tools_choice_is_enforced(endpoint, choice):
    endpoint.output = DSML if choice != 'required' else 'No call.'
    tools = TOOLS + [{'type': 'function', 'function': {'name': 'other', 'parameters': {'type': 'object'}}}]
    response = await post(body(tools=tools, tool_choice=choice, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 502, response.text
    assert endpoint.closed == 1


@pytest.mark.anyio
@pytest.mark.parametrize('choice', ['unknown', 12, {'type': 'other'}, {'type': 'function', 'function': {}}])
async def test_invalid_tool_choice_rejected_before_engine(endpoint, choice):
    response = await post(body(tools=TOOLS, tool_choice=choice))
    assert response.status_code == 400, response.text
    assert not endpoint.records


@pytest.mark.anyio
async def test_required_without_tools_rejected_before_engine(endpoint):
    response = await post(body(tool_choice='required'))
    assert response.status_code == 400, response.text
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_truncated_tool_block_reports_length_without_partial_call(endpoint, stream):
    endpoint.output = 'some text ' * 4 + DSML[:90]
    response = await post(body(stream=stream, tools=TOOLS, max_tokens=3,
        chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 200, response.text
    if stream:
        content, reasoning, calls, finishes = accumulate(chunks(response))
        assert finishes == ['length'] and calls == []
    else:
        result = response.json()['choices'][0]
        assert result['finish_reason'] == 'length' and 'tool_calls' not in result['message']
        content = result['message']['content']
    assert 'DSML' not in content


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_explicit_eos_finishes_once_and_closes_upstream(endpoint, stream):
    endpoint.output = 'Answer<｜end▁of▁sentence｜>ignored' + ' long' * 30
    endpoint.step = 1
    response = await post(body(stream=stream, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 200, response.text
    if stream:
        content, _, calls, finishes = accumulate(chunks(response))
        assert finishes == ['stop'] and not calls
    else:
        content = response.json()['choices'][0]['message']['content']
    assert content == 'Answer'
    assert endpoint.closed == 1


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_upstream_error_never_becomes_clean_eof(endpoint, stream):
    endpoint.error = True
    response = await post(body(stream=stream))
    if stream:
        events = chunks(response)
        assert any('error' in e for e in events)
        assert not [c for e in events for c in e.get('choices', []) if c.get('finish_reason')]
    else:
        assert response.status_code == 502
    assert endpoint.closed == 1


@pytest.mark.anyio
@pytest.mark.parametrize('mode', ['hidden', 'separate'])
async def test_cancel_closes_parser_and_upstream_without_finish(endpoint, mode):
    from gateway.deepseek import DeepSeekOutput
    (endpoint.path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': mode}}))
    source = await generation.generate(generation.ChatCompletionRequest(**body(stream=True)))
    seen = []; closed = []; finished = []
    original_close = DeepSeekOutput.close
    original_finish = DeepSeekOutput.finish
    def close(parser):
        closed.append(True)
        return original_close(parser)
    def finish(parser, **kwargs):
        finished.append(True)
        return original_finish(parser, **kwargs)
    with patch.object(DeepSeekOutput, 'close', close), patch.object(DeepSeekOutput, 'finish', finish):
        async for event in source.events:
            seen.append(event)
            if event.delta.get('content') or event.delta.get('reasoning_content'):
                break
        await source.aclose()
    assert closed == [True] and not finished
    assert endpoint.closed == 1
    assert not any(e.finish_reason for e in seen)


@pytest.mark.anyio
@pytest.mark.parametrize('route,extra', [
    ('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,eA=='}}]}]}),
    ('/v1/responses', {'input': [{'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'data:image/png;base64,eA=='}]}]}),
    ('/v1/embeddings', {'input': 'hello'}),
])
async def test_unsupported_media_embedding_rejected_before_tokenizer(endpoint, route, extra):
    original = generation.registry.get_tokenizer_async
    generation.registry.get_tokenizer_async = AsyncMock(side_effect=AssertionError('unsupported route tokenized'))
    try:
        response = await post({'model': 'deepseek', **extra}, route)
    finally:
        generation.registry.get_tokenizer_async = original
    assert response.status_code == 400, response.text
    assert not endpoint.records


def test_recipe_adapter_is_independent_of_all_chunk_boundaries(endpoint):
    from gateway.deepseek import DeepSeekOutput
    settings = load_reasoning_settings(endpoint.path, tokenizer=endpoint.tokenizer)
    raw = 'Inspect first.</think>Answer\n' + DSML
    for split in range(len(raw) + 1):
        parser = DeepSeekOutput('chatcmpl-fixed', 'deepseek', settings, TOOLS)
        try:
            parser.start()
            parser.feed(raw[:split]); parser.feed(raw[split:])
            _, reason = parser.finish()
            assert parser.reasoning == 'Inspect first.'
            assert parser.content == 'Answer'
            assert reason == 'tool_calls'
            assert len(parser.calls) == 1
            assert json.loads(parser.calls[0]['function']['arguments']) == ARGUMENTS
        finally:
            parser.close()


@pytest.mark.anyio
async def test_hidden_server_policy_cannot_be_overridden_by_visibility(endpoint):
    (endpoint.path / 'gateway.json').write_text('{"reasoning":{"mode":"hidden"}}')
    response = await post(body(include_reasoning=True, chat_template_kwargs={'enable_thinking': True}))
    assert response.status_code == 200, response.text
    assert response.json()['reasoning_status']['mode'] == 'hidden'
    assert response.json()['choices'][0]['message'] == {'role': 'assistant', 'content': 'Final answer.'}
    assert endpoint.records[0][0].endswith('<think>')


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('mode', ['disabled', 'hidden', 'separate'])
async def test_disabled_generation_never_exposes_accidental_thinking(endpoint, stream, mode):
    (endpoint.path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': mode}}))
    endpoint.output = '<think>secret</think>Answer'
    response = await post(body(stream=stream, chat_template_kwargs={'enable_thinking': False}))
    if stream:
        assert any('error' in e for e in chunks(response))
        assert 'secret' not in response.text
    else:
        assert response.status_code == 502, response.text


@pytest.mark.anyio
@pytest.mark.parametrize('mode', ['hidden', 'separate'])
async def test_cancel_nonstream_request_closes_upstream_without_finish(endpoint, mode):
    from gateway.deepseek import DeepSeekOutput
    (endpoint.path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': mode}}))
    started = asyncio.Event(); blocked = asyncio.Event(); closed = []
    async def source(*args, **kwargs):
        try:
            yield 'some reasoning'
            started.set()
            await blocked.wait()
        finally:
            closed.append('upstream')
    original = DeepSeekOutput.close
    def close(parser):
        closed.append('parser')
        original(parser)
    with patch.object(tc, 'stream_triton_text_deltas', source), patch.object(DeepSeekOutput, 'close', close):
        task = asyncio.create_task(post(body()))
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed == ['upstream', 'parser']


def test_external_metadata_selects_builtin_parser_and_direct_local_tokenizer(tmp_path):
    import sys
    from types import ModuleType
    external = tmp_path / 'external'; external.mkdir()
    (external / 'config.json').write_text('{"model_type":"deepseek_v41"}')
    version = tmp_path / '1'; version.mkdir()
    (version / 'model.json').write_text(json.dumps({'model': str(external), 'tokenizer': str(external)}))
    (tmp_path / 'config.pbtxt').write_text('backend: "vllm_deepseek"')
    registry = ModelRegistry()
    observed = []
    def direct(path, **kwargs):
        observed.append((path, kwargs))
        return Tokenizer()
    def forbidden(*args, **kwargs):
        raise AssertionError('DeepSeek must not use AutoTokenizer/remote model code')
    transformers = ModuleType('transformers')
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=forbidden)
    transformers.TokenizersBackend = SimpleNamespace(from_pretrained=direct)
    with patch.object(registry, 'resolve', return_value=version), patch.dict(sys.modules, {'transformers': transformers}):
        tokenizer, path = registry.get_tokenizer('alias')
        assert path == version and isinstance(tokenizer, Tokenizer)
        assert observed == [(str(external), {'local_files_only': True, 'trust_remote_code': False})]
        assert registry.get_tool_parser('alias') == 'deepseek_v41'
        assert registry.get_capabilities('alias') == frozenset({'chat'})


def test_fixed_template_rejects_unverified_replacement(endpoint):
    from fastapi import HTTPException
    endpoint.tokenizer.chat_template += '\n'
    with pytest.raises(HTTPException) as error:
        load_reasoning_settings(endpoint.path, tokenizer=endpoint.tokenizer)
    assert error.value.status_code == 500


@pytest.mark.anyio
async def test_delta_transport_preserves_shared_prefixes_and_closes_source():
    closed = []
    async def results(*args, **kwargs):
        try:
            for text in ('ha', 'ha!', 'ha'):
                yield SimpleNamespace(as_numpy=lambda name, text=text: np.asarray([text.encode()], dtype=np.object_))
        finally:
            closed.append(True)
    with (patch.object(tc, 'get_grpc_client', AsyncMock(return_value=object())),
          patch.object(tc, '_stream_grpc_results', results)):
        actual = [text async for text in tc.stream_triton_text_deltas('alias', 'prompt', {'max_tokens': 20})]
    assert actual == ['ha', 'ha!', 'ha']
    assert closed == [True]


@pytest.mark.anyio
async def test_explicit_eos_takes_priority_over_estimated_length(endpoint):
    endpoint.output = 'Answer<｜end▁of▁sentence｜>'
    response = await post(body(max_tokens=1, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 200, response.text
    assert response.json()['choices'][0]['finish_reason'] == 'stop'


@pytest.mark.anyio
async def test_typed_json_overflow_is_rejected(endpoint):
    endpoint.output = DSML.replace('12.5', '1e999')
    response = await post(body(tools=TOOLS, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 502, response.text


@pytest.mark.anyio
async def test_context_summary_uses_deepseek_grpc_and_disabled_thinking(endpoint):
    from gateway.context_compression import load_context_compression_settings
    settings = load_context_compression_settings(endpoint.path)
    with patch.object(generation, 'call_triton', side_effect=AssertionError('DeepSeek summary fell back to HTTP')):
        summary = await generation._generate_context_summary('deepseek', settings, None, 'old conversation')
    assert summary.text == 'Final answer.'
    assert endpoint.records[-1][0].endswith('</think>')
    assert 'Reasoning Effort:' not in endpoint.records[-1][0]


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_multiple_tool_ids_and_false_boolean_remain_typed(endpoint, stream):
    endpoint.output = DSML.replace('>true</｜DSML｜ parameter>', '>false</｜DSML｜ parameter>').replace(
        '</｜DSML｜ calls>', '<｜DSML｜ invoke name="typed"></｜DSML｜ invoke></｜DSML｜ calls>')
    response = await post(body(stream=stream, tools=TOOLS, chat_template_kwargs={'enable_thinking': False}))
    assert response.status_code == 200, response.text
    if stream:
        _, _, calls, finishes = accumulate(chunks(response))
        assert finishes == ['tool_calls']
    else:
        calls = [{'id': c['id'], **c['function']} for c in response.json()['choices'][0]['message']['tool_calls']]
    assert len(calls) == 2 and calls[0]['id'] != calls[1]['id']
    assert json.loads(calls[0]['arguments'])['b'] is False
    assert json.loads(calls[1]['arguments']) == {}


@pytest.mark.anyio
@pytest.mark.parametrize('cumulative,wire,whole', [
    (['Hel', 'Hello', 'Hello world'], ['Hel', 'lo', ' world'], 'Hello world'),
    (['ha', 'haha', 'hahaha'], ['ha', 'ha', 'ha'], 'hahaha'),
    (['a', 'a', 'aa'], ['a', '', 'a'], 'aa'),
    (['prompt', 'promptprompt'], ['prompt', 'prompt'], 'promptprompt'),
])
async def test_production_serializer_deltas_reach_consumer_exactly_once(cumulative, wire, whole):
    from deepseek_backend_wire import production_text_wire
    received = []
    closed = []
    with production_text_wire() as serialize:
        async def results(*args, **kwargs):
            try:
                for text in cumulative:
                    response = serialize(text)
                    received.extend(value.decode() for value in response.as_numpy('text_output'))
                    yield response
            finally:
                closed.append(True)
        with (patch.object(tc, 'get_grpc_client', AsyncMock(return_value=object())),
          patch.object(tc, '_stream_grpc_results', results)):
            actual = [delta async for delta in tc.stream_triton_text_deltas('alias', 'prompt', {'max_tokens': 20})]
    assert received == wire
    assert ''.join(actual) == whole
    assert actual == [value for value in wire if value]
    assert closed == [True]


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('step', [1, 7, 10000])
@pytest.mark.parametrize('mode,visible', [('hidden', True), ('separate', False), ('separate', True)])
@pytest.mark.parametrize('raw', [
    'first</think>Answer<think>secret</think>tail',
    'first<think>secret</think>Answer',
])
async def test_repeated_or_nested_thinking_is_rejected_before_secret_emission(endpoint, stream, step, mode, visible, raw):
    (endpoint.path / 'gateway.json').write_text(json.dumps({'reasoning': {'mode': mode}}))
    endpoint.step = step
    endpoint.output = raw
    response = await post(body(stream=stream, include_reasoning=visible))
    if stream:
        events = chunks(response)
        assert any('error' in event for event in events), response.text
        assert not any(c.get('finish_reason') for e in events for c in e.get('choices', []))
        assert 'secret' not in response.text
    else:
        assert response.status_code == 502, response.text
        assert 'secret' not in response.text
    assert endpoint.closed == 1


@pytest.mark.anyio
@pytest.mark.parametrize('tools', [None, []])
async def test_named_choice_without_tools_rejected_before_tokenizer(endpoint, tools):
    original = generation.registry.get_tokenizer_async
    generation.registry.get_tokenizer_async = AsyncMock(side_effect=AssertionError('invalid named tool choice tokenized'))
    try:
        response = await post(body(tools=tools, tool_choice={'type': 'function', 'function': {'name': 'missing'}}))
    finally:
        generation.registry.get_tokenizer_async = original
    assert response.status_code == 400, response.text
    assert not endpoint.records


@pytest.mark.anyio
@pytest.mark.parametrize('stream', [False, True])
async def test_dsml_string_parameter_preserves_literal_thinking_tags(endpoint, stream):
    literal = 'Documentation: <think>example text</think> remains literal.'
    endpoint.output = 'first</think>' + DSML.replace('hello "world"\nline', literal)
    response = await post(body(stream=stream, include_reasoning=False, tools=TOOLS))
    assert response.status_code == 200, response.text
    if stream:
        _, reasoning, calls, finish = accumulate(chunks(response))
        assert reasoning == '' and finish == ['tool_calls']
        arguments = calls[0]['arguments']
    else:
        result = response.json()['choices'][0]
        assert result['finish_reason'] == 'tool_calls'
        assert 'reasoning_content' not in result['message']
        arguments = result['message']['tool_calls'][0]['function']['arguments']
    assert json.loads(arguments)['s'] == literal
