# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V4.1's fixed HF prompt contract and optional output-only parser.

The recipe is imported only when a DeepSeek response is generated. It never
owns tokenization, request conversion, sampling, transport, or usage.
"""
from __future__ import annotations

from contextlib import aclosing
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import uuid

from fastapi import HTTPException

from .cpu_work import run_cpu
from .generation_types import GenerationEvent, GenerationStream
from .prompt import build_usage, completion_reached_token_limit
from .qwen import TemplateCapabilities, _metadata_locations, _read_json, _resolved_template

TEMPLATE_SHA256 = 'd959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3'
EOS = '<｜end▁of▁sentence｜>'
_CALLS = re.compile(r'<｜DSML｜\s*(calls|tool_calls)>')
_END_CALLS = re.compile(r'</｜DSML｜\s*(calls|tool_calls)>')
_INVOKE = re.compile(r'<｜DSML｜\s*invoke name="([^"<>]+)">')
_END_INVOKE = re.compile(r'</｜DSML｜\s*invoke>')
_PARAMETER = re.compile(r'<｜DSML｜\s*parameter name="([^"<>]+)" string="(true|false)">')
_END_PARAMETER = re.compile(r'</｜DSML｜\s*parameter>')
_MARKERS = ('<｜DSML｜', '</｜DSML｜', EOS, '<think>', '</think>', '<｜', '</｜')


def is_deepseek_profile(model_path: Path, tokenizer=None) -> bool:
    for location in _metadata_locations(model_path, tokenizer):
        config = _read_json(location / 'config.json')
        if config.get('model_type') in {'deepseek_v41', 'deepseek_v41_text'}:
            return True
        if 'DeepseekV41ForCausalLM' in (config.get('architectures') or []):
            return True
        model = _read_json(location / 'model.json').get('model')
        if isinstance(model, str):
            if Path(model).name.lower() == 'deepseek-v4.1-flash':
                return True
            if Path(model).is_dir():
                metadata = _read_json(Path(model) / 'config.json')
                if metadata.get('model_type') == 'deepseek_v41':
                    return True
    return False


def template_capabilities(model_path, tokenizer, *, tools=None):
    source = _resolved_template(model_path, tokenizer, tools=tools)
    digest = sha256(source.encode()).hexdigest() if isinstance(source, str) else ''
    if digest != TEMPLATE_SHA256:
        raise HTTPException(500, 'DeepSeek V4.1 requires the pinned upstream chat template')
    return TemplateCapabilities(template=source, fingerprint=digest, thinking='switch',
        reads_enable_thinking=True, efforts=frozenset({'low', 'medium', 'high', 'max'}), has_think_tags=True)


def validate_request(request):
    """Reject unsupported tool policies before loading/tokenizing the request."""
    choice = request.tool_choice
    if isinstance(choice, dict):
        function = choice.get("function")
        if choice.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise HTTPException(400, "tool_choice requires a named function")
    elif choice is not None and choice not in ("auto", "none", "required"):
        raise HTTPException(400, "Unsupported tool_choice")
    names = set()
    for tool in request.tools or []:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise HTTPException(400, "DeepSeek supports named function tools only")
        if function["name"] in names:
            raise HTTPException(400, "Function names must be unique")
        names.add(function["name"])
    if choice == "required" and not names:
        raise HTTPException(400, "tool_choice=required requires tools")
    if isinstance(choice, dict) and choice["function"]["name"] not in names:
        raise HTTPException(400, "tool_choice function must be present in tools")


def _invalid(message):
    raise HTTPException(502, f'Invalid DeepSeek output: {message}')


def _json(text):
    def reject_constant(value):
        raise ValueError(f'Non-finite JSON value: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate argument: {key}')
            result[key] = value
        return result
    try:
        value = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique)
        json.dumps(value, allow_nan=False)
        return value
    except (ValueError, TypeError) as exc:
        _invalid(f'invalid JSON arguments ({exc})')


class DeepSeekOutput:
    """One request's parser. Buffer DSML until structurally complete/validated.

    Text and reasoning stream immediately; tool fragments are released after a
    complete DSML block. Truncated tools never become successful partial calls.
    """
    def __init__(self, response_id, model, settings, tools, tool_choice=None):
        try:
            import deepseek_recipe as recipe
        except ImportError as exc:
            raise HTTPException(503, 'DeepSeek output parser is unavailable; use the DeepSeek gateway image') from exc
        self.recipe = recipe
        self.settings = settings
        self.tools = {tool['function']['name']: tool['function'] for tool in tools or []}
        self.required = tool_choice == 'required' or isinstance(tool_choice, dict)
        conversation = recipe.Conversation(thinking_mode=settings.enable_thinking,
            tools=[recipe.ToolDefinition(name, parameters=fn.get('parameters', {})) for name, fn in self.tools.items()],
            reasoning_effort=None)
        request = recipe.ConversationRequest(conversation, recipe.ModelVersion.DeepseekV41Flash)
        generator = recipe.ChatCompletionRequest.chunk_generator(request, response_id, model)
        self.processor = recipe.StreamProcessor(generator, recipe.ParsingOptions(
            parse_tool_calls=bool(self.tools), starts_in_tool_calls=False, parse_json_output=False,
            thinking_initial_stage=recipe.ThinkingInitialStage.ReasoningStart if settings.enable_thinking else None,
            stop_sequences=[EOS]))
        # The canonical prompt owns the only opening tag. Generated output
        # starts inside its prefilled reasoning stage when thinking is on.
        self.in_reasoning = settings.enable_thinking
        self.pending = ''; self.raw = ''; self.content = ''; self.reasoning = ''
        self.calls = {}; self.expected = []; self.after_tools = False; self.stopped = False

    @property
    def finished(self):
        return self.processor.finished

    def _chunks(self, chunks):
        events = []
        for chunk in chunks:
            payload = json.loads(chunk.to_json())
            for choice in payload['choices']:
                # Recipe usage/terminal events never bypass gateway validation.
                if choice.get('finish_reason'):
                    continue
                delta = choice['delta']
                self.content += delta.get('content') or ''
                self.reasoning += delta.get('reasoning_content') or ''
                for call in delta.get('tool_calls') or []:
                    dst = self.calls.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                    dst['id'] += call.get('id') or ''
                    for key in ('name', 'arguments'):
                        dst['function'][key] += call.get('function', {}).get(key) or ''
                visible = {k: v for k, v in delta.items() if v is not None}
                reasoning = visible.pop('reasoning_content', None)
                if reasoning and self.settings.expose_reasoning:
                    visible[self.settings.response_field] = reasoning
                if visible:
                    events.append(visible)
        return events

    def start(self):
        return self._chunks(self.processor.push(self.recipe.InferenceChunk.ready()))

    def _push(self, text):
        return self._chunks(self.processor.push(self.recipe.InferenceChunk.text(text))) if text else []

    def _validate_block(self, block):
        if not self.tools:
            _invalid('tool call is forbidden by tools/tool_choice')
        opening = _CALLS.match(block)
        closing = _END_CALLS.search(block)
        if not opening or not closing or opening.group(1) != closing.group(1):
            _invalid('malformed DSML calls block')
        cursor = opening.end(); end = closing.start(); calls = []
        while block[cursor:end].strip():
            cursor += len(block[cursor:end]) - len(block[cursor:end].lstrip())
            invoke = _INVOKE.match(block, cursor)
            if not invoke or invoke.group(1) not in self.tools:
                _invalid('unknown or undeclared tool')
            name = invoke.group(1); cursor = invoke.end(); arguments = {}
            while True:
                cursor += len(block[cursor:end]) - len(block[cursor:end].lstrip())
                ending = _END_INVOKE.match(block, cursor)
                if ending:
                    cursor = ending.end()
                    break
                parameter = _PARAMETER.match(block, cursor)
                if not parameter:
                    _invalid('malformed DSML parameter or invoke')
                ending = _END_PARAMETER.search(block, parameter.end(), end)
                if not ending:
                    _invalid('unterminated DSML parameter')
                key, string = parameter.groups()
                value = block[parameter.end():ending.start()]
                if key in arguments or '｜DSML｜' in value:
                    _invalid('duplicate or nested DSML parameter')
                arguments[key] = value if string == 'true' else _json(value)
                if string == 'false' and isinstance(arguments[key], str):
                    _invalid('string parameters require string="true"')
                cursor = ending.end()
            calls.append((name, arguments))
        if not calls:
            _invalid('empty DSML calls block')
        self.expected.extend(calls)

    def feed(self, text):
        if self.finished:
            return []
        self.raw += text
        self.pending += text
        events = []
        while self.pending and not self.finished:
            # Hold a possible split service token, including just '<'/'</'.
            index = next((i for i, char in enumerate(self.pending) if char == '<' and
                          any(self.pending[i:].startswith(m) or m.startswith(self.pending[i:]) for m in _MARKERS)), -1)
            if index < 0:
                if self.after_tools and self.pending.strip():
                    _invalid('text after tool calls')
                events.extend(self._push(self.pending)); self.pending = ''
                break
            if index:
                prefix = self.pending[:index]
                if self.after_tools and prefix.strip():
                    _invalid('text after tool calls')
                events.extend(self._push(prefix)); self.pending = self.pending[index:]
            if any(m.startswith(self.pending) and m != self.pending for m in _MARKERS):
                break
            if self.pending.startswith('<｜DSML｜'):
                ending = _END_CALLS.search(self.pending)
                if ending is None:
                    break
                block = self.pending[:ending.end()]
                self._validate_block(block)
                events.extend(self._push(block))
                self.pending = self.pending[ending.end():]; self.after_tools = True
            else:
                marker = next((m for m in _MARKERS if self.pending.startswith(m)), None)
                if marker == '<think>':
                    if not self.settings.enable_thinking:
                        _invalid('thinking emitted while disabled')
                    _invalid('nested reasoning opening' if self.in_reasoning else 'repeated reasoning opening')
                if marker == '</think>':
                    self.in_reasoning = False
                if marker in ('<｜', '</｜'):
                    _invalid('unexpected service token')
                if marker == '</｜DSML｜':
                    _invalid('unexpected DSML closing marker')
                if marker is None:
                    break
                events.extend(self._push(marker)); self.pending = self.pending[len(marker):]
                if marker == EOS:
                    self.stopped = True
        return events

    def finish(self, *, length=False):
        length = length and not self.stopped
        if self.pending and not self.finished:
            if not length:
                _invalid('incomplete service marker or DSML block')
            self.pending = ''
        events = []
        if not self.finished:
            reason = self.recipe.InferenceFinishReason.Length if length else self.recipe.InferenceFinishReason.Stop
            events.extend(self._chunks(self.processor.push(self.recipe.InferenceChunk.finish(reason))))
        events.extend(self._chunks(self.processor.finish()))
        calls = list(self.calls.values())
        if len(calls) != len(self.expected):
            _invalid('parser did not preserve complete tool calls')
        for call, (name, arguments) in zip(calls, self.expected):
            actual = _json(call['function']['arguments'])
            if call['function']['name'] != name or json.dumps(actual, sort_keys=True) != json.dumps(arguments, sort_keys=True):
                _invalid('parser changed tool name or typed arguments')
        if self.required and not calls and not length:
            _invalid('required tool call is missing')
        reason = 'length' if length or self.in_reasoning else 'tool_calls' if calls else 'stop'
        return events, reason

    def close(self):
        self.processor.close()


async def generate(request, tokenizer, prompt, sampling, settings, tools, *, headers, extensions):
    from . import triton_client as tc
    response_id = f'chatcmpl-{uuid.uuid4().hex}'; created = int(time.time())

    async def events():
        parser = DeepSeekOutput(response_id, request.model, settings, tools, request.tool_choice)
        def event(delta, **kwargs):
            return GenerationEvent(response_id, created, request.model, delta, **kwargs)
        try:
            for delta in parser.start():
                yield event(delta)
            async with aclosing(tc.stream_triton_text_deltas(request.model, prompt, sampling)) as source:
                async for text in source:
                    for delta in parser.feed(text):
                        yield event(delta)
                    if parser.finished:
                        break
            usage = await run_cpu(build_usage, tokenizer, prompt, parser.raw, reasoning_text=parser.reasoning)
            counted_reasoning = parser.reasoning
            deltas, reason = parser.finish(length=completion_reached_token_limit(usage, sampling))
            if parser.reasoning != counted_reasoning:
                usage = await run_cpu(build_usage, tokenizer, prompt, parser.raw, reasoning_text=parser.reasoning)
            for delta in deltas:
                yield event(delta)
            yield event({}, finish_reason=reason, usage=usage)
        finally:
            # Cancellation/upstream failure closes without synthesizing success.
            parser.close()

    if request.stream:
        return GenerationStream(events(), headers)
    message = {'role': 'assistant', 'content': ''}; calls = {}; terminal = None
    async with aclosing(events()) as source:
        async for event in source:
            for key, value in event.delta.items():
                if key == 'tool_calls':
                    for call in value:
                        dst = calls.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        dst['id'] += call.get('id') or ''
                        for field in ('name', 'arguments'):
                            dst['function'][field] += call.get('function', {}).get(field) or ''
                elif key != 'role':
                    message[key] = message.get(key, '') + value
            if event.finish_reason:
                terminal = event
    if calls:
        message['tool_calls'] = list(calls.values())
    return {'id': response_id, 'object': 'chat.completion', 'created': created, 'model': request.model,
        'choices': [{'index': 0, 'message': message, 'finish_reason': terminal.finish_reason}],
        'usage': terminal.usage, '_generation_headers': headers, **extensions}
