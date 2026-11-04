# SPDX-License-Identifier: Apache-2.0
"""Offline parity checks against pinned, independently generated token references.

No checkpoint Python or network access is used. Matching a reference qualifies
prompt encoding only, not engine execution, weights, or GPU readiness.
"""
from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any


ARTIFACT_NAMES = frozenset({'config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'})
SPECIAL_IDS = ('bos_token_id', 'eos_token_id', 'pad_token_id')
MAX_REFERENCE_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def _regular(path: Path, limit: int):
    """Open a stable regular blob, following HF snapshot links only at resolution."""
    path = Path(path)
    descriptor = None
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f'{path.name}: expected a regular local file')
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f'{path.name}: expected a regular local file')
        if _identity(before) != _identity(opened):
            raise ValueError(f'{path.name}: file changed before reading')
        if opened.st_size > limit:
            raise ValueError(f'{path.name}: local byte limit exceeded')
        yield descriptor, opened
        if (_identity(opened) != _identity(os.fstat(descriptor))
                or path.resolve(strict=True) != resolved
                or _identity(opened) != _identity(resolved.stat())):
            raise ValueError(f'{path.name}: file changed during reading; require immutable artifacts')
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{path.name}: cannot read a stable regular local file ({type(error).__name__})') from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _chunks(descriptor: int, size: int):
    offset = 0
    while offset < size:
        count = min(READ_CHUNK_BYTES, size - offset)
        block = os.pread(descriptor, count, offset)
        if not block or len(block) > count:
            raise ValueError('local file changed or was truncated during bounded reading')
        offset += len(block)
        yield block


def read_reference(path: Path) -> dict[str, Any]:
    path = Path(path)
    with _regular(path, MAX_REFERENCE_BYTES) as (descriptor, info):
        raw = b''.join(_chunks(descriptor, info.st_size))
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate reference key: {key}')
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError('non-finite JSON numbers are not permitted in token references')
    def finite_float(raw):
        value = float(raw)
        if not math.isfinite(value):
            reject_constant(raw)
        return value
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                       parse_constant=reject_constant, parse_float=finite_float)
    if not isinstance(value, dict):
        raise ValueError('token reference must be a JSON object')
    return value


def _directory(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f'{name} must be an absolute local directory; remote downloads are disabled')
    result = Path(value).resolve(strict=True)
    if not result.is_dir():
        raise ValueError(f'{name} must be a local directory')
    return result


def _artifact_hash(path: Path) -> str:
    digest = sha256()
    with _regular(path, MAX_ARTIFACT_BYTES) as (descriptor, info):
        for chunk in _chunks(descriptor, info.st_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_reference(reference: dict[str, Any]) -> None:
    if not isinstance(reference, dict) or type(reference.get('version')) is not int or reference['version'] != 1:
        raise ValueError('unsupported token reference version')
    checkpoint = reference.get('checkpoint')
    if (not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('repo_id'), str)
            or not checkpoint['repo_id'] or not re.fullmatch('[a-f0-9]{40}', str(checkpoint.get('revision', '')))):
        raise ValueError('reference requires a checkpoint identity and immutable revision')
    source = reference.get('source')
    if (not isinstance(source, dict) or not isinstance(source.get('kind'), str)
            or source['kind'] not in {'reviewed-checkpoint-encoder', 'synthetic-test-fixture'}):
        raise ValueError('unsupported reference source; checkpoint code is never executed')
    if source['kind'] == 'reviewed-checkpoint-encoder' and not re.fullmatch('[a-f0-9]{64}', str(source.get('encoder_sha256', ''))):
        raise ValueError('reviewed reference source requires its encoder SHA256')
    artifacts = reference.get('artifacts')
    if (not isinstance(artifacts, dict) or set(artifacts) != ARTIFACT_NAMES
            or any(not isinstance(v, str) or not re.fullmatch('[a-f0-9]{64}', v) for v in artifacts.values())):
        raise ValueError('reference artifacts must pin config, tokenizer files, and chat template by SHA256')
    special = reference.get('special_token_ids')
    if (not isinstance(special, dict) or set(special) != set(SPECIAL_IDS)
            or any(v is not None and (type(v) is not int or v < 0) for v in special.values())):
        raise ValueError('reference must declare BOS/EOS/PAD token IDs')
    cases = reference.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('reference requires nonempty token cases')
    names = set()
    for case in cases:
        if (not isinstance(case, dict) or not isinstance(case.get('name'), str) or not case['name']
                or len(case['name']) > 128 or case['name'] in names):
            raise ValueError('token reference case names must be unique nonempty strings')
        names.add(case['name'])
        if not isinstance(case.get('request'), dict) or not isinstance(case.get('expected_prompt'), str):
            raise ValueError(f'{case["name"]}: expected request and independent prompt reference')
        ids = case.get('expected_token_ids')
        if not isinstance(ids, list) or not ids or any(type(v) is not int or v < 0 for v in ids):
            raise ValueError(f'{case["name"]}: reference token IDs must be nonnegative integers')


def _render_case(tokenizer: Any, case: dict[str, Any], model_root: Path,
                 effort_map: dict[str, str] | None = None) -> str:
    from fastapi import HTTPException
    from .deepseek import template_capabilities, validate_request
    from .prompt import RequestTokenizer, build_conversation, render_chat_prompt, selected_tools
    from .schemas import ChatCompletionRequest
    from .reasoning import resolve_reasoning_effort

    # Exercise the same schema, role/tool translation and request-local template
    # binding as the serving path. Context truncation is deliberately not applied.
    try:
        request = ChatCompletionRequest.model_validate({**case['request'], 'model': 'deepseek-token-qualification'})
    except ValueError:
        raise ValueError(f'{case["name"]}: invalid reference request schema') from None
    try:
        validate_request(request)
        controls = request.chat_template_kwargs
        if controls is None or controls.enable_thinking is None:
            raise ValueError('token reference cases must explicitly select enable_thinking')
        thinking = controls.enable_thinking
        tools = selected_tools(request.tools, request.tool_choice)
        capabilities = template_capabilities(model_root, tokenizer, tools=tools)
        native_effort = resolve_reasoning_effort(request.reasoning_effort, capabilities.efforts, effort_map)
        if controls.preserve_thinking is not None and not capabilities.preserves_thinking:
            raise ValueError('DeepSeek template does not support preserve_thinking')
        if request.reasoning_effort is not None and not thinking:
            raise ValueError('invalid DeepSeek reasoning_effort in token reference')
        if request.response_format is not None and request.response_format.type != 'text' and thinking:
            raise ValueError('structured output cannot enable thinking')
        conversation = build_conversation(request.messages, preserve_reasoning=True)
        if any(not isinstance(message.get('content'), str) for message in conversation):
            raise ValueError('token qualification supports text-only conversations')
        conversation = [{**message, 'role': 'system' if message['role'] == 'developer' else message['role']}
                        for message in conversation]
        kwargs = {'enable_thinking': thinking}
        if native_effort is not None:
            kwargs['reasoning_effort'] = native_effort
        wrapped = RequestTokenizer(tokenizer, kwargs, strict=True,
            chat_template=capabilities.template, tools=tools)
        return render_chat_prompt(wrapped, conversation, tools, enable_thinking=thinking)
    except ValueError:
        raise
    except HTTPException as error:
        raise ValueError(f'{case["name"]}: {error.detail}') from None
    except Exception as error:
        # Do not echo a potentially private reference conversation in CLI errors.
        raise ValueError(f'{case["name"]}: request/template qualification failed ({type(error).__name__})') from None


def _first_difference(expected: list[int], actual: list[int]) -> dict[str, int | None] | None:
    for index in range(max(len(expected), len(actual))):
        left = expected[index] if index < len(expected) else None
        right = actual[index] if index < len(actual) else None
        if left != right:
            return {'index': index, 'expected': left, 'actual': right}
    return None


def validate_tokenizer(engine_config: dict[str, Any], reference: dict[str, Any], *, tokenizer: Any = None,
                       model_config_dir: Path | None = None) -> dict[str, Any]:
    """Compare local gateway rendering/encoding with literal reference outputs.

    ``source`` records the operator's reviewed encoder provenance; it is not an
    authenticity signature. The caller must trust the chosen reference file.
    ``tokenizer`` permits a controlled in-process runtime qualification/test.
    ``model_config_dir`` explicitly selects gateway.json model-local controls;
    the checkpoint/tokenizer directories never implicitly supply these aliases.
    """
    _validate_reference(reference)
    effort_map = {}
    if model_config_dir is not None:
        from fastapi import HTTPException
        from .reasoning import read_reasoning_effort_map
        try:
            effort_map = read_reasoning_effort_map(_directory(str(model_config_dir), 'model_config_dir'))
        except HTTPException as error:
            raise ValueError(str(error.detail)) from None
    if engine_config.get('revision') != reference['checkpoint']['revision']:
        raise ValueError('checkpoint revision differs from the token reference')
    model_root = _directory(engine_config.get('model'), 'model')
    tokenizer_root = _directory(engine_config.get('tokenizer', str(model_root)), 'tokenizer')
    paths = {name: (model_root if name == 'config.json' else tokenizer_root) / name for name in ARTIFACT_NAMES}
    for name, path in paths.items():
        if _artifact_hash(path) != reference['artifacts'][name]:
            raise ValueError(f'{name}: artifact hash differs from the token reference')
    config = read_reference(paths['config.json'])
    if config.get('model_type') != 'deepseek_v41':
        raise ValueError('token qualification requires the supported deepseek_v41 checkpoint profile')
    if tokenizer is None:
        # Match ModelRegistry's DeepSeek path: no AutoTokenizer/remote Python.
        from transformers import TokenizersBackend
        tokenizer = TokenizersBackend.from_pretrained(str(tokenizer_root), local_files_only=True, trust_remote_code=False)
    mismatches = {field: {'expected': reference['special_token_ids'][field], 'actual': getattr(tokenizer, field, None)}
                  for field in SPECIAL_IDS if getattr(tokenizer, field, None) != reference['special_token_ids'][field]}
    cases = []
    for case in reference['cases']:
        prompt = _render_case(tokenizer, case, model_root, effort_map)
        actual = list(tokenizer.encode(prompt, add_special_tokens=False))
        if any(type(value) is not int or value < 0 for value in actual):
            raise ValueError(f'{case["name"]}: runtime tokenizer returned invalid token IDs')
        expected = case['expected_token_ids']
        same_prompt, same_ids = prompt == case['expected_prompt'], actual == expected
        cases.append({'name': case['name'], 'matched': same_prompt and same_ids,
                      'prompt_match': same_prompt, 'token_ids_match': same_ids,
                      'expected_token_count': len(expected), 'actual_token_count': len(actual),
                      'first_token_difference': _first_difference(expected, actual)})
    for name, path in paths.items():
        if _artifact_hash(path) != reference['artifacts'][name]:
            raise ValueError(f'{name}: artifact hash changed during token qualification')
    return {'validation_level': 'prompt_token_id_parity_only',
            'matched': not mismatches and all(case['matched'] for case in cases),
            'reference_kind': reference['source']['kind'],
            'checkpoint': reference['checkpoint'], 'source': reference['source'],
            'special_token_mismatches': mismatches, 'case_count': len(cases), 'cases': cases}
