# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Capabilities of the *resolved* chat template, independent of model size.

Model metadata selects Qwen history handling only. Control support comes from
Jinja variable reads and bounded renders, cached by source hash and renderer
class. The tokenizer and its configuration are never modified.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import inspect
import json
from pathlib import Path
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class TemplateCapabilities:
    template: str | None = None
    fingerprint: str = ''
    thinking: str = 'unknown'  # switch, on, off, unknown
    reads_enable_thinking: bool = False
    preserves_thinking: bool = False
    efforts: frozenset[str] = frozenset()
    has_think_tags: bool = False


_CACHE: OrderedDict[tuple[str, type, bool, str], TemplateCapabilities] = OrderedDict()
_CACHE_LOCK = Lock()


def qwen_template_capabilities(
    model_path: Path, tokenizer: Any = None, *, tools: list[dict[str, Any]] | None = None,
) -> tuple[bool, TemplateCapabilities]:
    locations = _metadata_locations(model_path, tokenizer)
    qwen_profile = _is_qwen_profile(locations)
    source = _resolved_template(model_path, tokenizer, tools=tools)
    if source is None:
        return qwen_profile, TemplateCapabilities()
    fingerprint = sha256(source.encode('utf-8')).hexdigest()
    renderer_type = type(tokenizer) if callable(getattr(tokenizer, 'apply_chat_template', None)) else type(None)
    tools_key = sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False,
                                 separators=(",", ":")).encode("utf-8")).hexdigest()
    key = (fingerprint, renderer_type, qwen_profile, tools_key)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return qwen_profile, cached
    capabilities = _probe_template(source, fingerprint, tokenizer, qwen_profile, tools)
    with _CACHE_LOCK:
        _CACHE[key] = capabilities
        _CACHE.move_to_end(key)
        while len(_CACHE) > 256:
            _CACHE.popitem(last=False)
    return qwen_profile, capabilities


def _probe_template(
    source: str, fingerprint: str, tokenizer: Any, qwen_profile: bool,
    tools: list[dict[str, Any]] | None,
) -> TemplateCapabilities:
    try:
        from jinja2 import meta, nodes
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols']
        )
        environment.globals['raise_exception'] = _raise_template_error
        tree = environment.parse(source)
        variables = meta.find_undeclared_variables(tree)
        literals = [node.value for node in tree.find_all(nodes.Const) if isinstance(node.value, str)]
        literals += [node.data for node in tree.find_all(nodes.TemplateData)]
    except (ImportError, ValueError, TypeError, AttributeError):
        # No template evidence means explicit controls are rejected, never
        # silently swallowed. Minimal legacy test tokenizers still work.
        return TemplateCapabilities(template=source, fingerprint=fingerprint)
    except Exception:
        # Invalid/unsupported Jinja must not turn into guessed capabilities.
        return TemplateCapabilities(template=source, fingerprint=fingerprint)

    has_tags = any(tag in literal for literal in literals for tag in ('<think>', '<thinking>'))
    reads_enable = 'enable_thinking' in variables
    renderer = getattr(tokenizer, 'apply_chat_template', None)
    if callable(renderer):
        # Use the same selection, strict controls, and tools translation as the
        # real request. In particular, a no-tools signature must be probed via
        # the existing system-instruction fallback rather than a different Jinja.
        from .prompt import RequestTokenizer, render_chat_prompt

        def render(*, add_generation_prompt=True, **kwargs):
            wrapped = RequestTokenizer(tokenizer, kwargs, strict=True,
                                       chat_template=source, tools=tools)
            try:
                result = render_chat_prompt(wrapped, [{'role': 'user', 'content': 'Hello'}],
                    tools, enable_thinking=kwargs['enable_thinking'],
                    add_generation_prompt=add_generation_prompt)
                return result if isinstance(result, str) else None
            except Exception:
                return None
    else:
        compiled = environment.from_string(source)

        def render(*, add_generation_prompt=True, **kwargs):
            try:
                result = compiled.render(messages=[{'role': 'user', 'content': 'Hello'}],
                    add_generation_prompt=add_generation_prompt, tools=tools, **kwargs)
                return result if isinstance(result, str) else None
            except Exception:
                return None

    on = render(enable_thinking=True)
    off = render(enable_thinking=False)
    on_prefix = _generation_suffix(on, render(enable_thinking=True, add_generation_prompt=False))
    off_prefix = _generation_suffix(off, render(enable_thinking=False, add_generation_prompt=False))
    on_state = _thinking_prefix(on_prefix)
    off_state = _thinking_prefix(off_prefix)
    thinking = 'unknown'
    # Qwen3 hybrid lets the model emit the opening tag when enabled,
    # whereas Qwen3.5+ prefill it. Both disable via an empty closed block.
    empty_block_off = (off_prefix is not None and on_prefix is not None and off_prefix != on_prefix
                       and any(tag in off_prefix for tag in ('</think>', '</thinking>'))
                       and not any(tag in on_prefix for tag in ('<think>', '<thinking>')))
    if reads_enable and off_state is False and (on_state is True or empty_block_off):
        thinking = 'switch'
    elif on_state is True and off_state is True:
        thinking = 'on'
    elif (not reads_enable and on_prefix and off_prefix
          and on_state is False and off_state is False and (qwen_profile or has_tags)):
        thinking = 'off'

    efforts: set[str] = set()
    if 'reasoning_effort' in variables:
        values = {value: render(enable_thinking=True, reasoning_effort=value)
                  for value in ('low', 'medium', 'high', 'xhigh')}
        # An unused variable or a fallback that silently accepts arbitrary
        # values is not evidence that an effort is implemented.
        invalid = render(enable_thinking=True, reasoning_effort='__gateway_unsupported_effort__')
        for value, rendered in values.items():
            if rendered is not None and rendered != invalid:
                efforts.add(value)

    return TemplateCapabilities(
        template=source,
        fingerprint=fingerprint,
        thinking=thinking,
        reads_enable_thinking=reads_enable,
        preserves_thinking='preserve_thinking' in variables,
        efforts=frozenset(efforts),
        has_think_tags=has_tags,
    )


def _generation_suffix(rendered: str | None, without_generation: str | None) -> str | None:
    if rendered is None or without_generation is None or not rendered.startswith(without_generation):
        return None
    return rendered[len(without_generation):]


def _thinking_prefix(rendered: str | None) -> bool | None:
    if rendered is None:
        return None
    # Only the generated prefix is evidence. Tool descriptions, system text,
    # and user content may legitimately contain either thinking tag.
    opening = max(rendered.rfind('<think>'), rendered.rfind('<thinking>'))
    closing = max(rendered.rfind('</think>'), rendered.rfind('</thinking>'))
    return opening > closing if opening >= 0 else False


def chat_template_accepts_keyword(method: Any, keyword: str) -> bool:
    if not callable(method):
        return False
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        # Unknown signatures are verified by the real bounded render probes.
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD
               or (parameter.name == keyword
                   and parameter.kind != inspect.Parameter.POSITIONAL_ONLY)
               for parameter in parameters)


def _resolved_template(
    model_path: Path, tokenizer: Any, *, tools: list[dict[str, Any]] | None = None,
) -> str | None:
    # A renderer without tools= eventually uses the tools-to-system fallback;
    # its getter must resolve the default that this fallback actually renders.
    selected_tools = tools if chat_template_accepts_keyword(
        getattr(tokenizer, 'apply_chat_template', None), 'tools') else None
    getter = getattr(tokenizer, 'get_chat_template', None)
    if callable(getter):
        try:
            template = getter(tools=selected_tools) if chat_template_accepts_keyword(getter, 'tools') else getter()
        except (TypeError, ValueError, AttributeError):
            template = getattr(tokenizer, 'chat_template', None)
    else:
        template = getattr(tokenizer, 'chat_template', None)
    if isinstance(template, dict):
        template = template.get('tool_use' if selected_tools and 'tool_use' in template else 'default', template)
    if isinstance(template, str) and template:
        return template
    if template is not None:
        # A real unresolved named-template mapping must not be replaced with
        # a stale disk default. get_chat_template normally resolves it for us.
        return None
    path = model_path / 'chat_template.jinja'
    if path.is_file():
        try:
            return path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return None
    payload = _read_json(model_path / 'tokenizer_config.json')
    template = payload.get('chat_template')
    if isinstance(template, str) and template:
        return template
    return None


def _metadata_locations(model_path: Path, tokenizer: Any) -> tuple[Path, ...]:
    # AutoTokenizer retains its actual loading path. Prefer that identity over
    # a registry setting that could have changed after this tokenizer loaded.
    locations: list[Path] = []
    loaded_path = getattr(tokenizer, 'name_or_path', None)
    if isinstance(loaded_path, (str, Path)):
        try:
            candidate = Path(loaded_path)
            if candidate.is_dir():
                locations.append(candidate)
        except (OSError, ValueError):
            pass
    from .registry import ModelRegistry
    try:
        locations.append(ModelRegistry().resolve_tokenizer_path(model_path))
    except (OSError, UnicodeDecodeError, ValueError, TypeError):
        pass
    locations.append(model_path)
    return tuple(dict.fromkeys(locations))


def _is_qwen_profile(locations: tuple[Path, ...]) -> bool:
    for location in locations:
        config = _read_json(location / 'config.json')
        model_type = config.get('model_type')
        architectures = config.get('architectures')
        architectures = architectures if isinstance(architectures, list) else []
        if isinstance(model_type, str) or architectures:
            return (isinstance(model_type, str) and model_type.lower().startswith('qwen3')
                    or any(isinstance(name, str) and name.lower().startswith('qwen3')
                           for name in architectures))
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


def _raise_template_error(message: str) -> None:
    raise ValueError(message)
