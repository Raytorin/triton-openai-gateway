# SPDX-License-Identifier: Apache-2.0
"""Version-specific native prompt preparation and trustworthy processor facts."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BuiltMediaPrompt:
    prompt: Any
    origins: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PreparedEnginePrompt:
    prompt: Any
    vision_usage: dict[str, Any] | None


def unavailable_vision_usage():
    return {'status': 'unavailable', 'source': 'engine',
            'unit': 'embedding_positions', 'by_source': {}}


def extract_vision_usage(prompt, origins):
    """Count processor embeddings, validating every positional origin first.

    Span lengths only validate positions; get_num_embeds owns the count.
    Encoder/decoder and unknown layouts deliberately have no actual statistic.
    """
    unavailable = unavailable_vision_usage()
    try:
        if (not isinstance(prompt, dict) or prompt.get('type') != 'multimodal'
                or 'encoder_prompt_token_ids' in prompt):
            return unavailable
        placeholders = prompt['mm_placeholders']
        token_count = len(prompt['prompt_token_ids'])
        if not isinstance(placeholders, dict) or not origins:
            return unavailable
        if {key for key in placeholders if key in {'image', 'video'}} != set(origins):
            return unavailable
        counts = {}
        occupied = []
        for modality, kinds in origins.items():
            if modality not in {'image', 'video'} or not kinds:
                return unavailable
            ranges = placeholders.get(modality)
            if not isinstance(ranges, (list, tuple)) or len(ranges) != len(kinds):
                return unavailable
            previous_end = 0
            for item, kind in zip(ranges, kinds):
                if kind not in {'image', 'pdf', 'video'} or modality == 'video' and kind != 'video':
                    return unavailable
                offset, length = item.offset, item.length
                if type(offset) is not int or type(length) is not int or offset < previous_end or length < 0 or offset + length > token_count:
                    return unavailable
                previous_end = offset + length
                occupied.append((offset, offset + length))
                mask = getattr(item, 'is_embed', None)
                if mask is not None:
                    if getattr(mask, 'ndim', 1) != 1 or len(mask) != length:
                        return unavailable
                    values = mask.tolist() if hasattr(mask, 'tolist') else list(mask)
                    if any(type(value) is not bool for value in values):
                        return unavailable
                count = item.get_num_embeds()
                if type(count) is not int or count < 0 or count > length:
                    return unavailable
                if mask is not None and count != sum(values):
                    return unavailable
                counts[kind] = counts.get(kind, 0) + count
        occupied.sort()
        if any(left[1] > right[0] for left, right in zip(occupied, occupied[1:])):
            return unavailable
        return {**unavailable, 'status': 'available', 'by_source': counts}
    except Exception:
        # Statistics must never cause an already prepared input to be re-rendered.
        return unavailable


def make_prompt_preparer(engine):
    """Return an adapter only when the pinned native EngineInput API exists."""
    renderer = getattr(engine, 'renderer', None)
    render = getattr(renderer, 'render_cmpl_async', None)
    tok_params = getattr(renderer, 'default_cmpl_tok_params', None)
    model_config = getattr(engine, 'model_config', None)
    if not callable(render) or tok_params is None or model_config is None:
        return None
    try:
        from vllm.renderers.inputs.preprocess import parse_model_prompt
    except ImportError:
        return None

    async def prepare(built: BuiltMediaPrompt) -> PreparedEnginePrompt:
        parsed = parse_model_prompt(model_config, built.prompt)
        (prepared,) = await render([parsed], tok_params)
        # No fallback after entering native preparation, including malformed output.
        if not isinstance(prepared, dict) or 'type' not in prepared:
            raise ValueError('Native renderer did not return an EngineInput')
        usage = extract_vision_usage(prepared, built.origins) if built.origins else None
        return PreparedEnginePrompt(prepared, usage)

    return prepare
