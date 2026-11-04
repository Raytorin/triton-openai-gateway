# SPDX-License-Identifier: Apache-2.0
"""DeepSeek pixel admission, canonical image ordering and context reservation."""
from dataclasses import dataclass
import json
from pathlib import Path

from fastapi import HTTPException


@dataclass(frozen=True)
class VisionPolicy:
    image_limit: int = 0
    max_image_tokens: int = 0
    image_token_id: int = 129264


def load_vision_policy(model_path):
    try:
        engine = json.loads((model_path / 'model.json').read_text())
        if engine.get('enable_mm_embeds', False) is not False:
            raise ValueError('DeepSeek requires enable_mm_embeds:false')
        lm_only = engine.get('language_model_only', True)
        if type(lm_only) is not bool:
            raise ValueError('language_model_only must be a boolean')
        if lm_only:
            return VisionPolicy()
        limits = engine.get('limit_mm_per_prompt')
        limit = limits.get('image') if isinstance(limits, dict) else None
        if type(limit) is not int or limit <= 0:
            raise ValueError('DeepSeek images require positive limit_mm_per_prompt.image')
        if any(value not in (0, None) for key, value in limits.items() if key != 'image'):
            raise ValueError('DeepSeek supports image modality only')
        if engine.get('mm_processor_kwargs'):
            raise ValueError('DeepSeek processor overrides are unsupported')
        model = engine.get('model')
        root = Path(model) if isinstance(model, str) and Path(model).is_absolute() else model_path
        config = json.loads((root / 'config.json').read_text())
        vision = config.get('vision_config')
        if (config.get('model_type') != 'deepseek_v41'
                or config.get('architectures') != ['DeepseekV41ForCausalLM']
                or not isinstance(vision, dict)
                or vision.get('model_type') != 'deepseek_v41_vision'
                or config.get('image_token_id') != 129264):
            raise ValueError('DeepSeek images require a verified V4.1 vision architecture')
        maximum = vision.get('max_image_tokens')
        if type(maximum) is not int or maximum <= 0:
            raise ValueError('vision_config.max_image_tokens must be a positive integer')
        return VisionPolicy(limit, maximum)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise HTTPException(500, f'Invalid DeepSeek vision policy: {error}') from error


def validate_media(policy, media, *, check_image_limit=True):
    if not media.has_any:
        return
    if not policy.image_limit:
        raise HTTPException(400, 'DeepSeek is configured for text generation only; image input requires explicit opt-in')
    if media.videos or media.audios or media.pdfs:
        raise HTTPException(400, 'DeepSeek supports images only; audio, video and PDF are unsupported')
    if check_image_limit and len(media.images) > policy.image_limit:
        raise HTTPException(400, f'DeepSeek received {len(media.images)} images; maximum is {policy.image_limit}')


def canonicalize_conversation(conversation):
    """Match the pinned template's stable sorting of tool-result block slots.

    Consecutive user/tool messages form one template group. User image slots
    stay in place while tool-result slots follow the preceding assistant calls.
    """
    result = []
    ids = []
    index = 0
    while index < len(conversation):
        message = conversation[index]
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            ids = [call.get('id') or call.get('function', {}).get('id', '')
                   for call in message['tool_calls']]
        if message.get('role') not in {'user', 'tool'}:
            result.append(message)
            index += 1
            continue
        end = index
        while end < len(conversation) and conversation[end].get('role') in {'user', 'tool'}:
            end += 1
        group = conversation[index:end]
        def rank(message):
            call_id = message.get('tool_call_id', '')
            return max((i for i, value in enumerate(ids) if value and value == call_id), default=0)
        ordered = iter(sorted((m for m in group if m.get('role') == 'tool'), key=rank))
        result.extend(next(ordered) if m.get('role') == 'tool' else m for m in group)
        index = end
    return result


class CanonicalVisionTokenizer:
    """Canonicalize render candidates while retaining their original turn order.

    Context fitting and history selection must see the original conversation:
    sorting a tool result into an older user turn could make it removable.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, conversation, **kwargs):
        return self._tokenizer.apply_chat_template(canonicalize_conversation(conversation), **kwargs)


def additional_image_tokens(policy, tokenizer, prompt, media):
    if not media.images:
        return 0
    # Pinned vLLM 5df63a7 common/mm_preprocess.py safe_resize bounds the
    # entire H*(W+1)+2 span by max_n_token, including its row/start/end tokens.
    bound = len(media.images) * policy.max_image_tokens
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    counted = sum(token == policy.image_token_id for token in tokens)
    if counted != len(media.images):
        raise HTTPException(500, 'DeepSeek image token IDs do not match pixel payloads')
    return max(0, bound - counted)
