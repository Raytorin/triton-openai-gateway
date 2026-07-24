# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict
import hashlib
import math
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .media_preprocessing import (
    VisualItem,
    VllmMediaProcessingError,
    VllmMediaSettings,
    _extract_video_frames,
    _extract_pdf_text_pages,
    _image_payload_to_item,
    _materialize_payload,
    _render_pdf_pages,
    _transcribe_audio,
    load_vllm_media_settings,
    select_video_frame_indices,
)
from .observability import log_event
from .metrics import PDF_EMBEDDING_CACHE
from .settings import logger


_pdf_embedding_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
_pdf_embedding_cache_lock = threading.Lock()


@dataclass(frozen=True)
class MediaSummary:
    label: str
    text: str


@dataclass(frozen=True)
class PdfEmbeddingContext:
    model_name: str
    tokenizer: Any | None = None
    dimensions: int | None = None


def has_extended_vllm_media(media: Any) -> bool:
    return bool(media.videos or media.audios or media.pdfs)


def requires_gateway_media_preprocessing(
    backend: str | None,
    media: Any,
    settings: VllmMediaSettings | None = None,
) -> bool:
    if backend == "vllm":
        return has_extended_vllm_media(media)
    if backend == "vllm_multimodal":
        # One PDF expands into an unknown number of image items. Process it in
        # bounded chunks so it cannot exceed limit_mm_per_prompt.image.
        # Audio-capable architectures keep the native path. Vision-only models
        # can opt into the gateway ASR path with audio_asr_model.
        return bool(media.pdfs) or bool(
            media.audios and settings and settings.audio_asr_model
        )
    return False


async def materialize_native_remote_media(
    model_path: Path,
    media: Any,
) -> Any:
    remote_count = sum(
        payload.data.startswith(("http://", "https://"))
        for payloads in (media.images, media.videos, media.audios, media.pdfs)
        for payload in payloads
    )
    if not remote_count:
        return media

    started_at = time.monotonic()
    settings = load_vllm_media_settings(model_path)
    temp_parent = settings.temp_dir
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="triton-native-media-",
        dir=str(temp_parent) if temp_parent else None,
    ) as directory:
        work_dir = Path(directory)
        materialized = {}
        for media_type, payloads in (
            ("image", media.images),
            ("video", media.videos),
            ("audio", media.audios),
            ("pdf", media.pdfs),
        ):
            updated = []
            for index, payload in enumerate(payloads, start=1):
                if not payload.data.startswith(("http://", "https://")):
                    updated.append(payload)
                    continue
                path = await asyncio.to_thread(
                    _materialize_payload,
                    payload,
                    media_type,
                    work_dir,
                    index,
                    settings,
                )
                media_format = payload.format or path.suffix.lstrip(".") or None
                encoded = await asyncio.to_thread(_path_as_base64, path)
                updated.append(
                    replace(
                        payload,
                        data=encoded,
                        format=media_format,
                    )
                )
            materialized[f"{media_type}s"] = updated

    result = replace(media, **materialized)
    log_event(
        logger,
        "media.remote_materialized",
        "Remote media materialized",
        media_count=remote_count,
        duration_ms=round((time.monotonic() - started_at) * 1000, 3),
    )
    return result


def _path_as_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


async def prepare_vllm_media_conversation(
    model_name: str,
    model_path: Path,
    tokenizer: Any,
    conversation: list[dict[str, Any]],
    media: Any,
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings | None = None,
    pdf_embedding: PdfEmbeddingContext | None = None,
) -> list[dict[str, Any]]:
    """Convert unsupported Triton vLLM media to image/text map-reduce input."""
    settings = settings or load_vllm_media_settings(model_path)
    user_request = _latest_user_text(conversation)
    system_messages = _system_text_messages(conversation)
    summaries: list[MediaSummary] = []

    temp_parent = settings.temp_dir
    if temp_parent is not None:
        temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="triton-vllm-media-",
        dir=str(temp_parent) if temp_parent else None,
    ) as directory:
        work_dir = Path(directory)

        # Existing images are included when a request also contains extended media.
        for image_index, payload in enumerate(media.images, start=1):
            item = await asyncio.to_thread(
                _image_payload_to_item,
                payload,
                work_dir,
                image_index,
                settings,
            )
            summaries.extend(
                await _summarize_visual_items(
                    model_name,
                    tokenizer,
                    [item],
                    user_request,
                    system_messages,
                    sampling_parameters,
                    settings,
                )
            )

        for document_index, payload in enumerate(media.pdfs, start=1):
            document_summaries: list[MediaSummary] = []
            text_pages: list[tuple[str, str]] = []
            if settings.pdf_text_mode != "visual":
                try:
                    text_pages = await asyncio.to_thread(
                        _extract_pdf_text_pages,
                        payload,
                        work_dir,
                        document_index,
                        settings,
                    )
                except VllmMediaProcessingError as exc:
                    logger.info(
                        "PDF %d text extraction failed; using visual fallback: %s",
                        document_index,
                        exc,
                    )
                extracted_chars = sum(len(text) for _, text in text_pages)
                if (
                    extracted_chars >= settings.pdf_rag_min_text_chars
                    and pdf_embedding is not None
                    and not _is_document_summary_request(user_request)
                ):
                    document_summaries = await _retrieve_pdf_text_chunks(
                        text_pages,
                        user_request,
                        pdf_embedding,
                        settings,
                    )
                    logger.info(
                        "Selected %d PDF text chunk(s) with embedding model '%s'",
                        len(document_summaries),
                        pdf_embedding.model_name,
                    )
                elif extracted_chars >= settings.pdf_rag_min_text_chars:
                    text = "\n\n".join(
                        f"[{label}]\n{page_text}" for label, page_text in text_pages
                    )
                    document_summaries = await _summarize_text_chunks(
                        model_name,
                        tokenizer,
                        text,
                        f"PDF {document_index}",
                        user_request,
                        system_messages,
                        sampling_parameters,
                        settings,
                    )
                    document_summaries = await _reduce_summaries(
                        model_name,
                        tokenizer,
                        document_summaries,
                        user_request,
                        system_messages,
                        sampling_parameters,
                        settings,
                        force_single=True,
                        document_label=f"PDF {document_index} synthesis",
                    )
                    logger.info(
                        "Prepared PDF %d from %d extracted text character(s)",
                        document_index,
                        extracted_chars,
                    )

            if not document_summaries:
                pages = await asyncio.to_thread(
                    _render_pdf_pages,
                    payload,
                    work_dir,
                    document_index,
                    settings,
                )
                logger.info(
                    "Prepared PDF %d with %d page(s) for vLLM chunking",
                    document_index,
                    len(pages),
                )
                document_summaries = await _summarize_visual_items(
                    model_name,
                    tokenizer,
                    pages,
                    user_request,
                    system_messages,
                    sampling_parameters,
                    settings,
                    chunk_size=settings.pdf_chunk_pages,
                )
                document_summaries = await _reduce_summaries(
                    model_name,
                    tokenizer,
                    document_summaries,
                    user_request,
                    system_messages,
                    sampling_parameters,
                    settings,
                    force_single=True,
                    document_label=f"PDF {document_index} synthesis",
                )
            summaries.extend(document_summaries)

        for video_index, payload in enumerate(media.videos, start=1):
            frames = await asyncio.to_thread(
                _extract_video_frames,
                payload,
                work_dir,
                video_index,
                settings,
            )
            logger.info(
                "Prepared video %d with %d sampled frame(s) for vLLM chunking",
                video_index,
                len(frames),
            )
            summaries.extend(
                await _summarize_visual_items(
                    model_name,
                    tokenizer,
                    frames,
                    user_request,
                    system_messages,
                    sampling_parameters,
                    settings,
                    chunk_size=settings.video_chunk_frames,
                )
            )

        for audio_index, payload in enumerate(media.audios, start=1):
            audio_path = await asyncio.to_thread(
                _materialize_payload,
                payload,
                "audio",
                work_dir,
                audio_index,
                settings,
            )
            transcript = await asyncio.to_thread(
                _transcribe_audio,
                audio_path,
                settings,
            )
            logger.info(
                "Transcribed audio %d into %d character(s)",
                audio_index,
                len(transcript),
            )
            summaries.extend(
                await _summarize_text_chunks(
                    model_name,
                    tokenizer,
                    transcript,
                    f"Audio {audio_index}",
                    user_request,
                    system_messages,
                    sampling_parameters,
                    settings,
                )
            )

    if not summaries:
        raise VllmMediaProcessingError("No usable media content was extracted")

    summaries = await _reduce_summaries(
        model_name,
        tokenizer,
        summaries,
        user_request,
        system_messages,
        sampling_parameters,
        settings,
    )
    logger.info(
        "Prepared %d final media summary block(s) for model '%s'",
        len(summaries),
        model_name,
    )
    return _conversation_with_media_evidence(conversation, user_request, summaries)


async def _summarize_visual_items(
    model_name: str,
    tokenizer: Any,
    items: list[VisualItem],
    user_request: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
    chunk_size: int | None = None,
) -> list[MediaSummary]:
    chunk_size = max(min(chunk_size or settings.image_limit, settings.image_limit), 1)
    chunks = [
        items[start : start + chunk_size]
        for start in range(0, len(items), chunk_size)
    ]
    semaphore = asyncio.Semaphore(settings.media_map_concurrency)

    async def summarize(chunk: list[VisualItem]) -> list[MediaSummary]:
        async with semaphore:
            return await _summarize_visual_chunk_adaptive(
                model_name,
                tokenizer,
                chunk,
                user_request,
                system_messages,
                sampling_parameters,
                settings,
            )

    groups = await asyncio.gather(*(summarize(chunk) for chunk in chunks))
    return [summary for group in groups for summary in group]


async def _summarize_visual_chunk_adaptive(
    model_name: str,
    tokenizer: Any,
    items: list[VisualItem],
    user_request: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
) -> list[MediaSummary]:
    try:
        summary = await _generate_visual_summary(
            model_name,
            tokenizer,
            items,
            user_request,
            system_messages,
            sampling_parameters,
            settings,
        )
        return [MediaSummary(label=_chunk_label(items), text=summary)]
    except Exception as exc:
        if len(items) > 1 and _is_context_length_error(exc):
            middle = max(len(items) // 2, 1)
            left = await _summarize_visual_chunk_adaptive(
                model_name,
                tokenizer,
                items[:middle],
                user_request,
                system_messages,
                sampling_parameters,
                settings,
            )
            right = await _summarize_visual_chunk_adaptive(
                model_name,
                tokenizer,
                items[middle:],
                user_request,
                system_messages,
                sampling_parameters,
                settings,
            )
            return [*left, *right]
        if _is_context_length_error(exc):
            raise VllmMediaProcessingError(
                f"Media chunk '{_chunk_label(items)}' exceeds the model context. "
                "Reduce media pixel limits or increase max_model_len.",
                status_code=413,
            ) from exc
        raise


async def _generate_visual_summary(
    model_name: str,
    tokenizer: Any,
    items: list[VisualItem],
    user_request: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
) -> str:
    from .prompt import render_chat_prompt
    from .sanitizer import sanitize_generated_text, strip_prompt_echo
    from .triton_client import call_triton_multimodal

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Analyze only the media items in this chunk. Preserve facts, numbers, "
                "visible text, page/frame references and temporal order needed for the "
                "user request. Do not claim to have seen other chunks.\n"
                f"User request: {user_request}"
            ),
        }
    ]
    for item in items:
        content.extend(
            [
                {"type": "text", "text": item.label},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"},
                },
            ]
        )

    chunk_conversation = [
        *system_messages,
        {"role": "user", "content": content},
    ]
    prompt = render_chat_prompt(tokenizer, chunk_conversation)
    generated = await call_triton_multimodal(
        model_name,
        prompt,
        _summary_sampling(sampling_parameters, settings),
        [item.data for item in items],
    )
    generated = strip_prompt_echo(prompt, generated)
    generated, _ = sanitize_generated_text(generated)
    if not generated.strip():
        raise VllmMediaProcessingError(
            f"Model returned an empty summary for '{_chunk_label(items)}'",
            status_code=502,
        )
    return generated.strip()


async def _summarize_text_chunks(
    model_name: str,
    tokenizer: Any,
    text: str,
    source_label: str,
    user_request: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
) -> list[MediaSummary]:
    chunks = _split_text(text, settings.text_chunk_chars)
    if len(chunks) == 1:
        return [MediaSummary(label=source_label, text=chunks[0])]

    semaphore = asyncio.Semaphore(settings.media_map_concurrency)

    async def summarize(index: int, chunk: str) -> MediaSummary:
        instruction = (
            f"User request: {user_request}\n\n"
            f"This is transcript chunk {index}/{len(chunks)} from {source_label}. "
            "Extract relevant facts and preserve names, numbers and timestamps. "
            "Do not add information.\n\n"
            f"{chunk}"
        )
        async with semaphore:
            summary = await _generate_text_summary(
                model_name,
                tokenizer,
                instruction,
                system_messages,
                sampling_parameters,
                settings,
            )
        return MediaSummary(
                label=f"{source_label}, transcript chunk {index}/{len(chunks)}",
                text=summary,
            )

    return list(
        await asyncio.gather(
            *(summarize(index, chunk) for index, chunk in enumerate(chunks, start=1))
        )
    )


async def _retrieve_pdf_text_chunks(
    pages: list[tuple[str, str]],
    user_request: str,
    embedding: PdfEmbeddingContext,
    settings: VllmMediaSettings,
) -> list[MediaSummary]:
    from .embeddings import tokenize_embedding_inputs
    from .triton_client import call_triton_embeddings

    chunks = _build_pdf_text_chunks(
        pages,
        settings.pdf_rag_chunk_chars,
        settings.pdf_rag_chunk_overlap,
    )
    if not chunks:
        return []

    query = user_request
    if settings.pdf_embedding_query_prefix:
        query = f"{settings.pdf_embedding_query_prefix}{user_request}"

    async def embed(value: str) -> list[float]:
        cache_key = _embedding_cache_key(
            embedding.model_name,
            embedding.dimensions,
            value,
        )
        cached = _embedding_cache_get(cache_key)
        if cached is not None:
            PDF_EMBEDDING_CACHE.labels(embedding.model_name, "hit").inc()
            return cached
        PDF_EMBEDDING_CACHE.labels(embedding.model_name, "miss").inc()
        model_input: str | list[int] = value
        if embedding.tokenizer is not None:
            model_input = tokenize_embedding_inputs(embedding.tokenizer, [value])[0]
        vector, _ = await call_triton_embeddings(
            embedding.model_name,
            model_input,
            embedding.dimensions,
        )
        _embedding_cache_put(
            cache_key,
            vector,
            settings.pdf_embedding_cache_size,
        )
        return vector

    query_vector = await embed(query)
    semaphore = asyncio.Semaphore(settings.pdf_rag_embedding_concurrency)

    async def score(index: int, summary: MediaSummary):
        async with semaphore:
            chunk_vector = await embed(summary.text)
        return _cosine_similarity(query_vector, chunk_vector), index, summary

    scored = list(
        await asyncio.gather(
            *(score(index, summary) for index, summary in enumerate(chunks))
        )
    )

    top_k = min(max(settings.pdf_rag_top_k, 1), len(scored))
    selected = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
    # Restore document order after relevance selection so the final model sees
    # a coherent excerpt rather than a score-sorted fragment list.
    selected.sort(key=lambda item: item[1])
    return [summary for _, _, summary in selected]


def _embedding_cache_key(
    model_name: str,
    dimensions: int | None,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{model_name}:{dimensions or 0}:{digest}"


def _embedding_cache_get(key: str) -> list[float] | None:
    with _pdf_embedding_cache_lock:
        value = _pdf_embedding_cache.get(key)
        if value is None:
            return None
        _pdf_embedding_cache.move_to_end(key)
        return list(value)


def _embedding_cache_put(key: str, value: list[float], max_size: int) -> None:
    with _pdf_embedding_cache_lock:
        _pdf_embedding_cache[key] = tuple(float(item) for item in value)
        _pdf_embedding_cache.move_to_end(key)
        while len(_pdf_embedding_cache) > max(max_size, 1):
            _pdf_embedding_cache.popitem(last=False)


def _build_pdf_text_chunks(
    pages: list[tuple[str, str]],
    chunk_chars: int,
    overlap: int,
) -> list[MediaSummary]:
    chunk_chars = max(int(chunk_chars), 256)
    overlap = min(max(int(overlap), 0), chunk_chars // 2)
    result: list[MediaSummary] = []

    for label, page_text in pages:
        text = page_text.strip()
        cursor = 0
        part = 1
        while cursor < len(text):
            end = min(cursor + chunk_chars, len(text))
            if end < len(text):
                boundary = max(text.rfind("\n", cursor, end), text.rfind(" ", cursor, end))
                if boundary > cursor + chunk_chars // 2:
                    end = boundary
            chunk = text[cursor:end].strip()
            if chunk:
                suffix = f", text part {part}" if len(text) > chunk_chars else ""
                result.append(MediaSummary(label=f"{label}{suffix}", text=chunk))
                part += 1
            if end >= len(text):
                break
            cursor = max(end - overlap, cursor + 1)
    return result


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return float("-inf")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return float("-inf")
    return dot / (left_norm * right_norm)


def _is_document_summary_request(user_request: str) -> bool:
    normalized = " ".join(user_request.lower().split())
    summary_markers = (
        "о чем документ",
        "о чём документ",
        "о чем данный документ",
        "о чём данный документ",
        "суть документа",
        "краткое содержание",
        "резюме документа",
        "суммариз",
        "summarize",
        "summary of",
        "what is this document about",
        "what is the document about",
        "overview of the document",
    )
    return any(marker in normalized for marker in summary_markers)


async def _reduce_summaries(
    model_name: str,
    tokenizer: Any,
    summaries: list[MediaSummary],
    user_request: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
    force_single: bool = False,
    document_label: str = "Media synthesis",
) -> list[MediaSummary]:
    result = summaries
    group_size = max(settings.reduce_group_size, 2)
    limit = 1 if force_single else max(settings.final_summary_limit, 2)
    requested_output_tokens = int(sampling_parameters.get("max_tokens") or 256)
    evidence_token_budget = max(
        settings.max_model_len - requested_output_tokens - 512,
        256,
    )
    synthesize_single = force_single and len(result) == 1

    while (
        len(result) > limit
        or _summary_token_count(tokenizer, result) > evidence_token_budget
        or synthesize_single
    ):
        if len(result) == 1:
            instruction = (
                f"User request: {user_request}\n\n"
                "Synthesize the following notes into one coherent document-level context. "
                "Preserve relevant facts, numbers and relationships. Do not reproduce "
                "technical chunk labels or enumerate pages unless the user explicitly asks "
                "for a page-by-page answer. Do not add "
                f"information.\n\n{_format_summaries(result)}"
            )
            summary = await _generate_text_summary(
                model_name,
                tokenizer,
                instruction,
                system_messages,
                sampling_parameters,
                settings,
            )
            result = [MediaSummary(label=document_label, text=summary)]
            break

        groups = [
            result[start : start + group_size]
            for start in range(0, len(result), group_size)
        ]
        semaphore = asyncio.Semaphore(settings.media_map_concurrency)

        async def reduce_group(group: list[MediaSummary]) -> MediaSummary:
            instruction = (
                f"User request: {user_request}\n\n"
                "Synthesize the following analysis notes into coherent document-level "
                "context without losing relevant facts, numbers or relationships. Do not "
                "mirror technical chunk labels. Keep page/frame references only when they "
                "help answer the request. Do not add information.\n\n"
                f"{_format_summaries(group)}"
            )
            async with semaphore:
                summary = await _generate_text_summary(
                    model_name,
                    tokenizer,
                    instruction,
                    system_messages,
                    sampling_parameters,
                    settings,
                )
            return MediaSummary(
                    label=document_label if force_single else (
                        f"Reduced notes: {group[0].label} - {group[-1].label}"
                    ),
                    text=summary,
                )
        result = list(await asyncio.gather(*(reduce_group(group) for group in groups)))
    return result


async def _generate_text_summary(
    model_name: str,
    tokenizer: Any,
    instruction: str,
    system_messages: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
) -> str:
    from .prompt import render_chat_prompt
    from .sanitizer import sanitize_generated_text, strip_prompt_echo
    from .triton_client import call_triton_multimodal

    conversation = [*system_messages, {"role": "user", "content": instruction}]
    prompt = render_chat_prompt(tokenizer, conversation)
    try:
        generated = await call_triton_multimodal(
            model_name,
            prompt,
            _summary_sampling(sampling_parameters, settings),
            [],
        )
    except Exception as exc:
        if _is_context_length_error(exc):
            raise VllmMediaProcessingError(
                "Intermediate media summaries exceed the model context. "
                "Reduce summary_max_tokens/reduce_group_size or increase max_model_len.",
                status_code=413,
            ) from exc
        raise

    generated = strip_prompt_echo(prompt, generated)
    generated, _ = sanitize_generated_text(generated)
    if not generated.strip():
        raise VllmMediaProcessingError(
            "Model returned an empty intermediate media summary",
            status_code=502,
        )
    return generated.strip()


def _conversation_with_media_evidence(
    conversation: list[dict[str, Any]],
    user_request: str,
    summaries: list[MediaSummary],
) -> list[dict[str, Any]]:
    text_only = []
    for message in conversation:
        updated = dict(message)
        updated["content"] = _content_text(message.get("content"))
        if updated["content"] or message.get("role") in {"assistant", "tool"}:
            text_only.append(updated)

    evidence = (
        "The attached media was processed into internal evidence. Answer the original "
        "request directly and coherently. Do not expose processing details, chunk labels, "
        "or list pages unless the user explicitly asks for that structure. Preserve useful "
        "page/frame/timestamp references, do not invent missing details, and answer in the "
        "user's language.\n\n"
        f"Original request: {user_request}\n\n{_format_final_evidence(summaries)}"
    )
    for message in reversed(text_only):
        if message.get("role") == "user":
            original = str(message.get("content", "")).strip()
            message["content"] = f"{original}\n\n{evidence}".strip()
            return text_only

    return [*text_only, {"role": "user", "content": evidence}]


def _system_text_messages(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": text}
        for message in conversation
        if message.get("role") == "system"
        if (text := _content_text(message.get("content")))
    ]


def _latest_user_text(conversation: list[dict[str, Any]]) -> str:
    for message in reversed(conversation):
        if message.get("role") == "user":
            text = _content_text(message.get("content"))
            if text:
                return text
    return "Analyze the attached media and provide a concise answer."


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")).strip()
        for part in content
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
    ).strip()


def _format_final_evidence(summaries: list[MediaSummary]) -> str:
    if len(summaries) == 1:
        return summaries[0].text
    return _format_summaries(summaries)


def _summary_sampling(
    sampling_parameters: dict[str, Any],
    settings: VllmMediaSettings,
) -> dict[str, Any]:
    max_tokens = int(sampling_parameters.get("max_tokens") or 256)
    result = {
        **sampling_parameters,
        "max_tokens": min(max(max_tokens, 64), settings.summary_max_tokens),
        "temperature": min(float(sampling_parameters.get("temperature", 0.2)), 0.3),
    }
    return result


def _split_text(text: str, chunk_chars: int) -> list[str]:
    chunk_chars = max(int(chunk_chars), 256)
    text = text.strip()
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + chunk_chars, len(text))
        if end < len(text):
            boundary = max(text.rfind("\n", cursor, end), text.rfind(" ", cursor, end))
            if boundary > cursor + chunk_chars // 2:
                end = boundary
        chunks.append(text[cursor:end].strip())
        cursor = end
    return [chunk for chunk in chunks if chunk]


def _format_summaries(summaries: list[MediaSummary]) -> str:
    return "\n\n".join(f"[{item.label}]\n{item.text}" for item in summaries)


def _summary_token_count(tokenizer: Any, summaries: list[MediaSummary]) -> int:
    text = _format_summaries(summaries)
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return max(len(text) // 4, 1)


def _chunk_label(items: list[VisualItem]) -> str:
    if not items:
        return "empty media chunk"
    if len(items) == 1:
        return items[0].label
    return f"{items[0].label} - {items[-1].label}"


def _is_context_length_error(exc: Exception) -> bool:
    text = str(getattr(exc, "detail", exc)).lower()
    return "maximum model length" in text or (
        "longer than" in text and "model length" in text
    )
