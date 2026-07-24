# Copyright 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Modifications Copyright 2026 Raytorin and Triton OpenAI Gateway contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import av
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import pymupdf


DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[^;,]+);base64,(?P<payload>.*)$",
    re.IGNORECASE | re.DOTALL,
)
DEFAULT_MAX_MEDIA_BYTES = 256 * 1024 * 1024
DEFAULT_MAX_REQUEST_BYTES = 768 * 1024 * 1024
DEFAULT_MAX_MEDIA_ITEMS = 64
DEFAULT_MAX_SOURCE_PIXELS = 100_000_000
DEFAULT_MAX_OUTPUT_PIXELS = 4_194_304
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_MAX_FRAMES = 128
DEFAULT_VIDEO_MAX_TOTAL_FRAMES = 256
DEFAULT_VIDEO_MAX_PIXELS = 512 * 512
DEFAULT_PDF_DPI = 144
DEFAULT_PDF_MAX_PIXELS = 512 * 512
DEFAULT_PDF_MAX_PAGES = 64
DEFAULT_AUDIO_MAX_SECONDS = 600


@dataclass(frozen=True)
class EncodedMedia:
    data: bytes
    mime_type: str | None = None
    format: str | None = None


def parse_media_parameters(value: Any | None) -> dict[str, Any]:
    if value is None:
        return {}

    text = _as_text(value)
    if not text:
        return {}

    parameters = json.loads(text)
    if not isinstance(parameters, dict):
        raise ValueError("media_parameters must be a JSON object")
    return parameters


def decode_image_values(values: Iterable[Any], parameters: dict[str, Any]) -> list[Image.Image]:
    max_pixels = _non_negative_int(parameters.get("image_max_pixels"), 0)
    max_pixels = _bounded_output_pixels(max_pixels)
    images = []
    for value in values:
        media = decode_media_value(value)
        with Image.open(BytesIO(media.data)) as source:
            _validate_source_pixels(source.width, source.height, "image")
            image = source.convert("RGB")
        images.append(_resize_image(image, max_pixels))
    return images


def decode_video_values(
    values: Iterable[Any],
    parameters: dict[str, Any],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    sample_fps = _positive_float(parameters.get("video_fps"), DEFAULT_VIDEO_FPS)
    max_frames = _positive_int(
        parameters.get("video_max_frames"), DEFAULT_VIDEO_MAX_FRAMES
    )
    max_frames = min(max_frames, _env_positive_int("VLLM_MULTIMODAL_MAX_VIDEO_FRAMES", 256))
    max_pixels = _non_negative_int(
        parameters.get("video_max_pixels"), DEFAULT_VIDEO_MAX_PIXELS
    )
    max_pixels = _bounded_output_pixels(max_pixels)

    max_total_frames = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_TOTAL_VIDEO_FRAMES", DEFAULT_VIDEO_MAX_TOTAL_FRAMES
    )
    videos = []
    total_frames = 0
    for value in values:
        video = _decode_video(
            decode_media_value(value), sample_fps, max_frames, max_pixels
        )
        total_frames += len(video[0])
        if total_frames > max_total_frames:
            raise ValueError(
                f"request produces {total_frames} video frames; maximum is "
                f"{max_total_frames}"
            )
        videos.append(video)
    return videos


def decode_audio_values(
    values: Iterable[Any],
    parameters: dict[str, Any],
) -> list[tuple[np.ndarray, int]]:
    sample_rate = _positive_int(parameters.get("audio_sample_rate"), 16_000)
    sample_rate = min(sample_rate, 96_000)
    hard_max_seconds = float(
        _env_positive_int(
            "VLLM_MULTIMODAL_MAX_AUDIO_SECONDS", DEFAULT_AUDIO_MAX_SECONDS
        )
    )
    max_seconds = min(
        _positive_float(parameters.get("audio_max_seconds"), hard_max_seconds),
        hard_max_seconds,
    )
    audios = []
    remaining_seconds = max_seconds
    for value in values:
        if remaining_seconds <= 0:
            raise ValueError(
                "total audio duration exceeds the configured maximum of "
                f"{max_seconds:g} seconds"
            )
        audio = _decode_audio(
            decode_media_value(value), sample_rate, remaining_seconds
        )
        remaining_seconds -= len(audio[0]) / audio[1]
        audios.append(audio)
    return audios


def decode_pdf_values(
    values: Iterable[Any],
    parameters: dict[str, Any],
) -> tuple[list[Image.Image], list[int]]:
    dpi = _positive_int(parameters.get("pdf_dpi"), DEFAULT_PDF_DPI)
    max_pixels = _non_negative_int(
        parameters.get("pdf_max_pixels"), DEFAULT_PDF_MAX_PIXELS
    )
    max_pixels = _bounded_output_pixels(max_pixels)
    configured_max_pages = _non_negative_int(
        parameters.get("pdf_max_pages"), DEFAULT_PDF_MAX_PAGES
    )
    hard_max_pages = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_PDF_PAGES", DEFAULT_PDF_MAX_PAGES
    )
    max_pages = min(configured_max_pages or hard_max_pages, hard_max_pages)

    pages: list[Image.Image] = []
    page_counts: list[int] = []
    for value in values:
        remaining_pages = max_pages - len(pages)
        if remaining_pages <= 0:
            raise ValueError(
                f"total PDF page count exceeds the configured maximum of {max_pages}"
            )
        document_pages = _render_pdf(
            decode_media_value(value),
            dpi=dpi,
            max_pixels=max_pixels,
            max_pages=remaining_pages,
        )
        pages.extend(document_pages)
        page_counts.append(len(document_pages))
    return pages, page_counts


def build_multimodal_prompt(
    prompt: str,
    *,
    image_values: Iterable[Any] = (),
    video_values: Iterable[Any] = (),
    audio_values: Iterable[Any] = (),
    pdf_values: Iterable[Any] = (),
    parameters: dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    parameters = parameters or {}
    image_values = list(image_values)
    video_values = list(video_values)
    audio_values = list(audio_values)
    pdf_values = list(pdf_values)

    _validate_request_limits(
        [*image_values, *video_values, *audio_values, *pdf_values]
    )

    images = decode_image_values(image_values, parameters)
    videos = decode_video_values(video_values, parameters)
    audios = decode_audio_values(audio_values, parameters)
    pdf_pages, pdf_page_counts = decode_pdf_values(pdf_values, parameters)

    if pdf_page_counts:
        prompt = expand_pdf_placeholders(
            prompt,
            pdf_page_counts,
            media_order=parameters.get("media_order"),
            regular_image_count=len(images),
        )
        images = _merge_image_modalities(
            images,
            pdf_pages,
            pdf_page_counts,
            parameters.get("media_order"),
        )

    multi_modal_data: dict[str, Any] = {}
    if images:
        multi_modal_data["image"] = images
    if videos:
        multi_modal_data["video"] = videos
    if audios:
        multi_modal_data["audio"] = audios

    if not multi_modal_data:
        return prompt

    prompt_payload: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": multi_modal_data,
    }
    mm_processor_kwargs = parameters.get("mm_processor_kwargs")
    if isinstance(mm_processor_kwargs, dict) and mm_processor_kwargs:
        prompt_payload["mm_processor_kwargs"] = mm_processor_kwargs
    return prompt_payload


def decode_media_value(value: Any) -> EncodedMedia:
    raw = _as_bytes(value)
    max_bytes = _positive_int(
        os.environ.get("VLLM_MULTIMODAL_MAX_MEDIA_BYTES"),
        DEFAULT_MAX_MEDIA_BYTES,
    )

    try:
        text = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        text = ""

    mime_type = None
    media_format = None
    encoded = text
    requires_base64 = False
    if text.startswith("{"):
        envelope = json.loads(text)
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), str):
            raise ValueError("media JSON envelope must contain a string 'data' field")
        encoded = envelope["data"].strip()
        requires_base64 = True
        mime_type = _optional_text(envelope.get("mime_type"))
        media_format = _optional_text(envelope.get("format"))

    match = DATA_URL_RE.match(encoded)
    if match:
        mime_type = mime_type or match.group("mime")
        encoded = match.group("payload")
        requires_base64 = True

    if encoded.startswith(("http://", "https://", "file://")):
        raise ValueError(
            "vllm_multimodal backend accepts embedded media bytes/base64 only; "
            "remote URLs must be materialized by the gateway"
        )

    if encoded:
        if requires_base64 and len(encoded) > ((max_bytes + 2) // 3) * 4 + 16:
            raise ValueError(
                "base64 media payload exceeds the configured decoded item limit"
            )
        try:
            decoded = base64.b64decode("".join(encoded.split()), validate=True)
        except (ValueError, binascii.Error):
            if requires_base64:
                raise ValueError("media payload is not valid base64")
            decoded = raw
    else:
        decoded = raw

    if len(decoded) > max_bytes:
        raise ValueError(
            f"media item is {len(decoded)} bytes; maximum is {max_bytes} bytes"
        )
    return EncodedMedia(decoded, mime_type=mime_type, format=media_format)


def select_video_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0]

    source_fps = source_fps if source_fps > 0 else sample_fps
    desired = max(int(math.ceil(total_frames * sample_fps / source_fps)), 2)
    desired = min(desired, max_frames, total_frames)
    return sorted(
        set(int(round(index)) for index in np.linspace(0, total_frames - 1, desired))
    )


def expand_pdf_placeholders(
    prompt: str,
    page_counts: list[int],
    *,
    media_order: Any = None,
    regular_image_count: int = 0,
) -> str:
    if not page_counts:
        return prompt

    slot_types = _image_slot_types(media_order)
    if not slot_types:
        slot_types = ["image"] * regular_image_count + ["pdf"] * len(page_counts)

    if slot_types.count("pdf") != len(page_counts):
        raise ValueError(
            "media_parameters.media_order does not match the number of PDF inputs"
        )

    marker = _detect_image_marker(prompt)
    matches = list(re.finditer(re.escape(marker), prompt))
    if len(matches) < len(slot_types):
        raise ValueError(
            "rendered prompt does not contain enough image placeholders for PDF pages"
        )

    replacements: dict[int, str] = {}
    pdf_index = 0
    for slot_index, slot_type in enumerate(slot_types):
        if slot_type != "pdf":
            continue
        page_count = page_counts[pdf_index]
        if page_count <= 0:
            raise ValueError("PDF document contains no pages")
        replacements[matches[slot_index].start()] = marker * page_count
        pdf_index += 1

    parts = []
    cursor = 0
    for match in matches:
        replacement = replacements.get(match.start())
        if replacement is None:
            continue
        parts.append(prompt[cursor : match.start()])
        parts.append(replacement)
        cursor = match.end()
    parts.append(prompt[cursor:])
    return "".join(parts)


def _decode_video(
    media: EncodedMedia,
    sample_fps: float,
    max_frames: int,
    max_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = _safe_suffix(media.format, media.mime_type, ".mp4")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="triton-vllm-video-",
            suffix=suffix,
            dir=os.environ.get("TRITON_MULTIMODAL_TMPDIR"),
            delete=False,
        ) as output:
            output.write(media.data)
            temp_path = Path(output.name)

        reader = VideoReader(str(temp_path), ctx=cpu(0))
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps() or sample_fps)
        indices = select_video_frame_indices(
            total_frames,
            source_fps,
            sample_fps,
            max_frames,
        )
        if not indices:
            raise ValueError("video contains no decodable frames")

        frames = []
        for index in indices:
            frame = reader[index].asnumpy()
            _validate_source_pixels(frame.shape[1], frame.shape[0], "video frame")
            image = _resize_image(Image.fromarray(frame).convert("RGB"), max_pixels)
            frames.append(np.asarray(image, dtype=np.uint8))
        frames = np.stack(frames)

        metadata = {
            "fps": source_fps,
            "duration": total_frames / source_fps,
            "total_num_frames": total_frames,
            "frames_indices": indices,
            "video_backend": "decord",
            "do_sample_frames": False,
        }
        return frames, metadata
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _decode_audio(
    media: EncodedMedia,
    sample_rate: int,
    max_seconds: float,
) -> tuple[np.ndarray, int]:
    chunks: list[np.ndarray] = []
    with av.open(BytesIO(media.data)) as container:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("media item does not contain an audio stream")

        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        sample_count = 0
        max_samples = int(sample_rate * max_seconds)
        for frame in container.decode(stream):
            for converted in _resampled_frames(resampler.resample(frame)):
                samples = converted.to_ndarray().astype(np.float32).reshape(-1)
                sample_count += samples.size
                if sample_count > max_samples:
                    raise ValueError(
                        "audio duration exceeds the configured maximum of "
                        f"{max_seconds:g} seconds"
                    )
                chunks.append(samples)
        for converted in _resampled_frames(resampler.resample(None)):
            samples = converted.to_ndarray().astype(np.float32).reshape(-1)
            sample_count += samples.size
            if sample_count > max_samples:
                raise ValueError(
                    "audio duration exceeds the configured maximum of "
                    f"{max_seconds:g} seconds"
                )
            chunks.append(samples)

    if not chunks:
        raise ValueError("audio contains no decodable samples")
    samples = np.concatenate(chunks)
    return samples, sample_rate


def _render_pdf(
    media: EncodedMedia,
    *,
    dpi: int,
    max_pixels: int,
    max_pages: int,
) -> list[Image.Image]:
    document = pymupdf.open(stream=media.data, filetype="pdf")
    try:
        if document.page_count == 0:
            raise ValueError("PDF document contains no pages")
        if document.page_count > max_pages:
            raise ValueError(
                f"PDF has {document.page_count} pages; configured maximum is {max_pages}"
            )

        pages = []
        for page in document:
            scale = dpi / 72.0
            target_pixels = page.rect.width * scale * page.rect.height * scale
            if max_pixels and target_pixels > max_pixels:
                scale *= math.sqrt(max_pixels / target_pixels)
            matrix = pymupdf.Matrix(scale, scale)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            pages.append(image)
        return pages
    finally:
        document.close()


def _resize_image(image: Image.Image, max_pixels: int) -> Image.Image:
    if not max_pixels or image.width * image.height <= max_pixels:
        return image
    scale = math.sqrt(max_pixels / (image.width * image.height))
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _validate_request_limits(values: list[Any]) -> None:
    max_items = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_MEDIA_ITEMS", DEFAULT_MAX_MEDIA_ITEMS
    )
    if len(values) > max_items:
        raise ValueError(
            f"request contains {len(values)} media items; maximum is {max_items}"
        )

    max_request_bytes = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_REQUEST_BYTES", DEFAULT_MAX_REQUEST_BYTES
    )
    wire_bytes = sum(len(_as_bytes(value)) for value in values)
    if wire_bytes > max_request_bytes:
        raise ValueError(
            f"encoded media request is {wire_bytes} bytes; maximum is {max_request_bytes} bytes"
        )


def _validate_source_pixels(width: int, height: int, media_type: str) -> None:
    max_pixels = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_SOURCE_PIXELS", DEFAULT_MAX_SOURCE_PIXELS
    )
    pixels = int(width) * int(height)
    if pixels > max_pixels:
        raise ValueError(
            f"{media_type} has {pixels} source pixels; maximum is {max_pixels}"
        )


def _detect_image_marker(prompt: str) -> str:
    markers = (
        "<|vision_start|><|image_pad|><|vision_end|>",
        "<|image_pad|>",
        "<image>",
        "<|image|>",
    )
    for marker in markers:
        if marker in prompt:
            return marker
    raise ValueError(
        "cannot expand PDF pages: no supported image placeholder found in prompt"
    )


def _image_slot_types(media_order: Any) -> list[str]:
    if not isinstance(media_order, list):
        return []
    return [str(item).lower() for item in media_order if str(item).lower() in {"image", "pdf"}]


def _merge_image_modalities(
    images: list[Image.Image],
    pdf_pages: list[Image.Image],
    pdf_page_counts: list[int],
    media_order: Any,
) -> list[Image.Image]:
    slot_types = _image_slot_types(media_order)
    if not slot_types:
        return [*images, *pdf_pages]

    merged: list[Image.Image] = []
    image_index = 0
    page_index = 0
    pdf_index = 0
    for slot_type in slot_types:
        if slot_type == "image":
            if image_index >= len(images):
                raise ValueError("media_order contains more image entries than image inputs")
            merged.append(images[image_index])
            image_index += 1
            continue

        page_count = pdf_page_counts[pdf_index]
        merged.extend(pdf_pages[page_index : page_index + page_count])
        page_index += page_count
        pdf_index += 1

    if image_index != len(images) or pdf_index != len(pdf_page_counts):
        raise ValueError("media_order does not match image and PDF inputs")
    return merged


def _resampled_frames(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _safe_suffix(media_format: str | None, mime_type: str | None, default: str) -> str:
    value = media_format or (mime_type.split("/", 1)[-1] if mime_type else "")
    value = re.sub(r"[^a-zA-Z0-9]", "", value)[:10]
    return f".{value.lower()}" if value else default


def _as_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.bytes_):
        return value.tobytes()
    if isinstance(value, str):
        return value.encode("utf-8")
    return bytes(value)


def _as_text(value: Any) -> str:
    return _as_bytes(value).decode("utf-8").strip()


def _optional_text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None


def _positive_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"expected a positive integer, got {parsed}")
    return parsed


def _non_negative_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"expected a non-negative integer, got {parsed}")
    return parsed


def _positive_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"expected a positive number, got {parsed}")
    return parsed


def _env_positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return _positive_int(value, default)


def _bounded_output_pixels(configured: int) -> int:
    hard_limit = _env_positive_int(
        "VLLM_MULTIMODAL_MAX_OUTPUT_PIXELS", DEFAULT_MAX_OUTPUT_PIXELS
    )
    return min(configured or hard_limit, hard_limit)
