# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import ipaddress
import json
import math
import os
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .settings import logger


_asr_pipelines: dict[tuple[str, str], Any] = {}
_asr_lock = threading.Lock()


DEFAULT_MAX_REMOTE_MEDIA_BYTES = 200 * 1024 * 1024
DEFAULT_PDF_CHUNK_PAGES = 2
DEFAULT_PDF_DPI = 144
DEFAULT_MAX_PIXELS = 262144
DEFAULT_VIDEO_FPS = 1.0
DEFAULT_VIDEO_MAX_FRAMES = 32
DEFAULT_VIDEO_CHUNK_FRAMES = 4
DEFAULT_TEXT_CHUNK_CHARS = 6000
DEFAULT_SUMMARY_MAX_TOKENS = 256
DEFAULT_REDUCE_GROUP_SIZE = 5
DEFAULT_FINAL_SUMMARY_LIMIT = 6
DEFAULT_PDF_RAG_CHUNK_CHARS = 1800
DEFAULT_PDF_RAG_CHUNK_OVERLAP = 200
DEFAULT_PDF_RAG_TOP_K = 8
DEFAULT_PDF_RAG_MIN_TEXT_CHARS = 200
DEFAULT_PDF_RAG_EMBEDDING_CONCURRENCY = 8
DEFAULT_PDF_EMBEDDING_CACHE_SIZE = 4096
DEFAULT_MEDIA_MAP_CONCURRENCY = 2
DEFAULT_MEDIA_HISTORY_MODE = "latest"
DEFAULT_MEDIA_HISTORY_MAX_TOKENS = 512


class VllmMediaProcessingError(Exception):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class VllmMediaSettings:
    max_model_len: int
    image_limit: int
    pdf_chunk_pages: int
    pdf_dpi: int
    pdf_max_pixels: int
    video_fps: float
    video_max_frames: int
    video_chunk_frames: int
    video_max_pixels: int
    image_max_pixels: int
    text_chunk_chars: int
    summary_max_tokens: int
    reduce_group_size: int
    final_summary_limit: int
    pdf_embedding_model: str
    pdf_embedding_dimensions: int | None
    pdf_rag_chunk_chars: int
    pdf_rag_chunk_overlap: int
    pdf_rag_top_k: int
    pdf_rag_min_text_chars: int
    pdf_embedding_query_prefix: str
    pdf_text_mode: str
    pdf_rag_embedding_concurrency: int
    pdf_embedding_cache_size: int
    media_map_concurrency: int
    media_history_mode: str
    reset_history_on_new_media: bool
    focus_current_media: bool
    media_history_max_tokens: int
    max_remote_media_bytes: int
    remote_media_timeout: float
    allow_private_remote_urls: bool
    audio_asr_model: str
    audio_asr_device: str
    audio_chunk_length_seconds: float
    audio_stride_length_seconds: float
    temp_dir: Path | None


@dataclass(frozen=True)
class VisualItem:
    label: str
    data: str
    mime_type: str = "image/jpeg"


def load_vllm_media_settings(model_path: Path) -> VllmMediaSettings:
    model_json = _read_json(model_path / "model.json")
    gateway_json = _read_json(model_path / "gateway.json")
    configured = gateway_json.get("vllm_multimodal", gateway_json.get("multimodal", {}))
    if not isinstance(configured, dict):
        configured = {}
    pdf_rag = configured.get("pdf_rag", {})
    if not isinstance(pdf_rag, dict):
        pdf_rag = {}

    image_limit = 1
    limit_mm = model_json.get("limit_mm_per_prompt")
    if isinstance(limit_mm, dict):
        image_limit = _positive_int(limit_mm.get("image"), image_limit)

    pdf_chunk_pages = min(
        _configured_int(
            configured,
            "pdf_chunk_pages",
            "VLLM_MEDIA_PDF_CHUNK_PAGES",
            DEFAULT_PDF_CHUNK_PAGES,
        ),
        image_limit,
    )
    video_chunk_frames = min(
        _configured_int(
            configured,
            "video_chunk_frames",
            "VLLM_MEDIA_VIDEO_CHUNK_FRAMES",
            DEFAULT_VIDEO_CHUNK_FRAMES,
        ),
        image_limit,
    )
    temp_dir_value = os.environ.get("TRITON_MEDIA_DIR", "").strip()

    return VllmMediaSettings(
        max_model_len=_positive_int(model_json.get("max_model_len"), 8192),
        image_limit=image_limit,
        pdf_chunk_pages=max(pdf_chunk_pages, 1),
        pdf_dpi=_configured_int(configured, "pdf_dpi", "VLLM_MEDIA_PDF_DPI", DEFAULT_PDF_DPI),
        pdf_max_pixels=_configured_int(
            configured,
            "pdf_max_pixels",
            "VLLM_MEDIA_PDF_MAX_PIXELS",
            DEFAULT_MAX_PIXELS,
        ),
        video_fps=_configured_float(
            configured,
            "video_fps",
            "VLLM_MEDIA_VIDEO_FPS",
            DEFAULT_VIDEO_FPS,
        ),
        video_max_frames=_configured_int(
            configured,
            "video_max_frames",
            "VLLM_MEDIA_VIDEO_MAX_FRAMES",
            DEFAULT_VIDEO_MAX_FRAMES,
        ),
        video_chunk_frames=max(video_chunk_frames, 1),
        video_max_pixels=_configured_int(
            configured,
            "video_max_pixels",
            "VLLM_MEDIA_VIDEO_MAX_PIXELS",
            DEFAULT_MAX_PIXELS,
        ),
        image_max_pixels=_configured_int(
            configured,
            "image_max_pixels",
            "VLLM_MEDIA_IMAGE_MAX_PIXELS",
            DEFAULT_MAX_PIXELS,
        ),
        text_chunk_chars=_configured_int(
            configured,
            "text_chunk_chars",
            "VLLM_MEDIA_TEXT_CHUNK_CHARS",
            DEFAULT_TEXT_CHUNK_CHARS,
        ),
        summary_max_tokens=max(
            _configured_int(
                configured,
                "summary_max_tokens",
                "VLLM_MEDIA_SUMMARY_MAX_TOKENS",
                DEFAULT_SUMMARY_MAX_TOKENS,
            ),
            64,
        ),
        reduce_group_size=_configured_int(
            configured,
            "reduce_group_size",
            "VLLM_MEDIA_REDUCE_GROUP_SIZE",
            DEFAULT_REDUCE_GROUP_SIZE,
        ),
        final_summary_limit=_configured_int(
            configured,
            "final_summary_limit",
            "VLLM_MEDIA_FINAL_SUMMARY_LIMIT",
            DEFAULT_FINAL_SUMMARY_LIMIT,
        ),
        pdf_embedding_model=str(
            os.environ.get(
                "VLLM_MEDIA_PDF_EMBEDDING_MODEL",
                pdf_rag.get("embedding_model", ""),
            )
        ).strip(),
        pdf_embedding_dimensions=_optional_positive_int(
            os.environ.get(
                "VLLM_MEDIA_PDF_EMBEDDING_DIMENSIONS",
                pdf_rag.get("dimensions"),
            )
        ),
        pdf_rag_chunk_chars=_configured_nested_int(
            pdf_rag,
            "chunk_chars",
            "VLLM_MEDIA_PDF_RAG_CHUNK_CHARS",
            DEFAULT_PDF_RAG_CHUNK_CHARS,
        ),
        pdf_rag_chunk_overlap=_configured_nested_non_negative_int(
            pdf_rag,
            "chunk_overlap",
            "VLLM_MEDIA_PDF_RAG_CHUNK_OVERLAP",
            DEFAULT_PDF_RAG_CHUNK_OVERLAP,
        ),
        pdf_rag_top_k=_configured_nested_int(
            pdf_rag,
            "top_k",
            "VLLM_MEDIA_PDF_RAG_TOP_K",
            DEFAULT_PDF_RAG_TOP_K,
        ),
        pdf_rag_min_text_chars=_configured_nested_int(
            pdf_rag,
            "min_text_chars",
            "VLLM_MEDIA_PDF_RAG_MIN_TEXT_CHARS",
            DEFAULT_PDF_RAG_MIN_TEXT_CHARS,
        ),
        pdf_embedding_query_prefix=str(
            os.environ.get(
                "VLLM_MEDIA_PDF_EMBEDDING_QUERY_PREFIX",
                pdf_rag.get("query_prefix", ""),
            )
        ).strip(),
        pdf_text_mode=_configured_choice(
            configured,
            "pdf_text_mode",
            "VLLM_MEDIA_PDF_TEXT_MODE",
            "auto",
            {"auto", "text", "visual"},
        ),
        pdf_rag_embedding_concurrency=_configured_nested_int(
            pdf_rag,
            "embedding_concurrency",
            "VLLM_MEDIA_PDF_RAG_EMBEDDING_CONCURRENCY",
            DEFAULT_PDF_RAG_EMBEDDING_CONCURRENCY,
        ),
        pdf_embedding_cache_size=_configured_nested_int(
            pdf_rag,
            "embedding_cache_size",
            "VLLM_MEDIA_PDF_EMBEDDING_CACHE_SIZE",
            DEFAULT_PDF_EMBEDDING_CACHE_SIZE,
        ),
        media_map_concurrency=_configured_int(
            configured,
            "map_concurrency",
            "VLLM_MEDIA_MAP_CONCURRENCY",
            DEFAULT_MEDIA_MAP_CONCURRENCY,
        ),
        media_history_mode=_configured_choice(
            configured,
            "media_history_mode",
            "VLLM_MEDIA_HISTORY_MODE",
            DEFAULT_MEDIA_HISTORY_MODE,
            {"latest", "all"},
        ),
        reset_history_on_new_media=_configured_bool(
            configured,
            "reset_history_on_new_media",
            "VLLM_MEDIA_RESET_HISTORY_ON_NEW_MEDIA",
            False,
        ),
        focus_current_media=_configured_bool(
            configured,
            "focus_current_media",
            "VLLM_MEDIA_FOCUS_CURRENT",
            True,
        ),
        media_history_max_tokens=_configured_int(
            configured,
            "media_history_max_tokens",
            "VLLM_MEDIA_HISTORY_MAX_TOKENS",
            DEFAULT_MEDIA_HISTORY_MAX_TOKENS,
        ),
        max_remote_media_bytes=_configured_int(
            configured,
            "max_remote_media_bytes",
            "VLLM_MEDIA_MAX_REMOTE_BYTES",
            DEFAULT_MAX_REMOTE_MEDIA_BYTES,
        ),
        remote_media_timeout=_configured_float(
            configured,
            "remote_media_timeout",
            "VLLM_MEDIA_REMOTE_TIMEOUT",
            60.0,
        ),
        allow_private_remote_urls=_configured_bool(
            configured,
            "allow_private_remote_urls",
            "VLLM_MEDIA_ALLOW_PRIVATE_URLS",
            False,
        ),
        audio_asr_model=str(
            os.environ.get(
                "VLLM_MEDIA_AUDIO_ASR_MODEL",
                configured.get("audio_asr_model", ""),
            )
        ).strip(),
        audio_asr_device=str(
            os.environ.get(
                "VLLM_MEDIA_AUDIO_ASR_DEVICE",
                configured.get("audio_asr_device", "-1"),
            )
        ).strip(),
        audio_chunk_length_seconds=_configured_float(
            configured,
            "audio_chunk_length_seconds",
            "VLLM_MEDIA_AUDIO_CHUNK_SECONDS",
            30.0,
        ),
        audio_stride_length_seconds=_configured_float(
            configured,
            "audio_stride_length_seconds",
            "VLLM_MEDIA_AUDIO_STRIDE_SECONDS",
            5.0,
        ),
        temp_dir=Path(temp_dir_value) if temp_dir_value else None,
    )


def _render_pdf_pages(
    payload: Any,
    work_dir: Path,
    document_index: int,
    settings: VllmMediaSettings,
) -> list[VisualItem]:
    try:
        import pymupdf
    except ImportError as exc:
        raise VllmMediaProcessingError("PDF input requires package 'pymupdf'") from exc

    pdf_path = _materialize_payload(payload, "pdf", work_dir, document_index, settings)
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise VllmMediaProcessingError(f"Cannot open PDF {document_index}: {exc}") from exc

    with document:
        if document.page_count <= 0:
            raise VllmMediaProcessingError(f"PDF {document_index} contains no pages")
        pages = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            scale = max(settings.pdf_dpi, 1) / 72.0
            target_pixels = page.rect.width * scale * page.rect.height * scale
            if settings.pdf_max_pixels and target_pixels > settings.pdf_max_pixels:
                scale *= math.sqrt(settings.pdf_max_pixels / target_pixels)
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                alpha=False,
            )
            image = _pil_image_from_bytes(pixmap.tobytes("png"))
            pages.append(
                VisualItem(
                    label=f"PDF {document_index}, page {page_index + 1}/{document.page_count}",
                    data=_encode_pil_image(image, settings.pdf_max_pixels, "PNG"),
                    mime_type="image/png",
                )
            )
    return pages


def _extract_pdf_text_pages(
    payload: Any,
    work_dir: Path,
    document_index: int,
    settings: VllmMediaSettings,
) -> list[tuple[str, str]]:
    try:
        import pymupdf
    except ImportError as exc:
        raise VllmMediaProcessingError("PDF input requires package 'pymupdf'") from exc

    pdf_path = _materialize_payload(payload, "pdf", work_dir, document_index, settings)
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise VllmMediaProcessingError(f"Cannot open PDF {document_index}: {exc}") from exc

    pages: list[tuple[str, str]] = []
    with document:
        for page_index in range(document.page_count):
            text = document.load_page(page_index).get_text("text").strip()
            if text:
                pages.append(
                    (
                        f"PDF {document_index}, page {page_index + 1}/{document.page_count}",
                        text,
                    )
                )
    return pages


def _extract_video_frames(
    payload: Any,
    work_dir: Path,
    video_index: int,
    settings: VllmMediaSettings,
) -> list[VisualItem]:
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise VllmMediaProcessingError("Video input requires package 'decord'") from exc

    video_path = _materialize_payload(payload, "video", work_dir, video_index, settings)
    try:
        reader = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps() or 1.0)
    except Exception as exc:
        raise VllmMediaProcessingError(f"Cannot decode video {video_index}: {exc}") from exc

    if total_frames <= 0:
        raise VllmMediaProcessingError(f"Video {video_index} contains no frames")

    indices = select_video_frame_indices(
        total_frames,
        source_fps,
        settings.video_fps,
        settings.video_max_frames,
    )
    items = []
    for sampled_index, frame_index in enumerate(indices, start=1):
        try:
            frame = reader[frame_index].asnumpy()
        except Exception as exc:
            raise VllmMediaProcessingError(
                f"Cannot extract frame {frame_index} from video {video_index}: {exc}"
            ) from exc
        from PIL import Image

        image = Image.fromarray(frame).convert("RGB")
        timestamp = frame_index / source_fps if source_fps > 0 else 0.0
        items.append(
            VisualItem(
                label=(
                    f"Video {video_index}, sampled frame {sampled_index}/{len(indices)}, "
                    f"timestamp {timestamp:.2f}s"
                ),
                data=_encode_pil_image(image, settings.video_max_pixels, "JPEG"),
            )
        )
    return items


def select_video_frame_indices(
    total_frames: int,
    source_fps: float,
    sample_fps: float,
    max_frames: int,
) -> list[int]:
    total_frames = max(int(total_frames), 1)
    source_fps = max(float(source_fps), 0.001)
    sample_fps = max(float(sample_fps), 0.001)
    max_frames = max(int(max_frames), 1)
    duration = total_frames / source_fps
    target_count = min(total_frames, max_frames, max(int(math.ceil(duration * sample_fps)), 1))
    if target_count == 1:
        return [0]
    return sorted(
        {
            min(round(index * (total_frames - 1) / (target_count - 1)), total_frames - 1)
            for index in range(target_count)
        }
    )


def _image_payload_to_item(
    payload: Any,
    work_dir: Path,
    image_index: int,
    settings: VllmMediaSettings,
) -> VisualItem:
    image_path = _materialize_payload(payload, "image", work_dir, image_index, settings)
    image = _pil_image_from_bytes(image_path.read_bytes())
    return VisualItem(
        label=f"Image {image_index}",
        data=_encode_pil_image(image, settings.image_max_pixels, "JPEG"),
    )


def _transcribe_audio(audio_path: Path, settings: VllmMediaSettings) -> str:
    if not settings.audio_asr_model:
        raise VllmMediaProcessingError(
            "Audio input for a vLLM backend requires VLLM_MEDIA_AUDIO_ASR_MODEL "
            "to point to a local ASR model"
        )

    pipeline = _get_asr_pipeline(settings)
    kwargs = {
        "chunk_length_s": settings.audio_chunk_length_seconds,
        "stride_length_s": settings.audio_stride_length_seconds,
    }
    try:
        result = pipeline(str(audio_path), **kwargs)
    except TypeError:
        result = pipeline(str(audio_path))
    except Exception as exc:
        raise VllmMediaProcessingError(f"ASR processing failed: {exc}") from exc

    text = result.get("text") if isinstance(result, dict) else result
    if not isinstance(text, str) or not text.strip():
        raise VllmMediaProcessingError("ASR model returned an empty transcript")
    return text.strip()


def _get_asr_pipeline(settings: VllmMediaSettings):
    key = (settings.audio_asr_model, settings.audio_asr_device)
    with _asr_lock:
        cached = _asr_pipelines.get(key)
        if cached is not None:
            return cached

        from transformers import pipeline

        try:
            device: int | str = int(settings.audio_asr_device)
        except ValueError:
            device = settings.audio_asr_device
        logger.info(
            "Loading gateway ASR model '%s' on device '%s'",
            settings.audio_asr_model,
            device,
        )
        asr = pipeline(
            "automatic-speech-recognition",
            model=settings.audio_asr_model,
            device=device,
        )
        _asr_pipelines[key] = asr
        return asr


def _materialize_payload(
    payload: Any,
    media_type: str,
    work_dir: Path,
    index: int,
    settings: VllmMediaSettings,
) -> Path:
    value = str(payload.data).strip()
    suffix = _media_suffix(media_type, getattr(payload, "format", None), value)
    path = work_dir / f"{media_type}_{index}{suffix}"

    if value.startswith(("http://", "https://")):
        _download_remote_media(value, path, settings)
        return path

    try:
        raw = base64.b64decode("".join(value.split()), validate=True)
    except Exception as exc:
        raise VllmMediaProcessingError(f"Invalid base64 {media_type} payload") from exc
    if len(raw) > settings.max_remote_media_bytes:
        raise VllmMediaProcessingError(
            f"{media_type} payload is too large: {len(raw)} bytes exceeds "
            f"{settings.max_remote_media_bytes} bytes",
            status_code=413,
        )
    path.write_bytes(raw)
    return path


def _download_remote_media(
    url: str,
    path: Path,
    settings: VllmMediaSettings,
) -> None:
    _validate_remote_url(url, settings.allow_private_remote_urls)
    request = Request(url, headers={"User-Agent": "triton-openai-gateway/0.1"})
    downloaded = 0
    try:
        opener = build_opener(
            _ValidatedRedirectHandler(settings.allow_private_remote_urls)
        )
        with (
            opener.open(request, timeout=settings.remote_media_timeout) as response,
            path.open("wb") as output,
        ):
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > settings.max_remote_media_bytes:
                raise VllmMediaProcessingError(
                    f"Remote media exceeds {settings.max_remote_media_bytes} bytes",
                    status_code=413,
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > settings.max_remote_media_bytes:
                    raise VllmMediaProcessingError(
                        f"Remote media exceeds {settings.max_remote_media_bytes} bytes",
                        status_code=413,
                    )
                output.write(chunk)
    except VllmMediaProcessingError:
        raise
    except Exception as exc:
        raise VllmMediaProcessingError(f"Cannot download remote media: {exc}") from exc


class _ValidatedRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allow_private: bool):
        self.allow_private = allow_private
        super().__init__()

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_remote_url(newurl, self.allow_private)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _validate_remote_url(url: str, allow_private: bool) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise VllmMediaProcessingError("Remote media URL must use HTTP or HTTPS")
    if parsed.username or parsed.password:
        raise VllmMediaProcessingError("Remote media URL must not contain credentials")
    if allow_private:
        return

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise VllmMediaProcessingError(
            f"Cannot resolve remote media host: {parsed.hostname}"
        ) from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise VllmMediaProcessingError(
                f"Remote media host resolves to a non-public address: {ip}"
            )


def _pil_image_from_bytes(value: bytes):
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(value))
        image.load()
        return image.convert("RGB")
    except Exception as exc:
        raise VllmMediaProcessingError(f"Cannot decode image: {exc}") from exc


def _encode_pil_image(image: Any, max_pixels: int, image_format: str) -> str:
    max_pixels = max(int(max_pixels), 1024)
    width, height = image.size
    pixels = width * height
    if pixels > max_pixels:
        scale = math.sqrt(max_pixels / pixels)
        target = (max(round(width * scale), 1), max(round(height * scale), 1))
        from PIL import Image

        image = image.resize(target, Image.Resampling.LANCZOS)

    output = io.BytesIO()
    if image_format == "JPEG":
        image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
    else:
        image.save(output, format=image_format, optimize=True)
    return base64.b64encode(output.getvalue()).decode("ascii")


def _media_suffix(media_type: str, media_format: Any, value: str) -> str:
    if isinstance(media_format, str) and media_format:
        suffix = media_format.lower().lstrip(".")
        return ".jpg" if suffix == "jpeg" else f".{suffix}"
    if value.startswith(("http://", "https://")):
        suffix = Path(urlparse(value).path).suffix.lower()
        if suffix:
            return suffix
    return {
        "image": ".jpg",
        "pdf": ".pdf",
        "video": ".mp4",
        "audio": ".wav",
    }.get(media_type, ".bin")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read gateway media config %s: %s", path, exc)
        return {}
    return value if isinstance(value, dict) else {}


def _configured_int(config: dict[str, Any], key: str, env: str, default: int) -> int:
    return _positive_int(os.environ.get(env, config.get(key, default)), default)


def _configured_float(config: dict[str, Any], key: str, env: str, default: float) -> float:
    try:
        value = float(os.environ.get(env, config.get(key, default)))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _configured_bool(config: dict[str, Any], key: str, env: str, default: bool) -> bool:
    value = os.environ.get(env, config.get(key, default))
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configured_choice(
    config: dict[str, Any],
    key: str,
    env: str,
    default: str,
    choices: set[str],
) -> str:
    value = str(os.environ.get(env, config.get(key, default))).strip().lower()
    return value if value in choices else default


def _configured_nested_int(config: dict[str, Any], key: str, env: str, default: int) -> int:
    return _positive_int(os.environ.get(env, config.get(key, default)), default)


def _configured_nested_non_negative_int(
    config: dict[str, Any], key: str, env: str, default: int
) -> int:
    try:
        value = int(os.environ.get(env, config.get(key, default)))
    except (TypeError, ValueError):
        return default
    return max(value, 0)


def _optional_positive_int(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default
