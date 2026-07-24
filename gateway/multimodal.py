# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import re
from typing import Any
from urllib.parse import urlparse

from fastapi import HTTPException


DATA_URL_RE = re.compile(r"^data:[^;,]+;base64,(?P<payload>.*)$", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class MediaPayload:
    data: str
    mime_type: str | None = None
    format: str | None = None


@dataclass(frozen=True)
class MediaPayloads:
    images: list[MediaPayload]
    videos: list[MediaPayload]
    audios: list[MediaPayload]
    pdfs: list[MediaPayload]
    order: list[str]
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def has_any(self) -> bool:
        return bool(self.images or self.videos or self.audios or self.pdfs)


def normalize_message_content(content: Any) -> str | list[dict[str, Any]]:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail="Unsupported message content format")

    normalized: list[dict[str, Any]] = []
    has_non_text_part = False
    for part in content:
        if not isinstance(part, dict):
            raise HTTPException(status_code=400, detail="Invalid message content part")

        part_type = part.get("type")
        if part_type == "text":
            normalized.append({"type": "text", "text": str(part.get("text", ""))})
            continue

        if part_type == "image_url" or "image_url" in part:
            has_non_text_part = True
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                detail = image_url.get("detail")
            else:
                url = image_url
                detail = part.get("detail")

            if not isinstance(url, str) or not url.strip():
                raise HTTPException(status_code=400, detail="image_url.url is required")

            normalized_image_url: dict[str, Any] = {"url": url.strip()}
            if detail is not None:
                normalized_image_url["detail"] = detail
            normalized.append({"type": "image_url", "image_url": normalized_image_url})
            continue

        if part_type in {"pdf", "pdf_url", "file"} or "pdf" in part or "pdf_url" in part or "file" in part:
            has_non_text_part = True
            file_payload = part.get("file") or part.get("pdf_url") or part.get("pdf")
            if isinstance(file_payload, dict):
                url = (
                    file_payload.get("url")
                    or file_payload.get("file_data")
                    or file_payload.get("data")
                )
            else:
                url = file_payload

            if not isinstance(url, str) or not url.strip():
                raise HTTPException(status_code=400, detail="pdf/file content must contain url or base64 data")

            # Keep OpenAI-compatible shape for tokenizer chat templates. The
            # backend detects application/pdf/.pdf and renders pages to images.
            normalized.append({"type": "image_url", "image_url": {"url": url.strip()}})
            continue

        if part_type == "image" or "image" in part:
            has_non_text_part = True
            image = part.get("image")
            if not isinstance(image, str) or not image.strip():
                raise HTTPException(status_code=400, detail="image content must contain base64 data")
            normalized.append({"type": "image", "image": image.strip()})
            continue

        if part_type == "video_url" or "video_url" in part:
            has_non_text_part = True
            video_url = part.get("video_url")
            if isinstance(video_url, dict):
                url = video_url.get("url")
            else:
                url = video_url

            if not isinstance(url, str) or not url.strip():
                raise HTTPException(status_code=400, detail="video_url.url is required")

            normalized.append({"type": "video", "video": url.strip()})
            continue

        if part_type == "video" or "video" in part:
            has_non_text_part = True
            video = part.get("video")
            if not isinstance(video, str) or not video.strip():
                raise HTTPException(status_code=400, detail="video content must contain base64 data")
            normalized.append({"type": "video", "video": video.strip()})
            continue

        if part_type == "audio_url" or "audio_url" in part:
            has_non_text_part = True
            audio_url = part.get("audio_url")
            if isinstance(audio_url, dict):
                url = audio_url.get("url")
            else:
                url = audio_url

            if not isinstance(url, str) or not url.strip():
                raise HTTPException(status_code=400, detail="audio_url.url is required")

            normalized.append({"type": "audio", "audio": url.strip()})
            continue

        if part_type in {"audio", "input_audio"} or "audio" in part or "input_audio" in part:
            has_non_text_part = True
            audio_payload = part.get("input_audio") or part.get("audio")
            if isinstance(audio_payload, dict):
                audio_data = audio_payload.get("data") or audio_payload.get("url")
                audio_format = audio_payload.get("format")
            else:
                audio_data = audio_payload
                audio_format = part.get("format")

            if not isinstance(audio_data, str) or not audio_data.strip():
                raise HTTPException(status_code=400, detail="audio content must contain base64 data")

            audio_part: dict[str, Any] = {"type": "audio", "audio": audio_data.strip()}
            if audio_format is not None:
                audio_part["format"] = audio_format
            normalized.append(audio_part)
            continue

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content part type: {part_type}",
        )

    if not has_non_text_part:
        return "\n".join(
            str(part.get("text", ""))
            for part in normalized
            if part.get("text")
        )

    return normalized


def extract_image_payloads(conversation: list[dict[str, Any]]) -> list[str]:
    return [payload.data for payload in extract_media_payloads(conversation).images]


def reclassify_media_content(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            rewritten.append(message)
            continue

        updated_parts: list[dict[str, Any]] = []
        changed = False
        for part in content:
            updated = _reclassify_content_part(part)
            updated_parts.append(updated)
            changed = changed or updated is not part
        if changed:
            updated_message = dict(message)
            updated_message["content"] = updated_parts
            rewritten.append(updated_message)
        else:
            rewritten.append(message)
    return rewritten


def scope_media_history(
    conversation: list[dict[str, Any]],
    mode: str = "latest",
) -> tuple[list[dict[str, Any]], int]:
    if mode == "all":
        return conversation, 0

    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return conversation, 0

    scoped: list[dict[str, Any]] = []
    removed = 0
    for index, message in enumerate(conversation):
        content = message.get("content")
        if index == latest_user_index or not isinstance(content, list):
            scoped.append(message)
            continue

        text_parts = [
            str(part.get("text", "")).strip()
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        removed += sum(
            1
            for part in content
            if not (isinstance(part, dict) and part.get("type") == "text")
        )
        updated = dict(message)
        updated["content"] = "\n".join(part for part in text_parts if part)
        if any(
            not (isinstance(part, dict) and part.get("type") == "text")
            for part in content
        ):
            updated["_gateway_historical_media"] = True
        scoped.append(updated)
    return scoped, removed


def isolate_latest_media_turn(
    conversation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return conversation, 0

    isolated = [
        message
        for index, message in enumerate(conversation)
        if message.get("role") == "system" or index >= latest_user_index
    ]
    return isolated, len(conversation) - len(isolated)


def focus_current_media_context(
    conversation: list[dict[str, Any]],
    tokenizer: Any,
    media: MediaPayloads,
    history_max_tokens: int,
) -> tuple[list[dict[str, Any]], int, int]:
    latest_user_index = next(
        (
            index
            for index in range(len(conversation) - 1, -1, -1)
            if conversation[index].get("role") == "user"
        ),
        None,
    )
    if latest_user_index is None:
        return conversation, 0, 0

    system_messages = [
        dict(message)
        for message in conversation[:latest_user_index]
        if message.get("role") == "system"
    ]
    history = [
        message
        for index, message in enumerate(conversation[:latest_user_index])
        if message.get("role") != "system"
    ]
    active_tail = conversation[latest_user_index:]
    history, _ = _neutralize_historical_media_turns(history)

    remaining = max(int(history_max_tokens), 0)
    selected_reversed: list[dict[str, Any]] = []
    for message in reversed(history):
        cost = _history_message_token_cost(tokenizer, message)
        if cost > remaining:
            break
        selected_reversed.append(message)
        remaining -= cost

    selected_history = list(reversed(selected_reversed))
    dropped_count = len(history) - len(selected_history)
    instruction = _current_media_instruction(media)
    if system_messages:
        first = system_messages[0]
        content = first.get("content", "")
        first["content"] = f"{content}\n\n{instruction}".strip()
    else:
        system_messages = [{"role": "system", "content": instruction}]

    return (
        _strip_gateway_metadata(
            [*system_messages, *selected_history, *active_tail]
        ),
        len(selected_history),
        dropped_count,
    )


def _neutralize_historical_media_turns(
    history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in history:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)

    retained: list[dict[str, Any]] = []
    neutralized = 0
    for turn in turns:
        if any(message.get("_gateway_historical_media") for message in turn):
            retained.extend(_neutralize_historical_media_turn(turn))
            neutralized += len(turn)
        else:
            retained.extend(turn)
    return retained, neutralized


def _neutralize_historical_media_turn(
    turn: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    neutralized: list[dict[str, Any]] = []
    for message in turn:
        updated = dict(message)
        role = updated.get("role")
        if role == "user":
            content = str(updated.get("content", "")).strip()
            prefix = (
                "[Historical attachment omitted; it is not the current attachment.]"
            )
            updated["content"] = f"{prefix} {content}".strip()
        elif role == "assistant":
            updated["content"] = (
                "[Previous assistant response about an earlier attachment omitted "
                "because it describes a different attachment.]"
            )
        neutralized.append(updated)
    return neutralized


def strip_gateway_metadata(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _strip_gateway_metadata(conversation)


def _strip_gateway_metadata(
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cleaned = []
    for message in conversation:
        updated = dict(message)
        updated.pop("_gateway_historical_media", None)
        cleaned.append(updated)
    return cleaned


def _history_message_token_cost(tokenizer: Any, message: dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, list):
        text = "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content)
    try:
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        token_count = len(tokenizer.encode(text))
    return token_count + 8


def _current_media_instruction(media: MediaPayloads) -> str:
    media_types = [
        *(["image"] * len(media.images)),
        *(["video"] * len(media.videos)),
        *(["audio"] * len(media.audios)),
        *(["PDF"] * len(media.pdfs)),
    ]
    description = ", ".join(media_types) or "media"
    return (
        "The latest user message contains new current attachment(s): "
        f"{description}. Treat these attachments as the primary source for the current "
        "answer. Resolve phrases such as 'this', 'here', 'on the image', or 'in the file' "
        "to the attachments in the latest user message. Earlier descriptions of images, "
        "videos, audio, or documents refer to different historical attachments and must "
        "not be used as descriptions of the current attachment unless the user explicitly "
        "asks to compare them. Historical attachment turns may be represented by omission "
        "markers; treat those markers only as conversation context, not as evidence about "
        "the current attachment. If the current attachment cannot be analyzed, state that "
        "clearly instead of substituting information from conversation history. Do not "
        "mention these routing instructions in the answer."
    )


def _reclassify_content_part(part: Any) -> Any:
    if not isinstance(part, dict) or part.get("type") == "text":
        return part

    declared_kind = None
    value = None
    media_format = part.get("format")
    if part.get("type") == "image_url" or "image_url" in part:
        declared_kind = "image"
        image_url = part.get("image_url")
        value = image_url.get("url") if isinstance(image_url, dict) else image_url
    elif part.get("type") == "image" or "image" in part:
        declared_kind = "image"
        value = part.get("image")
    elif part.get("type") == "video" or "video" in part:
        declared_kind = "video"
        value = part.get("video")
    elif part.get("type") == "audio" or "audio" in part:
        declared_kind = "audio"
        value = part.get("audio")

    if declared_kind is None:
        return part
    payload = _extract_base64_payload(value, default_mime_type=declared_kind)
    detected_kind = _detect_media_kind(payload, declared_kind)
    if detected_kind == declared_kind:
        return part
    if detected_kind == "pdf":
        return {"type": "image_url", "image_url": {"url": value}}
    if detected_kind == "image":
        return {"type": "image_url", "image_url": {"url": value}}
    if detected_kind == "video":
        return {"type": "video", "video": value}

    updated = {"type": "audio", "audio": value}
    if media_format:
        updated["format"] = media_format
    return updated


def extract_media_payloads(conversation: list[dict[str, Any]]) -> MediaPayloads:
    images: list[MediaPayload] = []
    videos: list[MediaPayload] = []
    audios: list[MediaPayload] = []
    pdfs: list[MediaPayload] = []
    order: list[str] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue

        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url" or "image_url" in part:
                image_url = part.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                payload = _extract_base64_payload(url, default_mime_type="image")
                _append_classified_payload(
                    payload, "image", images, videos, audios, pdfs, order
                )
            elif part.get("type") == "image" or "image" in part:
                payload = _extract_base64_payload(part.get("image"), default_mime_type="image")
                _append_classified_payload(
                    payload, "image", images, videos, audios, pdfs, order
                )
            elif part.get("type") == "video" or "video" in part:
                payload = _extract_base64_payload(part.get("video"), default_mime_type="video")
                _append_classified_payload(
                    payload, "video", images, videos, audios, pdfs, order
                )
            elif part.get("type") == "audio" or "audio" in part:
                payload = _extract_base64_payload(part.get("audio"), default_mime_type="audio")
                if part.get("format") and payload.format is None:
                    payload = MediaPayload(
                        data=payload.data,
                        mime_type=payload.mime_type,
                        format=str(part.get("format")),
                    )
                _append_classified_payload(
                    payload, "audio", images, videos, audios, pdfs, order
                )

    return MediaPayloads(
        images=images,
        videos=videos,
        audios=audios,
        pdfs=pdfs,
        order=order,
    )


def _extract_base64_payload(value: Any, default_mime_type: str) -> MediaPayload:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail=f"{default_mime_type} payload must be a base64 string")

    payload = value.strip()
    mime_type = None
    match = DATA_URL_RE.match(payload)
    if match:
        mime_type = _extract_mime_type(payload)
        payload = match.group("payload").strip()

    if payload.startswith(("http://", "https://")):
        return MediaPayload(
            data=payload,
            mime_type=f"{default_mime_type}/remote-url",
            format=_guess_format_from_url(payload),
        )

    payload = "".join(payload.split())

    try:
        base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 {default_mime_type} payload") from exc

    return MediaPayload(
        data=payload,
        mime_type=mime_type,
        format=_guess_format(mime_type, default_mime_type),
    )


def _is_pdf_payload(payload: MediaPayload) -> bool:
    return _detect_media_kind(payload, "image") == "pdf"


def _append_classified_payload(
    payload: MediaPayload,
    declared_kind: str,
    images: list[MediaPayload],
    videos: list[MediaPayload],
    audios: list[MediaPayload],
    pdfs: list[MediaPayload],
    order: list[str],
) -> None:
    kind = _detect_media_kind(payload, declared_kind)
    collections = {
        "image": images,
        "video": videos,
        "audio": audios,
        "pdf": pdfs,
    }
    collections[kind].append(_payload_with_detected_metadata(payload, kind))
    order.append(kind)


def _detect_media_kind(payload: MediaPayload, declared_kind: str) -> str:
    mime_type = (payload.mime_type or "").lower()
    media_format = (payload.format or "").lower().lstrip(".")
    if mime_type == "application/pdf" or media_format == "pdf":
        return "pdf"

    if payload.data.startswith(("http://", "https://")):
        return _kind_from_format(media_format) or declared_kind

    prefix = _decoded_payload_prefix(payload.data)
    stripped = prefix.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"%PDF-"):
        return "pdf"
    if (
        prefix.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"))
        or (prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP")
    ):
        return "image"
    if (
        prefix.startswith(b"\x1aE\xdf\xa3")
        or (prefix.startswith(b"RIFF") and prefix[8:12] == b"AVI ")
        or (len(prefix) >= 12 and prefix[4:8] == b"ftyp" and declared_kind != "audio")
    ):
        return "video"
    if (
        prefix.startswith((b"ID3", b"fLaC", b"OggS"))
        or (prefix.startswith(b"RIFF") and prefix[8:12] == b"WAVE")
        or (len(prefix) >= 2 and prefix[0] == 0xFF and prefix[1] & 0xE0 == 0xE0)
        or (len(prefix) >= 12 and prefix[4:8] == b"ftyp" and declared_kind == "audio")
    ):
        return "audio"
    return _kind_from_format(media_format) or declared_kind


def _decoded_payload_prefix(payload: str, size: int = 96) -> bytes:
    try:
        encoded = payload[:size]
        encoded += "=" * (-len(encoded) % 4)
        return base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return b""


def _kind_from_format(media_format: str) -> str | None:
    if media_format == "pdf":
        return "pdf"
    if media_format in {"png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff"}:
        return "image"
    if media_format in {"mp4", "mov", "mkv", "webm", "avi", "mpeg", "mpg"}:
        return "video"
    if media_format in {"wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"}:
        return "audio"
    return None


def _payload_with_detected_metadata(payload: MediaPayload, kind: str) -> MediaPayload:
    detected_format = payload.format if _kind_from_format(payload.format or "") == kind else None
    defaults = {
        "pdf": ("application/pdf", "pdf"),
        "image": ("image/unknown", detected_format),
        "video": ("video/unknown", detected_format or "mp4"),
        "audio": ("audio/unknown", detected_format),
    }
    mime_type, media_format = defaults[kind]
    current_major = (payload.mime_type or "").split("/", 1)[0].lower()
    expected_major = "application" if kind == "pdf" else kind
    if current_major == expected_major:
        mime_type = payload.mime_type
    return MediaPayload(
        data=payload.data,
        mime_type=mime_type,
        format=media_format,
    )


def _extract_mime_type(data_url: str) -> str | None:
    header = data_url.split(",", 1)[0]
    if not header.startswith("data:"):
        return None
    return header[5:].split(";", 1)[0] or None


def _guess_format(mime_type: str | None, default_mime_type: str) -> str | None:
    if not mime_type or "/" not in mime_type:
        return None

    if mime_type.lower() == "application/pdf":
        return "pdf"

    media_type, subtype = mime_type.split("/", 1)
    if media_type != default_mime_type:
        return None
    if subtype == "jpeg":
        return "jpg"
    if subtype in {"x-wav", "wave"}:
        return "wav"
    return subtype.split("+", 1)[0]


def _guess_format_from_url(url: str) -> str | None:
    path = urlparse(url).path
    suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if suffix == "jpeg":
        return "jpg"
    return suffix or None
