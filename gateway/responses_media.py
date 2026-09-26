# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Bounded image preparation for the Responses input contract."""
from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile

from fastapi import HTTPException

from .media_preprocessing import _image_payload_to_item, VllmMediaProcessingError
from .multimodal import extract_media_payloads
from .token_budget import _json


def validate_vision(model_path: Path, tokenizer_path: Path) -> None:
    configured = _json(model_path / "gateway.json").get("generation", {}).get("supports_vision")
    if configured is not None and type(configured) is not bool:
        raise HTTPException(500, "generation.supports_vision must be a boolean")
    if configured is False:
        raise HTTPException(400, "This model does not support image input")
    if configured is True:
        return
    config = _json(tokenizer_path / "config.json")
    if isinstance(config.get("vision_config"), dict) or config.get("vision_tower"):
        return
    raise HTTPException(400, "Image capability is not configured for this model: "
                        "use a vision model or set generation.supports_vision=true for a verified vision runtime")


async def prepare_images(conversation, settings):
    def prepare():
        if settings.temp_dir is not None:
            settings.temp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="triton-responses-", dir=settings.temp_dir) as directory:
            result = []
            for message in conversation:
                updated = dict(message)
                if isinstance(message.get("content"), list):
                    parts = []
                    for part in message["content"]:
                        if part.get("type") == "image_url":
                            payload = extract_media_payloads([{"role": "user", "content": [part]}]).images[0]
                            item = _image_payload_to_item(payload, Path(directory), 0, settings)
                            parts.append({"type": "image_url", "image_url": {
                                "url": "data:image/jpeg;base64," + item.data,
                                "detail": part["image_url"].get("detail", "auto")}})
                        else:
                            parts.append(part)
                    updated["content"] = parts
                result.append(updated)
            return result
    try:
        return await asyncio.to_thread(prepare)
    except VllmMediaProcessingError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc
