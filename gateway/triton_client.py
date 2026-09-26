# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpc_aio
from tritonclient.utils import InferenceServerException
from fastapi import HTTPException
from grpc.aio import AioRpcError

from .debug import log_chat_response_debug
from .multimodal import MediaPayload, MediaPayloads
from .metrics import (
    TRITON_STREAM_CANCELLED,
    TRITON_STREAMS_ACTIVE,
    triton_call,
)
from .observability import get_request_id, get_traceparent, log_event
from .generation_telemetry import mark_first_generation_output
from .prompt import build_usage, completion_reached_token_limit
from .reasoning import (
    DISABLED_REASONING_SETTINGS,
    ReasoningResult,
    ReasoningSettings,
    observe_reasoning_result,
    reasoning_message_fields,
    split_reasoning_output,
)
from .sanitizer import sanitize_generated_text, strip_prompt_echo
from .schemas import ChatCompletionRequest
from .settings import (
    REQUEST_TIMEOUT_SECONDS,
    STREAM_HOLDBACK_CHARS,
    TRITON_BASE_URL,
    TRITON_GRPC_URL,
)
from .tool_parsers import extract_tool_calls


_http_client: httpx.AsyncClient | None = None
_grpc_client: grpc_aio.InferenceServerClient | None = None
_grpc_client_lock = asyncio.Lock()
logger = logging.getLogger("triton-chat-gateway")
REASONING_STREAM_HOLDBACK_CHARS = 12


def _process_completed_generation(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    generated_text: str,
    reasoning_settings: ReasoningSettings | None,
) -> tuple[str, ReasoningResult, dict[str, Any]]:
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS
    raw_generated_text = strip_prompt_echo(prompt, generated_text)
    reasoning_result = split_reasoning_output(raw_generated_text, settings)
    content, _ = sanitize_generated_text(reasoning_result.content)
    usage = build_usage(
        tokenizer,
        prompt,
        raw_generated_text,
        reasoning_text=reasoning_result.reasoning,
    )
    reasoning_tokens, content_tokens = observe_reasoning_result(
        request.model,
        tokenizer,
        settings,
        reasoning_result,
    )
    log_event(
        logger,
        "chat.reasoning_processed",
        "Reasoning policy applied to chat response",
        model=request.model,
        reasoning_mode=settings.mode,
        reasoning_parser=settings.parser,
        reasoning_detected=reasoning_result.detected,
        reasoning_incomplete=reasoning_result.incomplete,
        reasoning_tokens=reasoning_tokens,
        content_tokens=content_tokens,
    )
    return content, reasoning_result, usage


def _reasoning_delta(
    reasoning_result: ReasoningResult,
    settings: ReasoningSettings,
    emitted_reasoning: str,
    *,
    final: bool,
) -> tuple[str, str]:
    if not settings.expose_reasoning:
        return "", emitted_reasoning
    safe_reasoning = reasoning_result.reasoning
    if (
        not final
        and reasoning_result.incomplete
        and len(safe_reasoning) > REASONING_STREAM_HOLDBACK_CHARS
    ):
        safe_reasoning = safe_reasoning[:-REASONING_STREAM_HOLDBACK_CHARS]
    elif not final and reasoning_result.incomplete:
        safe_reasoning = ""

    if safe_reasoning.startswith(emitted_reasoning):
        delta = safe_reasoning[len(emitted_reasoning) :]
    else:
        # Cumulative output should be monotonic. Suppress a non-monotonic
        # parser transition rather than duplicating reasoning in the client.
        delta = ""
    return delta, safe_reasoning if delta else emitted_reasoning


def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS),
        )
    return _http_client


async def close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
    _http_client = None


async def get_grpc_client() -> grpc_aio.InferenceServerClient:
    global _grpc_client
    if _grpc_client is not None:
        return _grpc_client
    async with _grpc_client_lock:
        if _grpc_client is None:
            _grpc_client = grpc_aio.InferenceServerClient(url=TRITON_GRPC_URL)
        return _grpc_client


async def close_grpc_client() -> None:
    global _grpc_client
    async with _grpc_client_lock:
        client = _grpc_client
        _grpc_client = None
    if client is not None:
        await client.close()


async def is_triton_ready() -> bool:
    try:
        response = await get_http_client().get(
            f"{TRITON_BASE_URL}/v2/health/ready",
            timeout=2.0,
            headers=_request_headers(),
        )
    except httpx.HTTPError:
        return False
    return response.status_code == 200


async def list_ready_triton_models() -> list[str] | None:
    try:
        response = await get_http_client().post(
            f"{TRITON_BASE_URL}/v2/repository/index",
            json={"ready": True},
            timeout=2.0,
            headers=_request_headers(),
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    if not isinstance(payload, list):
        return None
    return sorted(
        {
            str(item["name"])
            for item in payload
            if isinstance(item, dict)
            and item.get("name")
            and str(item.get("state", "READY")).upper() == "READY"
        }
    )


def _is_context_length_error(text: str) -> bool:
    lowered = text.lower()
    return all(marker.lower() in lowered for marker in ("longer than", "maximum model length"))


def _friendly_error_message(detail: Any) -> str:
    detail_text = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
    if _is_context_length_error(detail_text):
        return (
            "Не удалось обработать запрос: PDF/изображения вместе с историей диалога "
            "не помещаются в контекст модели. Начните новый диалог без старой истории, "
            "уменьшите `pdf_chunk_pages` или `pdf_max_pixels`, либо увеличьте "
            "`max_model_len` для модели."
        )
    return f"Не удалось обработать запрос через Triton: {detail_text}"


def _triton_grpc_error(operation: str, exc: Exception) -> HTTPException:
    detail = str(exc)
    lowered = detail.lower()
    limit_markers = (
        "maximum model length",
        "exceeds the configured maximum",
        "configured maximum",
        "media item is",
        "media items; maximum",
        "source pixels; maximum",
        "video frames; maximum",
        "pdf has",
    )
    invalid_media_markers = (
        "not valid base64",
        "contains no decodable",
        "contains no pages",
        "no supported image placeholder",
        "remote urls must be materialized",
    )
    task_mismatch_markers = (
        "does not support 'generate' request",
        "does not support 'embed' request",
    )
    lora_unrecognized_markers = (
        "is not supported, we currently support",
        "lora feature is not enabled",
    )
    if any(marker in lowered for marker in limit_markers):
        status_code = 413
    elif any(
        marker in lowered
        for marker in (*invalid_media_markers, *task_mismatch_markers, *lora_unrecognized_markers)
    ):
        status_code = 400
    else:
        status_code = 502
    return HTTPException(
        status_code=status_code,
        detail=f"Triton {operation} failed: {detail}",
    )


def extract_text_output(response_json: dict[str, Any]) -> str:
    text_output = response_json.get("text_output")
    if text_output is not None:
        return str(text_output)

    for output in response_json.get("outputs", []):
        if output.get("name") != "text_output":
            continue
        data = output.get("data") or []
        if not data:
            return ""
        value = data[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    raise HTTPException(status_code=500, detail="Triton response does not contain text_output")


def extract_uint_output(response_json: dict[str, Any], name: str) -> int:
    for output in response_json.get("outputs", []):
        if output.get("name") != name:
            continue
        data = output.get("data") or []
        if not data:
            return 0
        return int(data[0])
    return 0


async def _stream_grpc_results(
    operation: str,
    model_name: str,
    inputs: list[grpcclient.InferInput],
    outputs: list[grpcclient.InferRequestedOutput],
    *,
    streaming: bool = False,
) -> AsyncIterator[grpcclient.InferResult]:
    client = await get_grpc_client()
    request_id = _new_triton_request_id(operation)

    async def request_iterator():
        yield {
            "model_name": model_name,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": request_id,
        }

    response_iterator = client.stream_infer(
        request_iterator(),
        stream_timeout=REQUEST_TIMEOUT_SECONDS,
        headers=_request_headers(grpc=True),
    )
    completed = False
    if streaming:
        TRITON_STREAMS_ACTIVE.labels(model_name).inc()
    try:
        with triton_call(operation, model_name, "grpc"):
            async for result, error in response_iterator:
                if error is not None:
                    raise error
                if result is not None:
                    yield result
        completed = True
    finally:
        if not completed:
            response_iterator.cancel()
            if streaming:
                TRITON_STREAM_CANCELLED.labels(model_name).inc()
        if streaming:
            TRITON_STREAMS_ACTIVE.labels(model_name).dec()


def _new_triton_request_id(operation: str) -> str:
    parent_id = get_request_id() or uuid.uuid4().hex
    return f"{parent_id}:{operation}:{uuid.uuid4().hex[:8]}"


def _request_headers(*, grpc: bool = False) -> dict[str, str]:
    request_id = get_request_id()
    headers = {"triton_grpc_error": "true"} if grpc else {}
    if request_id:
        headers.update(
            {
                "x-request-id": request_id,
                "triton-request-id": request_id,
            }
        )
    traceparent = get_traceparent()
    if traceparent:
        headers["traceparent"] = traceparent
    return headers


def _with_request_id(payload: dict[str, Any]) -> dict[str, Any]:
    request_id = get_request_id()
    return {"id": request_id, **payload} if request_id else payload


def _decode_numpy_first(value: np.ndarray | None) -> Any:
    if value is None:
        return None

    flattened = value.reshape(-1)
    if flattened.size == 0:
        return None

    item = flattened[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.bytes_):
        return item.tobytes().decode("utf-8")
    if hasattr(item, "item"):
        return item.item()
    return item


def _decode_numpy_strings(value: np.ndarray | None) -> list[str]:
    if value is None:
        return []

    decoded: list[str] = []
    for item in value.reshape(-1):
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        elif isinstance(item, np.bytes_):
            decoded.append(item.tobytes().decode("utf-8"))
        elif item is not None:
            decoded.append(str(item.item() if hasattr(item, "item") else item))
    return decoded


def _is_triton_final_response(result: grpcclient.InferResult) -> bool:
    get_response = getattr(result, "get_response", None)
    if get_response is None:
        return False

    try:
        response = get_response(as_json=True)
    except TypeError:
        try:
            response = get_response()
        except Exception:
            return False
    except Exception:
        return False

    parameters = {}
    if isinstance(response, dict):
        parameters = response.get("parameters") or {}
    else:
        parameters = getattr(response, "parameters", {}) or {}

    final_parameter = parameters.get("triton_final_response")
    if isinstance(final_parameter, dict):
        return bool(
            final_parameter.get("bool_param")
            or final_parameter.get("boolParam")
            or final_parameter.get("bool_value")
        )
    if hasattr(final_parameter, "bool_param"):
        return bool(final_parameter.bool_param)
    if isinstance(final_parameter, bool):
        return final_parameter
    return False


def _build_grpc_embedding_inputs(
    model_input: str | list[int],
    dimensions: int | None,
    *,
    output_types: list[str] | None = None,
    sparse_top_k: int | None = None,
    lora_name: str | None = None,
) -> list[grpcclient.InferInput]:
    embedding_request: dict[str, Any] = {"input": model_input, "pooling_params": {}}
    if dimensions is not None:
        embedding_request["pooling_params"]["dimensions"] = [dimensions]
    if output_types is not None:
        embedding_request["output_types"] = output_types
        embedding_request["sparse_format"] = "indices_values"
    if sparse_top_k is not None:
        embedding_request["sparse_top_k"] = sparse_top_k
    if lora_name is not None:
        embedding_request["lora_name"] = lora_name

    embedding_request_json = json.dumps(embedding_request, ensure_ascii=False)

    embedding_input = grpcclient.InferInput("embedding_request", [1], "BYTES")
    embedding_input.set_data_from_numpy(
        np.asarray([embedding_request_json.encode("utf-8")], dtype=np.object_)
    )

    return_input_tokens = grpcclient.InferInput("return_num_input_tokens", [1], "BOOL")
    return_input_tokens.set_data_from_numpy(np.asarray([True], dtype=np.bool_))

    return_output_tokens = grpcclient.InferInput("return_num_output_tokens", [1], "BOOL")
    return_output_tokens.set_data_from_numpy(np.asarray([True], dtype=np.bool_))

    return [embedding_input, return_input_tokens, return_output_tokens]


def _build_grpc_generate_inputs(
    prompt: str,
    sampling_parameters: dict[str, Any],
    images: list[str],
    stream: bool,
    media: MediaPayloads | None = None,
) -> list[grpcclient.InferInput]:
    inputs = _build_grpc_text_generation_base_inputs(prompt, sampling_parameters, stream)
    if media is None:
        _append_bytes_input(inputs, "image", [image.encode("utf-8") for image in images])
        return inputs

    _append_media_input(inputs, "image", media.images)
    _append_media_input(inputs, "video", media.videos)
    _append_media_input(inputs, "audio", media.audios)
    _append_media_input(inputs, "pdf", media.pdfs)

    media_parameters = grpcclient.InferInput("media_parameters", [1], "BYTES")
    parameters = {**media.parameters, "media_order": media.order}
    media_parameters.set_data_from_numpy(
        np.asarray(
            [json.dumps(parameters).encode("utf-8")],
            dtype=np.object_,
        )
    )
    inputs.append(media_parameters)
    return inputs


def _append_media_input(
    inputs: list[grpcclient.InferInput],
    name: str,
    payloads: list[MediaPayload],
) -> None:
    values = [
        json.dumps(
            {
                "data": payload.data,
                "mime_type": payload.mime_type,
                "format": payload.format,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        for payload in payloads
    ]
    _append_bytes_input(inputs, name, values)


def _append_bytes_input(
    inputs: list[grpcclient.InferInput],
    name: str,
    values: list[bytes],
) -> None:
    if not values:
        return
    tensor = grpcclient.InferInput(name, [len(values)], "BYTES")
    tensor.set_data_from_numpy(np.asarray(values, dtype=np.object_))
    inputs.append(tensor)


def _build_grpc_text_generation_base_inputs(
    prompt: str,
    sampling_parameters: dict[str, Any],
    stream: bool,
) -> list[grpcclient.InferInput]:
    text_input = grpcclient.InferInput("text_input", [1], "BYTES")
    text_input.set_data_from_numpy(np.asarray([prompt.encode("utf-8")], dtype=np.object_))

    stream_input = grpcclient.InferInput("stream", [1], "BOOL")
    stream_input.set_data_from_numpy(np.asarray([stream], dtype=np.bool_))

    sampling_input = grpcclient.InferInput("sampling_parameters", [1], "BYTES")
    sampling_input.set_data_from_numpy(
        np.asarray(
            [json.dumps(sampling_parameters, ensure_ascii=False).encode("utf-8")],
            dtype=np.object_,
        )
    )

    exclude_input = grpcclient.InferInput("exclude_input_in_output", [1], "BOOL")
    exclude_input.set_data_from_numpy(np.asarray([True], dtype=np.bool_))

    return [text_input, stream_input, sampling_input, exclude_input]


async def call_triton_embeddings(
    model_name: str,
    model_input: str | list[int],
    dimensions: int | None,
    lora_name: str | None = None,
) -> tuple[list[float], int]:
    inputs = _build_grpc_embedding_inputs(model_input, dimensions, lora_name=lora_name)
    outputs = [
        grpcclient.InferRequestedOutput("text_output"),
        grpcclient.InferRequestedOutput("num_input_tokens"),
        grpcclient.InferRequestedOutput("num_output_tokens"),
    ]
    try:
        embedding: list[float] | None = None
        prompt_tokens = 0
        async for item in _stream_grpc_results(
            "embeddings",
            model_name,
            inputs,
            outputs,
        ):
            embedding_json = _decode_numpy_first(item.as_numpy("text_output"))
            if embedding_json is None:
                continue
            parsed = json.loads(str(embedding_json))
            if not isinstance(parsed, list):
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Unexpected embeddings payload type from Triton: "
                        f"{type(parsed).__name__}"
                    ),
                )
            embedding = [float(value) for value in parsed]
            prompt_tokens_value = _decode_numpy_first(item.as_numpy("num_input_tokens"))
            prompt_tokens = int(prompt_tokens_value or 0)
        if embedding is None:
            raise HTTPException(
                status_code=502,
                detail="Triton embeddings response does not contain text_output",
            )
        return embedding, prompt_tokens
    except HTTPException:
        raise
    except (InferenceServerException, AioRpcError) as exc:
        raise _triton_grpc_error("embeddings gRPC infer", exc) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid embeddings payload from Triton: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Triton embeddings gRPC request failed: {exc}",
        ) from exc


async def call_triton_hybrid_embeddings(
    model_name: str,
    model_input: str | list[int],
    dimensions: int | None,
    output_types: list[str],
    sparse_top_k: int | None,
) -> tuple[dict[str, Any], int]:
    inputs = _build_grpc_embedding_inputs(
        model_input,
        dimensions,
        output_types=output_types,
        sparse_top_k=sparse_top_k,
    )
    outputs = [
        grpcclient.InferRequestedOutput("text_output"),
        grpcclient.InferRequestedOutput("num_input_tokens"),
        grpcclient.InferRequestedOutput("num_output_tokens"),
    ]
    try:
        result: dict[str, Any] | None = None
        prompt_tokens = 0
        async for item in _stream_grpc_results(
            "hybrid_embeddings",
            model_name,
            inputs,
            outputs,
        ):
            embedding_json = _decode_numpy_first(item.as_numpy("text_output"))
            if embedding_json is None:
                continue
            parsed = json.loads(str(embedding_json))
            result = _validate_hybrid_embedding_payload(parsed, output_types)
            prompt_tokens_value = _decode_numpy_first(
                item.as_numpy("num_input_tokens")
            )
            prompt_tokens = int(prompt_tokens_value or 0)
        if result is None:
            raise HTTPException(
                status_code=502,
                detail="Triton hybrid embeddings response does not contain text_output",
            )
        return result, prompt_tokens
    except HTTPException:
        raise
    except InferenceServerException as exc:
        raise _triton_grpc_error("hybrid embeddings gRPC infer", exc) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid hybrid embeddings payload from Triton: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Triton hybrid embeddings gRPC request failed: {exc}",
        ) from exc


def _validate_hybrid_embedding_payload(
    payload: Any,
    output_types: list[str],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=(
                "Unexpected hybrid embeddings payload type from Triton: "
                f"{type(payload).__name__}"
            ),
        )

    result: dict[str, Any] = {}
    if "dense" in output_types:
        dense = payload.get("dense")
        if not isinstance(dense, list):
            raise HTTPException(
                status_code=502,
                detail="Hybrid embeddings response is missing dense output",
            )
        try:
            normalized_dense = [float(value) for value in dense]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Hybrid embeddings dense output contains non-numeric values",
            ) from exc
        if not all(math.isfinite(value) for value in normalized_dense):
            raise HTTPException(
                status_code=502,
                detail="Hybrid embeddings dense output contains non-finite values",
            )
        result["dense"] = normalized_dense

    if "sparse" in output_types:
        sparse = payload.get("sparse")
        if not isinstance(sparse, dict):
            raise HTTPException(
                status_code=502,
                detail="Hybrid embeddings response is missing sparse output",
            )
        indices = sparse.get("indices")
        values = sparse.get("values")
        if not isinstance(indices, list) or not isinstance(values, list):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding must contain indices and values arrays",
            )
        if len(indices) != len(values):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding indices and values lengths differ",
            )
        if not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in indices
        ):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding indices must be integers",
            )
        try:
            normalized_indices = list(indices)
            normalized_values = [float(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding contains invalid index or weight values",
            ) from exc
        if any(index < 0 for index in normalized_indices):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding indices must be non-negative",
            )
        if len(set(normalized_indices)) != len(normalized_indices):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding indices must be unique",
            )
        if not all(math.isfinite(value) for value in normalized_values):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding weights must be finite",
            )
        if any(value < 0 for value in normalized_values):
            raise HTTPException(
                status_code=502,
                detail="Sparse embedding weights must be non-negative",
            )
        result["sparse"] = {
            "indices": normalized_indices,
            "values": normalized_values,
        }

    return result


async def call_triton_multimodal(
    model_name: str,
    prompt: str,
    sampling_parameters: dict[str, Any],
    images: list[str],
    media: MediaPayloads | None = None,
) -> str:
    inputs = _build_grpc_generate_inputs(
        prompt,
        sampling_parameters,
        images,
        stream=False,
        media=media,
    )
    outputs = [grpcclient.InferRequestedOutput("text_output")]
    last_text = ""

    try:
        async for item in _stream_grpc_results(
            "generate",
            model_name,
            inputs,
            outputs,
        ):
            text_chunks = _decode_numpy_strings(item.as_numpy("text_output"))
            if text_chunks:
                last_text = "".join(text_chunks)
        return last_text
    except HTTPException:
        raise
    except (InferenceServerException, AioRpcError) as exc:
        raise _triton_grpc_error("generation gRPC infer", exc) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Triton multimodal gRPC request failed: {exc}",
        ) from exc


async def call_triton_native_multimodal(
    model_name: str,
    prompt: str,
    sampling_parameters: dict[str, Any],
    media: MediaPayloads,
) -> str:
    return await call_triton_multimodal(
        model_name,
        prompt,
        sampling_parameters,
        [],
        media,
    )


async def stream_triton_multimodal_to_openai(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    images: list[str],
    media: MediaPayloads | None = None,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    emitted_text = ""
    emitted_reasoning = ""
    raw_generated_text = ""
    role_sent = False
    stop_requested = False
    inputs = _build_grpc_generate_inputs(
        prompt,
        sampling_parameters,
        images,
        stream=True,
        media=media,
    )
    outputs = [grpcclient.InferRequestedOutput("text_output")]
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {"role": "assistant"},
        )
    )
    role_sent = True

    try:
        async with aclosing(
            _stream_grpc_results(
                "generate-stream",
                request.model,
                inputs,
                outputs,
                streaming=True,
            )
        ) as results:
            async for item in results:
                for text_chunk in _decode_numpy_strings(item.as_numpy("text_output")):
                    event_text = strip_prompt_echo(prompt, text_chunk)
                    if event_text.startswith(raw_generated_text):
                        raw_generated_text = event_text
                    else:
                        raw_generated_text += event_text

                    reasoning_result = split_reasoning_output(
                        raw_generated_text,
                        settings,
                    )
                    current_text, should_stop = sanitize_generated_text(
                        reasoning_result.content,
                        streaming=True,
                    )
                    safe_text = current_text
                    if not should_stop and len(safe_text) > STREAM_HOLDBACK_CHARS:
                        safe_text = safe_text[:-STREAM_HOLDBACK_CHARS]
                    elif not should_stop:
                        safe_text = ""

                    if safe_text.startswith(emitted_text):
                        delta_text = safe_text[len(emitted_text) :]
                    else:
                        delta_text = safe_text

                    reasoning_delta, emitted_reasoning = _reasoning_delta(
                        reasoning_result,
                        settings,
                        emitted_reasoning,
                        final=False,
                    )

                    if not role_sent:
                        yield sse_event(
                            build_openai_chunk(
                                response_id,
                                created,
                                request.model,
                                {"role": "assistant"},
                            )
                        )
                        role_sent = True

                    if reasoning_delta:
                        yield sse_event(
                            build_openai_chunk(
                                response_id,
                                created,
                                request.model,
                                {settings.response_field: reasoning_delta},
                            )
                        )

                    if delta_text:
                        yield sse_event(
                            build_openai_chunk(
                                response_id,
                                created,
                                request.model,
                                {"content": delta_text},
                            )
                        )
                        emitted_text = safe_text

                    if should_stop:
                        stop_requested = True
                        break
                if stop_requested:
                    break
    except HTTPException:
        raise
    except InferenceServerException as exc:
        raise _triton_grpc_error("generation gRPC stream", exc) from exc
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Triton multimodal gRPC stream failed: {exc}",
        ) from exc

    reasoning_result = split_reasoning_output(raw_generated_text, settings)
    final_text, _ = sanitize_generated_text(reasoning_result.content)
    reasoning_delta, emitted_reasoning = _reasoning_delta(
        reasoning_result,
        settings,
        emitted_reasoning,
        final=True,
    )
    if reasoning_delta:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {settings.response_field: reasoning_delta},
            )
        )
    if final_text.startswith(emitted_text):
        delta_text = final_text[len(emitted_text) :]
    else:
        delta_text = ""

    if delta_text:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": delta_text},
            )
        )
        emitted_text = final_text

    usage = build_usage(
        tokenizer,
        prompt,
        raw_generated_text,
        reasoning_text=reasoning_result.reasoning,
    )
    reasoning_tokens, content_tokens = observe_reasoning_result(
        request.model,
        tokenizer,
        settings,
        reasoning_result,
    )
    log_event(
        logger,
        "chat.reasoning_processed",
        "Reasoning policy applied to chat response",
        model=request.model,
        reasoning_mode=settings.mode,
        reasoning_parser=settings.parser,
        reasoning_detected=reasoning_result.detected,
        reasoning_incomplete=reasoning_result.incomplete,
        reasoning_tokens=reasoning_tokens,
        content_tokens=content_tokens,
    )
    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {},
            finish_reason=(
                "length"
                if (
                    (reasoning_result.incomplete and not final_text)
                    or completion_reached_token_limit(
                        usage,
                        sampling_parameters,
                    )
                )
                else "stop"
            ),
            usage=usage,
        )
    )
    yield "data: [DONE]\n\n"


async def stream_triton_native_multimodal_to_openai(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    media: MediaPayloads,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    async with aclosing(stream_triton_multimodal_to_openai(
        request,
        tokenizer,
        prompt,
        sampling_parameters,
        [],
        media,
        reasoning_settings,
    )) as events:
        async for event in events:
            yield event


async def call_triton_rerank(
    model_name: str,
    query: str,
    documents: list[str],
    max_length: int | None,
    batch_size: int | None,
    normalize: bool,
) -> list[float]:
    infer_url = f"{TRITON_BASE_URL}/v2/models/{quote(model_name, safe='')}/infer"
    rerank_request: dict[str, Any] = {
        "query": query,
        "documents": documents,
        "normalize": normalize,
    }
    if max_length is not None:
        rerank_request["max_length"] = max_length
    if batch_size is not None:
        rerank_request["batch_size"] = batch_size

    payload = _with_request_id({
        "inputs": [
            {
                "name": "rerank_request",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(rerank_request, ensure_ascii=False)],
            }
        ],
        "outputs": [
            {"name": "text_output"},
        ],
    })

    with triton_call("rerank", model_name, "http"):
        response = await get_http_client().post(
            infer_url,
            json=payload,
            headers=_request_headers(),
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Triton rerank infer failed with status "
                    f"{response.status_code}: {response.text}"
                ),
            )

    scores_json = extract_text_output(response.json())
    try:
        scores = json.loads(scores_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid rerank payload from Triton: {scores_json}",
        ) from exc

    if not isinstance(scores, list):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected rerank payload type from Triton: {type(scores).__name__}",
        )

    return [float(score) for score in scores]


def build_openai_chunk(
    response_id: str,
    created: int,
    model_name: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    if any(
        value not in (None, "", [], {})
        for key, value in delta.items()
        if key != "role"
    ):
        mark_first_generation_output()
    payload: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def iter_sse_json_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    data_lines: list[str] = []

    async for line in response.aiter_lines():
        if line == "":
            if not data_lines:
                continue

            data = "\n".join(data_lines)
            data_lines = []
            if data == "[DONE]":
                return

            try:
                yield json.loads(data)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=502, detail=f"Invalid SSE payload from Triton: {data}") from exc
            continue

        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            try:
                yield json.loads(data)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=502, detail=f"Invalid SSE payload from Triton: {data}") from exc


async def stream_triton_to_openai(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    emitted_text = ""
    emitted_reasoning = ""
    raw_generated_text = ""
    role_sent = False
    generate_stream_url = f"{TRITON_BASE_URL}/v2/models/{quote(request.model, safe='')}/generate_stream"
    payload = {
        "text_input": prompt,
        "parameters": {
            **sampling_parameters,
            "stream": True,
        },
    }
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS

    with triton_call("generate-stream", request.model, "http"):
        async with get_http_client().stream(
            "POST",
            generate_stream_url,
            json=payload,
            headers=_request_headers(),
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Triton generate_stream failed with status "
                        f"{response.status_code}: {error_text.decode('utf-8', errors='replace')}"
                    ),
                )

            yield sse_event(
                build_openai_chunk(
                    response_id,
                    created,
                    request.model,
                    {"role": "assistant"},
                )
            )
            role_sent = True

            async for event in iter_sse_json_events(response):
                if "error" in event:
                    raise HTTPException(
                        status_code=502,
                        detail=f"Triton generate_stream returned error: {event['error']}",
                    )

                event_text = strip_prompt_echo(prompt, extract_text_output(event))
                if event_text.startswith(raw_generated_text):
                    raw_generated_text = event_text
                else:
                    raw_generated_text += event_text

                reasoning_result = split_reasoning_output(
                    raw_generated_text,
                    settings,
                )
                current_text, should_stop = sanitize_generated_text(
                    reasoning_result.content,
                    streaming=True,
                )
                safe_text = current_text
                if not should_stop and len(safe_text) > STREAM_HOLDBACK_CHARS:
                    safe_text = safe_text[:-STREAM_HOLDBACK_CHARS]
                elif not should_stop:
                    safe_text = ""

                if safe_text.startswith(emitted_text):
                    delta_text = safe_text[len(emitted_text) :]
                else:
                    delta_text = safe_text

                reasoning_delta, emitted_reasoning = _reasoning_delta(
                    reasoning_result,
                    settings,
                    emitted_reasoning,
                    final=False,
                )

                if not role_sent:
                    yield sse_event(
                        build_openai_chunk(
                            response_id,
                            created,
                            request.model,
                            {"role": "assistant"},
                        )
                    )
                    role_sent = True

                if reasoning_delta:
                    yield sse_event(
                        build_openai_chunk(
                            response_id,
                            created,
                            request.model,
                            {settings.response_field: reasoning_delta},
                        )
                    )

                if delta_text:
                    yield sse_event(
                        build_openai_chunk(
                            response_id,
                            created,
                            request.model,
                            {"content": delta_text},
                        )
                    )
                    emitted_text = safe_text

                if should_stop:
                    break

    reasoning_result = split_reasoning_output(raw_generated_text, settings)
    final_text, _ = sanitize_generated_text(reasoning_result.content)
    reasoning_delta, emitted_reasoning = _reasoning_delta(
        reasoning_result,
        settings,
        emitted_reasoning,
        final=True,
    )
    if reasoning_delta:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {settings.response_field: reasoning_delta},
            )
        )
    if final_text.startswith(emitted_text):
        delta_text = final_text[len(emitted_text) :]
    else:
        delta_text = ""

    if delta_text:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": delta_text},
            )
        )
        emitted_text = final_text

    usage = build_usage(
        tokenizer,
        prompt,
        raw_generated_text,
        reasoning_text=reasoning_result.reasoning,
    )
    reasoning_tokens, content_tokens = observe_reasoning_result(
        request.model,
        tokenizer,
        settings,
        reasoning_result,
    )
    log_event(
        logger,
        "chat.reasoning_processed",
        "Reasoning policy applied to chat response",
        model=request.model,
        reasoning_mode=settings.mode,
        reasoning_parser=settings.parser,
        reasoning_detected=reasoning_result.detected,
        reasoning_incomplete=reasoning_result.incomplete,
        reasoning_tokens=reasoning_tokens,
        content_tokens=content_tokens,
    )
    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {},
            finish_reason=(
                "length"
                if (
                    (reasoning_result.incomplete and not final_text)
                    or completion_reached_token_limit(
                        usage,
                        sampling_parameters,
                    )
                )
                else "stop"
            ),
            usage=usage,
        )
    )
    yield "data: [DONE]\n\n"


async def call_triton(model_name: str, prompt: str, sampling_parameters: dict[str, Any]) -> str:
    generate_url = f"{TRITON_BASE_URL}/v2/models/{quote(model_name, safe='')}/generate"
    payload = {
        "text_input": prompt,
        "parameters": {
            **sampling_parameters,
            "stream": False,
        },
    }

    with triton_call("generate", model_name, "http"):
        response = await get_http_client().post(
            generate_url,
            json=payload,
            headers=_request_headers(),
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=(
                    f"Triton infer failed with status {response.status_code}: "
                    f"{response.text}"
                ),
            )

    return extract_text_output(response.json())


async def call_triton_python_chat(
    model_name: str,
    prompt: str,
    conversation: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
) -> str:
    infer_url = f"{TRITON_BASE_URL}/v2/models/{quote(model_name, safe='')}/infer"
    chat_request = {
        "prompt": prompt,
        "messages": conversation,
        "sampling_parameters": sampling_parameters,
    }
    payload = _with_request_id({
        "inputs": [
            {
                "name": "chat_request",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(chat_request, ensure_ascii=False)],
            }
        ],
        "outputs": [
            {"name": "text_output"},
        ],
    })

    with triton_call("python-chat", model_name, "http"):
        response = await get_http_client().post(
            infer_url,
            json=payload,
            headers=_request_headers(),
        )
        if response.status_code != 200:
            detail = (
                "Triton python chat infer failed with status "
                f"{response.status_code}: {response.text}"
            )
            raise HTTPException(
                status_code=413 if _is_context_length_error(detail) else 502,
                detail=detail,
            )

    return extract_text_output(response.json())


async def stream_python_chat_to_openai(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    conversation: list[dict[str, Any]],
    sampling_parameters: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    tool_parser: str | None = None,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    try:
        generated_text = await call_triton_python_chat(
            request.model,
            prompt,
            conversation,
            sampling_parameters,
        )
    except HTTPException as exc:
        error_text = _friendly_error_message(exc.detail)
        usage = build_usage(tokenizer, prompt, "")

        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"role": "assistant"},
            )
        )
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": error_text},
            )
        )
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {},
                finish_reason="stop",
                usage=usage,
            )
        )
        yield "data: [DONE]\n\n"
        return

    generated_text, reasoning_result, usage = _process_completed_generation(
        request,
        tokenizer,
        prompt,
        generated_text,
        reasoning_settings,
    )
    tool_calls, remaining_text = (
        extract_tool_calls(generated_text, tools, tool_parser)
        if tools
        else ([], generated_text)
    )

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {"role": "assistant"},
        )
    )

    finish_reason = (
        "length"
        if (
            (reasoning_result.incomplete and not remaining_text)
            or completion_reached_token_limit(usage, sampling_parameters)
        )
        else "stop"
    )
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS
    if settings.expose_reasoning and reasoning_result.reasoning:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                reasoning_message_fields(reasoning_result, settings),
            )
        )
    if tool_calls:
        if finish_reason != "length":
            finish_reason = "tool_calls"
        for index, tool_call in enumerate(tool_calls):
            yield sse_event(
                build_openai_chunk(
                    response_id,
                    created,
                    request.model,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                **tool_call,
                            }
                        ]
                    },
                )
            )
    if remaining_text:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": remaining_text},
            )
        )

    log_chat_response_debug(
        request,
        generated_text,
        remaining_text,
        tool_calls,
        finish_reason,
    )

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {},
            finish_reason=finish_reason,
            usage=usage,
        )
    )
    yield "data: [DONE]\n\n"


async def stream_tool_aware_response(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    tools: list[dict[str, Any]],
    tool_parser: str | None = None,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    generated_text = await call_triton(request.model, prompt, sampling_parameters)
    generated_text, reasoning_result, usage = _process_completed_generation(
        request,
        tokenizer,
        prompt,
        generated_text,
        reasoning_settings,
    )
    tool_calls, remaining_text = extract_tool_calls(generated_text, tools, tool_parser)

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {"role": "assistant"},
        )
    )

    finish_reason = (
        "length"
        if (
            (reasoning_result.incomplete and not remaining_text)
            or completion_reached_token_limit(usage, sampling_parameters)
        )
        else "stop"
    )
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS
    if settings.expose_reasoning and reasoning_result.reasoning:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                reasoning_message_fields(reasoning_result, settings),
            )
        )
    if tool_calls:
        if finish_reason != "length":
            finish_reason = "tool_calls"
        for index, tool_call in enumerate(tool_calls):
            yield sse_event(
                build_openai_chunk(
                    response_id,
                    created,
                    request.model,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                **tool_call,
                            }
                        ]
                    },
                )
            )
    if remaining_text:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": remaining_text},
            )
        )

    log_chat_response_debug(
        request,
        generated_text,
        remaining_text,
        tool_calls,
        finish_reason,
    )

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {},
            finish_reason=finish_reason,
            usage=usage,
        )
    )
    yield "data: [DONE]\n\n"


async def stream_tool_aware_multimodal_response(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    images: list[str],
    tools: list[dict[str, Any]],
    tool_parser: str | None = None,
    media: MediaPayloads | None = None,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    response_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    generated_text = await call_triton_multimodal(
        request.model,
        prompt,
        sampling_parameters,
        images,
        media,
    )
    generated_text, reasoning_result, usage = _process_completed_generation(
        request,
        tokenizer,
        prompt,
        generated_text,
        reasoning_settings,
    )
    tool_calls, remaining_text = extract_tool_calls(generated_text, tools, tool_parser)

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {"role": "assistant"},
        )
    )

    finish_reason = (
        "length"
        if (
            (reasoning_result.incomplete and not remaining_text)
            or completion_reached_token_limit(usage, sampling_parameters)
        )
        else "stop"
    )
    settings = reasoning_settings or DISABLED_REASONING_SETTINGS
    if settings.expose_reasoning and reasoning_result.reasoning:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                reasoning_message_fields(reasoning_result, settings),
            )
        )
    if tool_calls:
        if finish_reason != "length":
            finish_reason = "tool_calls"
        for index, tool_call in enumerate(tool_calls):
            yield sse_event(
                build_openai_chunk(
                    response_id,
                    created,
                    request.model,
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                **tool_call,
                            }
                        ]
                    },
                )
            )
    if remaining_text:
        yield sse_event(
            build_openai_chunk(
                response_id,
                created,
                request.model,
                {"content": remaining_text},
            )
        )

    log_chat_response_debug(
        request,
        generated_text,
        remaining_text,
        tool_calls,
        finish_reason,
    )

    yield sse_event(
        build_openai_chunk(
            response_id,
            created,
            request.model,
            {},
            finish_reason=finish_reason,
            usage=usage,
        )
    )
    yield "data: [DONE]\n\n"


async def stream_tool_aware_native_multimodal_response(
    request: ChatCompletionRequest,
    tokenizer,
    prompt: str,
    sampling_parameters: dict[str, Any],
    media: MediaPayloads,
    tools: list[dict[str, Any]],
    tool_parser: str | None = None,
    reasoning_settings: ReasoningSettings | None = None,
) -> AsyncIterator[str]:
    async with aclosing(stream_tool_aware_multimodal_response(
        request,
        tokenizer,
        prompt,
        sampling_parameters,
        [],
        tools,
        tool_parser,
        media,
        reasoning_settings,
    )) as events:
        async for event in events:
            yield event
