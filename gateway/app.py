# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import replace
from functools import wraps
import asyncio
import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from . import __author__, __repository__
from .admission import AdmissionController, AdmissionLease
from .debug import (
    log_chat_prompt_debug,
    log_chat_request_debug,
    log_chat_response_debug,
)
from .embeddings import build_embedding_inputs, encode_embedding, tokenize_embedding_inputs
from .metrics import EMBEDDING_BATCH_SIZE, MEDIA_PROCESSING, metrics_response
from .multimodal import (
    extract_media_payloads,
    focus_current_media_context,
    isolate_latest_media_turn,
    reclassify_media_content,
    scope_media_history,
    strip_gateway_metadata,
)
from .observability import RequestContextMiddleware, log_event, set_request_model
from .prompt import (
    add_system_instruction,
    build_conversation,
    build_sampling_parameters,
    build_usage,
    fit_conversation_to_context,
    has_tool_result,
    selected_tools,
    tool_choice_instruction,
)
from .registry import ModelRegistry
from .rerank import build_rerank_documents, build_rerank_response
from .sanitizer import sanitize_generated_text, strip_prompt_echo
from .schemas import ChatCompletionRequest, EmbeddingsRequest, RerankRequest
from .settings import (
    EMBEDDING_MAX_CONCURRENCY,
    LOG_PROMPT_PREVIEW,
    TOKENIZER_PRELOAD,
    logger,
)
from .tool_parsers import extract_tool_calls
from .triton_client import (
    call_triton,
    call_triton_embeddings,
    call_triton_multimodal,
    call_triton_native_multimodal,
    call_triton_python_chat,
    call_triton_rerank,
    close_grpc_client,
    close_http_client,
    is_triton_ready,
    list_ready_triton_models,
    stream_python_chat_to_openai,
    stream_tool_aware_multimodal_response,
    stream_tool_aware_native_multimodal_response,
    stream_tool_aware_response,
    stream_triton_multimodal_to_openai,
    stream_triton_native_multimodal_to_openai,
    stream_triton_to_openai,
)
from .vllm_media import (
    PdfEmbeddingContext,
    VllmMediaProcessingError,
    load_vllm_media_settings,
    materialize_native_remote_media,
    prepare_vllm_media_conversation,
    requires_gateway_media_preprocessing,
)


registry = ModelRegistry()
admission = AdmissionController()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    preload_task = None
    if TOKENIZER_PRELOAD:
        preload_task = asyncio.create_task(registry.preload_tokenizers())
    try:
        yield
    finally:
        if preload_task is not None:
            if not preload_task.done():
                preload_task.cancel()
            try:
                await preload_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Tokenizer preload task failed during shutdown")
        await close_grpc_client()
        await close_http_client()


app = FastAPI(
    title="Triton OpenAI Gateway",
    description="OpenAI-compatible API and multimodal orchestration for NVIDIA Triton.",
    version="0.1.0",
    contact={"name": __author__, "url": __repository__},
    license_info={
        "name": "Apache-2.0",
        "url": f"{__repository__}/blob/main/LICENSE",
    },
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(HTTPException)
async def log_http_exception(_request, exc: HTTPException):
    log_event(
        logger,
        "http.request.rejected" if exc.status_code < 500 else "http.request.error",
        "HTTP request rejected" if exc.status_code < 500 else "HTTP request failed",
        level=logging.WARNING if exc.status_code < 500 else logging.ERROR,
        status_code=exc.status_code,
        detail=str(exc.detail),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def log_request_validation_exception(_request, exc: RequestValidationError):
    diagnostics = [
        {
            "type": error.get("type"),
            "loc": list(error.get("loc") or []),
            "msg": error.get("msg"),
        }
        for error in exc.errors()
    ]
    log_event(
        logger,
        "http.request.validation_failed",
        "HTTP request validation failed",
        level=logging.WARNING,
        status_code=422,
        validation_errors=diagnostics,
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )


async def _release_stream(
    source: AsyncIterator[Any],
    lease: AdmissionLease,
) -> AsyncIterator[Any]:
    try:
        async for item in source:
            yield item
    finally:
        await lease.release()


def admitted(route: str):
    def decorator(endpoint):
        @wraps(endpoint)
        async def wrapped(request):
            set_request_model(request.model)
            model_path = registry.resolve(request.model)
            registry.validate_route(request.model, route, model_path)
            resolved_route = route
            if route == "chat":
                conversation = reclassify_media_content(
                    build_conversation(request.messages)
                )
                if extract_media_payloads(conversation).has_any:
                    resolved_route = "media"

            lease = await admission.acquire(
                resolved_route,
                request.model,
                model_path,
            )
            try:
                response = await endpoint(request)
            except BaseException:
                await lease.release()
                raise

            if isinstance(response, StreamingResponse):
                response.body_iterator = _release_stream(
                    response.body_iterator,
                    lease,
                )
                return response

            await lease.release()
            return response

        return wrapped

    return decorator


def _estimate_media_context_tokens(media, media_settings) -> int:
    image_tokens = len(media.images) * max(
        (int(media_settings.image_max_pixels) + 783) // 784,
        1,
    )
    video_tokens = len(media.videos) * int(media_settings.video_max_frames) * max(
        (int(media_settings.video_max_pixels) + 783) // 784,
        1,
    )
    audio_tokens = len(media.audios) * 512
    return image_tokens + video_tokens + audio_tokens


async def _guard_openai_stream(source: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        async for event in source:
            yield event
    except HTTPException as exc:
        log_event(
            logger,
            "stream.failed",
            "Streaming inference failed",
            level=logging.ERROR,
            status_code=exc.status_code,
            error_type=type(exc).__name__,
        )
        payload = {
            "error": {
                "message": str(exc.detail),
                "type": "triton_error",
                "code": exc.status_code,
            }
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.exception(
            "Unexpected streaming inference failure",
            extra={
                "event": "stream.failed",
                "status_code": 500,
                "error_type": type(exc).__name__,
            },
        )
        payload = {
            "error": {
                "message": "Unexpected streaming inference failure",
                "type": "internal_error",
                "code": 500,
            }
        }
        yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/ready")
async def readinesscheck():
    if not await is_triton_ready():
        raise HTTPException(status_code=503, detail="Triton is not ready")
    return {"status": "ok"}


@app.get("/metrics")
async def prometheus_metrics():
    return metrics_response()


@app.get("/v1/models")
async def list_models():
    ready_models = await list_ready_triton_models()
    model_names = ready_models if ready_models is not None else registry.list_models()
    models = [
        {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "triton-openai-gateway",
        }
        for model_name in model_names
    ]
    return {"object": "list", "data": models}


@app.post("/v1/embeddings")
@admitted("embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    backend = registry.get_backend(request.model)
    model_inputs = build_embedding_inputs(request)
    encoding_format = request.encoding_format or "float"

    if backend in {"vllm", "vllm_multimodal"}:
        tokenizer, _ = await registry.get_tokenizer_async(request.model)
        model_inputs = tokenize_embedding_inputs(tokenizer, model_inputs)
        log_event(
            logger,
            "embeddings.routed",
            "Routing embeddings request to vLLM",
            model=request.model,
            backend=backend,
            transport="grpc",
        )
    else:
        log_event(
            logger,
            "embeddings.routed",
            "Routing embeddings request",
            model=request.model,
            backend=backend or "unknown",
            transport="grpc",
        )

    EMBEDDING_BATCH_SIZE.labels(request.model).observe(len(model_inputs))
    semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)

    async def embed(index: int, model_input):
        async with semaphore:
            embedding, prompt_tokens = await call_triton_embeddings(
                request.model,
                model_input,
                request.dimensions,
            )
        return index, embedding, prompt_tokens

    results = await asyncio.gather(
        *(embed(index, model_input) for index, model_input in enumerate(model_inputs))
    )
    results.sort(key=lambda item: item[0])
    total_prompt_tokens = sum(item[2] for item in results)
    data = [
        {
            "object": "embedding",
            "embedding": encode_embedding(embedding, encoding_format),
            "index": index,
        }
        for index, embedding, _ in results
    ]

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "total_tokens": total_prompt_tokens,
        },
    }


@app.post("/rerank")
@app.post("/v1/rerank")
@app.post("/v2/rerank")
@admitted("rerank")
async def rerank(request: RerankRequest):
    documents = build_rerank_documents(request)
    log_event(
        logger,
        "rerank.routed",
        "Routing rerank request",
        model=request.model,
        backend=registry.get_backend(request.model) or "unknown",
        transport="http",
    )

    scores = await call_triton_rerank(
        request.model,
        request.query,
        documents,
        request.max_length,
        request.batch_size,
        bool(request.normalize),
    )
    return build_rerank_response(request, documents, scores)


@app.post("/v1/chat/completions")
@admitted("chat")
async def create_chat_completion(request: ChatCompletionRequest):
    backend = registry.get_backend(request.model)
    is_vllm_backend = backend in {"vllm", "vllm_multimodal"}
    tokenizer, model_path = await registry.get_tokenizer_async(request.model)
    tools = selected_tools(request.tools, request.tool_choice)
    tool_parser = registry.get_tool_parser(request.model) if tools else None
    sampling_parameters = build_sampling_parameters(request)
    conversation = build_conversation(request.messages)
    log_chat_request_debug(request, conversation, tools, tool_parser)
    media_settings = load_vllm_media_settings(model_path)
    conversation, removed_historical_media = scope_media_history(
        conversation,
        media_settings.media_history_mode,
    )
    if removed_historical_media:
        log_event(
            logger,
            "media.history_scoped",
            "Historical media removed from active request",
            model=request.model,
            mode=media_settings.media_history_mode,
            removed_media_count=removed_historical_media,
        )
    conversation = reclassify_media_content(conversation)
    request_media = extract_media_payloads(conversation)
    if request_media.has_any and media_settings.reset_history_on_new_media:
        conversation, removed_history_messages = isolate_latest_media_turn(conversation)
        if removed_history_messages:
            log_event(
                logger,
                "media.history_reset",
                "Conversation history removed for new media request",
                model=request.model,
                removed_message_count=removed_history_messages,
            )
    elif request_media.has_any and media_settings.focus_current_media:
        conversation, kept_history_messages, dropped_history_messages = (
            focus_current_media_context(
                conversation,
                tokenizer,
                request_media,
                media_settings.media_history_max_tokens,
            )
        )
        log_event(
            logger,
            "media.context_focused",
            "Conversation focused on current media",
            model=request.model,
            history_token_budget=media_settings.media_history_max_tokens,
            kept_history_messages=kept_history_messages,
            dropped_history_messages=dropped_history_messages,
        )
    conversation = strip_gateway_metadata(conversation)

    if requires_gateway_media_preprocessing(
        backend,
        request_media,
        media_settings,
    ):
        media_kind = "pdf" if request_media.pdfs else (
            "video" if request_media.videos else "audio"
        )
        media_started_at = time.monotonic()
        try:
            pdf_embedding = None
            if media_settings.pdf_embedding_model and request_media.pdfs:
                embedding_backend = registry.get_backend(
                    media_settings.pdf_embedding_model
                )
                embedding_tokenizer = None
                if embedding_backend in {"vllm", "vllm_multimodal"}:
                    embedding_tokenizer, _ = await registry.get_tokenizer_async(
                        media_settings.pdf_embedding_model
                    )
                pdf_embedding = PdfEmbeddingContext(
                    model_name=media_settings.pdf_embedding_model,
                    tokenizer=embedding_tokenizer,
                    dimensions=media_settings.pdf_embedding_dimensions,
                )
            conversation = await prepare_vllm_media_conversation(
                request.model,
                model_path,
                tokenizer,
                conversation,
                request_media,
                sampling_parameters,
                settings=media_settings,
                pdf_embedding=pdf_embedding,
            )
        except VllmMediaProcessingError as exc:
            MEDIA_PROCESSING.labels(request.model, media_kind, "error").observe(
                time.monotonic() - media_started_at
            )
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        else:
            MEDIA_PROCESSING.labels(request.model, media_kind, "success").observe(
                time.monotonic() - media_started_at
            )

    conversation = add_system_instruction(
        conversation,
        tool_choice_instruction(
            request.tool_choice,
            has_tool_result=has_tool_result(conversation),
        )
        if tools
        else None,
    )
    media = extract_media_payloads(conversation)
    if backend == "vllm_multimodal" and media.has_any:
        media = await materialize_native_remote_media(model_path, media)
        native_settings = load_vllm_media_settings(model_path)
        media = replace(
            media,
            parameters={
                "image_max_pixels": native_settings.image_max_pixels,
                "video_fps": native_settings.video_fps,
                "video_max_frames": native_settings.video_max_frames,
                "video_max_pixels": native_settings.video_max_pixels,
                "pdf_dpi": native_settings.pdf_dpi,
                "pdf_max_pixels": native_settings.pdf_max_pixels,
            },
        )
    images = [payload.data for payload in media.images]
    reserved_media_tokens = _estimate_media_context_tokens(media, media_settings)
    conversation, prompt, prompt_tokens, dropped_context_messages = (
        fit_conversation_to_context(
            tokenizer,
            conversation,
            tools,
            media_settings.max_model_len,
            int(sampling_parameters.get("max_tokens") or 256),
            reserved_media_tokens=reserved_media_tokens,
        )
    )
    if dropped_context_messages:
        log_event(
            logger,
            "chat.context_trimmed",
            "Oldest conversation turns removed to fit model context",
            model=request.model,
            dropped_message_count=dropped_context_messages,
            prompt_tokens=prompt_tokens,
            max_model_len=media_settings.max_model_len,
            reserved_media_tokens=reserved_media_tokens,
        )

    log_event(
        logger,
        "chat.routed",
        "Routing chat request",
        model=request.model,
        backend=backend or "unknown",
        transport="grpc" if is_vllm_backend else "http",
        stream=bool(request.stream),
        tool_count=len(tools or []),
        image_count=len(request_media.images),
        video_count=len(request_media.videos),
        audio_count=len(request_media.audios),
        pdf_count=len(request_media.pdfs),
        media_formats=[
            payload.format
            for payloads in (
                request_media.images,
                request_media.videos,
                request_media.audios,
                request_media.pdfs,
            )
            for payload in payloads
        ],
        media_mime_types=[
            payload.mime_type
            for payloads in (
                request_media.images,
                request_media.videos,
                request_media.audios,
                request_media.pdfs,
            )
            for payload in payloads
        ],
        prompt_chars=len(prompt),
        prompt_tokens=prompt_tokens,
        reserved_media_tokens=reserved_media_tokens,
    )
    if LOG_PROMPT_PREVIEW:
        logger.debug(
            "Rendered prompt preview: %r",
            prompt[:500],
            extra={"event": "chat.prompt_preview"},
        )
    log_chat_prompt_debug(
        request,
        prompt,
        prompt_tokens=prompt_tokens,
        reserved_media_tokens=reserved_media_tokens,
    )

    if request.stream:
        if backend == "python":
            return StreamingResponse(
                _guard_openai_stream(stream_python_chat_to_openai(
                    request,
                    tokenizer,
                    prompt,
                    conversation,
                    sampling_parameters,
                    tools,
                    tool_parser,
                )),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if backend == "vllm_multimodal" and media.has_any:
            if tools:
                return StreamingResponse(
                    _guard_openai_stream(stream_tool_aware_native_multimodal_response(
                        request,
                        tokenizer,
                        prompt,
                        sampling_parameters,
                        media,
                        tools,
                        tool_parser,
                    )),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            return StreamingResponse(
                _guard_openai_stream(stream_triton_native_multimodal_to_openai(
                    request,
                    tokenizer,
                    prompt,
                    sampling_parameters,
                    media,
                )),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if tools and is_vllm_backend:
            return StreamingResponse(
                _guard_openai_stream(stream_tool_aware_multimodal_response(
                    request,
                    tokenizer,
                    prompt,
                    sampling_parameters,
                    images,
                    tools,
                    tool_parser,
                )),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if tools:
            return StreamingResponse(
                _guard_openai_stream(stream_tool_aware_response(
                    request,
                    tokenizer,
                    prompt,
                    sampling_parameters,
                    tools,
                    tool_parser,
                )),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        if is_vllm_backend:
            return StreamingResponse(
                _guard_openai_stream(stream_triton_multimodal_to_openai(
                    request,
                    tokenizer,
                    prompt,
                    sampling_parameters,
                    images,
                )),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return StreamingResponse(
            _guard_openai_stream(stream_triton_to_openai(
                request,
                tokenizer,
                prompt,
                sampling_parameters,
            )),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if backend == "python":
        generated_text = await call_triton_python_chat(
            request.model,
            prompt,
            conversation,
            sampling_parameters,
        )
    elif backend == "vllm_multimodal" and media.has_any:
        generated_text = await call_triton_native_multimodal(
            request.model,
            prompt,
            sampling_parameters,
            media,
        )
    elif is_vllm_backend or images:
        generated_text = await call_triton_multimodal(
            request.model,
            prompt,
            sampling_parameters,
            images,
        )
    else:
        generated_text = await call_triton(request.model, prompt, sampling_parameters)
    generated_text = strip_prompt_echo(prompt, generated_text)
    generated_text, _ = sanitize_generated_text(generated_text)
    tool_calls, remaining_text = (
        extract_tool_calls(generated_text, tools, tool_parser)
        if tools
        else ([], generated_text)
    )
    usage = build_usage(tokenizer, prompt, generated_text)
    finish_reason = "tool_calls" if tool_calls else "stop"
    log_chat_response_debug(
        request,
        generated_text,
        remaining_text,
        tool_calls,
        finish_reason,
    )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": remaining_text if remaining_text else "",
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }
