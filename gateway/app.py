# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from functools import wraps
import asyncio
import json
import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, StreamingResponse

from . import __author__, __repository__, __version__
from .admission import AdmissionLease
from .embeddings import (
    build_embedding_inputs,
    encode_embedding,
    load_hybrid_embedding_settings,
    reject_hybrid_options_on_dense_endpoint,
    tokenize_embedding_inputs,
    validate_hybrid_embedding_request,
)
from .generation_telemetry import (
    configure_generation_telemetry,
    get_generation_telemetry,
    observe_generation_usage,
)
from .metrics import EMBEDDING_BATCH_SIZE, RERANK_STRATEGY_SELECTIONS, SPARSE_EMBEDDING_SIZE, metrics_response
from .multimodal import extract_media_payloads, reclassify_media_content
from .observability import RequestContextMiddleware, log_event, set_request_model
from .prompt import build_conversation
from .rerank import (
    build_rerank_documents,
    build_rerank_response,
    iter_rerank_batches,
    plan_rerank_execution,
)
from .rerank_strategies import resolve_rerank_strategy
from .schemas import (
    ChatCompletionRequest,
    EmbeddingsRequest,
    HybridEmbeddingsRequest,
    RerankRequest,
)
from .settings import EMBEDDING_MAX_CONCURRENCY, TOKENIZER_PRELOAD, logger
from .triton_client import call_triton_embeddings, call_triton_hybrid_embeddings, call_triton_rerank, close_grpc_client, close_http_client, is_triton_ready, list_ready_triton_models
from .tracing import shutdown_tracing


from .generation import registry, admission, generate
from .generation_types import GenerationStream, chat_events, ManagedStreamingResponse


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
        shutdown_tracing()


app = FastAPI(
    title="Triton OpenAI Gateway",
    description="OpenAI-compatible API and multimodal orchestration for NVIDIA Triton.",
    version=__version__,
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
            configure_generation_telemetry(
                route=resolved_route,
                model=request.model,
                backend=registry.get_backend(request.model) or "unknown",
            )
            if telemetry := get_generation_telemetry():
                telemetry.admitted(lease.wait_seconds)
            try:
                response = await endpoint(request)
            except BaseException as exc:
                if telemetry := get_generation_telemetry():
                    telemetry.fail(exc)
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


async def _guard_openai_stream(source: AsyncIterator[str]) -> AsyncIterator[str]:
    try:
        async for event in source:
            yield event
    except HTTPException as exc:
        if telemetry := get_generation_telemetry():
            telemetry.fail(exc)
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
        if telemetry := get_generation_telemetry():
            telemetry.fail(exc)
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
    reject_hybrid_options_on_dense_endpoint(request)
    backend = registry.get_backend(request.model)
    if request.lora_name is not None and backend != "vllm_multimodal":
        raise HTTPException(
            status_code=400,
            detail="Embedding LoRA requires the bundled vllm_multimodal backend. "
            "Set backend: \"vllm_multimodal\" in config.pbtxt and configure "
            "enable_lora and multi_lora.json for this model.",
        )
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
                lora_name=request.lora_name,
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

    usage = {
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_prompt_tokens,
    }
    observe_generation_usage(usage)
    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": usage,
    }


@app.post("/v1/embeddings/hybrid", include_in_schema=False)
@app.post("/v1/hybrid_embeddings")
@admitted("embeddings")
async def create_hybrid_embeddings(request: HybridEmbeddingsRequest):
    backend = registry.get_backend(request.model)
    model_path = registry.resolve(request.model)
    settings = load_hybrid_embedding_settings(model_path)
    model_inputs = build_embedding_inputs(request)
    sparse_top_k = validate_hybrid_embedding_request(
        request,
        settings,
        len(model_inputs),
    )

    if backend == "vllm":
        raise HTTPException(
            status_code=400,
            detail=(
                "/v1/hybrid_embeddings is not supported by the stock Triton vLLM "
                "backend; use the bundled vllm_multimodal backend or the BGE-M3 "
                "Python backend"
            ),
        )

    output_types = list(request.output_types)
    encoding_format = request.encoding_format or "float"
    log_event(
        logger,
        "hybrid_embeddings.routed",
        "Routing hybrid embeddings request",
        model=request.model,
        backend=backend or "unknown",
        transport="grpc",
        output_types=output_types,
        sparse_top_k=sparse_top_k,
        input_count=len(model_inputs),
    )
    EMBEDDING_BATCH_SIZE.labels(request.model).observe(len(model_inputs))
    semaphore = asyncio.Semaphore(EMBEDDING_MAX_CONCURRENCY)

    async def embed(index: int, model_input):
        async with semaphore:
            result, prompt_tokens = await call_triton_hybrid_embeddings(
                request.model,
                model_input,
                request.dimensions,
                output_types,
                sparse_top_k,
            )
        return index, result, prompt_tokens

    results = await asyncio.gather(
        *(embed(index, model_input) for index, model_input in enumerate(model_inputs))
    )
    results.sort(key=lambda item: item[0])
    total_prompt_tokens = sum(item[2] for item in results)
    data = []
    for index, result, _ in results:
        item: dict[str, Any] = {
            "object": "hybrid_embedding",
            "index": index,
        }
        if "dense" in result:
            item["embedding"] = encode_embedding(
                result["dense"],
                encoding_format,
            )
        if "sparse" in result:
            item["sparse_embedding"] = result["sparse"]
            SPARSE_EMBEDDING_SIZE.labels(request.model).observe(
                len(result["sparse"]["indices"])
            )
        data.append(item)

    usage = {
        "prompt_tokens": total_prompt_tokens,
        "total_tokens": total_prompt_tokens,
    }
    observe_generation_usage(usage)
    return {
        "object": "hybrid_embedding.list",
        "data": data,
        "model": request.model,
        "output_types": output_types,
        "usage": usage,
    }


@app.post("/rerank")
@app.post("/v1/rerank")
@app.post("/v2/rerank")
@admitted("rerank")
async def rerank(request: RerankRequest):
    documents = build_rerank_documents(request)
    model_path = registry.resolve(request.model)
    execution = plan_rerank_execution(request, documents, model_path)
    strategy = resolve_rerank_strategy(request, model_path)
    RERANK_STRATEGY_SELECTIONS.labels(
        request.model,
        strategy.name,
        strategy.method,
        strategy.source,
    ).inc()
    log_event(
        logger,
        "rerank.routed",
        "Routing rerank request",
        model=request.model,
        backend=registry.get_backend(request.model) or "unknown",
        transport="http",
        strategy=strategy.name,
        strategy_method=strategy.method,
        strategy_source=strategy.source,
        strategy_version=strategy.version,
        candidate_count=len(documents),
        batch_size=execution.batch_size,
        batch_count=execution.batch_count,
        max_length=execution.max_length,
    )

    scores: list[float] = []
    for document_batch in iter_rerank_batches(documents, execution):
        scores.extend(
            await call_triton_rerank(
                request.model,
                request.query,
                document_batch,
                execution.max_length,
                min(execution.batch_size, len(document_batch)),
                bool(request.normalize),
            )
        )
    return build_rerank_response(
        request,
        documents,
        scores,
        strategy,
        execution,
    )


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, response: Response = None):
    result = await generate(request)
    if isinstance(result, GenerationStream):
        return ManagedStreamingResponse(
            _guard_openai_stream(chat_events(result)),
            media_type="text/event-stream", headers=result.headers,
            close=result.aclose,
        )
    if response is not None:
        response.headers.update(result.headers)
    return result.to_chat()
