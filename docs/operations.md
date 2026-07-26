# Operations

[Home](../README.md) / Operations

**Language:** English | [Русский](operations.ru.md)

This guide covers runtime checks and failure modes for the combined Triton and
OpenAI gateway image.

## Ports And Health

| Port | Service | Recommended exposure |
| --- | --- | --- |
| `8000` | Raw Triton HTTP and repository API | Internal operators only |
| `8001` | Raw Triton gRPC | Gateway and trusted clients only |
| `8002` | Triton, model, and optional GPU metrics | Monitoring network only |
| `8080` | OpenAI-compatible API and gateway metrics | Authenticated proxy |

Gateway probes:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

`/health` checks the FastAPI process. `/ready` returns success only when local
Triton reports ready. A model can still be loading when strict Triton readiness
is disabled; inspect the repository index for per-model state.

## Model Lifecycle

List repository state:

```bash
curl -sS http://127.0.0.1:8000/v2/repository/index
```

Load and unload explicitly:

```bash
curl -fsS -X POST \
  http://127.0.0.1:8000/v2/repository/models/MODEL_NAME/load

curl -fsS -X POST \
  http://127.0.0.1:8000/v2/repository/models/MODEL_NAME/unload
```

During S3 load, the watcher logs the temporary `model.json` update and active
symlink. After unload, the temporary directory disappears and the stale link is
removed.

## Metrics

### Gateway

`http://HOST:8080/metrics` exposes Prometheus series for:

- HTTP requests and latency;
- admission in-flight, queued, rejected, and wait time;
- Triton calls, duration, and active streams;
- tokenizer loading and cache behavior;
- media preprocessing and PDF embedding cache activity.

### Triton And vLLM

`http://HOST:8002/metrics` exposes Triton model metrics. The included
`vllm_multimodal` backend reports custom vLLM metrics by default; set
`REPORT_CUSTOM_METRICS=true` in `config.pbtxt` to make the intent explicit.

### GPU And MIG

Triton's embedded DCGM requires privileges that may conflict with a non-root pod.
Choose one mode:

- `triton.gpuMetrics.mode=builtin`: Triton runs as root with dropped Linux
  capabilities and publishes `nv_gpu_*` on port `8002`.
- `triton.gpuMetrics.mode=external`: Triton stays non-root and GPU/MIG data comes
  from NVIDIA DCGM Exporter on port `9400`.

Do not deploy a second exporter if NVIDIA GPU Operator already provides one.
The standalone chart in [`helm/dcgm-exporter`](../helm/dcgm-exporter) is for
clusters without an existing exporter or for a node-pinned test endpoint.

DCGM metrics describe a GPU or MIG device, not a model. They represent one model
only when that device is dedicated to the model for the measurement interval.

## Logging

JSON is the default:

```text
LOG_FORMAT=json
LOG_LEVEL=INFO
```

CEF and human-readable text are also available. Logs include a request ID,
model, route, transport, status, duration, and media counts where applicable.
The gateway returns the same ID in `X-Request-ID`; clients may supply their own
valid ID in the request header.

For one chat request:

```json
{
  "model": "MODEL_NAME",
  "messages": [{"role": "user", "content": "Diagnose this request"}],
  "debug": true
}
```

Safe debug mode logs roles, tool-call IDs, parser selection, token counts, and
finish reason. It does not log prompts or tool results. Enable
`DEBUG_LOG_PAYLOADS=true` only for controlled diagnostics because payload
previews may contain sensitive user data.

## Tracing

The Helm chart can enable Triton OpenTelemetry export:

```yaml
triton:
  tracing:
    enabled: true
    endpoint: http://otel-collector.observability.svc:4318/v1/traces
    level: TIMESTAMPS
    rate: 0
    count: -1
```

With `rate: 0`, Triton traces requests carrying a W3C trace context. Use a
positive sampling rate only after considering trace volume and collector load.
Metrics remain the primary source for capacity and saturation analysis.

## Common Failures

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Model exists in Triton but gateway says not found | Active symlink was not created | Check watcher logs, numeric version layout, and write access to the selected watcher root |
| `AsyncEngineArgs` rejects a key | `model.json` contains an argument unsupported by pinned vLLM | Remove or rename the key; keep gateway settings in `gateway.json` |
| `KIND_GPU is currently for single-GPU models` | A tensor-parallel engine uses Triton `KIND_GPU` | Use one `KIND_MODEL` instance and assign the required devices |
| Shared memory pool cannot grow | Container `/dev/shm` is too small | Increase Docker `--shm-size` or pod `/dev/shm` memory volume |
| Prompt exceeds context | Text plus media tokens plus output reservation exceed `max_model_len` | Reduce media resolution/chunk size, history, or requested output; increase context only if memory permits |
| Audio is rejected by a VL model | The architecture has no audio modality and no ASR is configured | Configure a local ASR model or use an audio-capable model |
| GPU series are absent on `:8002` | Embedded DCGM could not initialize | Use root built-in mode or an external DCGM Exporter |
| `pip` reports vLLM conflicts for `apache-tvm-ffi`, `openai`, or `pydantic` during the image build | The pinned NVIDIA `26.06` base image already contains these package-metadata mismatches | Keep the tested base digest and verified versions; do not upgrade core vLLM dependencies independently |
| Gateway returns `429` | In-flight and queue capacity are full | Retry with backoff or tune tested admission limits |
| Gateway returns `413` | Request body, media item, pages, pixels, frames, or audio duration exceeds a hard limit | Reduce the input or raise a specific limit after capacity testing |

## Capacity And Resource Guidance

- Size `/dev/shm` for concurrent Triton IPC payloads, especially large media.
- Treat `max_num_seqs` as a vLLM concurrency ceiling, not a guaranteed user
  count and not a GPU thread-block multiple.
- Keep media preprocessing concurrency lower than chat concurrency; video and
  PDF decoding consume CPU and host memory before GPU inference.
- Tune context length and KV cache together. A configured maximum context does
  not mean every concurrent request can occupy that full length.
- Use representative input/output lengths and media sizes for load tests.
- Dedicate GPU/MIG devices when model-level GPU attribution is required.
- Keep enough termination grace time for stream cancellation and engine cleanup.

## Production Checklist

- Put authentication, authorization, TLS, and tenant quotas in front of port
  `8080`.
- Keep raw Triton and metrics ports off public networks.
- Pin the built image by digest in deployment manifests.
- Store S3 credentials in Kubernetes Secrets or an external secret manager.
- Set request-body, remote-download, media, queue, and timeout limits.
- Keep private remote URL access disabled unless the trust boundary requires it.
- Use NetworkPolicy for Triton, metrics, object storage, and OTLP endpoints.
- Scrape gateway, Triton, vLLM, and DCGM metrics with alerts for errors, queue
  growth, latency, memory pressure, and model availability.
- Test load, unload, pod restart, client cancellation, and long media requests.
- Review `trust_remote_code=true` for every model; model repository contents are
  executable code when this option is enabled.
