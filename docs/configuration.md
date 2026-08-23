# Configuration

[Home](../README.md) / Configuration

**Language:** English | [Русский](configuration.ru.md)

Configuration is split by ownership. Triton reads `config.pbtxt`; vLLM reads
`model.json`; the gateway reads the optional `gateway.json`. Keeping those files
separate prevents gateway-only keys from being passed to vLLM engine arguments.

## Runtime Compatibility

The Dockerfile pins the NVIDIA Triton `26.07-vllm-python-py3` base image by
digest. The build verifies Triton client `2.71.0`, vLLM `0.24.0`, Transformers
`5.6.1`, Torch, FlashInfer, and all added media/runtime dependencies. Changing the base
image is an explicit compatibility migration, not a routine package upgrade.

Build with the pinned default:

```bash
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:26.07 .
```

Override the base only when you have validated the complete runtime matrix:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:TAG \
  -f Dockerfile.triton-gateway \
  -t triton-openai-gateway:custom .
```

`docker/verify-runtime.py` intentionally fails the build when a package version
does not match the tested matrix.

## Model Repository Layout

```text
MODEL_NAME/
├── config.pbtxt
└── 1/
    ├── model.json
    ├── gateway.json              # optional
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── weights...
```

The gateway reads tokenizer and model metadata from the active numeric version.
For the S3 repository flow, the watcher creates a stable link under
`<watcher-root>/models-active/MODEL_NAME` after Triton has materialized the
version. The root follows `WATCHER_MODEL_DIR`, `TMP_ROOT`, Triton's
`TMPDIR`, then `/tmp`, in that order.

## `config.pbtxt`

Use Triton `KIND_MODEL` for a vLLM engine that owns multiple GPUs. A minimal
custom multimodal configuration is included in
[`examples/vllm-multimodal-config.pbtxt`](../examples/vllm-multimodal-config.pbtxt):

```protobuf
name: "Qwen3-VL-Example"
backend: "vllm_multimodal"
max_batch_size: 0

model_transaction_policy { decoupled: true }

instance_group [{ kind: KIND_MODEL count: 1 }]

parameters [
  { key: "GPU_DEVICE_IDS" value: { string_value: "0,1" } },
  { key: "REPORT_CUSTOM_METRICS" value: { string_value: "true" } }
]
```

Use `backend: "vllm"` for the stock backend. `GPU_DEVICE_IDS` selects devices
visible inside the container; `tensor_parallel_size` in `model.json` must match
the number of GPUs assigned to that engine. Do not create one Triton instance
per GPU for a single tensor-parallel engine.

Python embedding and rerank models define their own input/output tensors and do
not use the vLLM example above.

## `model.json`

`model.json` is passed to vLLM `AsyncEngineArgs`. Use only arguments supported
by the pinned vLLM version:

```json
{
  "gpu_memory_utilization": 0.8,
  "max_model_len": 16384,
  "tensor_parallel_size": 2,
  "dtype": "auto",
  "trust_remote_code": true,
  "max_num_seqs": 16,
  "limit_mm_per_prompt": {
    "image": 8,
    "video": 1,
    "audio": 1
  }
}
```

The watcher writes the temporary local `model` path at load time. For GGUF,
set `load_format` to `gguf` and place the `.gguf` file in the version directory;
the watcher writes the exact file path.

Engine capacity values are model- and hardware-specific. Validate
`gpu_memory_utilization`, context length, parallelism, KV cache dtype,
`max_num_seqs`, and multimodal limits under representative load rather than
copying the example unchanged.

### Structured Output

Clients request constrained JSON through the OpenAI `response_format` field.
The gateway translates `json_object` and `json_schema` into Triton 26.07's
serialized vLLM `structured_outputs` parameter for both `vllm` and
`vllm_multimodal` backends.

No extra engine setting is required in the stock image. vLLM `0.24.0` uses the
`auto` backend by default and the pinned Triton image includes `xgrammar`. To
make the selection explicit, add one of these blocks to `model.json`:

```json
{
  "structured_outputs_config": {
    "backend": "auto"
  }
}
```

Use `"backend": "xgrammar"` to force the bundled implementation. Do not add
the removed request option `guided_decoding_backend`, and do not select
`guidance` unless you build and validate a custom image containing that
dependency.

## `gateway.json`

`gateway.json` is optional and belongs next to `model.json`. The complete
example is [`examples/gateway.vllm-multimodal.json`](../examples/gateway.vllm-multimodal.json).

```json
{
  "admission": {
    "chat": {
      "max_inflight": 64,
      "max_queue": 256,
      "queue_timeout_seconds": 30
    },
    "media": {
      "max_inflight": 4,
      "max_queue": 16,
      "queue_timeout_seconds": 30
    }
  },
  "vllm_multimodal": {
    "media_history_mode": "latest",
    "focus_current_media": true,
    "media_history_max_tokens": 512,
    "pdf_chunk_pages": 2,
    "video_fps": 1.0,
    "video_max_frames": 32,
    "audio_asr_model": ""
  }
}
```

Important groups:

| Group | Purpose |
| --- | --- |
| `admission` | Per-model route concurrency, queue size, and timeout |
| `media_history_*` | How prior media and text history are retained |
| `pdf_*` | PDF extraction mode, rendering, chunking, and reduce limits |
| `pdf_rag` | Optional embedding model and retrieval parameters |
| `video_*` | Sampling FPS, frame count, pixel budget, and chunk size |
| `audio_*` | Local ASR model, device, and chunk overlap |
| `max_remote_media_bytes` | Download limit for remote content |
| `context_compression` | Context overflow, rolling-summary, and fallback policy |
| `rerank` | Execution limits, post-score strategies, and optional SQLite source |
| `reasoning` | Thinking mode, output parser, and OpenAI-compatible response field |

The `pdf_rag.embedding_model` value must name an embedding model already loaded
in the same Triton server. Retrieval is used only when enough text can be
extracted from the PDF.

### Context Compression

Context handling is configured per chat model. The default `truncate` mode
preserves the previous behavior; `disabled` returns HTTP `400` on overflow;
`summarize` replaces the oldest complete turns with a rolling summary while
preserving system messages, recent turns, and the current user request.

```json
{
  "context_compression": {
    "mode": "summarize",
    "fallback_mode": "truncate",
    "summary_model": "",
    "summary_max_tokens": 256,
    "summary_input_max_tokens": 4096,
    "preserve_recent_messages": 4,
    "max_summary_calls": 4,
    "summary_timeout_seconds": 120,
    "summary_temperature": 0.0,
    "cache_size": 256,
    "version": "structured-memory-v2",
    "safety_margin_tokens": 64,
    "trigger_ratio": 0.8,
    "target_ratio": 0.6,
    "evidence_max_tokens": 384
  }
}
```

In `summarize` mode, compaction starts when the rendered prompt reaches
`trigger_ratio` of the available prompt budget and aims for `target_ratio`.
This proactive margin prevents every subsequent turn from immediately invoking
another summary. `evidence_max_tokens` reserves verbatim, redacted evidence for
important paths, identifiers, numbers, tool results, and corrected values that
an abstractive summary could lose.

An empty `summary_model` uses the requested chat model; otherwise it must name
another loaded chat model. Internal summary calls have separate Prometheus
counters and are not added to the client response's `usage`. Non-streaming
responses expose `context_status`; streaming responses expose equivalent
`X-Context-*` headers.

See
[`examples/gateway.context-compression.json`](../examples/gateway.context-compression.json)
for all commonly used settings.

### Rerank Selection

The rerank model always scores every supplied document first. The gateway then
sorts the scores and applies the selected post-processing strategy. Existing
requests without `selection` retain the `top_n` behavior.

```json
{
  "rerank": {
    "default_strategy": "top_n",
    "execution": {
      "max_documents_per_request": 256,
      "default_batch_size": 4,
      "max_batch_size": 8,
      "max_batch_tokens": 8192,
      "default_max_length": 512,
      "max_length": 8192
    },
    "strategies": {
      "strict": {
        "method": "top_n_and_threshold",
        "parameters": {
          "score_threshold": 0.5,
          "top_n": 5
        },
        "allow_request_parameters": ["top_n"],
        "version": "1"
      }
    }
  }
}
```

The `execution` block bounds peak reranker memory. The gateway splits one
request into sequential micro-batches, preserves document indices, and merges
all scores before sorting. Effective batch size is also bounded by
`max_batch_tokens / max_length`, so longer pairs automatically reduce GPU
concurrency.

Client values above `max_batch_size` or `max_length` return HTTP `400`; more
than `max_documents_per_request` documents returns HTTP `413`. The default
rerank admission policy allows one active request and 64 queued requests per
model. Increase `admission.rerank.max_inflight` only after validating the
specific reranker, GPU, and Triton model instance count under load.

Clients select a named policy with
`"selection": {"strategy": "strict", "parameters": {"top_n": 2}}`.
Built-in methods include `top_n`, `score_threshold`,
`top_n_and_threshold`, `metadata_filter`, and `diversity`. Strategies can
also be refreshed from a read-only SQLite database configured under
`rerank.database`; clients cannot submit executable code or SQL.

See [`examples/gateway.rerank.json`](../examples/gateway.rerank.json) for the
complete static and SQLite configuration.

### Reasoning Output

Reasoning policy is disabled by default and configured per model:

```json
{
  "reasoning": {
    "mode": "separate",
    "parser": "auto",
    "response_field": "reasoning_content"
  }
}
```

`disabled` requests a non-thinking chat template and strips accidental
reasoning blocks. `hidden` allows model thinking but omits the text from the
client response. `separate` returns reasoning independently from final
`content` in both JSON and SSE. A client may downgrade `separate` to
`hidden` for one request with `"include_reasoning": false`, but cannot
elevate a stricter server policy.

Hiding reasoning does not stop the model from generating it. Reasoning and the
final answer share `max_tokens`; if the model consumes the complete budget
before producing final content, the gateway returns empty `content` with
`finish_reason: "length"`.

`parser: "auto"` detects supported Qwen chat templates. Explicit
`qwen3` and `think_tags` parsers are also available. Use
`response_field: "reasoning_content"` for LiteLLM/OpenAI-compatible clients or
`"reasoning"` for clients following the vLLM schema.

See [`examples/gateway.reasoning.json`](../examples/gateway.reasoning.json).

## Environment Variables

Helm values are the preferred Kubernetes interface. Direct container users can
set the corresponding environment variables.

### Gateway And Transport

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRITON_BASE_URL` | `http://127.0.0.1:8000` | Triton HTTP endpoint |
| `TRITON_GRPC_URL` | `127.0.0.1:8001` | Triton gRPC endpoint |
| `WATCHER_MODEL_DIR` | unset | Explicit local directory containing Triton's temporary `folder*` checkouts |
| `TMP_ROOT` | unset | Backward-compatible watcher root alias |
| `TMPDIR` | system value | Triton temporary directory and automatic watcher fallback |
| `MODELS_ACTIVE_DIR` | `<watcher-root>/models-active` | Stable active-model links |
| `GATEWAY_PORT` | `8080` | FastAPI listen port |
| `REQUEST_TIMEOUT_SECONDS` | `600` | Upstream request timeout |
| `GATEWAY_MAX_REQUEST_BODY_BYTES` | `268435456` | Maximum HTTP request body |
| `TOKENIZER_PRELOAD` | `true` | Preload active tokenizers at startup |
| `TOKENIZER_TRUST_REMOTE_CODE` | `true` | Allow model tokenizer remote code |

### Admission Control

Global defaults use `GATEWAY_MAX_INFLIGHT_REQUESTS`,
`GATEWAY_MAX_QUEUE_SIZE`, and `GATEWAY_QUEUE_TIMEOUT_SECONDS`. Route overrides
use `GATEWAY_CHAT_*`, `GATEWAY_MEDIA_*`, `GATEWAY_EMBEDDINGS_*`, and
`GATEWAY_RERANK_*`. Per-model `gateway.json` takes precedence for configured
routes.

### Logging And Debugging

| Variable | Default | Purpose |
| --- | --- | --- |
| `LOG_FORMAT` | `json` | `json`, `cef`, or `text` |
| `LOG_LEVEL` | `INFO` | Python log level |
| `GATEWAY_DEBUG` | `false` | Safe debug metadata for all chat requests |
| `DEBUG_LOG_PAYLOADS` | `false` | Include bounded payload previews; may expose user data |
| `DEBUG_PREVIEW_CHARS` | `2000` | Debug preview limit |
| `UVICORN_ACCESS_LOG` | `false` | Uvicorn access log |

A client can set `"debug": true` on one chat request without enabling global
debug mode.

### Media

Stock backend preprocessing uses `VLLM_MEDIA_*` variables. Native backend hard
limits use `VLLM_MULTIMODAL_*`. The authoritative list and defaults are exposed
as `gateway.vllmMultimodal` and `gateway.nativeBackend` in
[`helm/triton-gateway/values.yaml`](../helm/triton-gateway/values.yaml).

For temporary media on a dedicated volume, set both `TRITON_MEDIA_DIR` and
`TRITON_MULTIMODAL_TMPDIR`. Temporary files are deleted after processing; this
volume is not an archive.

## Helm

The chart supports:

- S3 model repositories and credential Secrets;
- physical GPU and MIG resource requests;
- model and temporary media PVCs;
- built-in or external DCGM metrics;
- Triton OTLP tracing;
- probes, security contexts, scheduling, and topology settings;
- bounded gateway and backend concurrency.

Start from [`helm/triton-gateway/values.yaml`](../helm/triton-gateway/values.yaml)
and keep environment-specific values in a separate file:

```bash
helm upgrade --install triton-openai-gateway ./helm/triton-gateway \
  --namespace inference --create-namespace \
  --values values.production.yaml
```

See the [chart README](../helm/triton-gateway/README.md) for focused deployment
examples.
