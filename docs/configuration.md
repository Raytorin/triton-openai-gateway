# Configuration

[Home](../README.md) / Configuration

**Language:** English | [Русский](configuration.ru.md)

Configuration is split by ownership. Triton reads `config.pbtxt`; vLLM reads
`model.json`; the gateway reads the optional `gateway.json`. Keeping those files
separate prevents gateway-only keys from being passed to vLLM engine arguments.

## Runtime Compatibility

The Dockerfile pins the NVIDIA Triton `26.09-vllm-python-py3` base image by
digest. The build verifies Triton client `2.73.0`, vLLM `0.29.0`, Transformers
`5.16.1`, Torch, FlashInfer, and all added media/runtime dependencies. Changing the base
image is an explicit compatibility migration, not a routine package upgrade.

Build with the pinned default:

```bash
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:26.09 .
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

The watcher retires replaced/unlinked versions and older, unpublished duplicates
of the same model, backend, and numeric version. Selection happens before path
repair and preserves the original `model.json` timestamp during repair, so a
rewritten copy cannot look newer, including after a watcher restart.
Retirement is recorded in `.watcher-retired.json`; these copies are not published
again after a watcher restart. Removal requires 300 seconds of unchanged file
metadata by default and is retried at most every 30 seconds. A resumed download
restarts the grace period. Active symlinks and process references through open
files, memory maps, or working directories defer removal; an unavailable process
inventory also defers it. Empty checkout metadata is collected after the same
grace period, without recursively deleting newly appeared versions.
Shared metadata is staged with a persistent journal before an empty checkout is
removed. If a download appears or cleanup is interrupted, the watcher restores
the metadata before model discovery, preserving any newer files written by the
downloader. If recovery must be deferred, cleanup pauses and copies without a
config are not published; unrelated models with valid configs remain available.

Other numeric versions, external checkpoints and symlink targets are preserved.
The existing conservative DeepSeek retirement policy is unchanged: shared
DeepSeek checkpoints require an operator-confirmed unload. Unidentifiable or
malformed configurations are retained; a repeated config failure is logged once
per file identity and retried when that file changes. The source S3 repository
is never modified. Retirement markers belong to local `folder*` checkouts and
must not be copied into a source model repository.

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
The gateway translates `json_object` and `json_schema` into Triton 26.09's
serialized vLLM `structured_outputs` parameter for both `vllm` and
`vllm_multimodal` backends.

For a structured-output request, the gateway disables model thinking for that
request so the schema constrains the visible `message.content`, not an internal
reasoning block. The model's configured reasoning policy remains unchanged for
ordinary chat requests. An explicit `chat_template_kwargs.enable_thinking: true`
with `json_object` or `json_schema` is rejected with HTTP 400; omitting it or
setting it to `false` retains the non-thinking structured-output path.

No extra engine setting is required in the stock image. vLLM `0.29.0` uses the
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
| `embeddings.hybrid` | Dense/sparse output types, batch limit, and sparse-vector limits |

The `pdf_rag.embedding_model` value must name an embedding model already loaded
in the same Triton server. Retrieval is used only when enough text can be
extracted from the PDF.

### Hybrid And Sparse Embeddings

The standard `POST /v1/embeddings` contract remains unchanged and returns a
dense vector. A model explicitly enables lexical sparse or combined output in
its `gateway.json`:

```json
{
  "embeddings": {
    "hybrid": {
      "enabled": true,
      "output_types": ["dense", "sparse"],
      "max_batch_size": 32,
      "default_sparse_top_k": null,
      "max_sparse_top_k": 8192
    }
  }
}
```

Use `POST /v1/hybrid_embeddings` with `output_types: ["dense", "sparse"]`.
The response keeps the dense vector in `embedding` and returns a compact sparse
vector as parallel `sparse_embedding.indices` and
`sparse_embedding.values` arrays. `sparse_top_k` can cap its non-zero entries.
The gateway rejects this route for the stock Triton `vllm` backend because its
request adapter returns dense vectors only. The bundled `vllm_multimodal`
backend supports BGE-M3 natively and selects vLLM `embed`, `token_classify`, or
`embed&token_classify` pooling for each request.

The legacy `/v1/embeddings/hybrid` alias remains available for direct gateway
clients. Do not use it as a LiteLLM pass-through target: LiteLLM 1.97 treats
URLs containing `/v1/embed` as Cohere routes while processing background logs.

The recommended
[`vllm_multimodal` profile](../examples/bge-m3-vllm-multimodal/README.md)
uses vLLM scheduling and loads `sparse_linear.pt` and `colbert_linear.pt`
through its native `BgeM3EmbeddingModel`. Its `model.json` must set
`runner: "pooling"` and override the upstream architecture name. Do not fix a
single `pooler_config.task`, because the backend chooses the task per request.

The separate [Python fallback](../examples/bge-m3-hybrid/README.md) loads the
same local encoder and sparse head with Transformers. Both profiles are fully
offline and do not contact Hugging Face at runtime.

An offline LiteLLM deployment can expose the custom route with the exact
`pass_through_endpoints` entry in
[`examples/litellm.config.yaml`](../examples/litellm.config.yaml). LiteLLM
authenticates the request and forwards its JSON and response without converting
the sparse payload. Internet access is not involved. An OpenAI Python client
configured with the LiteLLM `base_url` continues to work for dense
`embeddings.create()` calls; its typed embeddings method always targets the
standard endpoint, so call the custom hybrid route with an ordinary HTTP POST.

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
`finish_reason: "length"`. An incomplete reasoning block or empty text alone
does not establish that the output budget was exhausted. When available, the
engine finish reason takes precedence over text-based heuristics.

`parser: "auto"` detects supported Qwen chat templates. Explicit
`qwen3` and `think_tags` parsers are also available. Use
`response_field: "reasoning_content"` for LiteLLM/OpenAI-compatible clients or
`"reasoning"` for clients following the vLLM schema.

See [`examples/gateway.reasoning.json`](../examples/gateway.reasoning.json).

#### Per-request Qwen thinking

Only `/v1/chat/completions` accepts the following generation controls:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "preserve_thinking": false
  }
}
```

`enable_thinking` selects whether the template generates a thinking turn.
`preserve_thinking` controls retention of earlier reasoning only for templates
that implement it; it does not enable thinking. Both fields require JSON
booleans (`true`/`false`), not strings or integers. `include_reasoning` controls
response visibility only: hiding reasoning does not turn generation off.

Server policy resolves environment variables before model `gateway.json`,
then defaults to `disabled` when neither is set. Missing request controls leave
that policy and the template's defaults in effect. `disabled` is a hard ban:
explicit `enable_thinking: true` fails with HTTP 400. `hidden` cannot be
revealed by `include_reasoning: true`. Under `separate`, a supporting switchable
Qwen template accepts both thinking and non-thinking requests for the same
loaded model. Request controls never change shared model/tokenizer state.

Capabilities are validated against the actual resolved chat template's
fingerprint and profile, not just the model name. A fixed non-thinking template
accepts `false` and rejects `true`; a fixed thinking template accepts `true`
when server policy permits it and rejects `false`. Unknown keys or invalid
subfield types in `chat_template_kwargs` fail with HTTP 422; an unsupported
control or value fails with HTTP 400 before inference.

Top-level `reasoning_effort` is accepted only when the template consumes it,
using that template's allowed values. `high` and `xhigh` are distinct and are
never globally aliased. For example, the supported Qwen3.8 template accepts
`low`, `medium`, and `xhigh`, with a template default of `xhigh`; older Qwen
templates may not consume effort at all. The Qwen3.6 examples intentionally
omit effort. `preserve_thinking` likewise requires template support.

The Responses API contract is unchanged and does not accept these reasoning
controls. Chat sampling also accepts `top_k` (`-1` or a positive integer) and
`presence_penalty` (from `-2` to `2`); these are passed to generation.

See the [paired Qwen3.6 requests](../examples/REQUEST_EXAMPLES.en.md#qwen36-27b-fp8-thinking-on-and-off)
for generation on/off and visibility examples. Local template and request
validation does not establish GPU performance or model quality.

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
| `WATCHER_CLEANUP_GRACE_SECONDS` | `300` | Unchanged-file grace period before collecting retired local copies |
| `WATCHER_CLEANUP_INTERVAL_SECONDS` | `30` | Minimum interval between cleanup attempts for one copy |
| `WATCHER_PROC_ROOT` | `/proc` | Process inventory mount used to protect in-use copies; use a complete container inventory |
| `GATEWAY_PORT` | `8080` | FastAPI listen port |
| `REQUEST_TIMEOUT_SECONDS` | `600` | Upstream request timeout |
| `GATEWAY_MAX_REQUEST_BODY_BYTES` | `268435456` | Maximum HTTP request body |
| `TOKENIZER_PRELOAD` | `true` | Preload active tokenizers at startup |
| `TOKENIZER_TRUST_REMOTE_CODE` | `true` | Allow model tokenizer remote code |

### Gateway Watchdog

A background supervisor probes the local `/health` endpoint, which does not call
Triton. After three consecutive failures it sends SIGTERM to Uvicorn, waits up to
10 seconds, uses SIGKILL if necessary, and starts a new process. A crashed process
is restarted without waiting for three probes. Startup allows 60 seconds until
the first successful response; a single failed probe does not trigger a restart.
When previously live Triton stops responding, the supervisor stops the gateway
and exits so container shutdown can finish. Restarting interrupts in-flight gateway requests.

| Variable | Default | Purpose |
| --- | --- | --- |
| `GATEWAY_HEALTH_INTERVAL_SECONDS` | `1` | Probe interval |
| `GATEWAY_HEALTH_TIMEOUT_SECONDS` | `1` | Per-request HTTP timeout |
| `GATEWAY_HEALTH_FAILURE_THRESHOLD` | `3` | Consecutive failure threshold |
| `GATEWAY_STARTUP_GRACE_SECONDS` | `60` | Startup grace until first successful response |
| `GATEWAY_STOP_GRACE_SECONDS` | `10` | Shutdown grace after SIGTERM |
| `GATEWAY_RESTART_DELAY_SECONDS` | `2` | Delay before restarting |
| `GATEWAY_RUN_DIR` | `/tmp/triton-gateway-<uid>` | Private supervisor/CLI state directory; same OS user, owner-only mode `0700` |

Use `triton-gateway status` and `triton-gateway restart` inside the container to
inspect or restart only the gateway. See [gateway process control](operations.md#gateway-status-and-restart)
for deadlines and exit codes.

### CPU Preparation

Chat and Responses render templates, tokenize prompts, fit context, and count
output usage in a bounded worker pool. The pool also handles CPU portions of
summary preparation; inference remains asynchronous. Dense embeddings tokenize
inputs in this pool. Dense and hybrid embedding responses are encoded one vector
at a time in the pool, without a second traversal by FastAPI on the event loop.
`GATEWAY_EMBEDDING_MAX_CONCURRENCY` (default `8`) bounds both inference calls and
worker tasks within each embedding request. A failed or cancelled batch drains
its outstanding backend calls before releasing admission capacity.

Tokenizer loading is shared by concurrent requests for the same model. Loading
another model does not hold a registry-wide cache lock. Cold loading can still
contend for Python's GIL; `TOKENIZER_PRELOAD=true` starts loading at startup.

| Environment variable | Default | Meaning |
| --- | ---: | --- |
| `GATEWAY_CPU_WORKERS` | `4` | Maximum concurrent CPU work items per gateway process |
| `GATEWAY_CPU_MAX_QUEUE` | `64` | Maximum waiting CPU work items; `0` disables waiting |
| `GATEWAY_CPU_QUEUE_TIMEOUT_SECONDS` | `30` | Maximum wait for a CPU slot |

A full queue or an expired wait returns HTTP `429` with `Retry-After: 1` before
streaming starts. An error after streaming starts follows the API's stream error
contract. On cancellation, queued work is removed; running preparation stops
between tokenizer calls. CPU and admission slots remain occupied until the
current native call finishes. Increasing Uvicorn workers multiplies these
process-local limits as well as the admission limits.

Monitor `triton_gateway_cpu_active`, `triton_gateway_cpu_queued`,
`triton_gateway_cpu_rejected_total`, `triton_gateway_cpu_wait_seconds`, and
`triton_gateway_cpu_duration_seconds`. CPU metrics contain no request or model IDs.

### Admission Control

A request receives its global and model/route inflight slots together. Waiting
for a busy model does not consume global inflight capacity needed by other
models. Every waiting request counts toward both queue limits and uses the
shorter of their queue timeouts. Queue overflow or expiry returns HTTP `429`.


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

### Output token budgets

Chat uses `max_completion_tokens` (or legacy `max_tokens`). Equal aliases are
accepted; conflicting values return 400. Limits must be positive integers.
Omitted/null limits select 4096 tokens normally or 8192 with thinking enabled,
reduced to the available context without discarding an already fitting history.
Explicit limits are never silently clamped. Thinking and function arguments
count toward the same output budget. Prompt and completion totals use cumulative
engine counts when the backend supports generation metadata; otherwise the
gateway estimates them by tokenization. Reasoning segmentation remains estimated.

Global environment defaults: `GATEWAY_DEFAULT_OUTPUT_TOKENS=4096`,
`GATEWAY_REASONING_DEFAULT_OUTPUT_TOKENS=8192`, `GATEWAY_MAX_OUTPUT_TOKENS=32768`.
Configure a model in `gateway.json` (not vLLM engine arguments):

```json
{"generation":{"default_output_tokens":4096,"reasoning_default_output_tokens":8192,"max_output_tokens":32768}}
```

The model cap cannot exceed the global cap. Automatic defaults are bounded by the cap.
Set both defaults to 256 to restore the old fallback. A positive runtime
`model.json.max_model_len` is required. If absent/automatic, explicitly set a
conservative `generation.context_window`; it never expands a known runtime limit.
Do not use tokenizer sentinel values as runtime context limits. The existing
context safety margin applies. Internal summary calls retain their own budgets.
`X-Output-Token-Limit`, `X-Output-Token-Limit-Source` and `X-Token-Usage-Source`
report the decision; structured logs include the input/media/context reserves.
`X-Token-Usage-Source: backend-preferred` describes this preference/fallback policy,
not a guarantee that every backend supplies every counter. The pre-inference
budget log still describes an estimate of the input size.

The shared vLLM backend advertises optional `return_generation_metadata` (BOOL)
and `generation_metadata` (BYTES) tensors. The version-1 JSON envelope carries
`finished`, `prompt_tokens` and per-choice engine finish/stop reasons, cumulative
completion counts and first/last token IDs. Existing `num_output_tokens` remains
a delta counter. Support is discovered before inference and cached for at most
30 seconds; generation is never retried to negotiate the extension. Legacy
backends receive the original request shape. Abort/error, malformed metadata or
a missing advertised terminal result become upstream errors, not successful
`length` completions.
