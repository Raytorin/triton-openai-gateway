# DeepSeek V4.1 Flash: text and opt-in images

Use the optional image described in [the build guide](../../docker/README.deepseek.md).
This example selects the isolated vLLM 0.32 CUDA 12.9 environment; Qwen and BGE
continue to select their existing backend. Text, reasoning and tools are supported
by the gateway profile. This example defaults to text-only; images require the
explicit configuration below. Audio, video, PDF and embedding requests are rejected.

1. Provide a complete snapshot of `deepseek-ai/DeepSeek-V4.1-Flash`, revision
   `314113f8a7dadb388215396ba5fbee2c610a04d6`, through a read-only immutable local
   mount or a complete Triton S3 model version directory. Include the complete
   safetensors index/shard set, `config.json`, `tokenizer_config.json`,
   `tokenizer.json` and `chat_template.jinja`. HF snapshot links to local cache
   blobs are supported; mount their targets too. A Git LFS pointer is insufficient.
2. Copy this directory to the Triton model repository. Edit `1/model.json`'s
   absolute `model` path to match the snapshot mount, or use the same-model
   repository aliases described below for S3 loading. A separate absolute local
   `tokenizer` directory is allowed. Relative paths and remote Hub/URI loads are
   rejected, so preflight checks the same root the engine opens. `model_weights`
   does not select the engine's primary weight source. `revision` records supplied
   provenance; it does not authenticate local files.
3. Set `GPU_DEVICE_IDS` and `tensor_parallel_size` for the validated deployment.
   The example's TP=1/device 0 are editable placeholders, **not a fit claim**.
   Select `DEEPSEEK_CACHE_ARCH=smXX` using the actual validated GPU architecture,
   and mount a writable private `DEEPSEEK_CACHE_DIR` (default
   `/tmp/triton-runtime-cache/vllm032-cu129`). For example, pass
   `-e DEEPSEEK_CACHE_ARCH=<validated-sm-token> -e DEEPSEEK_CACHE_DIR=/runtime-cache`
   with a writable `/runtime-cache` mount. These are process environment settings,
   not model.json engine keys. Do not use `cpu`/`unqualified` for engine startup.
4. Run read-only structural admission before handing the directory to Triton:

   ```sh
   python3 scripts/validate_deepseek_checkpoint.py /models/DeepSeek-V4.1-Flash/1/model.json --json
   ```

   In the image use `/opt/tritonserver/tools/deepseek/validate_deepseek_checkpoint.py`.
   `--check-runtime` also checks native architecture/tokenizer registry membership;
   run it in the selected isolated environment. It does not inspect native model
   classes or launch engine workers. The backend repeats admission before
   constructing engine arguments/configuration. No network calls or tensor payload
   reads occur. Limits are local policy: JSON config 2 MiB, index 32 MiB,
   header 16 MiB per shard, 4096 shards, 1,000,000 tensors and nesting depth 64.
5. Query the selected model after loading, and require HTTP 200:

   ```sh
   curl --fail http://127.0.0.1:8000/v2/models/DeepSeek-V4.1-Flash/versions/1/ready
   ```

   Gateway `/health` and successful structural admission do not establish this
   model's READY state. Then qualify actual text/thinking/tools requests and
   cancellation through the gateway, and verify model unload/reload.

`lazy` loading avoids eagerly copying a complete shard into host memory. Engram
`cpu_offload:true` uses host/pinned memory. `serialized_bytes` and
`allocated_bytes` describe logical file size and filesystem allocation, including
sparse-file differences. Neither measures peak host RAM or per-rank GPU HBM.
Offloaded host tensors, loader buffers, GPU weights, KV cache and workspaces each
need a separate capacity check. Header validation proves neither file hashes,
content correctness, memory fit nor GPU kernel availability. FP4 expert packing
may be serialized as I8; logical parameter counts are not file-byte counts.

`language_model_only:true` with `enable_mm_embeds:false` disables vision tower
weights and vision profiling in the selected native implementation. Imports and
checkpoint weight iteration can still touch vision-related metadata/shards;
structural admission requires the **full** snapshot closure. Conservative context
and sequence settings do not establish GPU fit. Expert parallelism,
speculative decoding and external KV connectors are not enabled.

The temporary-download watcher preflights this backend before publishing its
link, preserves an explicit external absolute snapshot path, and retries failed
admission on later scans. Retired DeepSeek checkouts receive a local
`.deepseek-retired` marker and are retained to protect a stub still loading them;
they stay retired after watcher restarts. Remove those temporary checkouts only
after verifying the prior instance has unloaded. This is not an automatic READY
handoff. Direct Triton local/S3 repository loads still need a complete immutable
upload handoff; the watcher does not control them all.

## Enable images

For a checkpoint with the supported DeepSeek vision encoder/projector and
`vision_config`, opt in through `1/model.json`:

```json
{
  "language_model_only": false,
  "enable_mm_embeds": false,
  "limit_mm_per_prompt": {"image": 4}
}
```

These are changes to the corresponding keys in the full engine configuration,
not a replacement for the file. Set both `model` and `tokenizer` to the intended
local snapshot. A positive integer image limit is required. `enable_mm_embeds`
stays false: ordinary image pixels do not require accepting client-supplied
embedding tensors. Loading the vision tower needs a model reload and a separate
GPU capacity check; the text-only context/sequence configuration is not a fit
claim for vision.

Chat `image_url` and Responses `input_image` use the existing image transport
and the native vLLM DeepSeek processor. Prompt placeholders and pixels preserve
the same order, including reordered function results. The processor owns image
resizing, patch grids and placeholder expansion. Gateway context reservation
uses the checkpoint's validated image-token bound; this reservation is not
reported as consumed visual tokens. See the [visual usage documentation](../../docs/visual-usage.md) for the
distinction between processor facts and unavailable accounting.

## Model-local reasoning effort

The included `1/gateway.json` enables separate reasoning and maps the request
alias `xhigh` to this model's native `high` (75). `max` remains 100. The mapping
is local to the model: Qwen's native `xhigh` is not changed by this setting.

```json
{
  "reasoning_effort": "xhigh",
  "chat_template_kwargs": {"enable_thinking": true}
}
```

The selected native effort is rendered into the prompt sent to the backend.
It is not an extra vLLM sampling parameter. `reasoning.effort_map` performs one
lookup, validates the target against the selected template, and does not alter
template defaults when the request omits effort. Thinking must be enabled;
constrained JSON output continues to disable thinking.

The included engine `model.json` declares `<think>` / `</think>` boundaries.
With thinking enabled, add `"thinking_token_budget": 128` and a larger total
limit such as `"max_completion_tokens": 4096` to close reasoning at its budget
and continue the answer. This numeric budget is separate from effort; it still
consumes the common completion limit, including the closing tokens.

Startup/shutdown deadlines are Triton parameters, not engine keys. The startup
deadline includes the supported-task query. Cooperative startup failures close
the engine context and retain the first cause; late factories cannot announce
READY. A stuck native constructor cannot be killed safely from a Python thread:
the owning stub/process must be stopped by its supervisor. Bounded cleanup does
not prove that such a constructor left no resources behind.

## S3 repository loading

Triton downloads an S3 model version before backend initialization. For repository
loading, `model` and/or `tokenizer` may name `/tmp/models-active/<model-name>` or
`/tmp/models-active/<model-name>/<version>`. The name and version must match the
current Triton model. `MODELS_ACTIVE_DIR` selects a different active directory
when configured. Only exact aliases are accepted; other names/versions, extra
suffixes, traversal and escaping version-directory symlinks are rejected.

Both the watcher and the DeepSeek backend bind these aliases to the actual
materialized version directory before checkpoint admission. An existing active
symlink cannot select an older copy. The backend passes the admitted configuration
to engine arguments in memory; it does not rewrite checkpoint metadata. The watcher
atomically repairs its temporary JSON and publishes the active link only after
structural admission succeeds. Incomplete copies retain their original JSON and
are retried on later scans. Model and tokenizer are resolved independently;
explicit external snapshots/tokenizers keep their paths and fail if unavailable.

This avoids an additional manually staged checkpoint for normal S3 loading.
Keep `DEEPSEEK_CACHE_ARCH=sm90` in the h200 container environment before startup;
this is not a model.json engine key. S3 path repair and CPU admission do not
qualify GPU startup, inference, or full-context capacity.

## Checkpoint format and prompt token qualification

Structural admission recognizes `F8_E8M0` (UE8M0) scale tensors as one byte per
element and reports observed dtypes/counts and the declared quantization profile.
It still reads only headers; this does not prove that the weights match every
expected model parameter or fit in memory. `--check-runtime` additionally checks
native registry membership and CPU decoding of a tiny synthetic UE8M0 tensor
when that format occurs. Missing or incompatible runtime support is an error.
No checkpoint tensor payload is used by that probe, and it is not GPU qualification.
For UE8M0, select `safetensors_load_strategy: "lazy"` explicitly as in this example.
The runtime probe follows the chosen loader API: lazy uses `safe_open`, eager
uses byte deserialization. One decoder can support a dtype while another rejects
it; no fallback silently changes the loading strategy. Unqualified or implicit
strategies are rejected for UE8M0 until checked separately.

The example pins a specific checkpoint revision. Do not assume a newer
`DeepSeek-V4.1-Flash-1017` snapshot has the same tokenizer, template or weights.
Qualify each intended snapshot against its own reviewed reference.

For this example's pinned revision, the repository includes independent prompt
and token-ID goldens generated offline with its reviewed `encoding/encoding.py`
and raw tokenizer BPE. The reference records the encoder SHA256, four artifact
hashes, BOS/EOS/PAD IDs and 16 synthetic requests covering Unicode, thinking
efforts, history, tools and tool results. The `medium` alias uses the pinned
template's numeric budget of 62. The reference is operator-reviewed provenance,
not an authenticity signature.

With the tokenizer/config/template files available locally, run from the checkout:

```sh
python3 scripts/validate_deepseek_tokenizer.py /models/DeepSeek-V4.1-Flash/1/model.json \
  --model-config-dir /models/DeepSeek-V4.1-Flash/1 \
  --reference tests/fixtures/deepseek/v41-flash/token-reference.json --json
```

The tool uses the gateway's `TokenizersBackend` and prompt renderer with
`local_files_only=True` and `trust_remote_code=False`. It neither downloads files
nor executes the checkpoint encoder. It compares literal expected prompts/token
IDs, verifies artifact hashes before and after, and returns nonzero on mismatch.
`--model-config-dir` selects the serving directory's `gateway.json` for effort
aliases. Without it, qualification does not infer a mapping from checkpoint or
tokenizer directories; native template values remain available.
Reports contain counts and mismatch positions, not conversation text. This checks
prompt encoding only; it does not run context fitting, inference or GPU kernels.
The installed script is `/opt/tritonserver/tools/deepseek/validate_deepseek_tokenizer.py`;
provide the reviewed reference separately when running it in an image.

To include the real-tokenizer parity case in local tests, set
`DEEPSEEK_TOKENIZER_SNAPSHOT` to the absolute directory containing the four pinned
artifacts. Without it, that optional test is skipped; the ordinary CPU suite uses
explicitly synthetic token references. No model weight files are needed.

DeepSeek JSON responses use a separate content parser so literal `<think>`, DSML
and EOS strings remain JSON data. Normally stopped output must be valid finite
JSON without duplicate keys (`json_object` also requires an object). A genuine
engine `length` completion may retain partial JSON. For DeepSeek, structured
JSON with `tool_choice=required` or a named mandatory tool is rejected before
inference; optional `auto` and `none` remain available. CPU protocol tests verify
forwarding and parsing, not enforcement of JSON Schema by a real GPU decoder.

For a local no-weight native failure/exit gate, run inside the optional image:

```sh
python3 /opt/tritonserver/tools/deepseek/smoke-deepseek-stub.py
```

Run it separately from a serving process. It requires the intended missing-file
preflight failure and Triton exit 1 without a reported SIGSEGV. This deliberately
does not reproduce a native vLLM constructor/destructor crash. Actual GPU model
startup, generation, shutdown and failed-constructor recovery remain deployment
qualification gates.
