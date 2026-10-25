# DeepSeek V4.1 Flash: opt-in text generation

Use the optional image described in [the build guide](../../docker/README.deepseek.md).
This example selects the isolated vLLM 0.32 CUDA 12.9 environment; Qwen and BGE
continue to select their existing backend. Text, reasoning and tools are supported
by the gateway profile. Media and embedding requests are rejected.

1. Mount a complete immutable local snapshot of `deepseek-ai/DeepSeek-V4.1-Flash`,
   revision `314113f8a7dadb388215396ba5fbee2c610a04d6`, read-only. Include the complete
   safetensors index/shard set, `config.json`, `tokenizer_config.json`,
   `tokenizer.json` and `chat_template.jinja`. HF snapshot links to local cache
   blobs are supported; mount their targets too. A Git LFS pointer is insufficient.
2. Copy this directory to the Triton model repository. Edit `1/model.json`'s
   absolute `model` path to match the snapshot mount. A separate absolute local
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

Startup/shutdown deadlines are Triton parameters, not engine keys. The startup
deadline includes the supported-task query. Cooperative startup failures close
the engine context and retain the first cause; late factories cannot announce
READY. A stuck native constructor cannot be killed safely from a Python thread:
the owning stub/process must be stopped by its supervisor. Bounded cleanup does
not prove that such a constructor left no resources behind.

For a local no-weight native failure/exit gate, run inside the optional image:

```sh
python3 /opt/tritonserver/tools/deepseek/smoke-deepseek-stub.py
```

Run it separately from a serving process. It requires the intended missing-file
preflight failure and Triton exit 1 without a reported SIGSEGV. This deliberately
does not reproduce a native vLLM constructor/destructor crash. Actual GPU model
startup, generation, shutdown and failed-constructor recovery remain deployment
qualification gates.
