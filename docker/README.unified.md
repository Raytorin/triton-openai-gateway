# Runtime tracks and the unified vLLM 0.32 image

[Home](../README.md) / Runtime tracks

**Language:** English | [Русский](README.unified.ru.md)

`Dockerfile.triton-gateway` remains the default, maintained official/legacy
track. `Dockerfile.triton-gateway-unified` is a separate, explicit choice for a
shared Python 3.12 / vLLM 0.32 / CUDA 12.9 execution environment. Both recipes
copy the current gateway and custom backend code from the checkout.

| Recipe | Runtime and role |
| --- | --- |
| `Dockerfile.triton-gateway` | Digest-pinned NVIDIA Triton `26.09-vllm-python-py3`, NVIDIA vLLM `0.29`; default official/legacy track |
| `Dockerfile.triton-gateway-deepseek` | Official track plus an independently locked vLLM `0.32` environment used by `vllm_deepseek`; also supplies the runtime artifact for the unified build |
| `Dockerfile.triton-gateway-unified` | Clean final OS stage with release-matched native Triton components, the separate vLLM `0.32` environment, and a hash-pinned gateway overlay |

The additive DeepSeek recipe remains available. Its default-model isolation and
checkpoint admission rules are described in [the DeepSeek guide](README.deepseek.md).
The unified image copies selected runtime artifacts from it rather than
inheriting its final filesystem or the older heavyweight Python inference stack.
The CUDA 12.9 engine toolkit does not replace the native Triton core's CUDA 13
ABI or driver requirements.

## Build locally

Build from the same checkout for all three steps. The optional environment and
unified image target `linux/amd64`. Model weights are not included.

1. Build the default official/legacy image:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway \
     -t triton-openai-gateway:official-local .
   ```

2. If a verified runtime artifact is not already available, build the additive
   DeepSeek image:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway-deepseek \
     --build-arg GATEWAY_IMAGE=triton-openai-gateway:official-local \
     -t triton-openai-gateway:deepseek-local .
   ```

3. Explicitly select the unified final image:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway-unified \
     --build-arg RUNTIME_IMAGE=triton-openai-gateway:deepseek-local \
     -t triton-openai-gateway:unified-vllm032-local .
   ```

`RUNTIME_IMAGE` defaults to `triton-openai-gateway:deepseek-local`. Supply the
recorded image ID or immutable digest when reproducing an artifact. Keep the
source checkout and runtime input records together; a mutable tag alone does
not identify the runtime. Build verification failures block use of that artifact.

The engine lock and provenance remain in `docker/runtime-locks/vllm032-cu129.*`.
The gateway overlay has its own `docker/runtime-locks/unified-vllm032.*` files.
Do not merge the official track's exact dependency pins into the new environment.
The overlay retains its newer Pydantic and NumPy while reconciling
`tritonclient==2.73.0` with `protobuf==6.33.6` and `ml-dtypes==0.5.4`.

## Runtime selection and compatibility

The unified image selects `/opt/envs/vllm032-cu129` for the gateway, watcher,
stock `vllm`, included `vllm_multimodal`, and generic Python backend launch paths.
Backend launchers select Python, native library paths, compiler paths, and caches
before executing the release-matched native Python stub. The native stub remains
a separate executable. Model backend names and the model repository layout stay
the same; selecting the new environment does not rewrite model configurations.

Generic Python/vLLM launches do not apply DeepSeek-only checkpoint or task
restrictions. Models selecting `vllm_deepseek` retain that backend's explicit
checkpoint admission and generation-only task contract, including explicitly
enabled checkpoint-native images. Gateway APIs, shared
multimodal processing, pooling/rerank routes, and existing model configurations
are compatibility targets; model support and engine arguments must still be
checked against the new vLLM version.

NVIDIA's manual Python frontends remain on the official/legacy track. Their
Pydantic `2.10.6` and NumPy `1.26.4` constraints conflict with the new environment;
the unified image does not claim to provide those frontends. See the
[official runtime migration guide](../docs/migration-26.09.md) for that track's
matrix and known metadata mismatches.

No GPU inference qualification has been completed for the unified image.
Build-time dependency checks, CPU imports, spawned-worker checks, and native stub
startup cover different failure modes and do not establish model correctness,
GPU compatibility, memory capacity, throughput, or latency. No size or
performance improvement is claimed here.

Local ASR keeps NVIDIA's `/opt/ffmpeg-safe` build and its codec/licensing
restrictions. The common overlay also includes SoundFile and SoxR for native
audio input and resampling. Writable JIT caches must allow loading compiled
objects; a `noexec` cache mount can prevent runtime compilation.

## Validate and choose an image

Run the no-weight CPU checks against the actual image from the repository root:

```sh
docker run --rm --network none --runtime runc --user 1000:1000 \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777,size=2g \
  --shm-size=512m -e NVIDIA_VISIBLE_DEVICES=void \
  -e TRITON_RUNTIME_CACHE_ARCH=cpu \
  -v "$PWD/tests:/opt/tritonserver/tests:ro" \
  --entrypoint /opt/envs/vllm032-cu129/bin/python3.12 \
  triton-openai-gateway:unified-vllm032-local -I \
  /opt/tritonserver/unified/launch.py --exec /bin/bash -ec \
  'python3 /opt/tritonserver/tests/runtime_unified_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_vllm_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_stock_vllm_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_audio_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_media_smoke.py'
```

Before using a unified candidate, verify its manifests and full dependency
closure, then run CPU import/spawn checks, gateway/client/media regressions, and
the real native Triton stub checks. Qualification on the target GPU must also
cover the following:

- Existing text, vision, audio/video/PDF, pooling, and rerank models used by the
  deployment, including streaming, cancellation, load/unload, and failure cleanup.
- EngineCore and tensor-parallel workers, CUDA/NCCL libraries, FlashInfer kernels,
  model-specific kernels, and the actual JIT compiler/toolkit combination.
- DeepSeek checkpoint admission and startup when that backend is used.
- Driver compatibility with both the retained native Triton core and the newer
  Python engine, plus writable runtime cache paths outside immutable artifacts.

Record input image identities, verifier results, model/hardware coverage, and
same-method image/filesystem size measurements with the candidate. Keep runtime
environments read-only; do not install packages at startup or mount over
`/opt/envs/vllm032-cu129`, `/opt/cuda/12.9`, or immutable artifact directories.

Use the qualified image tag or digest in the existing Docker or Helm deployment
configuration. The [quick start](../README.md#quick-start) and
[Helm guide](../helm/triton-gateway/README.md) describe the deployment parameters.
To roll back, select the previously qualified official image tag or digest and
its matching model configuration. Keep that artifact until the new candidate
has passed the required model checks.

For example, after stopping the candidate, run the qualified official image
with the same environment file and model repository as the previous deployment
(replace the repository placeholders):

```sh
docker run --rm --gpus all --shm-size=8g \
  --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8080:8080 \
  triton-openai-gateway:official-local \
  tritonserver --model-repository=s3://S3_ENDPOINT/BUCKET/PREFIX \
  --model-control-mode=explicit --strict-readiness=false
```

Models configured with `backend: "vllm_deepseek"` require the previously qualified
additive legacy image built with `Dockerfile.triton-gateway-deepseek`. Replace
`triton-openai-gateway:official-local` in the example with that image's tag or
digest. The plain official image does not include `vllm_deepseek`; select a Helm
revision using the additive image for these models as well.

For Helm, roll back to the recorded qualified release revision and verify the
selected models again:

```sh
helm history triton-openai-gateway --namespace inference
helm rollback triton-openai-gateway QUALIFIED_REVISION --namespace inference
```

## Maintain the official track and review promotion

Dependabot retains recurring Docker updates for the repository root, targeting
`dev`. Review official NVIDIA image updates regularly while the unified track is
available. For each update, inspect the candidate image's actual package matrix,
update the official digest and coherent dependency pins, and run the official
runtime verifier plus gateway/client/media, backend, and target-GPU checks.
Do not upgrade vLLM, Torch, Transformers, or other native runtime components
individually inside the official image to imitate the unified environment.

An official NVIDIA image containing vLLM `0.32` or later is a candidate for
convergence. Its version number alone does not prove compatibility or switch
the default. Compare the runtime matrix, Python/backend ABI, gateway dependencies,
manual frontends, and required model features; repeat CPU/native and GPU
qualification before explicitly changing the default or retiring an alternative.
Updating pins, merging an update, and publishing an image are separate reviewed
actions; neither track publishes or replaces the other automatically.
