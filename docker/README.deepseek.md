# Optional DeepSeek execution environment

The default `Dockerfile.triton-gateway` retains its NVIDIA vLLM 0.29 environment.
Build it first, then opt into the separate Python 3.12 / vLLM 0.32 CUDA 12.9
execution environment:

```sh
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:qwen-local .
docker build --platform linux/amd64 -f Dockerfile.triton-gateway-deepseek \
  --build-arg GATEWAY_IMAGE=triton-openai-gateway:qwen-local \
  -t triton-openai-gateway:deepseek-local .
```

Use a recorded local image ID/digest for `GATEWAY_IMAGE` when reproducing an
artifact. No model checkpoint is part of either image. The optional build needs
space for the CUDA toolchain, wheelhouse and installed environment. It downloads
only public runtime artifacts; the installation and import/spawn checks run with
network disabled. The independently extracted Python distribution owns its
`EXTERNALLY-MANAGED` marker, which is removed before installing its locked wheels.
The system Python installation is unchanged. Every DeepSeek pip command goes
through the process-local launcher, which removes inherited `PIP_*` constraints
and disables pip configuration files; Qwen retains the base image's pip policy.

Before starting Triton for GPU inference, set `DEEPSEEK_CACHE_ARCH` to the
validated architecture (for example `sm90`). CPU qualification can use `cpu`;
`cpu` and `unqualified` cannot launch the GPU engine. Set `DEEPSEEK_CACHE_DIR` to
a writable persistent directory if needed; its default is
`/tmp/triton-runtime-cache/vllm032-cu129`, with a separate subdirectory per
architecture. Each mutable runtime cache is selected there before exec/import.

Select the new backend per model in `config.pbtxt`:

```protobuf
backend: "vllm_deepseek"
model_transaction_policy { decoupled: true }
instance_group [{ kind: KIND_MODEL count: 1 }]
parameters {
  key: "EXECUTION_ENV_PATH"
  value { string_value: "/opt/envs/vllm032-cu129" }
}
```

The backend accepts text generation, including gateway-rendered thinking/tool
prompts. It rejects embedding, image, audio, video, PDF and media-parameter
inputs. Keep model-specific checkpoint validation and engine settings in the
model configuration. Existing Qwen/BGE models keep their original backend and
execution environment.

`docker/runtime-locks/vllm032-cu129.requirements.txt` is a pre-materialization lock: its 204 core packages were resolved from actual
wheel metadata. The large FlashInfer data-wheel closure comes from the inspected
JIT shim plus exact official-index versions/hashes; their full wheel metadata and
installation remain a build gate. An unexpected dependency fails the hash-required
download/install or `pip check`; never bypass those failures. The separate
`vllm032-cu129.kernel-artifacts.json` records each artifact and verification state. The input provenance JSON
pins the independent Python artifact, CUDA image and engine source/wheels.
`flashinfer-cubin` and `flashinfer-jit-cache` package the precompiled kernels;
there is no runtime package-install/download command. Torch's wheel-required
NCCL 2.29.7 is retained; optional reference-image overrides are not applied.

The build creates `/opt/runtime-manifests/vllm032-cu129.json`, containing every
installed distribution version and hashes of the environment/toolkit/kernel
artifacts. Its derived content digest identifies this composed environment; the
upstream reference-image digest is a separate provenance input.

For the exact locked Decord 0.6.0 wheel, the build corrects its installed WHEEL
tag to the upstream filename's `py3-none-manylinux2010_x86_64`, after checking
the original archive hash and every installed payload byte. This metadata-only
derivation is restricted to Python 3.12.15, NumPy 2.3.5 and Linux x86_64 with
glibc; the original wheel and native code remain unchanged. The build regenerates
RECORD, records `decord-metadata-repair.json` in the manifest's artifact tree,
and still requires the complete `pip check` to pass.

The custom backend installs an executable `triton_python_backend_stub` wrapper
and a sibling symlink to stock `triton_python_backend_utils.py`; Triton uses both
to select this per-backend launcher. It execs the release-matched stock C++ stub
with native library paths already selected. Bootstrap rejects a missed launcher
and foreign Torch/CUDA/cuDNN/NCCL mappings. The pinned stock stub's exact
`libcudart.so.13` DT_NEEDED dependency is a hash-verified core ABI exception;
Python workers receive no such exception, and engine CUDA12 libraries stay in
the derived roots. Compiler/header overrides are also replaced locally.

CUDA12.9 identifies the selected external compiler toolkit and Torch wheel.
CUTLASS DSL also packages CUDA13 resources and can select a flavor from the
CUDA driver's capabilities, or the newest installed flavor without a driver.
The verifier's `selected_cutlass_ir` reports an already-loaded extension's path
and CUDA major, or `not_loaded`; `available` lists installed candidates
separately. Reporting does not import CUTLASS or initialize a GPU. GPU/JIT
qualification must assess the actual selected combination.

Repeat CPU validation of the built image without weights, GPUs or networking:

```sh
docker run --rm --network none --entrypoint env \
  triton-openai-gateway:deepseek-local \
  DEEPSEEK_CACHE_ARCH=cpu \
  /opt/envs/vllm032-cu129/bin/python3.12 -I \
  /opt/tritonserver/backends/vllm_deepseek/launch.py --exec \
  /opt/envs/vllm032-cu129/bin/python3.12 \
  /opt/tritonserver/tools/deepseek/verify-deepseek-runtime.py --imports --spawn

docker run --rm --network none --entrypoint \
  /opt/envs/vllm032-cu129/bin/python3.12 triton-openai-gateway:deepseek-local \
  /opt/tritonserver/tools/deepseek/deepseek-runtime-manifest.py verify \
  --manifest /opt/runtime-manifests/vllm032-cu129.json
```

The verifier reports parent/child module origins, libpython and native maps,
compiler/header paths and tool versions, cache paths, exact package versions,
immutable FlashInfer artifact origins and `pip check`. Mutable caches are outside
the manifested `/opt/runtime-artifacts` tree and immutable installed environment.
Inherited `FLASHINFER_CUBIN_DIR` overrides are cleared so the installed cubin
package remains discoverable; `FLASHINFER_NO_DOWNLOAD=1` makes missing artifacts
fail offline. Never mount a writable directory over the execution environment.

These CPU checks do not qualify inference. Acceptance additionally requires a
real Triton Python stub using `EXECUTION_ENV_PATH`, clean unload after failed
startup, and GPU tests of EngineCore/TP workers, CUDA/NCCL/FlashMLA/DeepGEMM
libraries and JIT compiler selection. Validate the retained Qwen/BGE behavior in
the composed image. The retained Triton core's driver requirement still applies.

Use the [DeepSeek V4.1 Flash model example](../examples/deepseek-v4.1-flash/README.md)
for an external immutable checkpoint mount, bounded structural admission,
operator-selected parallelism/cache architecture and selected-model READY checks.
It also documents the local no-weight native preflight-failure smoke gate.
