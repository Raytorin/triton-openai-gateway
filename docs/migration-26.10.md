# Migration To Triton 26.10

[Home](../README.md) / Triton 26.10 migration

**Language:** English | [Русский](migration-26.10.ru.md)

The classic image moves to this digest-pinned NVIDIA runtime:

```text
nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3@sha256:06803bff18e43379a7ecb4ce46feed537ca717c06c5472b354aec94bf20466e2
```

Upgrade the vendor vLLM, Torch, Transformers, FlashInfer and native Triton
components together. The [26.09 matrix](migration-26.09.md) remains available
for the previous runtime.

## Runtime Matrix

These versions identify the inspected image and gateway overlay. Exact vendor
build suffixes are intentional: NVIDIA's
[compatibility table](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html)
lists vLLM `0.31.0+8977dc6f.nv26.10.72757140`, while this digest contains the
`0.31.0+88641915.nv26.10.75644909` build below.

| Component | Pinned version |
| --- | --- |
| Triton Server | 2.74.0 |
| NVIDIA vLLM | 0.31.0+88641915.nv26.10.75644909 |
| Transformers | 5.17.0 |
| PyTorch | 2.15.0a0+875d815502.nv26.10.72655024 |
| FlashInfer | 0.7.1rc5+c6a6b324.nvinternal.26.10.fi.rc5.5943.pr6394.cu.73382939 |
| compressed-tensors | 0.17.0 |
| Python | 3.12.3 |
| Image CUDA version | 13.4.3.001 |
| Triton client | 2.74.0 |
| grpcio | 1.84.0 |
| ml-dtypes | 0.5.4 |
| protobuf | 6.33.6 |
| NumPy | 1.26.4 |
| Pydantic | 2.14.1 |
| OpenAI SDK | 2.54.0 |
| FastAPI | 0.136.3 |
| Starlette | 1.9.0 |
| Uvicorn / uvloop | 0.54.0 / 0.23.0 |
| Pillow | 13.0.0 |
| OpenTelemetry API, SDK, OTLP exporter, and OTLP HTTP exporter | 1.45.2 |

`docker/verify-runtime.py` verifies the runtime and overlay versions during
the build. Triton frontend 2.74 accepts Pydantic `>=2.12,<3`; the old exact
`2.10.6` pin is no longer needed. The Triton client overlay retains protobuf
`<7` and ml-dtypes `<=0.5.4` compatibility.

## Changes Relevant To This Gateway

NVIDIA's [2.74 release notes](https://github.com/triton-inference-server/server/releases/tag/v2.74.0)
describe:

- Configurable S3 addressing style.
- Earlier request-cancellation checks and HTTP input/buffer fixes.
- Relaxed OpenAI/Pydantic dependency constraints.
- vLLM `>=0.30` support, with prompt rendering before generation and encoding.

The included multimodal backend follows that rendering boundary for generation
and pooling. Keep model engine arguments in `model.json` and gateway-only
settings in `gateway.json`. Native media support remains model-specific;
updating the image does not add a modality to an unsupported architecture.
Configurable S3 addressing is not a substitute for checking a repository's
download and file-override behavior.

## Remaining Compatibility Constraints

The classic image retains vendor metadata conflicts:

- Triton's Python binding requires NumPy `<2`, while librosa 1.0.0 and
  OpenCV 5.0.0.93 require NumPy 2.x. NumPy stays at `1.26.4`.
- NVIDIA supplies CUTLASS DSL `4.8.0.dev0`, while the installed vLLM metadata
  requests `4.7.1`. The vendor package is retained.

The added Decord `0.6.0` wheel also retains an upstream internal platform-tag
mismatch: `pip check` reports it as unsupported despite the passing CPU video
decode check. The narrowly qualified metadata correction for the independent
DeepSeek environment is not applied to the classic system Python.

The gateway overlay corrects the base ml-dtypes `0.6.0`/NumPy mismatch by
pinning ml-dtypes `0.5.4`. Version verification does **not** imply a clean
`pip check` for the classic environment. Do not fix the remaining conflicts by
upgrading or downgrading individual core packages without qualifying the full
runtime.

Upstream also lists DCGM/NSCQ shutdown faults on NVSwitch and limitations for
stock vLLM tensor parallelism with explicit model control. Validate the chosen
distributed configuration and clean unload/shutdown. This migration does not
automatically select Ray.

## Optional DeepSeek Environment

The [additive DeepSeek recipe](../docker/README.deepseek.md) keeps its separate
Python 3.12 / vLLM `0.32.0+cu129` / Torch `2.13.0+cu129` / CUDA 12.9 environment.
The classic upgrade does not remove or independently upgrade that engine.

Compared with the previous pinned 26.09 image, the inspected NVIDIA fork also
registers `DeepseekV41ForCausalLM` and contains native vision and UE8M0 loading
code. This makes the official engine a candidate for a separate compatibility
test; it does not qualify a checkpoint on GPU or switch this project's backend
to that engine. Existing model-local chat templates and reasoning settings stay
unchanged.

Its exact native ABI exception now matches the 26.10 stock Python stub and
`libcudart.so.13.4.92`. Compiler paths, engine libraries, caches and spawned
Python workers remain isolated. Revalidate the real stub and GPU behavior when
composing the new base with this environment; matching Python package versions
alone does not establish native ABI compatibility.

## Deployment And Rollback

Build a new image rather than installing packages in a running container.
The Helm default tag is `26.10-vllm-gateway`; record an immutable digest and
preserve the previously qualified image plus its model configurations.

Before switching, test the existing model profiles: generation, SSE/tools,
JSON Schema, embeddings/rerank, enabled media, cancellation and load/unload.
Include target-GPU workers, CUDA/JIT, memory and clean server shutdown. CPU
build/import tests do not qualify model inference or performance.

To roll back, restore the recorded prior image and matching configurations,
reload the previously loaded models, and check model-specific READY plus an
inference response. DeepSeek requires its previously qualified additive image;
the plain classic image does not include that backend.

See [Configuration](configuration.md) and [Operations](operations.md) for
deployment settings and operational checks.
