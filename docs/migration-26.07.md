# Migration To Triton 26.07

[Home](../README.md) / Triton 26.07 migration

**Language:** English | [Русский](migration-26.07.ru.md)

This project uses one digest-pinned NVIDIA runtime. Triton, vLLM, PyTorch,
Transformers, FlashInfer, and `compressed-tensors` are upgraded as one tested
matrix rather than as independent packages.

## Runtime Matrix

```text
nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3@sha256:31e20bfbc65055d6b85553a00d45342a62c506874aa996c11ae53650151f05b1
```

| Component | 26.06 | 26.07 |
| --- | --- | --- |
| Triton Server | 2.70.0 | 2.71.0 |
| NVIDIA vLLM | 0.22.1 | 0.24.0 |
| Transformers | 5.6.0 | 5.6.1 |
| FlashInfer | 0.6.12 | 0.6.14 |
| Python | 3.12 | 3.12 |
| CUDA | 13.3.0 | 13.3.4.1 |
| NCCL | 2.30.4 | 2.30.7 |

`docker/verify-runtime.py` checks the full installed matrix while the image is
built. `tritonclient[grpc]==2.71.0` requires `grpcio < 1.68` and
`protobuf < 7`, so the gateway deliberately pins `grpcio==1.67.1` and
`protobuf==6.33.6` instead of retaining the base-image versions.

## Relevant Changes

Triton 26.07 improves model-readiness reporting and fixes a deadlock and a
use-after-free in Python backend asynchronous/decoupled execution. Those fixes
are directly relevant to explicit model load/unload and the project's
decoupled `vllm_multimodal` backend.

The NVIDIA OpenAI frontend also gained a streaming tool-call buffer limit, but
this project does not use that frontend. Gateway limits and tool-call parsing
therefore remain necessary.

## Multimodal Compatibility

The standard Triton `vllm` wrapper still does not provide this project's native
video, audio, and PDF tensor contract. The included `vllm_multimodal` backend
and bounded gateway preprocessing remain part of the supported architecture.
The custom backend follows vLLM 0.24 interfaces shipped in the pinned image.

## Production Validation

Before promotion, run the following checks on the target GPU platform:

1. Build the image without overriding `BASE_IMAGE`; runtime verification must pass.
2. Check `/ready`, `/v1/models`, Triton readiness, and both metrics endpoints.
3. Test chat, streaming, reasoning, tools, embeddings, and rerank.
4. Test image, video, audio, and PDF through `vllm_multimodal`.
5. Run model load/unload cycles and verify child-process cleanup.
6. Canary every TP/PP/DP/EP topology used in production.
7. Compare TTFT, token throughput, GPU memory, and response quality with 26.06.

## Primary References

- [NVIDIA Triton 26.07 release notes](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-07.html)
- [NVIDIA vLLM 26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-07.html)
