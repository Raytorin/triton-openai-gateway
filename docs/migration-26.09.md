# Migration To Triton 26.09

[Home](../README.md) / Triton 26.09 migration

**Language:** English | [Русский](migration-26.09.ru.md)

The gateway uses one digest-pinned NVIDIA runtime. Upgrade Triton, vLLM,
PyTorch, Transformers, FlashInfer, and `compressed-tensors` together with this
base image:

```text
nvcr.io/nvidia/tritonserver:26.09-vllm-python-py3@sha256:ff41460e8fb1a831e37d368f531e5f40761baae03a07adac2f4cde2fc2c953d1
```

## Runtime Matrix

| Component | Pinned version |
| --- | --- |
| Triton Server | 2.73.0 |
| NVIDIA vLLM | 0.29.0+5013de39.nv26.9.69442229 |
| Transformers | 5.16.1 |
| PyTorch | 2.14.0a0+b2c75dd062.nv26.9.68203377 |
| FlashInfer | 0.6.18.post1+f8a38c73.nv26.9.cu.68241617 |
| compressed-tensors | 0.17.0 |
| Python | 3.12.3 |
| Triton client | 2.73.0 |
| grpcio | 1.84.0 |
| ml-dtypes | 0.5.4 |
| protobuf | 6.33.6 |
| NumPy | 1.26.4 |
| Pydantic | 2.10.6 |
| FastAPI | 0.136.3 |
| Pillow | 13.0.0 |
| OpenTelemetry API, SDK, OTLP exporter, and OTLP HTTP exporter | 1.45.1 |

`docker/verify-runtime.py` checks installed versions when the image is built.
`tritonclient[grpc]==2.73.0` requires `grpcio >= 1.81.1`,
`ml-dtypes <= 0.5.4`, and `protobuf < 7`; the gateway pins the versions above.

## Compatibility Constraints

The NVIDIA base image contains dependency-metadata mismatches involving
vLLM's Pydantic, OpenAI, and CUTLASS requirements and librosa's NumPy requirement.
The gateway retains NumPy `1.26.4` for Triton and Pydantic `2.10.6` for its
frontend. Version verification does not imply a clean `pip check`. Do not
resolve these warnings by independently upgrading core runtime packages.

The stock Triton `vllm` wrapper does not provide this project's native video,
audio, and PDF tensor contract. Continue using the included `vllm_multimodal`
backend and gateway preprocessing for those inputs. Its engine arguments
follow vLLM `0.29` from the pinned image; keep gateway-only settings in
`gateway.json`, outside `model.json`.

Use the rebuilt gateway image in Docker or Helm deployments. Update
environment-specific image tags together with the runtime; the Helm chart
defaults to `26.09-vllm-gateway`. Model capacity, GPU parallelism, and media
limits still depend on the target model and hardware.

See [Configuration](configuration.md) for model files and
[Operations](operations.md) for deployment and capacity guidance.
