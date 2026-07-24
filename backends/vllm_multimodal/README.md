# `vllm_multimodal` Backend

[Project home](../../README.md) / Custom backend

**Language:** English | [Русский](README.ru.md)

`vllm_multimodal` is a Python-based Triton backend derived from NVIDIA's Triton
`vllm` backend. It preserves the asynchronous vLLM engine, decoupled streaming,
cancellation, embeddings, LoRA, device selection, and custom metrics while
adding explicit image, video, audio, PDF, and media-parameter inputs.

Use it when media should cross the Triton boundary as native data rather than be
fully converted into text or image chunks by the gateway.

## Installation

The project Dockerfile installs this directory at:

```text
/opt/tritonserver/backends/vllm_multimodal
```

The backend reuses the Python backend runtime already present in the pinned
Triton image. Do not set `runtime` in a model configuration; Triton discovers
the global backend by its `model.py`.

## Model Configuration

```protobuf
name: "Qwen3-VL-Example"
backend: "vllm_multimodal"
max_batch_size: 0

model_transaction_policy { decoupled: true }

instance_group [{ kind: KIND_MODEL count: 1 }]

parameters [
  { key: "GPU_DEVICE_IDS" value: { string_value: "0,1" } },
  { key: "REPORT_CUSTOM_METRICS" value: { string_value: "true" } },
  { key: "ENABLE_VLLM_HEALTH_CHECK" value: { string_value: "true" } }
]
```

Use one `KIND_MODEL` instance for a multi-GPU engine. The number of selected
devices must agree with vLLM parallelism in `model.json`.

Complete model files are available in [`examples/`](../../examples/).

## Input Contract

The backend keeps the standard `text_input`, `stream`, and generation inputs and
adds optional Triton BYTES tensors:

| Input | Contents |
| --- | --- |
| `image` | One or more encoded images |
| `video` | One or more encoded videos |
| `audio` | One or more encoded audio items |
| `pdf` | One or more encoded PDF documents |
| `media_parameters` | One JSON object controlling decoding |

Each media item may be raw bytes, base64 text, a data URL, or a JSON envelope:

```json
{
  "data": "BASE64_DATA",
  "mime_type": "video/mp4",
  "format": "mp4"
}
```

Remote URLs are rejected. The bundled gateway downloads and validates remote
content before forwarding materialized bytes over local gRPC.

## Media Parameters

```json
{
  "media_order": ["image", "pdf", "video"],
  "image_max_pixels": 0,
  "video_fps": 2.0,
  "video_max_frames": 128,
  "video_max_pixels": 262144,
  "audio_sample_rate": 16000,
  "pdf_dpi": 144,
  "pdf_max_pixels": 262144,
  "pdf_max_pages": 64,
  "mm_processor_kwargs": {}
}
```

PDF pages are rendered to images for a direct Triton request. On the OpenAI
endpoint, long-document map/reduce remains in the gateway because hidden engine
requests would make streaming, usage accounting, and scheduling misleading.

Video is decoded into frames and metadata required by the vLLM renderer. Audio
works only for architectures with native audio support. A vision-only model
still requires a separate ASR stage.

## Resource Limits

CPU preprocessing runs outside the vLLM event loop and is bounded by
`VLLM_MULTIMODAL_PREPROCESS_CONCURRENCY`. Hard limits include:

- `VLLM_MULTIMODAL_MAX_MEDIA_BYTES`
- `VLLM_MULTIMODAL_MAX_REQUEST_BYTES`
- `VLLM_MULTIMODAL_MAX_MEDIA_ITEMS`
- `VLLM_MULTIMODAL_MAX_SOURCE_PIXELS`
- `VLLM_MULTIMODAL_MAX_OUTPUT_PIXELS`
- `VLLM_MULTIMODAL_MAX_VIDEO_FRAMES`
- `VLLM_MULTIMODAL_MAX_TOTAL_VIDEO_FRAMES`
- `VLLM_MULTIMODAL_MAX_PDF_PAGES`
- `VLLM_MULTIMODAL_MAX_AUDIO_SECONDS`

Response and metrics queues are bounded by
`VLLM_MULTIMODAL_RESPONSE_QUEUE_SIZE` and
`VLLM_MULTIMODAL_METRICS_QUEUE_SIZE`.

Custom vLLM metrics are enabled by default. Override them with
`REPORT_CUSTOM_METRICS` in `config.pbtxt` or
`VLLM_MULTIMODAL_REPORT_CUSTOM_METRICS` in the environment.

## Limitations

- Native media support depends on the model architecture and pinned vLLM.
- Media tokens plus text and output must fit `max_model_len`.
- `limit_mm_per_prompt` must permit every modality used by a request.
- Direct large-PDF inference is context-bound; use the gateway endpoint for
  bounded text/vision map-reduce.
- Temporary video files are removed after decoding and are not persisted.

## License

Files in this directory are distributed under BSD-3-Clause and include
NVIDIA-derived code plus project modifications. Preserve all source headers and see
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
