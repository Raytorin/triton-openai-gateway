# Triton Gateway Request Examples

**Language:** English | [Русский](REQUEST_EXAMPLES.md)

These examples provide quick gateway checks. Replace model names, paths, hosts,
and tokens for your environment.

```bash
export GATEWAY_URL="http://127.0.0.1:8080"
export MODEL="Qwen3-32B-FP8"
```

## Text Chat

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "user", "content": "Who are you?"}
    ],
    "max_tokens": 256,
    "temperature": 0.2
  }' | jq
```

## Image: Base64 Data URL

```bash
base64 -w0 /path/to/image.jpg > /tmp/image.b64

python3 - <<'PY'
import json
from pathlib import Path

image_b64 = Path("/tmp/image.b64").read_text().strip()
payload = {
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + image_b64
                    },
                },
            ],
        }
    ],
    "max_tokens": 256,
    "temperature": 0.2,
}

Path("/tmp/vl-image-request.json").write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding="utf-8",
)
PY

curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vl-image-request.json | jq
```

## Video: Base64 Data URL

Do not pass a large base64 value through `jq --arg`; the shell may return
`Argument list too long`. Write base64 to a file and build JSON in Python.

```bash
base64 -w0 /path/to/video.mp4 > /tmp/video.b64

python3 - <<'PY'
import json
from pathlib import Path

video_b64 = Path("/tmp/video.b64").read_text().strip()
payload = {
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What happens in this video?"},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "data:video/mp4;base64," + video_b64
                    },
                },
            ],
        }
    ],
    "max_tokens": 256,
    "temperature": 0.2,
}

Path("/tmp/vl-video-request.json").write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding="utf-8",
)
PY

curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vl-video-request.json | jq
```

With `backend: "vllm_multimodal"`, the backend decodes the original video and
passes native video frames plus metadata to vLLM. With the stock
`backend: "vllm"`, the gateway samples frames, analyzes bounded chunks, and
reduces intermediate results. Relevant settings include
`VLLM_MEDIA_VIDEO_FPS`, `VLLM_MEDIA_VIDEO_MAX_FRAMES`,
`VLLM_MEDIA_VIDEO_CHUNK_FRAMES`, and `VLLM_MEDIA_VIDEO_MAX_PIXELS`.

## PDF: Base64 Data URL

PDF uses the `data:application/pdf;base64,...` MIME prefix. The gateway processes
the document in bounded chunks and builds one final answer for both vLLM
backends. Configure it with `VLLM_MEDIA_PDF_*` or model-local `gateway.json`.

```bash
base64 -w0 /path/to/document.pdf > /tmp/document.b64

python3 - <<'PY'
import json
from pathlib import Path

pdf_b64 = Path("/tmp/document.b64").read_text().strip()
payload = {
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Summarize the document and list its main points.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:application/pdf;base64," + pdf_b64
                    },
                },
            ],
        }
    ],
    "max_tokens": 512,
    "temperature": 0.2,
}

Path("/tmp/vl-pdf-request.json").write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding="utf-8",
)
PY

curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vl-pdf-request.json | jq
```

Pages do not have to fit into `max_model_len` simultaneously. One processing
chunk and the reduced intermediate results must fit. For very large document
collections, use a persistent RAG index instead.

`pdf_chunk_pages` should not exceed `limit_mm_per_prompt.image`, because every
rendered PDF page is an image:

```json
{
  "pdf_chunk_pages": 2,
  "pdf_max_pixels": 262144,
  "pdf_chunk_max_tokens": 256,
  "limit_mm_per_prompt": {
    "image": 2,
    "video": 1
  }
}
```

If one chunk does not fit, set `pdf_chunk_pages` to `1` or reduce
`pdf_max_pixels`. If final reduction does not fit, lower
`pdf_chunk_max_tokens` or increase `max_model_len` when GPU memory permits.

## PDF: OpenAI-Style `file`

Use this form for clients that send PDF as a `file` content part. An explicit
`application/pdf` MIME is preferred. The gateway also validates the `%PDF-`
signature when a client sends an incorrect MIME.

```bash
base64 -w0 /path/to/document.pdf > /tmp/document.b64

python3 - <<'PY'
import json
from pathlib import Path

pdf_b64 = Path("/tmp/document.b64").read_text().strip()
payload = {
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize this PDF and list risks."},
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": "data:application/pdf;base64," + pdf_b64,
                    },
                },
            ],
        }
    ],
    "max_tokens": 512,
    "temperature": 0.2,
}

Path("/tmp/vl-pdf-file-request.json").write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding="utf-8",
)
PY

curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vl-pdf-file-request.json | jq
```

## PDF: Remote URL

The gateway can download a PDF for a stock `vllm` model:

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Find the amount, deadlines, and obligations."
          },
          {
            "type": "pdf_url",
            "pdf_url": {
              "url": "https://example.local/docs/contract.pdf"
            }
          }
        ]
      }
    ],
    "max_tokens": 512,
    "temperature": 0.2
  }' | jq
```

The equivalent file form is
`{"type": "file", "file": {"url": "https://example.local/docs/contract.pdf"}}`.
`vllm_multimodal` accepts only embedded bytes/base64, so the gateway or client
must materialize the URL before the backend call.

## Multiple PDF Documents

Add multiple PDF parts to one `content` array. Processing passes remain bounded
by `pdf_chunk_pages` and `limit_mm_per_prompt.image`.

```json
[
  {
    "type": "text",
    "text": "Compare these contracts by amount, term, and penalties."
  },
  {
    "type": "file",
    "file": {
      "file_data": "data:application/pdf;base64,<FIRST_PDF_BASE64>"
    }
  },
  {
    "type": "file",
    "file": {
      "file_data": "data:application/pdf;base64,<SECOND_PDF_BASE64>"
    }
  }
]
```

Common failures:

- `Invalid base64 image payload`: verify unwrapped base64 and the
  `data:application/pdf;base64,` prefix.
- HTTP `413` for one page: lower `VLLM_MEDIA_PDF_MAX_PIXELS` or increase the
  model context when memory permits.
- `max_model_len` error: start a new conversation without old history, lower
  `pdf_chunk_pages` or `pdf_max_pixels`, or increase context.
- Prefer an explicit PDF data URL when a remote URL has no `.pdf` suffix.

## Audio

Qwen3-VL does not support audio regardless of backend. Use an ASR model to
produce text before chat. Native `audio` input is intended for architectures
that actually support audio in vLLM.

```bash
base64 -w0 /path/to/audio.wav > /tmp/audio.b64

python3 - <<'PY'
import json
from pathlib import Path

audio_b64 = Path("/tmp/audio.b64").read_text().strip()
payload = {
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe and summarize the audio."},
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": "data:audio/wav;base64," + audio_b64
                    },
                },
            ],
        }
    ],
    "max_tokens": 256,
    "temperature": 0.2,
}

Path("/tmp/vl-audio-request.json").write_text(
    json.dumps(payload, ensure_ascii=False),
    encoding="utf-8",
)
PY

curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/vl-audio-request.json | jq
```

For MP3, use `data:audio/mpeg;base64,`.

## Embeddings

```bash
export EMBEDDING_MODEL="Qwen3-Embedding-4B"

curl -s "$GATEWAY_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$EMBEDDING_MODEL"'",
    "input": "Text to embed"
  }' | jq
```

Batch request:

```bash
curl -s "$GATEWAY_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$EMBEDDING_MODEL"'",
    "input": [
      "First text",
      "Second text"
    ]
  }' | jq
```

## Rerank

```bash
export RERANK_MODEL="bge-reranker-v2-m3-python"

curl -s "$GATEWAY_URL/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$RERANK_MODEL"'",
    "query": "What is Docker?",
    "documents": [
      "Docker is an application containerization platform.",
      "Kubernetes orchestrates containers in a cluster.",
      "PostgreSQL is a relational database."
    ],
    "top_n": 2
  }' | jq
```

Through an authenticated LiteLLM proxy:

```bash
export LITELLM_URL="https://example.local"
export LITELLM_TOKEN="sk-..."

curl -s "$LITELLM_URL/v1/rerank" \
  -H "Authorization: Bearer $LITELLM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3-python",
    "query": "What is Docker?",
    "documents": [
      "Docker is an application containerization platform.",
      "Kubernetes orchestrates containers in a cluster.",
      "PostgreSQL is a relational database."
    ],
    "top_n": 2
  }' | jq
```

## Tool Calling

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {
        "role": "user",
        "content": "What is the current weather in London in celsius?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_current_weather",
          "description": "Get the current weather in a given location",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {
                "type": "string",
                "description": "The city and country, for example London, UK"
              },
              "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
              }
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "temperature": 0.1
  }' | jq
```

A successful response contains `choices[0].message.tool_calls` with the function
name and JSON arguments. The gateway formats the call but does not execute the
function. Your orchestration layer must run it and send a `role: "tool"` result
back to the model.
