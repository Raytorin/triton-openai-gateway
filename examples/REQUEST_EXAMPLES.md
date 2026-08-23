# Примеры запросов к Triton Gateway

**Язык:** [English](REQUEST_EXAMPLES.en.md) | Русский

Ниже примеры для быстрого тестирования gateway. Заменяй `MODEL`, `HOST`, пути к файлам и токены под свое окружение.

Базовые переменные:

```bash
export GATEWAY_URL="http://127.0.0.1:8080"
export MODEL="Qwen3-32B-FP8"
```

## Chat: обычный текст

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "user", "content": "Кто ты?"}
    ],
    "max_tokens": 256,
    "temperature": 0.2
  }' | jq
```

## Structured JSON Output

`json_schema` передаётся в constrained decoding, а не только добавляется в
prompt как инструкция. Поле `message.content` остаётся JSON-строкой, которую
можно разобрать через `jq`:

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "user", "content": "Верни город и температуру: Москва, 18 C."}
    ],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "weather",
        "strict": true,
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "temperature_c": {"type": "number"}
          },
          "required": ["city", "temperature_c"],
          "additionalProperties": false
        }
      }
    },
    "max_tokens": 128
  }' | jq -r '.choices[0].message.content | fromjson'
```

## Reasoning-модели

Серверная policy задаётся в `gateway.json` модели. При
`reasoning.mode=separate` рассуждение возвращается в
`message.reasoning_content`, а итоговый ответ остаётся в
`message.content`:

```bash
curl -sS "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL"'",
    "messages": [
      {"role": "user", "content": "Вычисли 37 * 48 и дай краткий ответ."}
    ],
    "max_tokens": 1024
  }' | jq '{
    reasoning: .choices[0].message.reasoning_content,
    answer: .choices[0].message.content,
    status: .reasoning_status
  }'
```

Передайте `"include_reasoning": false`, чтобы скрыть reasoning в одном
запросе, не меняя серверную policy.

## Image: base64/data URL

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
                {"type": "text", "text": "Опиши изображение."},
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

## Video: base64/data URL

Для больших видео не передавай base64 через `jq --arg`: можно получить `Argument list too long`. Сначала сохрани base64 в файл, затем собери JSON через Python.

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
                {"type": "text", "text": "Что происходит на видео?"},
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

Для `backend: "vllm_multimodal"` исходное video декодируется новым backend и
передается в нативную video-модальность vLLM вместе с metadata. Для штатного
`backend: "vllm"` gateway выбирает кадры, анализирует их чанками и объединяет
промежуточные результаты. Настройки обоих режимов:
`VLLM_MEDIA_VIDEO_FPS`, `VLLM_MEDIA_VIDEO_MAX_FRAMES`,
`VLLM_MEDIA_VIDEO_CHUNK_FRAMES` и `VLLM_MEDIA_VIDEO_MAX_PIXELS`.

## PDF: base64/data URL

PDF передается как `data:application/pdf;base64,...`. Для штатного `vllm` и
`vllm_multimodal` gateway рендерит страницы, анализирует их ограниченными чанками
и собирает итоговый ответ. Параметры задаются через `VLLM_MEDIA_PDF_*` или
`gateway.json` рядом с моделью.

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
                {"type": "text", "text": "Кратко перескажи документ и выдели основные пункты."},
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

Все страницы обрабатываются последовательно и не должны одновременно помещаться
в `max_model_len`. В контекст должен помещаться один чанк и итоговые сокращенные
результаты; для очень больших документов предпочтительна RAG-индексация.

Размер одного чанка регулируется параметром `pdf_chunk_pages`. Он должен соответствовать `limit_mm_per_prompt.image`, потому что каждая PDF-страница становится отдельным изображением:

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

Если один чанк не помещается в контекст, снизь `pdf_chunk_pages` до `1` или уменьши `pdf_max_pixels`. Если итоговая сборка по большому документу не помещается, снизь `pdf_chunk_max_tokens` или увеличь `max_model_len`.

## PDF: OpenAI-style `file`

Удобно для клиентов, которые отправляют PDF как content-part `file`. Предпочтителен
явный MIME `application/pdf`; при неверном MIME gateway дополнительно проверяет
сигнатуру `%PDF-` в содержимом файла.

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
                {"type": "text", "text": "Сделай краткую выжимку PDF и перечисли риски."},
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

## PDF: URL документа

В режиме штатного `vllm` gateway может скачать URL на `.pdf`:

```bash
curl -s "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-32B-Instruct-FP8",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Найди в документе сумму, сроки и обязательства сторон."},
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

Та же ссылка принимается в форме `{"type": "file", "file": {"url": "https://example.local/docs/contract.pdf"}}`.
`vllm_multimodal` намеренно принимает только встроенные bytes/base64; URL для
него необходимо заранее материализовать в gateway или клиенте.

## PDF: несколько документов

Добавь несколько PDF parts в один `content`. Для обоих vLLM backend размер
каждого прохода ограничивается `pdf_chunk_pages` и
`limit_mm_per_prompt.image`.

```json
[
  {"type": "text", "text": "Сравни два договора и найди отличия по суммам, срокам и штрафам."},
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

Типовые проблемы:

- `Invalid base64 image payload`: проверь, что base64 без переносов или собран через Python, и есть префикс `data:application/pdf;base64,`.
- HTTP 413 для одной страницы: уменьши `VLLM_MEDIA_PDF_MAX_PIXELS` или увеличь `max_model_len`.
- Ошибка про `max_model_len`: начни новый диалог без старой истории, снизь `pdf_chunk_pages`, снизь `pdf_max_pixels` или увеличь `max_model_len`.
- URL без суффикса `.pdf` лучше заменить на data URL с явным MIME `application/pdf`.

## Audio: примечание

Qwen3-VL не поддерживает audio независимо от выбранного backend. Для него
рабочий сценарий: ASR-модель переводит аудио в текст, затем текст отправляется в
chat-модель. Нативный `audio` input нового backend предназначен для архитектур,
которые действительно поддерживают audio в vLLM.

Если gateway настроен на обработку `audio_url`, тестовый запрос может выглядеть так:

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
                {"type": "text", "text": "Расшифруй аудио и кратко опиши смысл."},
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

Для `mp3` поменяй MIME на `data:audio/mpeg;base64,`.

## Embeddings

```bash
export EMBEDDING_MODEL="Qwen3-Embedding-4B"

curl -s "$GATEWAY_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$EMBEDDING_MODEL"'",
    "input": "Тестовый текст для embedding"
  }' | jq
```

Пакетный запрос:

```bash
curl -s "$GATEWAY_URL/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$EMBEDDING_MODEL"'",
    "input": [
      "Первый текст",
      "Второй текст"
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
    "query": "Что такое Docker?",
    "documents": [
      "Docker - это платформа контейнеризации приложений.",
      "Kubernetes управляет контейнерами в кластере.",
      "PostgreSQL - это реляционная база данных."
    ],
    "top_n": 2
  }' | jq
```

Если используется LiteLLM с авторизацией:

```bash
export LITELLM_URL="https://example.local"
export LITELLM_TOKEN="sk-..."

curl -s "$LITELLM_URL/v1/rerank" \
  -H "Authorization: Bearer $LITELLM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-reranker-v2-m3-python",
    "query": "Что такое Docker?",
    "documents": [
      "Docker - это платформа контейнеризации приложений.",
      "Kubernetes управляет контейнерами в кластере.",
      "PostgreSQL - это реляционная база данных."
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
                "description": "The city and state, e.g. San Francisco, CA"
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

Ожидаемый успешный результат: `choices[0].message.tool_calls` с именем функции и JSON-аргументами. Gateway только формирует tool call. Саму функцию, например получение погоды, должен выполнить внешний orchestration слой.
