# Backend `vllm_multimodal`

[Главная проекта](../../README.ru.md) / Собственный backend

**Язык:** [English](README.md) | Русский

`vllm_multimodal` является Python-based Triton backend, созданным на основе
NVIDIA Triton `vllm` backend. Он сохраняет асинхронный engine vLLM, decoupled
streaming, cancellation, embeddings, LoRA, выбор устройств и custom metrics, а
также добавляет явные input для изображений, видео, аудио, PDF и параметров media.

Используйте его, когда media должны передаваться через границу Triton как
нативные данные, а не полностью преобразовываться gateway в текст или чанки
изображений.

## Установка

Dockerfile проекта размещает директорию в:

```text
/opt/tritonserver/backends/vllm_multimodal
```

Backend повторно использует runtime Python backend из закреплённого образа
Triton. Не задавайте `runtime` в конфигурации модели: Triton обнаруживает
глобальный backend по его `model.py`.

## Конфигурация модели

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

Для multi-GPU engine используйте один instance `KIND_MODEL`. Число выбранных
devices должно соответствовать параметрам parallelism в `model.json`.

Полные примеры находятся в директории [`examples/`](../../examples/).

## Контракт input

Backend сохраняет стандартные `text_input`, `stream` и generation inputs и
добавляет опциональные Triton BYTES tensors:

| Input | Содержимое |
| --- | --- |
| `image` | Одно или несколько закодированных изображений |
| `video` | Одно или несколько закодированных видео |
| `audio` | Одно или несколько закодированных аудио |
| `pdf` | Один или несколько PDF-документов |
| `media_parameters` | Один JSON-объект с настройками декодирования |

Media item может быть raw bytes, base64, data URL или JSON envelope:

```json
{
  "data": "BASE64_DATA",
  "mime_type": "video/mp4",
  "format": "mp4"
}
```

Remote URL отклоняются. Gateway самостоятельно скачивает и проверяет удалённый
контент, после чего передаёт подготовленные bytes через локальный gRPC.

## Параметры media

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

При прямом Triton request страницы PDF рендерятся в изображения. На OpenAI
endpoint map/reduce больших документов остаётся в gateway, поскольку скрытые
engine-запросы сделали бы streaming, usage accounting и scheduling неточными.

Видео преобразуется в кадры и metadata, необходимые renderer-у vLLM. Аудио
работает только для архитектур с нативной поддержкой audio. Vision-only модели
всё равно требуют отдельного ASR этапа.

## Ограничения ресурсов

CPU preprocessing выполняется вне event loop vLLM и ограничивается
`VLLM_MULTIMODAL_PREPROCESS_CONCURRENCY`. Жёсткие лимиты:

- `VLLM_MULTIMODAL_MAX_MEDIA_BYTES`
- `VLLM_MULTIMODAL_MAX_REQUEST_BYTES`
- `VLLM_MULTIMODAL_MAX_MEDIA_ITEMS`
- `VLLM_MULTIMODAL_MAX_SOURCE_PIXELS`
- `VLLM_MULTIMODAL_MAX_OUTPUT_PIXELS`
- `VLLM_MULTIMODAL_MAX_VIDEO_FRAMES`
- `VLLM_MULTIMODAL_MAX_TOTAL_VIDEO_FRAMES`
- `VLLM_MULTIMODAL_MAX_PDF_PAGES`
- `VLLM_MULTIMODAL_MAX_AUDIO_SECONDS`

Response и metrics queues ограничиваются
`VLLM_MULTIMODAL_RESPONSE_QUEUE_SIZE` и
`VLLM_MULTIMODAL_METRICS_QUEUE_SIZE`.

Custom vLLM metrics включены по умолчанию. Их можно переопределить через
`REPORT_CUSTOM_METRICS` в `config.pbtxt` или
`VLLM_MULTIMODAL_REPORT_CUSTOM_METRICS` в environment.

## Ограничения

- Нативная поддержка media зависит от архитектуры модели и версии vLLM.
- Media tokens, text и output должны помещаться в `max_model_len`.
- `limit_mm_per_prompt` должен разрешать каждую используемую модальность.
- Прямой запрос большого PDF ограничен контекстом; для map/reduce используйте
  endpoint gateway.
- Временные video files удаляются после декодирования и не сохраняются.

## Лицензия

Файлы в этой директории распространяются по BSD-3-Clause и включают код NVIDIA
с изменениями проекта. Сохраняйте исходные заголовки и смотрите
[`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).
