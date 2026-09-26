# Конфигурация

[Главная](../README.ru.md) / Конфигурация

**Язык:** [English](configuration.md) | Русский

Настройки разделены по владельцам. Triton читает `config.pbtxt`, vLLM читает
`model.json`, а gateway читает опциональный `gateway.json`. Такое разделение не
позволяет передать gateway-only ключи в аргументы engine vLLM.

## Совместимость runtime

Dockerfile закрепляет базовый образ NVIDIA Triton
`26.07-vllm-python-py3` по digest. При сборке проверяются Triton client `2.71.0`,
vLLM `0.24.0`, Transformers `5.6.1`, Torch, FlashInfer и все добавленные
зависимости для media и runtime. Замена базового образа является отдельной миграцией
совместимости, а не обычным обновлением пакета.

Сборка с закреплённым образом:

```bash
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:26.07 .
```

Меняйте базовый образ только после проверки полной матрицы runtime:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/tritonserver:TAG \
  -f Dockerfile.triton-gateway \
  -t triton-openai-gateway:custom .
```

`docker/verify-runtime.py` намеренно завершает сборку с ошибкой, если версия
пакета не совпадает с проверенной.

## Структура model repository

```text
MODEL_NAME/
├── config.pbtxt
└── 1/
    ├── model.json
    ├── gateway.json              # опционально
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── веса модели...
```

Gateway читает tokenizer и metadata модели из активной числовой версии. В S3
сценарии watcher создаёт стабильную ссылку
`<watcher-root>/models-active/MODEL_NAME` после того, как Triton
материализовал версию. Корень выбирается по приоритету `WATCHER_MODEL_DIR`,
`TMP_ROOT`, `TMPDIR` Triton, затем `/tmp`.

Watcher отслеживает опубликованные ссылки. После удаления ссылки он удаляет
соответствующую временную версию на следующем проходе (обычно в течение секунды),
а при замене ссылки — сразу после переключения. Каталог `folder*` вместе с
metadata удаляется, когда в нём больше нет числовых версий и активных ссылок.
Другие версии, общие активные ссылки и пути вне корня watcher сохраняются;
исходный S3-репозиторий не изменяется. Каталоги, которые ещё ни разу не были
опубликованы, не считаются мусором: Triton может ещё загружать их.

## `config.pbtxt`

Используйте Triton `KIND_MODEL`, если один vLLM engine занимает несколько GPU.
Минимальный пример custom multimodal-конфигурации находится в
[`examples/vllm-multimodal-config.pbtxt`](../examples/vllm-multimodal-config.pbtxt):

```protobuf
name: "Qwen3-VL-Example"
backend: "vllm_multimodal"
max_batch_size: 0

model_transaction_policy { decoupled: true }

instance_group [{ kind: KIND_MODEL count: 1 }]

parameters [
  { key: "GPU_DEVICE_IDS" value: { string_value: "0,1" } },
  { key: "REPORT_CUSTOM_METRICS" value: { string_value: "true" } }
]
```

Для штатного backend используйте `backend: "vllm"`. `GPU_DEVICE_IDS` выбирает
устройства, видимые внутри контейнера. Значение `tensor_parallel_size` в
`model.json` должно соответствовать числу GPU, выделенных одному engine. Не
создавайте отдельный Triton instance на каждый GPU одного tensor-parallel engine.

Python-модели embeddings и rerank определяют собственные input/output tensors и
не используют приведённый vLLM-конфиг.

## `model.json`

Содержимое `model.json` передаётся в `AsyncEngineArgs` vLLM. Используйте только
аргументы, поддерживаемые закреплённой версией vLLM:

```json
{
  "gpu_memory_utilization": 0.8,
  "max_model_len": 16384,
  "tensor_parallel_size": 2,
  "dtype": "auto",
  "trust_remote_code": true,
  "max_num_seqs": 16,
  "limit_mm_per_prompt": {
    "image": 8,
    "video": 1,
    "audio": 1
  }
}
```

Watcher записывает временный локальный путь `model` во время загрузки. Для GGUF
укажите `load_format: "gguf"` и положите `.gguf` файл в директорию версии;
watcher запишет путь до конкретного файла.

Параметры ёмкости зависят от модели и оборудования. Проверяйте
`gpu_memory_utilization`, context length, parallelism, KV cache dtype,
`max_num_seqs` и мультимодальные лимиты под реалистичной нагрузкой, а не
копируйте пример без изменений.

### Structured Output

Клиент запрашивает ограниченный JSON через OpenAI-поле `response_format`.
Gateway переводит `json_object` и `json_schema` в сериализованный параметр
Triton 26.07/vLLM `structured_outputs` для backend `vllm` и
`vllm_multimodal`.

Для запроса со structured output gateway отключает thinking только на время
этого запроса, чтобы схема ограничивала видимое поле `message.content`, а не
внутренний reasoning block. Настроенная policy модели продолжает действовать
для обычных chat-запросов.

Для штатного образа дополнительная настройка engine не нужна. В vLLM `0.24.0`
по умолчанию используется backend `auto`, а в закреплённом образе Triton уже
есть `xgrammar`. При необходимости выбор можно явно указать в `model.json`:

```json
{
  "structured_outputs_config": {
    "backend": "auto"
  }
}
```

Чтобы зафиксировать встроенную реализацию, используйте
`"backend": "xgrammar"`. Не добавляйте удалённый request-параметр
`guided_decoding_backend` и не выбирайте `guidance`, если собственный образ с
этой зависимостью не был отдельно собран и проверен.

## `gateway.json`

Опциональный `gateway.json` находится рядом с `model.json`. Полный пример:
[`examples/gateway.vllm-multimodal.json`](../examples/gateway.vllm-multimodal.json).

```json
{
  "admission": {
    "chat": {
      "max_inflight": 64,
      "max_queue": 256,
      "queue_timeout_seconds": 30
    },
    "media": {
      "max_inflight": 4,
      "max_queue": 16,
      "queue_timeout_seconds": 30
    }
  },
  "vllm_multimodal": {
    "media_history_mode": "latest",
    "focus_current_media": true,
    "media_history_max_tokens": 512,
    "pdf_chunk_pages": 2,
    "video_fps": 1.0,
    "video_max_frames": 32,
    "audio_asr_model": ""
  }
}
```

Основные группы:

| Группа | Назначение |
| --- | --- |
| `admission` | Concurrency, размер очереди и таймаут по модели и route |
| `media_history_*` | Правила сохранения прошлых media и текстовой истории |
| `pdf_*` | Режим извлечения, rendering, chunking и reduce limits PDF |
| `pdf_rag` | Опциональная embedding-модель и параметры retrieval |
| `video_*` | FPS, число кадров, pixel budget и размер чанка |
| `audio_*` | Локальная ASR-модель, device и overlap чанков |
| `max_remote_media_bytes` | Лимит скачивания remote content |
| `context_compression` | Политика переполнения контекста, rolling summary и fallback |
| `rerank` | Лимиты выполнения, стратегии после scoring и опциональный SQLite source |
| `reasoning` | Режим thinking, parser ответа и OpenAI-совместимое поле |
| `embeddings.hybrid` | Типы dense/sparse результата, batch limit и лимиты sparse-вектора |

`pdf_rag.embedding_model` должен содержать имя embedding-модели, уже загруженной
в том же Triton. Retrieval используется только для PDF, из которых удалось
извлечь достаточный объём текста.

### Гибридные и sparse embeddings

Стандартный контракт `POST /v1/embeddings` не меняется и возвращает dense-вектор.
Лексический sparse либо комбинированный результат модель включает явно в своём
`gateway.json`:

```json
{
  "embeddings": {
    "hybrid": {
      "enabled": true,
      "output_types": ["dense", "sparse"],
      "max_batch_size": 32,
      "default_sparse_top_k": null,
      "max_sparse_top_k": 8192
    }
  }
}
```

В `POST /v1/hybrid_embeddings` передаётся
`output_types: ["dense", "sparse"]`. Dense-вектор находится в `embedding`, а
компактный sparse-вектор возвращается параллельными массивами
`sparse_embedding.indices` и `sparse_embedding.values`. Параметр
`sparse_top_k` ограничивает число ненулевых элементов. Для штатного Triton
backend `vllm` gateway отклоняет этот маршрут, поскольку его request adapter
возвращает только dense-вектор. Встроенный `vllm_multimodal` поддерживает
BGE-M3 нативно и выбирает pooling-задачу vLLM `embed`, `token_classify` или
`embed&token_classify` для каждого запроса.

Старый alias `/v1/embeddings/hybrid` остаётся доступен для прямых клиентов
gateway. Не используйте его как target pass-through в LiteLLM: LiteLLM 1.97
считает URL с подстрокой `/v1/embed` маршрутом Cohere при фоновом логировании.

Рекомендуемый
[профиль `vllm_multimodal`](../examples/bge-m3-vllm-multimodal/README.ru.md)
использует scheduling vLLM и загружает `sparse_linear.pt` и
`colbert_linear.pt` через нативный `BgeM3EmbeddingModel`. В `model.json` нужно
указать `runner: "pooling"` и переопределить исходное имя архитектуры. Не
фиксируйте один `pooler_config.task`: backend выбирает задачу для каждого
запроса.

Отдельный [Python fallback](../examples/bge-m3-hybrid/README.ru.md) загружает
локальный encoder и sparse-head через Transformers. Оба профиля полностью
офлайн и не обращаются к Hugging Face во время запуска.

В закрытом контуре LiteLLM публикует этот маршрут через точную настройку
`pass_through_endpoints` из
[`examples/litellm.config.yaml`](../examples/litellm.config.yaml). LiteLLM
проверяет авторизацию и без преобразования перенаправляет JSON и ответ;
доступ в интернет для этого не нужен. `OpenAI` Python client с `base_url`
LiteLLM продолжает работать для dense-вызовов `embeddings.create()`. Его
типизированный embeddings-метод всегда обращается к стандартному endpoint,
поэтому hybrid route вызывается обычным HTTP POST.

### Сжатие контекста

Обработка истории настраивается для каждой chat-модели. Режим `truncate` по
умолчанию сохраняет прежнее поведение; `disabled` возвращает HTTP `400` при
переполнении; `summarize` заменяет старейшие полные turn-ы rolling summary,
сохраняя system messages, недавнюю историю и текущий запрос пользователя.

```json
{
  "context_compression": {
    "mode": "summarize",
    "fallback_mode": "truncate",
    "summary_model": "",
    "summary_max_tokens": 256,
    "summary_input_max_tokens": 4096,
    "preserve_recent_messages": 4,
    "max_summary_calls": 4,
    "summary_timeout_seconds": 120,
    "summary_temperature": 0.0,
    "cache_size": 256,
    "version": "structured-memory-v2",
    "safety_margin_tokens": 64,
    "trigger_ratio": 0.8,
    "target_ratio": 0.6,
    "evidence_max_tokens": 384
  }
}
```

В режиме `summarize` сжатие начинается, когда сформированный prompt достигает
`trigger_ratio` доступного бюджета, и стремится уменьшить его до
`target_ratio`. Такой запас не позволяет каждому следующему сообщению сразу
запускать новое summary. `evidence_max_tokens` резервирует дословные, очищенные
от секретов доказательства для важных путей, идентификаторов, чисел, результатов
tools и исправленных значений, которые может потерять абстрактивное summary.

Пустой `summary_model` использует запрошенную chat-модель; иначе нужно указать
другую загруженную chat-модель. Внутренние summary-вызовы учитываются отдельными
Prometheus-метриками и не добавляются в `usage` ответа клиенту. Обычный ответ
содержит `context_status`, а streaming-ответ передаёт эквивалентные заголовки
`X-Context-*`.

Распространённые параметры приведены в
[`examples/gateway.context-compression.json`](../examples/gateway.context-compression.json).

### Отбор результатов rerank

Rerank-модель сначала оценивает каждый переданный документ. Затем gateway
сортирует score и применяет выбранную стратегию постобработки. Запросы без
`selection` сохраняют прежнее поведение `top_n`.

```json
{
  "rerank": {
    "default_strategy": "top_n",
    "execution": {
      "max_documents_per_request": 256,
      "default_batch_size": 4,
      "max_batch_size": 8,
      "max_batch_tokens": 8192,
      "default_max_length": 512,
      "max_length": 8192
    },
    "strategies": {
      "strict": {
        "method": "top_n_and_threshold",
        "parameters": {
          "score_threshold": 0.5,
          "top_n": 5
        },
        "allow_request_parameters": ["top_n"],
        "version": "1"
      }
    }
  }
}
```

Блок `execution` ограничивает пиковую память reranker. Gateway разбивает один
запрос на последовательные micro-batch, сохраняет индексы документов и
объединяет все score перед сортировкой. Фактический batch дополнительно
ограничивается значением `max_batch_tokens / max_length`, поэтому длинные пары
автоматически уменьшают одновременную нагрузку на GPU.

Значения клиента выше `max_batch_size` или `max_length` возвращают HTTP `400`,
а превышение `max_documents_per_request` возвращает HTTP `413`. По умолчанию
admission допускает один активный rerank-запрос и очередь из 64 запросов на
модель. Увеличивайте `admission.rerank.max_inflight` только после нагрузочной
проверки конкретного reranker, GPU и числа model instance Triton.

Клиент выбирает именованную политику через
`"selection": {"strategy": "strict", "parameters": {"top_n": 2}}`.
Встроены методы `top_n`, `score_threshold`,
`top_n_and_threshold`, `metadata_filter` и `diversity`. Стратегии также
можно обновлять из read-only SQLite, настроенной в `rerank.database`; клиент
не может передать исполняемый код или SQL.

Полный статический и SQLite-пример:
[`examples/gateway.rerank.json`](../examples/gateway.rerank.json).

### Вывод reasoning

По умолчанию reasoning выключен и настраивается для каждой модели:

```json
{
  "reasoning": {
    "mode": "separate",
    "parser": "auto",
    "response_field": "reasoning_content"
  }
}
```

`disabled` запрашивает chat template без thinking и удаляет случайно
возвращённые reasoning blocks. `hidden` разрешает модели рассуждать, но не
возвращает текст клиенту. `separate` отделяет reasoning от итогового
`content` в JSON и SSE. Клиент может понизить `separate` до `hidden` для
одного запроса через `"include_reasoning": false`, но не может ослабить более
строгую серверную политику.

Скрытие reasoning не останавливает его генерацию моделью. Рассуждение и итоговый
ответ используют общий `max_tokens`; если модель исчерпала бюджет до
формирования ответа, gateway возвращает пустой `content` и
`finish_reason: "length"`.

`parser: "auto"` определяет поддерживаемые Qwen chat templates. Также
доступны явные parser-ы `qwen3` и `think_tags`. Для LiteLLM и
OpenAI-совместимых клиентов используйте
`response_field: "reasoning_content"`, а для схемы vLLM — `"reasoning"`.

Пример: [`examples/gateway.reasoning.json`](../examples/gateway.reasoning.json).

## Переменные окружения

В Kubernetes предпочтительно использовать Helm values. При прямом запуске
контейнера доступны соответствующие environment variables.

### Gateway и transport

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `TRITON_BASE_URL` | `http://127.0.0.1:8000` | Triton HTTP endpoint |
| `TRITON_GRPC_URL` | `127.0.0.1:8001` | Triton gRPC endpoint |
| `WATCHER_MODEL_DIR` | не задано | Явная локальная директория временных checkout `folder*` Triton |
| `TMP_ROOT` | не задано | Обратно совместимый alias корня watcher |
| `TMPDIR` | системное значение | Временная директория Triton и автоматический fallback watcher |
| `MODELS_ACTIVE_DIR` | `<watcher-root>/models-active` | Стабильные ссылки активных моделей |
| `GATEWAY_PORT` | `8080` | Порт FastAPI |
| `REQUEST_TIMEOUT_SECONDS` | `600` | Таймаут upstream-запроса |
| `GATEWAY_MAX_REQUEST_BODY_BYTES` | `268435456` | Максимальный размер HTTP request body |
| `TOKENIZER_PRELOAD` | `true` | Предзагрузка tokenizer активных моделей |
| `TOKENIZER_TRUST_REMOTE_CODE` | `true` | Разрешение remote code токенайзера |

### Watchdog Gateway

Фоновый supervisor проверяет локальный `/health`, который отвечает без обращения
к Triton. После трёх последовательных сбоев он завершает Uvicorn через SIGTERM,
ждёт до 10 секунд, при необходимости отправляет SIGKILL и запускает новый процесс.
При падении процесса перезапуск выполняется без ожидания трёх проверок. На запуск
до первого успешного ответа отводится 60 секунд; единичный сбой не вызывает рестарт.
После остановки ранее доступного Triton supervisor завершает gateway и выходит,
чтобы не мешать завершению контейнера. Перезапуск обрывает текущие запросы gateway.

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `GATEWAY_HEALTH_INTERVAL_SECONDS` | `1` | Интервал проверок |
| `GATEWAY_HEALTH_TIMEOUT_SECONDS` | `1` | Таймаут одного HTTP-запроса |
| `GATEWAY_HEALTH_FAILURE_THRESHOLD` | `3` | Число последовательных сбоев |
| `GATEWAY_STARTUP_GRACE_SECONDS` | `60` | Ожидание первого успешного ответа |
| `GATEWAY_STOP_GRACE_SECONDS` | `10` | Ожидание завершения после SIGTERM |
| `GATEWAY_RESTART_DELAY_SECONDS` | `2` | Пауза перед повторным запуском |

### Admission control

Глобальные значения задаются через `GATEWAY_MAX_INFLIGHT_REQUESTS`,
`GATEWAY_MAX_QUEUE_SIZE` и `GATEWAY_QUEUE_TIMEOUT_SECONDS`. Route-specific
переопределения используют `GATEWAY_CHAT_*`, `GATEWAY_MEDIA_*`,
`GATEWAY_EMBEDDINGS_*` и `GATEWAY_RERANK_*`. Настройки route из `gateway.json`
имеют приоритет для конкретной модели.

### Логи и debug

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `LOG_FORMAT` | `json` | `json`, `cef` или `text` |
| `LOG_LEVEL` | `INFO` | Уровень Python logs |
| `GATEWAY_DEBUG` | `false` | Безопасные debug metadata для всех chat-запросов |
| `DEBUG_LOG_PAYLOADS` | `false` | Ограниченный preview payload; может раскрыть данные пользователя |
| `DEBUG_PREVIEW_CHARS` | `2000` | Максимальный размер debug preview |
| `UVICORN_ACCESS_LOG` | `false` | Access log Uvicorn |

Для одного chat-запроса клиент может передать `"debug": true`, не включая
глобальный debug mode.

### Media

Preprocessing штатного backend использует переменные `VLLM_MEDIA_*`. Жёсткие
лимиты native backend задаются через `VLLM_MULTIMODAL_*`. Полный список и
значения по умолчанию находятся в `gateway.vllmMultimodal` и
`gateway.nativeBackend` файла
[`helm/triton-gateway/values.yaml`](../helm/triton-gateway/values.yaml).

Для временных media на отдельном volume задайте `TRITON_MEDIA_DIR` и
`TRITON_MULTIMODAL_TMPDIR`. Временные файлы удаляются после обработки; этот
volume не является архивом.

## Helm

Chart поддерживает:

- S3 model repository и Secret с credentials;
- физические GPU и MIG resources;
- PVC для моделей и временных media;
- встроенные или внешние DCGM metrics;
- OTLP tracing Triton;
- probes, security context, scheduling и topology;
- ограничение concurrency gateway и backend.

Возьмите за основу
[`helm/triton-gateway/values.yaml`](../helm/triton-gateway/values.yaml), а
environment-specific значения храните отдельно:

```bash
helm upgrade --install triton-openai-gateway ./helm/triton-gateway \
  --namespace inference --create-namespace \
  --values values.production.yaml
```

Примеры установки находятся в [README Helm chart](../helm/triton-gateway/README.ru.md).

### Лимит выходных токенов

Chat принимает `max_completion_tokens` и legacy-алиас `max_tokens`. Равные
значения допустимы, конфликтующие дают 400. Лимиты — только целые числа > 0.
Без параметра/null выбирается 4096 токенов, при thinking — 8192, с уменьшением
до свободного контекста без удаления уже помещающейся истории. Явный лимит
молча не уменьшается. Thinking и аргументы функций входят в общий бюджет.
Usage пока оценивается повторной токенизацией.

Глобальные defaults: `GATEWAY_DEFAULT_OUTPUT_TOKENS=4096`,
`GATEWAY_REASONING_DEFAULT_OUTPUT_TOKENS=8192`, `GATEWAY_MAX_OUTPUT_TOKENS=32768`.
Настройки модели находятся в `gateway.json`, не в аргументах движка vLLM:

```json
{"generation":{"default_output_tokens":4096,"reasoning_default_output_tokens":8192,"max_output_tokens":32768}}
```

Потолок модели не превышает глобальный; оба default должны помещаться в потолок.
Для прежнего поведения задайте оба default равными 256. Нужен положительный
`model.json.max_model_len`; если он не задан/автоматический, явно укажите
консервативный `generation.context_window`. Он не расширяет известный runtime
context. Служебное огромное значение tokenizer не считается runtime context.
Применяется существующий safety margin. Внутренние summary-вызовы сохраняют
свои бюджеты. Заголовки `X-Output-Token-Limit`, `X-Output-Token-Limit-Source`,
`X-Token-Usage-Source` и структурированные логи показывают принятое решение.
