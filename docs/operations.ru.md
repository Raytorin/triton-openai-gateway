# Эксплуатация

[Главная](../README.ru.md) / Эксплуатация

**Язык:** [English](operations.md) | Русский

Раздел описывает runtime-проверки и типовые ошибки общего образа Triton и
OpenAI gateway.

## Порты и health checks

| Порт | Сервис | Рекомендуемый доступ |
| --- | --- | --- |
| `8000` | Raw Triton HTTP и repository API | Только внутренние операторы |
| `8001` | Raw Triton gRPC | Gateway и доверенные клиенты |
| `8002` | Метрики Triton, моделей и опционально GPU | Только сеть мониторинга |
| `8080` | OpenAI-совместимый API и метрики gateway | Через аутентифицированный proxy |

Проверки gateway:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

`/health` проверяет процесс FastAPI. `/ready` возвращает успех, только если
локальный Triton находится в состоянии ready. Если strict readiness Triton
выключен, отдельная модель всё ещё может загружаться; её состояние нужно
проверять через repository index.

## Жизненный цикл модели

Получение состояния repository:

```bash
curl -sS http://127.0.0.1:8000/v2/repository/index
```

Явная загрузка и выгрузка:

```bash
curl -fsS -X POST \
  http://127.0.0.1:8000/v2/repository/models/MODEL_NAME/load

curl -fsS -X POST \
  http://127.0.0.1:8000/v2/repository/models/MODEL_NAME/unload
```

Во время S3 load watcher логирует изменение временного `model.json` и создание
активной ссылки. После unload временная директория исчезает, и watcher удаляет
устаревшую ссылку.

## Метрики

### Gateway

`http://HOST:8080/metrics` публикует Prometheus series для:

- HTTP requests и latency;
- admission inflight, queue, rejected и wait time;
- Triton calls, duration и активных streams;
- загрузки tokenizer и поведения cache;
- media preprocessing и PDF embedding cache;
- context compression, внутренних summary-вызовов, выбора rerank-стратегий и
  числа reasoning tokens.

### Triton и vLLM

`http://HOST:8002/metrics` публикует model metrics Triton. Встроенный backend
`vllm_multimodal` по умолчанию отправляет custom vLLM metrics. Добавьте
`REPORT_CUSTOM_METRICS=true` в `config.pbtxt`, чтобы явно закрепить это поведение.

### GPU и MIG

Встроенный DCGM Triton требует прав, которые могут конфликтовать с non-root pod.
Выберите один режим:

- `triton.gpuMetrics.mode=builtin`: Triton запускается от root со сброшенными
  Linux capabilities и публикует `nv_gpu_*` на порту `8002`.
- `triton.gpuMetrics.mode=external`: Triton остаётся non-root, а GPU/MIG metrics
  поступают от NVIDIA DCGM Exporter на порту `9400`.

Не устанавливайте второй exporter, если NVIDIA GPU Operator уже развернул его.
Отдельный chart [`helm/dcgm-exporter`](../helm/dcgm-exporter) предназначен для
кластеров без exporter или для привязанного к узлу тестового endpoint.

DCGM измеряет GPU или MIG device, а не конкретную модель. Его показатели можно
считать метриками модели, только если устройство выделено этой модели на весь
интервал измерения.

## Логирование

Формат по умолчанию: JSON.

```text
LOG_FORMAT=json
LOG_LEVEL=INFO
```

Также доступны CEF и человекочитаемый text. Логи содержат request ID, модель,
route, transport, status, duration и количество media, когда это применимо.
Gateway возвращает тот же ID в `X-Request-ID`; клиент может передать собственный
валидный ID в header запроса.

Debug одного chat-запроса:

```json
{
  "model": "MODEL_NAME",
  "messages": [{"role": "user", "content": "Проверь этот запрос"}],
  "debug": true
}
```

Безопасный debug mode логирует роли, tool-call IDs, выбранный parser, token
counts и finish reason. Prompt и tool results не логируются. Включайте
`DEBUG_LOG_PAYLOADS=true` только для контролируемой диагностики: preview может
содержать чувствительные данные пользователя.

### Проверка сжатия контекста

После изменения summary-модели или политики сжатия запустите целевой evaluator.
Он принудительно вызывает compaction и проверяет сохранность точных
идентификаторов, путей, исправленных значений и результатов tools:

```bash
python scripts/evaluate-context-memory.py \
  --model MODEL_NAME \
  --strict
```

Это функциональная проверка сохранения фактов, а не benchmark качества. Её
нужно выполнять с той же моделью и `gateway.json`, которые пойдут в deployment.

## Tracing

Helm chart может включить экспорт OpenTelemetry из Triton:

```yaml
triton:
  tracing:
    enabled: true
    endpoint: http://otel-collector.observability.svc:4318/v1/traces
    level: TIMESTAMPS
    rate: 0
    count: -1
```

При `rate: 0` Triton трассирует запросы, содержащие W3C trace context. Включайте
положительный sampling rate только с учётом объёма traces и нагрузки на
collector. Для анализа ёмкости и насыщения основным источником остаются метрики.

## Типовые ошибки

| Симптом | Возможная причина | Действие |
| --- | --- | --- |
| Модель есть в Triton, но gateway сообщает `not found` | Не создана активная symlink | Проверьте watcher logs, числовую версию и права записи в выбранный корень watcher |
| `AsyncEngineArgs` отклоняет ключ | `model.json` содержит аргумент, которого нет в закреплённом vLLM | Удалите или переименуйте ключ; настройки gateway храните в `gateway.json` |
| `KIND_GPU is currently for single-GPU models` | Tensor-parallel engine использует Triton `KIND_GPU` | Используйте один `KIND_MODEL` и назначьте ему нужные devices |
| Shared memory pool не увеличивается | Контейнеру не хватает `/dev/shm` | Увеличьте Docker `--shm-size` или memory volume `/dev/shm` в pod |
| Prompt превышает context | Text, media tokens и output reserve больше `max_model_len` | Уменьшите resolution, chunk, history или output; увеличивайте context только при наличии памяти |
| VL-модель отклоняет аудио | Архитектура не поддерживает аудио и ASR не настроен | Настройте локальную ASR-модель или используйте audio-capable модель |
| На `:8002` нет GPU series | Не инициализировался встроенный DCGM | Используйте root builtin mode или внешний DCGM Exporter |
| `pip` сообщает о конфликтах vLLM для `apache-tvm-ffi`, `openai` или `pydantic` при сборке образа | Закреплённый базовый образ NVIDIA `26.06` уже содержит эти расхождения package metadata | Сохраняйте проверенные digest и версии; не обновляйте core-зависимости vLLM независимо |
| Gateway возвращает `429` | Заполнены inflight и queue | Повторите с backoff или настройте проверенные admission limits |
| Gateway возвращает `413` | Превышен лимит body, media, страниц, pixels, кадров или длительности | Уменьшите input либо поднимите конкретный лимит после нагрузочного теста |

## Ёмкость и ресурсы

- Выделяйте `/dev/shm` с учётом параллельных Triton IPC payload, особенно для
  больших media.
- Рассматривайте `max_num_seqs` как предел concurrency vLLM, а не гарантированное
  число пользователей и не кратность GPU thread blocks.
- Делайте media preprocessing concurrency ниже chat concurrency: декодирование
  PDF и видео расходует CPU и host memory до GPU inference.
- Настраивайте context length вместе с KV cache. Максимальный context одной
  последовательности не означает, что все параллельные запросы смогут занять его.
- Используйте реалистичные длины input/output и размеры media в load tests.
- Выделяйте отдельные GPU/MIG, если нужна корректная атрибуция GPU к модели.
- Оставляйте достаточно termination grace time для отмены stream и очистки engine.

## Production checklist

- Разместите authentication, authorization, TLS и tenant quotas перед портом
  `8080`.
- Не публикуйте raw Triton и metrics ports во внешнюю сеть.
- Закрепляйте собранный image по digest в deployment manifests.
- Храните S3 credentials в Kubernetes Secrets или внешнем secret manager.
- Настройте лимиты request body, remote download, media, queue и timeout.
- Не разрешайте private remote URL без необходимости в trust boundary.
- Используйте NetworkPolicy для Triton, metrics, object storage и OTLP.
- Собирайте gateway, Triton, vLLM и DCGM metrics и настройте alerts на errors,
  рост queue, latency, memory pressure и доступность модели.
- Проверьте load, unload, restart pod, client cancellation и большие media.
- Проверяйте `trust_remote_code=true` для каждой модели: содержимое model
  repository может исполнять Python-код внутри контейнера.
