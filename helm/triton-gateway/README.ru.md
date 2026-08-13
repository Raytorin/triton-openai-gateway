# Helm Chart Triton OpenAI Gateway

[Главная проекта](../../README.ru.md) / Helm deployment

**Язык:** [English](README.md) | Русский

Chart разворачивает NVIDIA Triton и OpenAI-совместимый gateway в одном pod. Он
поддерживает S3 model repository, GPU и MIG resources, хранилище моделей и media,
admission control, Prometheus metrics, DCGM и OTLP tracing Triton.

## Сервисы

| Порт | Имя | Назначение |
| --- | --- | --- |
| `8000` | `http` | Raw Triton HTTP и model repository API |
| `8001` | `grpc` | Raw Triton gRPC |
| `8002` | `metrics` | Triton, vLLM и опциональные встроенные GPU metrics |
| `8080` | `gateway` | OpenAI-совместимый API и метрики gateway |

## Требования

- Kubernetes с NVIDIA drivers и device plugin либо GPU Operator.
- Helm 3.
- Собранный образ Triton OpenAI Gateway в доступном cluster registry.
- Triton model repository и credentials для него.

## Установка

Создайте отдельный values-файл для окружения:

```yaml
image:
  repository: registry.example.com/ml/triton-openai-gateway
  tag: "26.07"

triton:
  modelRepository: s3://object-store.example.com/models/triton
  modelControlMode: explicit
  loadModels:
    - chat-model
    - embedding-model
  gpuMetrics:
    mode: external

s3:
  existingSecret: triton-s3-credentials

resources:
  requests:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: 64Gi
    nvidia.com/gpu: "1"
```

Установка или обновление:

```bash
helm upgrade --install triton-openai-gateway ./helm/triton-gateway \
  --namespace inference \
  --create-namespace \
  --values values.production.yaml
```

[`values.production.example.yaml`](values.production.example.yaml) содержит
полный стартовый пример без реальных имён registry, Secret и моделей.

S3 Secret должен содержать environment variables, которые ожидает S3 repository
agent Triton. Не храните credentials в `values.yaml`.

## Обновление и health probes

По умолчанию используется стратегия `Recreate`. Rolling update может
заблокироваться, если текущий Triton pod уже занял все GPU, необходимые новому
pod. Используйте `updateStrategy.type: RollingUpdate` только при наличии
свободных GPU для одновременной работы двух pod.

Startup, liveness и readiness по умолчанию проверяют endpoint gateway
`/health`. Поэтому UI и repository API остаются доступны во время load/unload
отдельной модели. Чтобы удалять pod из Service endpoints, когда Triton не готов:

```yaml
readinessProbe:
  path: /ready
  port: gateway
```

Элементы `triton.loadModels` преобразуются в повторяющиеся аргументы
`--load-model=<name>`. Оставьте список пустым, если lifecycle моделей
управляется только через repository API Triton.

## GPU и MIG

Для MIG profile запросите resource, опубликованный device plugin:

```yaml
resources:
  requests:
    cpu: "10"
    memory: 80Gi
    nvidia.com/mig-3g.40gb: "1"
  limits:
    cpu: "10"
    memory: 80Gi
    nvidia.com/mig-3g.40gb: "1"
```

Node selectors, tolerations, affinity и topology spread constraints задаются в
соответствующих top-level values.

## Хранилище моделей

По умолчанию рабочая директория моделей является `emptyDir`, смонтированным в
`/models` и `/tmp`. Подключите существующий PVC, если скачанные данные модели
должны пережить пересоздание pod:

```yaml
modelStorage:
  persistence:
    enabled: true
    existingClaim: triton-model-workspace
```

Создание нового PVC chart-ом:

```yaml
modelStorage:
  persistence:
    enabled: true
    size: 200Gi
    storageClassName: fast-rwo
    retain: true
```

Перед увеличением `replicaCount` используйте ReadWriteMany или отдельный PVC для
каждого pod.

По умолчанию watcher использует временную директорию Triton и создаёт внутри
неё `models-active`. Задавайте `gateway.watcherModelDir`, только если
временные checkout `folder*` создаются в другом месте, а
`gateway.modelsActiveDir` — если активным ссылкам нужен отдельный путь.
`gateway.tmpRoot` сохранён как совместимый alias.

## Временное хранилище media

Большим видео и PDF может не хватить writable layer контейнера. Подключите
отдельный media PVC:

```yaml
mediaPersistence:
  enabled: true
  size: 100Gi
  storageClassName: fast-rwo
  mountPath: /var/lib/triton/media
  useAsTmpDir: true
  retain: true
```

Временные файлы удаляются после обработки запроса. PVC является workspace, а не
архивом media.

## Shared memory

Chart монтирует memory-backed `emptyDir` в `/dev/shm`. Это устраняет маленький
контейнерный default, из-за которого ломаются IPC Python backend Triton и
большие мультимодальные запросы:

```yaml
sharedMemory:
  enabled: true
  sizeLimit: 8Gi
```

Использование shared memory учитывается в памяти pod. Для конкурентной
обработки больших файлов увеличивайте одновременно memory limit pod и
`sharedMemory.sizeLimit`.

## Admission и media limits

Начинайте с консервативных значений и настраивайте их под реалистичной нагрузкой:

```yaml
gateway:
  admission:
    maxInflight: "256"
    maxQueue: "512"
    chatMaxInflight: "64"
    chatMaxQueue: "256"
    mediaMaxInflight: "4"
    mediaMaxQueue: "16"
  nativeBackend:
    maxMediaBytes: "268435456"
    maxRequestBytes: "805306368"
    maxTotalVideoFrames: "256"
    maxPdfPages: "64"
    preprocessConcurrency: "2"
  vllmMultimodal:
    pdfChunkPages: 2
    videoFps: 1.0
    videoMaxFrames: 32
    videoChunkFrames: 4
```

Эти лимиты защищают CPU и host memory до поступления запроса на GPU.

## Логи и debug

```yaml
gateway:
  observability:
    logFormat: json
    logLevel: INFO
    debugEnabled: false
    debugLogPayloads: false
    uvicornAccessLog: false
```

Передайте `debug: true` в одном chat request для безопасных metadata цепочки.
`debugLogPayloads` может раскрыть prompts, tool results и user data, поэтому его
не следует включать в production.

## GPU metrics

### Встроенный DCGM

```yaml
triton:
  gpuMetrics:
    mode: builtin
```

Triton публикует `nv_gpu_*` на порту `8002`. Встроенный DCGM требует root; chart
использует root security context с выключенным privilege escalation и
сброшенными Linux capabilities.

### Внешний DCGM Exporter

```yaml
triton:
  gpuMetrics:
    mode: external
```

Используйте режим, если Triton должен работать non-root. Если GPU Operator уже
предоставляет DCGM Exporter, используйте его Service. Иначе включите сохранённую
dependency:

```yaml
dcgm-exporter:
  enabled: true
  serviceMonitor:
    enabled: false
```

Архив chart dependency включён в repository, поэтому для установки не нужен
`helm dependency update`.

## OpenTelemetry

```yaml
gateway:
  observability:
    generationTelemetry: true
    otel:
      enabled: true
      endpoint: http://otel-collector.observability.svc:4318/v1/traces
      sampleRatio: "0.05"
      serviceName: triton-openai-gateway

triton:
  tracing:
    enabled: true
    endpoint: http://otel-collector.observability.svc:4318/v1/traces
    level: TIMESTAMPS
    rate: 0
    count: -1
    serviceName: triton-inference-server
```

Gateway создаёт root span и передаёт W3C `traceparent` в Triton. При `rate: 0`
Triton трассирует только requests, выбранные sampler gateway. Prompts, media,
сгенерированный текст, результаты tools и reasoning content не экспортируются.

## Проверка

```bash
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
```

После установки:

```bash
kubectl -n inference get pods,service,pvc
kubectl -n inference port-forward service/triton-openai-gateway 8080:8080
curl -fsS http://127.0.0.1:8080/ready
```

Файлы моделей описаны в разделе [Конфигурация](../../docs/configuration.ru.md), а
runtime-рекомендации в разделе [Эксплуатация](../../docs/operations.ru.md).
