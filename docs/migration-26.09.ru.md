# Миграция на Triton 26.09

[Главная](../README.ru.md) / Миграция на Triton 26.09

**Язык:** [English](migration-26.09.md) | Русский

Gateway использует единый NVIDIA runtime, закреплённый по digest. Обновляйте
Triton, vLLM, PyTorch, Transformers, FlashInfer и `compressed-tensors` вместе с
базовым образом:

```text
nvcr.io/nvidia/tritonserver:26.09-vllm-python-py3@sha256:ff41460e8fb1a831e37d368f531e5f40761baae03a07adac2f4cde2fc2c953d1
```

## Матрица runtime

| Компонент | Закреплённая версия |
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
| OpenTelemetry API, SDK, OTLP exporter и OTLP HTTP exporter | 1.45.1 |

`docker/verify-runtime.py` проверяет установленные версии при сборке образа.
`tritonclient[grpc]==2.73.0` требует `grpcio >= 1.81.1`,
`ml-dtypes <= 0.5.4` и `protobuf < 7`; gateway закрепляет версии из таблицы.

## Ограничения совместимости

Базовый образ NVIDIA содержит расхождения dependency metadata для требований
vLLM к Pydantic, OpenAI и CUTLASS и требования librosa к NumPy. Gateway
сохраняет NumPy `1.26.4` для Triton и Pydantic `2.10.6` для своего frontend.
Проверка версий не означает чистый `pip check`. Не устраняйте эти предупреждения
независимым обновлением ключевых пакетов runtime.

Штатный wrapper Triton `vllm` не предоставляет нативный tensor contract проекта
для видео, аудио и PDF. Для таких входов используйте встроенный backend
`vllm_multimodal` и preprocessing gateway. Его engine arguments соответствуют
vLLM `0.29` из закреплённого образа; храните настройки gateway в
`gateway.json`, отдельно от `model.json`.

Используйте заново собранный образ gateway в Docker или Helm. Обновляйте
image tags конкретного окружения вместе с runtime; по умолчанию Helm chart
использует `26.09-vllm-gateway`. Ёмкость модели, GPU parallelism и media limits
по-прежнему зависят от выбранной модели и оборудования.

Файлы модели описаны в [конфигурации](configuration.ru.md), а рекомендации по
развёртыванию и ёмкости — в [эксплуатации](operations.ru.md).
