# Миграция на Triton 26.10

[Главная](../README.ru.md) / Миграция на Triton 26.10

**Язык:** [English](migration-26.10.md) | Русский

Классический образ переходит на runtime NVIDIA с закреплённым digest:

```text
nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3@sha256:06803bff18e43379a7ecb4ce46feed537ca717c06c5472b354aec94bf20466e2
```

Обновляйте vendor vLLM, Torch, Transformers, FlashInfer и нативные компоненты
Triton вместе. [Матрица 26.09](migration-26.09.ru.md) сохранена для предыдущего
runtime.

## Матрица runtime

Версии ниже соответствуют исследованному образу и gateway overlay. Точные
vendor build suffix закреплены намеренно:
[таблица совместимости NVIDIA](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/introduction/compatibility.html)
указывает vLLM `0.31.0+8977dc6f.nv26.10.72757140`, а этот digest содержит build
`0.31.0+88641915.nv26.10.75644909`, приведённый ниже.

| Компонент | Закреплённая версия |
| --- | --- |
| Triton Server | 2.74.0 |
| NVIDIA vLLM | 0.31.0+88641915.nv26.10.75644909 |
| Transformers | 5.17.0 |
| PyTorch | 2.15.0a0+875d815502.nv26.10.72655024 |
| FlashInfer | 0.7.1rc5+c6a6b324.nvinternal.26.10.fi.rc5.5943.pr6394.cu.73382939 |
| compressed-tensors | 0.17.0 |
| Python | 3.12.3 |
| Версия CUDA в образе | 13.4.3.001 |
| Triton client | 2.74.0 |
| grpcio | 1.84.0 |
| ml-dtypes | 0.5.4 |
| protobuf | 6.33.6 |
| NumPy | 1.26.4 |
| Pydantic | 2.14.1 |
| OpenAI SDK | 2.54.0 |
| FastAPI | 0.136.3 |
| Starlette | 1.9.0 |
| Uvicorn / uvloop | 0.54.0 / 0.23.0 |
| Pillow | 13.0.0 |
| OpenTelemetry API, SDK, OTLP exporter и OTLP HTTP exporter | 1.45.1 |

`docker/verify-runtime.py` проверяет версии runtime и overlay при сборке.
Triton frontend 2.74 допускает Pydantic `>=2.12,<3`; прежний точный pin
`2.10.6` больше не нужен. Overlay Triton client сохраняет совместимость с
protobuf `<7` и ml-dtypes `<=0.5.4`.

## Изменения, важные для gateway

В [release notes NVIDIA 2.74](https://github.com/triton-inference-server/server/releases/tag/v2.74.0)
указаны:

- Настраиваемый S3 addressing style.
- Более ранняя проверка отмены и исправления HTTP input/buffers.
- Смягчение ограничений зависимостей OpenAI/Pydantic.
- Поддержка vLLM `>=0.30` с рендерингом prompt перед generation и encoding.

Встроенный multimodal backend учитывает этот этап рендеринга для generation и
pooling. Аргументы engine остаются в `model.json`, настройки gateway — в
`gateway.json`. Нативная поддержка media зависит от модели: замена образа не
добавляет модальность неподдерживаемой архитектуре. Настройка S3 addressing не
заменяет проверку скачивания репозитория и file override.

## Оставшиеся ограничения совместимости

Классический образ сохраняет расхождения vendor metadata:

- Python binding Triton требует NumPy `<2`, а librosa 1.0.0 и OpenCV
  5.0.0.93 — NumPy 2.x. NumPy остаётся `1.26.4`.
- NVIDIA поставляет CUTLASS DSL `4.8.0.dev0`, хотя metadata установленного
  vLLM требует `4.7.1`. Vendor-пакет сохраняется.

Добавленный wheel Decord `0.6.0` также сохраняет upstream-ошибку внутреннего
platform tag: `pip check` считает его неподдерживаемым, хотя CPU-проверка
декодирования видео проходит. Узко квалифицированное исправление metadata для
отдельного окружения DeepSeek не применяется к системному Python классического
образа.

Gateway overlay исправляет конфликт базового ml-dtypes `0.6.0` с NumPy,
закрепляя ml-dtypes `0.5.4`. Проверка версий **не означает** чистый `pip check`
классического окружения. Не устраняйте оставшиеся конфликты отдельным обновлением
или откатом ключевых пакетов без проверки всего runtime.

Upstream также отмечает ошибки завершения DCGM/NSCQ на NVSwitch и ограничения
stock vLLM tensor parallelism при explicit model control. Проверяйте выбранную
distributed-конфигурацию, unload и завершение сервера. Миграция не переключает
engine на Ray автоматически.

## Дополнительное окружение DeepSeek

[Аддитивная сборка DeepSeek](../docker/README.deepseek.md) сохраняет отдельное
окружение Python 3.12 / vLLM `0.32.0+cu129` / Torch `2.13.0+cu129` / CUDA 12.9.
Обновление классической базы не удаляет этот engine и не обновляет его отдельно.

По сравнению с прежним закреплённым образом 26.09 проверенный NVIDIA fork также
регистрирует `DeepseekV41ForCausalLM` и содержит код native vision и загрузки
UE8M0. Официальный engine становится кандидатом на отдельную проверку
совместимости; это не подтверждает работу checkpoint на GPU и не переключает
backend проекта на этот engine. Chat templates и настройки reasoning отдельных
моделей сохраняются.

Точное исключение native ABI теперь соответствует stock Python stub из 26.10
и `libcudart.so.13.4.92`. Пути компилятора, библиотеки engine, кеши и дочерние
Python workers остаются изолированными. При сборке с новой базой повторно
проверьте настоящий stub и GPU: совпадение версий Python-пакетов само по себе
не доказывает совместимость native ABI.

## Развёртывание и откат

Собирайте новый образ вместо установки пакетов в работающий контейнер.
Тег Helm по умолчанию — `26.10-vllm-gateway`; фиксируйте неизменяемый digest и
сохраняйте ранее проверенный образ вместе с конфигурациями его моделей.

Перед переключением проверьте существующие профили: generation, SSE/tools,
JSON Schema, embeddings/rerank, включённые media, отмену и load/unload.
Включите проверку GPU workers, CUDA/JIT, памяти и завершения сервера. CPU
build/import тесты не подтверждают GPU inference и производительность.

Для отката верните записанный прежний образ и соответствующие конфиги,
загрузите исходный набор моделей, проверьте model-specific READY и ответ
инференса. Для DeepSeek нужен ранее проверенный аддитивный образ: обычный
классический образ этот backend не содержит.

Параметры развёртывания и эксплуатации описаны в
[конфигурации](configuration.ru.md) и [операциях](operations.ru.md).
