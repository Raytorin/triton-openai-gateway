# Миграция на Triton 26.07

[Главная](../README.ru.md) / Миграция на Triton 26.07

**Язык:** [English](migration-26.07.md) | Русский

Проект использует один NVIDIA runtime, закреплённый по digest. Triton, vLLM,
PyTorch, Transformers, FlashInfer и `compressed-tensors` обновляются как единая
проверяемая матрица, а не как независимые пакеты.

## Матрица runtime

```text
nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3@sha256:31e20bfbc65055d6b85553a00d45342a62c506874aa996c11ae53650151f05b1
```

| Компонент | 26.06 | 26.07 |
| --- | --- | --- |
| Triton Server | 2.70.0 | 2.71.0 |
| NVIDIA vLLM | 0.22.1 | 0.24.0 |
| Transformers | 5.6.0 | 5.6.1 |
| FlashInfer | 0.6.12 | 0.6.14 |
| Python | 3.12 | 3.12 |
| CUDA | 13.3.0 | 13.3.4.1 |
| NCCL | 2.30.4 | 2.30.7 |

`docker/verify-runtime.py` проверяет полную установленную матрицу при сборке
образа. `tritonclient[grpc]==2.71.0` требует `grpcio < 1.68` и `protobuf < 7`,
поэтому gateway намеренно закрепляет `grpcio==1.67.1` и `protobuf==6.33.6`, а не
оставляет версии базового образа.

## Значимые изменения

Triton 26.07 улучшает model readiness и исправляет deadlock и use-after-free в
асинхронном/decoupled выполнении Python backend. Эти исправления важны для
явного load/unload моделей и decoupled backend `vllm_multimodal`.

В NVIDIA OpenAI frontend также появился лимит буфера streaming tool calls, но
проект не использует этот frontend. Собственные лимиты gateway и parsing tools
по-прежнему необходимы.

## Совместимость multimodal

Штатный wrapper Triton `vllm` не предоставляет нативный tensor contract проекта
для видео, аудио и PDF. Поэтому встроенный backend `vllm_multimodal` и
ограниченный по ресурсам preprocessing gateway остаются частью поддерживаемой
архитектуры. Custom backend следует интерфейсам vLLM 0.24 из закреплённого образа.

## Проверка перед production

Перед продвижением выполните проверки на целевой GPU-платформе:

1. Соберите образ без переопределения `BASE_IMAGE`; runtime verification должен пройти.
2. Проверьте `/ready`, `/v1/models`, readiness Triton и оба metrics endpoint.
3. Проверьте chat, streaming, reasoning, tools, embeddings и rerank.
4. Проверьте image, video, audio и PDF через `vllm_multimodal`.
5. Выполните циклы load/unload и проверьте завершение дочерних процессов.
6. Проведите canary для каждой production-топологии TP/PP/DP/EP.
7. Сравните TTFT, token throughput, GPU memory и качество ответов с 26.06.

## Первичные источники

- [Release notes NVIDIA Triton 26.07](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-07.html)
- [Release notes NVIDIA vLLM 26.07](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/rel-26-07.html)
