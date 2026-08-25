# Нативные hybrid embeddings BGE-M3

**Язык:** [English](README.md) | Русский

Этот профиль запускает `BAAI/bge-m3` через встроенный backend
`vllm_multimodal`. Стандартный `/v1/embeddings` сохраняет dense-ответ, а
`/v1/hybrid_embeddings` возвращает dense, лексический sparse либо оба
представления.

Поместите `config.pbtxt` рядом с version directory Triton. Файлы `model.json` и
`gateway.json` поместите в version directory вместе с локальными файлами
модели. Repository watcher также может автоматически заменить `model` на путь
до материализованных весов.

Ожидаемая структура:

```text
bge-m3/
|-- config.pbtxt
`-- 1/
    |-- model.json
    |-- gateway.json
    |-- config.json
    |-- model.safetensors        # либо совместимые shards
    |-- sparse_linear.pt
    |-- colbert_linear.pt
    |-- tokenizer.json
    `-- ...
```

Параметр `hf_overrides.architectures` обязателен: исходный config BGE-M3
указывает `XLMRobertaModel`, а для загрузки sparse- и ColBERT-head vLLM должен
выбрать нативную реализацию `BgeM3EmbeddingModel`. Не задавайте фиксированный
`pooler_config.task`: backend выбирает `embed`, `token_classify` или
`embed&token_classify` для каждого запроса.

Пример использует одну GPU с консервативными параметрами. Меняйте
`GPU_DEVICE_IDS`, parallelism, memory utilization, batch- и admission-лимиты
только после тестирования на целевом оборудовании. BGE-M3 не поддерживает
Matryoshka truncation, поэтому `dimensions` следует опустить либо указать
нативное значение `1024`.

В консервативном профиле chunked prefill выключен. Включайте его только после
GPU load test для dense, sparse и комбинированного pooling на настроенной
максимальной длине последовательности.

Требуется закреплённый runtime Triton 26.07/vLLM 0.24. Перед production
проведите GPU smoke test и сравните dense/sparse результаты с
[Python fallback](../bge-m3-hybrid/README.ru.md) на репрезентативных
многоязычных запросах.
