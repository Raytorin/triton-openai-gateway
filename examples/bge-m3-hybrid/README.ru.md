# Гибридные embeddings BGE-M3

**Язык:** [English](README.md) | Русский

Этот fallback на Python backend возвращает dense и лексические sparse
embeddings `BAAI/bge-m3` через два endpoint gateway. Если нужны scheduling и
continuous batching vLLM, используйте
[нативный профиль `vllm_multimodal`](../bge-m3-vllm-multimodal/README.ru.md).

Fallback публикует:

- `/v1/embeddings` сохраняет существующий OpenAI-совместимый dense-ответ;
- `/v1/hybrid_embeddings` возвращает dense, sparse либо оба представления.

Поместите файлы модели Hugging Face в version directory `1/` рядом с
`model.py`. В каталоге должен находиться обученный sparse-head
`sparse_linear.pt`, входящий в состав модели. Backend загружает только локальные
файлы, не скачивает веса во время запуска и не разрешает `trust_remote_code`.

Ожидаемая структура:

```text
bge-m3/
|-- config.pbtxt
`-- 1/
    |-- gateway.json
    |-- model.py
    |-- config.json
    |-- pytorch_model.bin         # либо совместимые safetensors/shards
    |-- sparse_linear.pt
    |-- tokenizer.json
    `-- ...
```

Перед развёртыванием укажите нужную GPU в `instance_group.gpus` файла
`config.pbtxt`. BGE-M3 не поддерживает Matryoshka embeddings, поэтому
`dimensions` следует опустить либо указать нативный hidden size модели.
