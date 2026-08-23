# Native BGE-M3 hybrid embeddings

**Language:** English | [Русский](README.ru.md)

This profile runs `BAAI/bge-m3` through the bundled `vllm_multimodal` backend.
It preserves the standard dense `/v1/embeddings` response and enables dense,
lexical sparse, or combined output on `/v1/hybrid_embeddings`.

Copy `config.pbtxt` next to the Triton version directory. Copy `model.json` and
`gateway.json` into that version directory beside the local model files, or let
the repository watcher rewrite `model` to the materialized weights path.

Expected repository layout:

```text
bge-m3/
|-- config.pbtxt
`-- 1/
    |-- model.json
    |-- gateway.json
    |-- config.json
    |-- model.safetensors        # or compatible shards
    |-- sparse_linear.pt
    |-- colbert_linear.pt
    |-- tokenizer.json
    `-- ...
```

`hf_overrides.architectures` is required because the upstream BGE-M3 config
declares `XLMRobertaModel`; vLLM must resolve its native
`BgeM3EmbeddingModel` implementation to load the sparse and ColBERT heads.
Do not set a fixed `pooler_config.task`: the backend selects `embed`,
`token_classify`, or `embed&token_classify` per request.

The example uses one GPU conservatively. Adjust `GPU_DEVICE_IDS`, parallelism,
memory utilization, batch limits, and admission limits only after testing on
the target hardware. BGE-M3 does not support Matryoshka truncation, so omit
`dimensions` or use the native dimension `1024`.

The conservative profile disables chunked prefill. Enable it only after a GPU
load test covers dense, sparse, and combined pooling at the configured maximum
sequence length.

The pinned Triton 26.07/vLLM 0.24 runtime is required. Run a GPU smoke test
before production deployment and compare dense/sparse output with the
[Python fallback](../bge-m3-hybrid/README.md) on representative multilingual
queries.
