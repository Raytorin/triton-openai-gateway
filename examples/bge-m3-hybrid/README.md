# BGE-M3 hybrid embeddings

**Language:** English | [Русский](README.ru.md)

This Python-backend fallback serves dense and lexical sparse embeddings from
`BAAI/bge-m3` through both gateway endpoints. Prefer the
[native `vllm_multimodal` profile](../bge-m3-vllm-multimodal/README.md) when
vLLM scheduling and continuous batching are required.

The fallback exposes:

- `/v1/embeddings` returns the existing OpenAI-compatible dense response.
- `/v1/hybrid_embeddings` returns dense, sparse, or both output types.

Copy the Hugging Face model files into version directory `1/` beside `model.py`.
The directory must include the trained `sparse_linear.pt` head distributed with
the model. The backend loads local files only and never downloads weights at
runtime. It also keeps `trust_remote_code` disabled.

Expected layout:

```text
bge-m3/
|-- config.pbtxt
`-- 1/
    |-- gateway.json
    |-- model.py
    |-- config.json
    |-- pytorch_model.bin         # or compatible safetensors/shards
    |-- sparse_linear.pt
    |-- tokenizer.json
    `-- ...
```

Adjust `instance_group.gpus` in `config.pbtxt` before deployment. BGE-M3 does
not provide Matryoshka embeddings, so `dimensions` must be omitted or equal to
the native hidden size.
