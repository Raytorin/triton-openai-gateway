# Sparse and hybrid embeddings

**Language:** English | [Русский](sparse-embeddings.ru.md)

## When to use a separate endpoint

The standard OpenAI endpoint `POST /v1/embeddings` returns only dense vectors.
The `return_sparse`, `output_type`, and `output_types` parameters do not apply
there.

For BGE-M3 lexical sparse vectors, use:

```text
POST /v1/hybrid_embeddings
```

This endpoint supports the following output modes:

| Mode | `output_types` | Result |
| --- | --- | --- |
| Sparse only | `["sparse"]` | `sparse_embedding` |
| Dense and sparse | `["dense", "sparse"]` | `embedding` and `sparse_embedding` |
| Dense only | `["dense"]` | `embedding` |

For ordinary dense embeddings, continue using `POST /v1/embeddings`.

## Requesting a sparse vector

If the LiteLLM base URL already ends in `/v1`:

```bash
export API_BASE="https://example.local/project-api/genai/litellm-webui/v1"
export API_TOKEN="sk-..."

curl -sS "$API_BASE/hybrid_embeddings" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": "Industrial equipment diagnostics",
    "output_types": ["sparse"],
    "sparse_top_k": 64
  }' | jq
```

For an internal TLS certificate, use `--cacert /path/to/company-ca.pem`.
The `-k` option disables certificate verification and is suitable only for
temporary diagnostics.

`sparse_top_k` is optional. It limits the result to the specified number of
highest nonzero weights. Smaller values reduce response and storage size but
may reduce retrieval recall.

## Requesting hybrid vectors

To obtain both representations in one request, change `output_types`:

```bash
curl -sS "$API_BASE/hybrid_embeddings" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "Industrial equipment diagnostics",
      "Predictive maintenance of pumps"
    ],
    "output_types": ["dense", "sparse"],
    "sparse_top_k": 128
  }' | jq
```

`input` can be a single string or an array of strings. For an array, the response
contains one `data` item per input text; `index` preserves the input order.

## Response format

A sparse-only response:

```json
{
  "object": "hybrid_embedding.list",
  "data": [
    {
      "object": "hybrid_embedding",
      "index": 0,
      "sparse_embedding": {
        "indices": [59, 559, 12769],
        "values": [0.0227, 0.0577, 0.1654]
      }
    }
  ],
  "model": "bge-m3",
  "output_types": ["sparse"],
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9
  }
}
```

`indices` and `values` are parallel arrays: `values[n]` is the weight for
`indices[n]`. Indices identify tokens in the model vocabulary, not word positions
in the input text. The numbers above illustrate the response structure.

In hybrid mode, the same item also contains a dense vector:

```json
{
  "object": "hybrid_embedding",
  "index": 0,
  "embedding": [-0.0272, -0.0493, 0.0158],
  "sparse_embedding": {
    "indices": [59, 559, 12769],
    "values": [0.0227, 0.0577, 0.1654]
  }
}
```

The example `embedding` array is shortened. The actual dimension depends on the
model.

## Calling from Python

This endpoint is an OpenAI API extension. The `client.embeddings.create(...)`
method calls `/v1/embeddings`, so it does not request sparse vectors. Use an
HTTP client:

```python
import os

import requests

api_base = os.environ["API_BASE"].rstrip("/")
response = requests.post(
    f"{api_base}/hybrid_embeddings",
    headers={"Authorization": f"Bearer {os.environ['API_TOKEN']}"},
    json={
        "model": "bge-m3",
        "input": "Industrial equipment diagnostics",
        "output_types": ["sparse"],
        "sparse_top_k": 64,
    },
    timeout=300,
)
response.raise_for_status()
sparse = response.json()["data"][0]["sparse_embedding"]
```

With a local LiteLLM and model deployment, this request does not require OpenAI
or Internet access.

## Compatible parameters

Use `output_types` for new integrations. Compatibility aliases are also accepted:

- `"output_type": "sparse"` — sparse only;
- `"output_type": "dense"` — dense only;
- `"output_type": "hybrid"` — dense and sparse;
- `"return_sparse": true` — dense and sparse.

Do not combine aliases with `output_types`: the server returns a validation
error. Sending these parameters through `extra_body` to `/v1/embeddings` does not
switch endpoints either; change the URL explicitly.

## If the endpoint is unavailable

`Pass-through endpoint /v1/hybrid_embeddings not found` means that LiteLLM has
no registered route for this endpoint. Ask the service administrator to configure
a passthrough route to Triton OpenAI Gateway.

An error saying that the model is not configured for hybrid embeddings means
that the selected model or backend does not support sparse output. Use a BGE-M3
model configured for hybrid embeddings by the service administrator.
