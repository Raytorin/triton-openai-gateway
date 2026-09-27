# Sparse и hybrid embeddings

## Когда нужен отдельный endpoint

Стандартный OpenAI endpoint `POST /v1/embeddings` возвращает только dense-вектор.
Параметры `return_sparse`, `output_type` и `output_types` в нём не применяются.

Для лексических sparse-векторов BGE-M3 используйте:

```text
POST /v1/hybrid_embeddings
```

Этот endpoint позволяет запросить:

| Режим | `output_types` | Результат |
| --- | --- | --- |
| Только sparse | `["sparse"]` | `sparse_embedding` |
| Dense и sparse | `["dense", "sparse"]` | `embedding` и `sparse_embedding` |
| Только dense | `["dense"]` | `embedding` |

Для обычного dense-вектора рекомендуется продолжать использовать
`POST /v1/embeddings`.

## Запрос только sparse-вектора

Если базовый адрес LiteLLM уже заканчивается на `/v1`:

```bash
export API_BASE="https://example.local/project-api/genai/litellm-webui/v1"
export API_TOKEN="sk-..."

curl -sS "$API_BASE/hybrid_embeddings" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": "Диагностика промышленного оборудования",
    "output_types": ["sparse"],
    "sparse_top_k": 64
  }' | jq
```

Для внутреннего TLS-сертификата используйте
`--cacert /path/to/company-ca.pem`. Параметр `-k` отключает проверку сертификата
и подходит только для временной диагностики.

`sparse_top_k` необязателен. Он ограничивает результат указанным количеством
наиболее значимых ненулевых весов. Меньшее значение сокращает размер ответа и
хранения, но может снизить полноту поиска.

## Запрос hybrid-вектора

Чтобы получить оба представления одним запросом, измените `output_types`:

```bash
curl -sS "$API_BASE/hybrid_embeddings" \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bge-m3",
    "input": [
      "Диагностика промышленного оборудования",
      "Предиктивное обслуживание насосов"
    ],
    "output_types": ["dense", "sparse"],
    "sparse_top_k": 128
  }' | jq
```

`input` может быть одной строкой или массивом строк. Для массива в ответе будет
по одному элементу `data` на каждый входной текст; поле `index` сохраняет их
порядок.

## Формат ответа

Ответ только со sparse-вектором:

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

`indices` и `values` являются параллельными массивами: значению
`indices[n]` соответствует вес `values[n]`. Индексы обозначают идентификаторы
токенов в словаре модели, а не позиции слов во входном тексте.

В hybrid-режиме тот же элемент дополнительно содержит dense-вектор:

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

Массив `embedding` в примере сокращён. Реальная размерность зависит от модели.

## Вызов из Python

Endpoint является расширением OpenAI API. Метод
`client.embeddings.create(...)` всегда обращается к `/v1/embeddings` и поэтому
не запрашивает sparse-вектор. Используйте обычный HTTP-клиент:

```python
import os

import requests

api_base = os.environ["API_BASE"].rstrip("/")
response = requests.post(
    f"{api_base}/hybrid_embeddings",
    headers={"Authorization": f"Bearer {os.environ['API_TOKEN']}"},
    json={
        "model": "bge-m3",
        "input": "Диагностика промышленного оборудования",
        "output_types": ["sparse"],
        "sparse_top_k": 64,
    },
    timeout=300,
)
response.raise_for_status()
sparse = response.json()["data"][0]["sparse_embedding"]
```

Этот запрос выполняется через локальный LiteLLM и не требует доступа к OpenAI
или интернету.

## Совместимые параметры

Рекомендуемый параметр — `output_types`. Для совместимости также принимаются:

- `"output_type": "sparse"` — только sparse;
- `"output_type": "dense"` — только dense;
- `"output_type": "hybrid"` — dense и sparse;
- `"return_sparse": true` — dense и sparse.

Не передавайте aliases одновременно с `output_types`: сервер вернёт ошибку
валидации. Передача этих параметров через `extra_body` в `/v1/embeddings` также
не переключает endpoint; URL необходимо изменить явно.

## Если endpoint недоступен

Сообщение `Pass-through endpoint /v1/hybrid_embeddings not found` означает, что
маршрут не зарегистрирован в LiteLLM. Обратитесь к администратору сервиса:
endpoint должен быть настроен как pass-through на Triton OpenAI Gateway.

Ошибка о том, что модель не настроена для hybrid embeddings, означает, что
выбранная модель или её backend не поддерживает sparse-выход. Используйте
предоставленную администратором модель BGE-M3 с поддержкой hybrid embeddings.
