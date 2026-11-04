# Visual input usage

The gateway exposes processor-reported visual embedding positions separately
from its conservative context reservation. Counts are model-specific input
positions, not a universal price or a measurement of GPU time.

Images, rendered PDF pages and video frames can all consume visual positions.
Their original source is retained as `image`, `pdf` or `video`, including when
pages/frames are passed to a model as ordinary images.

## API fields

Chat, Responses and their final SSE usage packets may include these additive
extensions:

```json
{
  "prompt_tokens": 512,
  "completion_tokens": 10,
  "total_tokens": 522,
  "vision_usage": {
    "main": {
      "status": "available",
      "source": "engine",
      "unit": "embedding_positions",
      "by_source": {"image": 256},
      "calls": 1
    },
    "preprocessing": {
      "status": "available",
      "source": "engine",
      "unit": "embedding_positions",
      "by_source": {"pdf": 768, "video": 256},
      "calls": 2
    }
  },
  "vision_context_reservation": {
    "source": "estimated",
    "unit": "context_positions",
    "by_scope": {"main": 1023}
  }
}
```

Responses uses `input_tokens`/`output_tokens` instead of
`prompt_tokens`/`completion_tokens` and preserves the same extensions.
These extensions are gateway-specific; applications should tolerate absent
fields and SDKs may handle custom usage fields differently.

`main` describes calls for the requested answer; `preprocessing` describes
auxiliary visual summaries, including executed adaptive splits or retries.
`calls` counts those actual transport attempts. With map/reduce preprocessing,
the final main prompt can be text-only while preprocessing has visual usage.
Existing totals continue to describe the main response. Do not add main visual
positions to `prompt_tokens` again, and do not treat preprocessing positions
as part of that final prompt's context.

- `available`: all observed calls in that scope supplied verified processor
  facts and completed normally.
- `partial`: some processor facts are known, but a call failed, was cancelled,
  lacked terminal metadata or had unavailable accounting. Counts represent
  only the observed portion.
- `unavailable`: no trustworthy visual breakdown was observed. An empty
  `by_source` is unknown accounting, not zero consumption.

Scopes without visual calls are omitted. Known zero counts remain distinguishable
from unavailable accounting. Reservation fields are supplied only where the
visual portion of the applied context budget can be identified; a combined
audio/vision reserve is not split into invented visual figures.

## Prometheus

| Metric | Meaning |
| --- | --- |
| `gateway_visual_embedding_positions_total` | Actual observed visual positions; labels `model`, `route`, `source_kind`, `scope`, `source="engine"` |
| `gateway_visual_usage_requests_total` | Request accounting availability by `model`, `route`, `scope`, `status` |
| `gateway_visual_context_positions_reserved_total` | Separate conservative reservation, including service delimiters; labels `model`, `route`, `scope`, `source="estimated"` |

For example, observed visual positions per second by model and input origin:

```promql
sum by (model, source_kind, scope) (
  rate(gateway_visual_embedding_positions_total{source="engine"}[5m])
)
```

Inspect coverage alongside that rate:

```promql
sum by (model, scope, status) (
  increase(gateway_visual_usage_requests_total[1h])
)
```

Repeated cumulative packets and repeated request finalization do not add the
same positions twice. Already observed facts remain counted after disconnects
or errors, with partial status where appropriate. Labels do not contain media
content, URLs, filenames or request identifiers.

## Runtime support

Exact accounting requires a supported native vLLM renderer and decoder-only
processed input. The backend prepares the input once, reads each validated
placeholder's `get_num_embeds()` (including its mask), and passes that same
input to generation. Unsupported APIs, encoder/decoder layouts or mismatched
origin/range data result in unavailable accounting. Pixel formulas, raw image
token IDs, range lengths and gateway reservation are never substituted as
consumed visual usage. Generation remains compatible with legacy runtimes.

Local tests verify protocol, ordering, masks, accumulation and SDK transport.
They do not establish a GPU memory fit or qualify image inference on a
particular GPU/checkpoint. See [DeepSeek opt-in images](../examples/deepseek-v4.1-flash/README.md#enable-images).
