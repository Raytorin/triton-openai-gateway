# Учёт визуальных токенов

Gateway передаёт фактическое число visual embedding positions, полученное от
процессора модели, отдельно от консервативного резерва контекста. Это входные
позиции конкретной модели, а не универсальная цена или время работы GPU.

Изображения, страницы PDF и видеокадры могут расходовать визуальные позиции.
Их происхождение сохраняется как `image`, `pdf` или `video`, даже если страницы
и кадры передаются модели как обычные картинки.

## Поля API

Chat, Responses и финальные usage-события SSE могут содержать расширения:

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

Responses использует `input_tokens`/`output_tokens` вместо
`prompt_tokens`/`completion_tokens` и сохраняет те же расширения. Эти поля
добавлены нашим gateway; приложение должно допускать их отсутствие. SDK могут
по-разному обрабатывать нестандартные поля usage.

`main` относится к вызовам для запрошенного ответа, `preprocessing` — к
вспомогательным визуальным сводкам, включая выполненные повторы и адаптивное
разбиение. `calls` — число этих фактических попыток обращения к backend. При
map/reduce финальный prompt может быть текстовым, хотя preprocessing расходовал
визуальные токены. Обычные totals продолжают описывать основной ответ.
Визуальные позиции main уже входят в prompt: повторно прибавлять их нельзя.
Расход preprocessing не входит в контекст финального prompt.

- `available`: все наблюдаемые вызовы этого scope передали проверенные факты
  процессора и завершились нормально.
- `partial`: часть фактов известна, но был отказ, отмена, отсутствие terminal
  metadata или недоступный учёт другого вызова. Числа отражают известную часть.
- `unavailable`: достоверная детализация не получена. Пустой `by_source`
  означает неизвестный расход, а не нулевой.

Scope без визуальных вызовов отсутствует. Известный ноль отличается от
недоступной статистики. Поле резерва появляется там, где можно выделить
визуальную часть применённого бюджета. Общий резерв audio/vision не делится
на вымышленные компоненты.

## Prometheus

| Метрика | Значение |
| --- | --- |
| `gateway_visual_embedding_positions_total` | Наблюдаемые фактические позиции; labels `model`, `route`, `source_kind`, `scope`, `source="engine"` |
| `gateway_visual_usage_requests_total` | Доступность учёта по `model`, `route`, `scope`, `status` |
| `gateway_visual_context_positions_reserved_total` | Отдельный консервативный резерв со служебными разделителями; labels `model`, `route`, `scope`, `source="estimated"` |

Пример количества наблюдаемых визуальных позиций в секунду:

```promql
sum by (model, source_kind, scope) (
  rate(gateway_visual_embedding_positions_total{source="engine"}[5m])
)
```

Доступность статистики следует проверять вместе с расходом:

```promql
sum by (model, scope, status) (
  increase(gateway_visual_usage_requests_total[1h])
)
```

Повторные cumulative-пакеты и повторное завершение запроса не удваивают числа.
Уже полученные факты сохраняются при disconnect и ошибках; при необходимости
указывается partial. В labels нет содержимого media, URL, имён файлов и ID
пользовательских запросов.

## Поддержка runtime

Точный учёт требует поддерживаемого native renderer vLLM и decoder-only input.
Backend готовит input один раз, читает `get_num_embeds()` каждого проверенного
placeholder с учётом mask и передаёт тот же input в генерацию. Неподдерживаемый
API, encoder/decoder layout или несовпадение origins/ranges дают unavailable.
Размеры pixels, сырые image token IDs, длина ranges и резерв gateway не
выдаются за фактический расход. Старые runtime продолжают генерировать по
совместимому пути.

Локальные тесты проверяют протокол, порядок, masks, накопление и передачу через
SDK. Для конкретного GPU/checkpoint отдельно нужны проверка памяти и реальная
vision inference. См. [включение изображений DeepSeek](../examples/deepseek-v4.1-flash/README.md#enable-images).
