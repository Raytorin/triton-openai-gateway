# Responses API

**Язык:** [English](responses.md) | Русский

`POST /v1/responses` и `POST /responses` выполняют генерацию без хранения,
с JSON или Responses SSE. Поддерживается описанное подмножество OpenAI API.
Chat, Responses, выбор LoRA и admission используют общий сервис генерации.

## Поддерживаемый контракт

| Параметр / возможность | Поведение |
| --- | --- |
| `model`, строковый `input` | Обязательные; применяется chat template модели |
| Список сообщений `input`, `instructions` | Роли user, assistant, system, developer; developer преобразуется в system |
| `input_text`, `output_text` | Текстовые части, включая историю ответов assistant |
| `input_image.image_url` | HTTP(S) или image data URL; только сообщения user |
| `tools` | Клиентские функции: name, description, parameters, strict=false/null |
| `tool_choice` | auto, none, required или `{type: function, name: ...}` |
| `function_call`, `function_call_output` | История вызовов и строковых результатов с совпадающим call_id; повторы и несвязанные вызовы отклоняются |
| `text.format` | text, json_object или json_schema с name/schema/strict; необходима поддержка constrained output в backend |
| `temperature`, `top_p` | Передаются существующему слою sampling |
| `max_output_tokens` | Положительное целое или null; общая политика с Chat |
| `lora_name` | Расширение gateway; маршрутизация как в Chat |
| `metadata` | Возвращается в ответе, не сохраняется |
| `truncation` | disabled по умолчанию или auto; скрытого суммаризирования нет |
| `stream` | События Responses, а не Chat chunks |
| `store`, `background` | Только отсутствие/null/false; хранения и фонового выполнения нет |
| `reasoning` | Только пустой объект/null; thinking настраивается для модели |
| `parallel_tool_calls` | Отсутствие/null/true; false отклоняется, backend не гарантирует последовательные вызовы |

Неподдерживаемые параметры возвращают 400 до инференса: встроенные инструменты,
strict для аргументов функций, file_id, входные audio/video/PDF, text verbosity,
непустые настройки reasoning, previous_response_id, conversation, store=true,
background=true. Audio/video/PDF доступны через Chat. Получение/удаление/отмена
по ID, Conversations, compact, генерация audio/images и legacy `/v1/completions`
не входят в этот релиз. Функции выполняет клиент.

**Отличие default от OpenAI:** опущенный `store` означает **false**. Каждый ход
передаёт собственную историю; ID не обозначают доступные для получения объекты.
В следующий input добавляются вызовы функций и результаты с тем же call_id.
Вызовы и результаты должны предшествовать следующему сообщению.
Instructions передаёт клиент.

## Политика токенов и контекста

Chat принимает `max_completion_tokens` или legacy-алиас `max_tokens`, Responses —
`max_output_tokens`. Равные Chat-алиасы допустимы, разные возвращают 400.
Boolean, строки, дробные, нулевые и отрицательные лимиты отклоняются.
Отсутствие/null выбирает минимум из default профиля, потолка сервера/модели и
свободного контекста. Defaults: 4096 обычно и 8192 с thinking; начальный глобальный
потолок — 32768. Это эксплуатационная настройка, а не характеристика модели.

Из настроенного runtime context вычитаются инструкции, tools, история,
дополнительный резерв изображений и safety margin (по умолчанию 64).
Из media-резерва вычитаются уже учтённые в prompt placeholders. Резерв изображений —
консервативная оценка по лимиту пикселей, а не точный подсчёт processor.
После сокращения истории оставшиеся media пересчитываются. Изображения проходят
существующий загрузчик с ограничениями и декодер, уменьшаются до image_max_pixels
и преобразуются в JPEG. Поле detail принимается, обработкой управляет лимит пикселей.
Vision определяется по локальному config.json; для других проверенных vision
runtime задайте generation.supports_vision=true в gateway.json.
Значение false отклоняет изображения до инференса.

Помещающаяся история сохраняется, даже если на вывод остаётся меньше default.
Явный лимит не меняется молча. `truncation=auto` удаляет только старейшие завершённые
ходы, сохраняя инструкции, последний запрос user и связанные вызовы/результаты
функций. При disabled переполнение возвращает ошибку. Thinking, текст и аргументы
функций расходуют общий бюджет. Исчерпание даёт status=incomplete и
incomplete_details.reason=max_output_tokens, в том числе до первого видимого
текста. Автоматического повтора генерации нет.

Текущие текстовые транспорты Triton не предоставляют достоверных usage и finish
metadata; usage и исчерпание лимита оцениваются повторной токенизацией.
Это обозначено `X-Token-Usage-Source: estimated`. Заголовки X-Output-Token-Limit
и X-Output-Token-Limit-Source показывают выбранный бюджет. Сырой thinking не
выдаётся за OpenAI reasoning summary. Defaults, caps и обязательная настройка
runtime context описаны в [конфигурации](configuration.ru.md). Есть
[пример gateway.json](../examples/responses/gateway.json); окно контекста
в нём необходимо привести к реальной настройке runtime.

## Пример Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="configured-by-proxy")
response = client.responses.create(
    model="your-model", input="Привет", store=False, max_output_tokens=128,
)
print(response.output_text)

with client.responses.stream(model="your-model", input="Привет", store=False) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
    final = stream.get_final_response()
```

Функции описываются плоским форматом Responses (`{"type":"function","name":"..."}`),
а не вложенной формой Chat. В многоходовом диалоге добавляйте response.output
в виде dict и соответствующие function_call_output к исходной истории input.

## Диалог без хранения

База данных и volume для Responses не нужны. Приложение клиента хранит историю
и повторно передаёт её в каждом запросе. `instructions` также передаются каждый
раз. Например, после ответа на первый вопрос:

```python
history = [{"role": "user", "content": "Назови столицу Франции"}]
first = client.responses.create(
    model="your-model", input=history, instructions="Отвечай кратко",
    store=False, max_output_tokens=128,
)
if first.status != "completed":
    raise RuntimeError(f"Ответ не завершён: {first.status}")
history.extend(item.model_dump(exclude_none=True) for item in first.output)
history.append({"role": "user", "content": "А какая река через неё протекает?"})
second = client.responses.create(
    model="your-model", input=history, instructions="Отвечай кратко",
    store=False, max_output_tokens=128,
)
print(second.output_text)
```

При использовании tools до следующего сообщения добавьте результаты всех вызовов
с соответствующими `call_id`. Не выполняйте незавершённый вызов из ответа
`status=incomplete`: аргументы могут быть обрезаны. `output_text` содержит только
текст; полный `output` нужен для сохранения вызовов функций.

`store=false` означает отсутствие сохраняемого объекта Responses. Это не настройка
логирования: gateway debug/prompt logging и журналы LiteLLM настраиваются отдельно.
ID ответа служит для согласования событий текущего ответа; получить результат
позже по ID или продолжить через `previous_response_id` нельзя. Повторный POST
создаёт новую генерацию, дедупликация запросов не предоставляется. Если приложение
автоматически повторяет запросы, оно должно отдельно контролировать повторное
выполнение клиентских функций.

## Streaming и ошибки

События: response.created/in_progress; output_item.added/done;
content_part.added/done; output_text.delta/done; function_call_arguments.delta/done;
и ровно одно терминальное response.completed/incomplete/failed. sequence_number
возрастает; item ID, call ID и output_index согласованы на протяжении потока.
После терминального события поток заканчивается без Chat-маркера `[DONE]`.
Некоторые Python и tool-aware пути Triton буферизуют генерацию перед выдачей.
На этих путях Responses не обещает инкрементальную генерацию аргументов функций.

Ошибки до заголовков возвращаются HTTP-ответом. После заголовков выдаются error
и response.failed. Disconnect закрывает backend iterator и освобождает admission,
включая этап подготовки. Отмена передаётся в gRPC; фактическую остановку GPU на
конкретном backend нужно проверить runtime smoke. Закрытие HTTP-транспорта
не гарантирует немедленную остановку GPU.

Request ID добавляет существующий middleware. Метрики содержат два фиксированных
маршрута; отдельные response/item ID в Prometheus labels не попадают.

## Совместимость и приёмка релиза

requirements-client.txt закрепляет OpenAI SDK 2.54.0 и LiteLLM 1.102.1.
Локальный HTTP fixture проверяет текст/output_text, сборку потока SDK, цикл
function calling, потоковые tools и JSON Schema обоими клиентами. Фиксируются
переданные маршруты и лимиты. При model="openai/test" и api_base=.../v1 LiteLLM
обращается к /v1/responses и не добавляет default лимита. У развёрнутого прокси
могут отличаться версия, маршрутизация и defaults.

```bash
python -m venv .local/client-venv
.local/client-venv/bin/python -m pip install -r requirements-client.txt
.local/client-venv/bin/python tests/client_compatibility.py --server-python .venv/bin/python
```

Fixture не проверяет реальный Triton GPU или развёрнутый LiteLLM-прокси.
До релиза нужен согласованный runtime smoke: текст, изображения, функции,
JSON Schema, длинный/thinking вывод и disconnect. Проверяются фактический маршрут,
max_output_tokens, остановка backend и освобождение admission.
Локальный CI не обращается к DevZone и ничего туда не развёртывает.
