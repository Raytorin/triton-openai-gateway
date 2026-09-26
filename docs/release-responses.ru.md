# Первый релиз Responses — не опубликован

**Язык:** [English](release-responses.md) | Русский

Добавлены POST /v1/responses и /responses без хранения: текст, изображения
с ограничениями, клиентские функции, JSON/JSON Schema и Responses SSE.
Генерация, LoRA, политика reasoning и admission общие с Chat.

## Миграция

Автоматический default вывода Chat меняется с 256 на 4096, с thinking — на 8192.
Для старого fallback задайте GATEWAY_DEFAULT_OUTPUT_TOKENS и
GATEWAY_REASONING_DEFAULT_OUTPUT_TOKENS равными 256. Потолок сервера по умолчанию —
32768; модель может его снизить. Явные лимиты не уменьшаются молча. Chat-алиасы
должны совпадать и принимать положительные целые числа. Отсутствие/null сохраняет
помещающуюся историю и уменьшает бюджет до свободного контекста. Runtime context
нужно задать в model.json max_model_len или gateway.json generation.context_window.

Responses использует store=false, в том числе при отсутствии параметра.
Хранение, background, продолжение серверного диалога, генерация audio/images,
compact и legacy completions не входят в релиз. Перед миграцией клиентов
изучите [таблицу совместимости](responses.ru.md).

## Проверки и поставка

Локальный CI включает регрессионные тесты и HTTP fixtures с закреплёнными
версиями OpenAI SDK/LiteLLM. Они проверяют маршрут /v1/responses и переданный
лимит вывода, но не доказывают совместимость развёрнутого прокси или GPU runtime.
Runtime smoke остаётся отдельным условием выпуска и требует явного согласования
для ресурсов DevZone. На используемых транспортах проверьте disconnect и остановку
backend. Stateful-возможности остаются следующим релизом; SQLite и контракт
идентичности доверенного прокси этим изменением не включаются.
