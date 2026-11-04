# Участие в разработке

**Язык:** [English](CONTRIBUTING.md) | Русский

Спасибо за вклад в Triton OpenAI Gateway.

## Перед началом

- Создайте issue для изменений public API, структуры моделей или поддерживаемых
  версий runtime.
- Не объединяйте обновление runtime с несвязанными функциями.
- Сохраняйте copyright NVIDIA и BSD-заголовки в производных backend-файлах.
- Поддерживайте актуальность `NOTICE`, `AUTHORS.md` и атрибуции стороннего кода.
- Не добавляйте веса моделей, credentials, generated cache и user payloads.

## Процесс разработки

1. Создайте fork и отдельную ветку для изменения.
2. Внесите минимальное законченное изменение.
3. Добавьте или обновите unit tests в `tests/`.
4. Обновите обе языковые версии документации и примеры.
5. Выполните CPU-проверки локально и проверки совместимости в закреплённом образе Triton.
6. Создайте pull request с описанием поведения, совместимости и тестов.

## Сборка

```bash
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:dev .
```

Сборка проверяет точные версии runtime-зависимостей. Обновление base image или
dependency должно одновременно менять `docker/verify-runtime.py`, requirements,
документацию совместимости и tests.

## Тесты

Создайте окружение Python 3.12 для CPU-safe проверок:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --requirement requirements-test.txt
.venv/bin/python -m pytest -q \
  --cov=gateway \
  --cov=backends/vllm_multimodal \
  --cov=backends/vllm_deepseek \
  --cov-fail-under=75
.venv/bin/python tests/runtime_media_smoke.py
.venv/bin/cffconvert --validate
.venv/bin/python scripts/audit_dependencies.py
```

GitHub Actions выполняет два Linux job для pull request, push в `dev`/`main`
и ручного запуска. В начале проверяется синтаксис tracked Python, JSON и Bash;
actionlint и ShellCheck проверяют workflow и shell-скрипты. Зависимости тестов,
runtime и SDK проходят аудит отдельно. Прежнее узкое исключение аудита runtime
описано в [SECURITY.ru.md](SECURITY.ru.md).

JUnit и coverage сохраняются на семь дней. Загружаются только эти два XML-файла;
локальные заметки, виртуальные окружения, пользовательские данные и runtime
dumps не включаются. Набор test-зависимостей намеренно меньше GPU runtime и не
должен использоваться для сборки production-образа.

Проверка совместимости SDK в отдельном клиентском окружении:

```bash
python3.12 -m venv .local/client-venv
.local/client-venv/bin/python -m pip install --requirement requirements-client.txt
.local/client-venv/bin/python tests/client_compatibility.py \
  --server-python "$PWD/.venv/bin/python"
```

Отдельный PDF/video preprocessing smoke test:

```bash
docker run --rm \
  --entrypoint python3 \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  triton-openai-gateway:dev \
  tests/runtime_media_smoke.py
```

Проверка контрактов реального API vLLM на CPU:

```bash
docker run --rm --network none \
  --entrypoint python3 \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  triton-openai-gateway:dev \
  tests/runtime_vllm_smoke.py
docker run --rm --network none \
  --entrypoint python3 \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  triton-openai-gateway:dev \
  tests/runtime_stock_vllm_smoke.py
```

Проверка охватывает контракты sampling, structured output, LoRA, pooling,
метрик и жизненного цикла engine через установленные API vLLM без GPU inference.
Отсутствие runtime-зависимостей завершает проверку с ошибкой, а не пропуском.

Проверка shell и Helm:

```bash
python3.12 scripts/check_repository.py
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
helm template test ./helm/triton-gateway \
  --values helm/triton-gateway/values.production.example.yaml >/dev/null
helm lint ./helm/dcgm-exporter
```

Обычный CI не скачивает веса моделей и не собирает большие образы Triton.
Проверки реального API vLLM выше требуют готового образа; native DeepSeek и
GPU inference проверяются отдельно от CPU suite на GitHub-hosted runner.

Изменения GPU и end-to-end media также требуют проверки на живом Triton с
моделью, поддерживающей нужную модальность. Укажите в pull request архитектуру
модели, тип GPU, форму запроса и очищенные от приватных данных логи.

## Требования к коду

- Ограничивайте использование ресурсов для queues, files, downloads и media.
- Не выполняйте блокирующую CPU-работу внутри async event loop.
- Сохраняйте client cancellation на всех уровнях transport.
- Не логируйте prompt, media, tool output и credentials по умолчанию.
- Возвращайте понятные OpenAI-совместимые ошибки без утечки traceback.
- Добавляйте комментарии только там, где причина не очевидна из кода.

Вклад принимается по [Apache License 2.0](LICENSE), кроме файлов на основе
upstream, сохраняющих собственные лицензионные условия.
