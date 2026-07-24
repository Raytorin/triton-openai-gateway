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
  --cov-fail-under=50
.venv/bin/python tests/runtime_media_smoke.py
.venv/bin/cffconvert --validate
.venv/bin/pip-audit --requirement requirements-test.txt --progress-spinner off
.venv/bin/pip-audit \
  --requirement docker/triton-chat-gateway-requirements.txt \
  --progress-spinner off
```

GitHub Actions выполняет те же тесты и аудит зависимостей для каждого pull request. Набор
test-зависимостей намеренно меньше GPU runtime и не должен использоваться для
сборки production-образа.

Для проверки на точном закреплённом runtime Triton запустите unit tests внутри
собранного образа:

```bash
docker run --rm \
  --entrypoint python3 \
  -v "$PWD:/workspace" \
  -w /workspace \
  triton-openai-gateway:dev \
  -m unittest discover -s tests -v
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

Проверка shell и Helm:

```bash
bash -n watch_triton.sh docker/80-watch-triton.sh docker/90-triton-chat-gateway.sh
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
helm lint ./helm/dcgm-exporter
```

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
