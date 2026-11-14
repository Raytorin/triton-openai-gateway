# Варианты runtime и unified-образ с vLLM 0.32

[Главная](../README.ru.md) / Варианты runtime

**Язык:** [English](README.unified.md) | Русский

`Dockerfile.triton-gateway` остаётся поддерживаемым вариантом по умолчанию на
официальном/legacy runtime. `Dockerfile.triton-gateway-unified` выбирается явно
и использует общее окружение Python 3.12 / vLLM 0.32 / CUDA 12.9. Оба рецепта
копируют актуальный код gateway и собственного backend из checkout.

| Рецепт | Runtime и назначение |
| --- | --- |
| `Dockerfile.triton-gateway` | NVIDIA Triton `26.10-vllm-python-py3` с закреплённым digest и NVIDIA vLLM `0.31`, официальный вариант по умолчанию |
| `Dockerfile.triton-gateway-deepseek` | Официальный вариант с дополнительным независимо закреплённым окружением vLLM `0.32` для `vllm_deepseek`; также источник runtime для unified-сборки |
| `Dockerfile.triton-gateway-unified` | Финальный слой Ubuntu с нативными компонентами Triton `26.10`, единым окружением vLLM `0.32` и зависимостями gateway с закреплёнными хешами |

Аддитивный DeepSeek-рецепт сохраняется. Изоляция исходных моделей и проверки
checkpoint описаны в [руководстве DeepSeek](README.deepseek.md). Unified-образ
копирует из него выбранные runtime-артефакты, не наследуя финальную файловую
систему и старый тяжёлый Python-стек инференса. Toolkit CUDA 12.9 для engine
не заменяет требования нативного ядра Triton к CUDA 13 ABI и драйверу.

## Локальная сборка

Все три шага выполняются из одного checkout gateway `0.7.0`. Дополнительное
окружение и unified-образ рассчитаны на `linux/amd64`. Веса моделей
подключаются отдельно.

1. Соберите официальный/legacy образ по умолчанию:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway \
     -t triton-openai-gateway:official-local .
   ```

2. Если проверенный runtime-артефакт ещё не собран, создайте аддитивный
   DeepSeek-образ:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway-deepseek \
     --build-arg GATEWAY_IMAGE=triton-openai-gateway:official-local \
     -t triton-openai-gateway:deepseek-local .
   ```

3. Явно выберите unified-вариант финального образа:

   ```sh
   docker build --platform linux/amd64 \
     -f Dockerfile.triton-gateway-unified \
     --build-arg RUNTIME_IMAGE=triton-openai-gateway:deepseek-local \
     -t triton-openai-gateway:unified-vllm032-local .
   ```

Значение `RUNTIME_IMAGE` по умолчанию — `triton-openai-gateway:deepseek-local`.
Для воспроизводимости передавайте записанный image ID или неизменяемый digest.
Сохраняйте checkout вместе с описанием входных runtime-артефактов: изменяемый тег
сам по себе не определяет окружение. Ошибки проверок при сборке блокируют
использование такого артефакта.

Lock и происхождение engine остаются в `docker/runtime-locks/vllm032-cu129.*`.
Gateway overlay использует отдельные файлы
`docker/runtime-locks/unified-vllm032.*`. Не переносите в новое окружение точные
версии зависимостей официального варианта. Overlay сохраняет новые Pydantic и
NumPy и согласует `tritonclient==2.74.0` с `protobuf==6.33.6` и
`ml-dtypes==0.5.4`.

## Выбор окружения и совместимость

В unified-образе `/opt/envs/vllm032-cu129` используется для gateway, watcher,
штатного `vllm`, встроенного `vllm_multimodal` и обычных Python backend.
Launcher каждого backend выбирает Python, пути нативных библиотек, компилятора
и кешей до запуска нативного Python stub соответствующей версии Triton.
Нативный stub остаётся отдельным исполняемым файлом. Имена backend и структура
репозитория моделей сохраняются; выбор нового окружения не переписывает
конфигурации моделей.

Обычные Python/vLLM backend не получают ограничения DeepSeek на checkpoint и
тип задачи. Модели с `vllm_deepseek` сохраняют его явные проверки checkpoint и
режим задач `generate`, включая явно включённые изображения checkpoint.
Совместимость API gateway, общего media-пути,
pooling/rerank и существующих конфигураций моделей остаётся целью; поддержку
конкретной модели и аргументы engine нужно проверять с новой версией vLLM.

Дополнительные Python frontend NVIDIA доступны в официальном варианте. В
unified-образе штатный и собственные бэкенды принимают `thinking_token_budget`
для моделей с подходящими настройками рассуждений. Подробнее в документах
по [конфигурации gateway](../docs/configuration.ru.md) и
[совместимости Triton 26.10](../docs/migration-26.10.ru.md).

Инференс в этом образе нужно отдельно проверить на целевых GPU. При сборке
проверяются зависимости и импорты на CPU. Дочерний процесс и запуск нативного
stub проверяются в готовом образе по инструкции ниже.

Для локального ASR сохраняется NVIDIA `/opt/ffmpeg-safe` с прежними ограничениями
кодеков и лицензирования. Общий overlay также содержит SoundFile и SoxR для
native audio и смены частоты дискретизации. Кеши JIT должны позволять загрузку
скомпилированных объектов: монтирование кеша с `noexec` может блокировать
runtime-компиляцию.

## Проверка и выбор образа

Проверки CPU без весов запускаются на фактическом образе из корня репозитория:

```sh
docker run --rm --network none --runtime runc --user 1000:1000 \
  --read-only --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777,size=2g \
  --shm-size=512m -e NVIDIA_VISIBLE_DEVICES=void \
  -e TRITON_RUNTIME_CACHE_ARCH=cpu \
  -v "$PWD/tests:/opt/tritonserver/tests:ro" \
  --entrypoint /opt/envs/vllm032-cu129/bin/python3.12 \
  triton-openai-gateway:unified-vllm032-local -I \
  /opt/tritonserver/unified/launch.py --exec /bin/bash -ec \
  'python3 /opt/tritonserver/tests/runtime_unified_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_vllm_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_stock_vllm_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_audio_smoke.py &&
   python3 /opt/tritonserver/tests/runtime_media_smoke.py'
```

Перед использованием unified-кандидата проверьте manifests и полное замыкание
зависимостей, затем CPU import/spawn, регрессии gateway/client/media и реальный
нативный stub Triton. Квалификация на целевой GPU также должна включать:

- Используемые text, vision, audio/video/PDF, pooling и rerank модели, включая
  streaming, отмену, load/unload и очистку после неудачного запуска.
- EngineCore и tensor-parallel workers, CUDA/NCCL, FlashInfer, ядра конкретной
  модели и фактически выбранную комбинацию JIT-компилятора и toolkit.
- Проверки checkpoint и запуск DeepSeek, если этот backend используется.
- Совместимость драйвера с нативным ядром Triton и новым Python engine, а также
  доступные для записи кеши вне неизменяемых runtime-артефактов.

Сохраняйте с кандидатом идентификаторы входных образов, результаты verifier,
перечень проверенных моделей/оборудования и измерения размера образов и файловых
систем одним методом. Runtime-окружения должны оставаться неизменяемыми:
не устанавливайте пакеты при старте и не монтируйте поверх
`/opt/envs/vllm032-cu129`, `/opt/cuda/12.9` или каталогов immutable-артефактов.

Укажите проверенный тег или digest в существующей Docker/Helm конфигурации.
Параметры развёртывания описаны в [быстром старте](../README.ru.md#быстрый-старт)
и [руководстве Helm](../helm/triton-gateway/README.ru.md). Для отката выберите
ранее проверенный официальный образ и согласованную с ним конфигурацию моделей.
Сохраняйте этот артефакт, пока новый кандидат не пройдёт необходимые проверки
моделей.

Например, после остановки кандидата запустите проверенный официальный образ с
тем же файлом окружения и репозиторием моделей, что использовались прежде
(замените параметры репозитория):

```sh
docker run --rm --gpus all --shm-size=8g \
  --env-file .env \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 -p 8080:8080 \
  triton-openai-gateway:official-local \
  tritonserver --model-repository=s3://S3_ENDPOINT/BUCKET/PREFIX \
  --model-control-mode=explicit --strict-readiness=false
```

Для моделей с `backend: "vllm_deepseek"` нужен ранее квалифицированный аддитивный
legacy-образ из `Dockerfile.triton-gateway-deepseek`. Замените
`triton-openai-gateway:official-local` в примере тегом или digest этого образа.
Обычный официальный образ не содержит `vllm_deepseek`; для этих моделей также
выбирайте Helm revision с аддитивным образом.

Для Helm вернитесь к записанной проверенной revision и снова проверьте выбранные
модели:

```sh
helm history triton-openai-gateway --namespace inference
helm rollback triton-openai-gateway QUALIFIED_REVISION --namespace inference
```

## Поддержка официального варианта и сближение runtime

В Dependabot сохраняются регулярные Docker-обновления корня репозитория с
целевой веткой `dev`. Продолжайте проверять официальные обновления NVIDIA,
пока доступен unified-вариант. Для каждого обновления изучайте фактическую
матрицу пакетов кандидата, обновляйте официальный digest и согласованные pins,
запускайте официальный verifier, тесты gateway/client/media, backend и проверки
на целевой GPU. Не обновляйте vLLM, Torch, Transformers и другие нативные
компоненты по одному внутри официального образа ради повторения unified-стека.

Официальный образ NVIDIA с vLLM `0.32` или новее станет кандидатом на сближение
вариантов. Один номер версии не доказывает совместимость и не меняет вариант
по умолчанию. Сравните матрицу runtime, Python/backend ABI, зависимости gateway,
ручные frontend и необходимые возможности моделей; повторите CPU/native и GPU
проверки до явной смены варианта по умолчанию или удаления альтернативы.
Обновление pins, merge и публикация образа — отдельные проверяемые действия;
ни один вариант не публикует и не заменяет другой автоматически.
