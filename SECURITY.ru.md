# Политика безопасности

**Язык:** [English](SECURITY.md) | Русский

## Сообщение об уязвимости

Не создавайте публичный issue для предполагаемой уязвимости. Используйте
private security advisory GitHub и укажите:

- затронутый endpoint или компонент;
- проверенную revision и способ deployment;
- шаги воспроизведения или минимальный request;
- предполагаемое влияние;
- логи без credentials и данных пользователя.

Если private advisories не включены, свяжитесь с владельцем repository через
приватный канал в его GitHub profile до публикации подробностей.

## Поддерживаемые версии

Исправления безопасности применяются к текущей default branch. Исторические
копии не поддерживаются, если это явно не указано в GitHub release.

## Граница ответственности deployment

Проект не предоставляет authentication, tenant isolation, TLS termination и
secret manager. Размещайте его за аутентифицированным proxy, а raw Triton,
metrics, DCGM и tracing оставляйте во внутренних сетях.

Model repository входит в доверенную вычислительную базу. При
`trust_remote_code=true` tokenizer или файлы модели могут исполнять Python-код
в контейнере Triton. Загружайте только проверенные модели из контролируемого
repository.

Remote media, большие request body, PDF parsing, video decoding и долгие streams
являются ресурсоёмкими input. Сохраняйте сетевые ограничения по умолчанию и
задавайте жёсткие лимиты размера, длительности, pixels, queues и timeout.
