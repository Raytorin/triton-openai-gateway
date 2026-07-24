# Contributing

**Language:** English | [Русский](CONTRIBUTING.ru.md)

Thank you for improving Triton OpenAI Gateway.

## Before You Start

- Open an issue for behavior changes that alter the public API, model layout, or
  supported runtime versions.
- Keep changes scoped. Do not mix runtime upgrades with unrelated features.
- Preserve all NVIDIA copyright and BSD license headers in derived backend
  files.
- Keep `NOTICE`, `AUTHORS.md`, and third-party attribution accurate.
- Never commit model weights, credentials, generated caches, or user payloads.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Make the smallest coherent change.
3. Add or update unit tests under `tests/`.
4. Update both English and Russian documentation and examples when behavior changes.
5. Run the CPU checks locally and compatibility checks in the pinned Triton image.
6. Submit a pull request describing behavior, compatibility impact, and tests.

## Build

```bash
docker build -f Dockerfile.triton-gateway -t triton-openai-gateway:dev .
```

The build verifies exact runtime dependency versions. A base-image or dependency
upgrade should update `docker/verify-runtime.py`, requirements, compatibility
documentation, and tests in one change.

## Tests

Create a Python 3.12 environment for CPU-safe checks:

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

GitHub Actions runs the same tests and dependency audits for every pull request. The test
requirements are intentionally smaller than the GPU runtime and must not be
used to build the production image.

To confirm behavior against the exact pinned Triton runtime, run the unit tests
inside the built image:

```bash
docker run --rm \
  --entrypoint python3 \
  -v "$PWD:/workspace" \
  -w /workspace \
  triton-openai-gateway:dev \
  -m unittest discover -s tests -v
```

Run the standalone PDF/video preprocessing smoke test:

```bash
docker run --rm \
  --entrypoint python3 \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  triton-openai-gateway:dev \
  tests/runtime_media_smoke.py
```

Validate shell and Helm files when those areas change:

```bash
bash -n watch_triton.sh docker/80-watch-triton.sh docker/90-triton-chat-gateway.sh
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
helm lint ./helm/dcgm-exporter
```

GPU and end-to-end media changes also require a live Triton test with a model
that supports the affected modality. Include model architecture, GPU type,
request shape, and relevant logs in the pull request without including private
payloads.

## Code Guidelines

- Prefer explicit, bounded resource use for queues, files, downloads, and media.
- Keep blocking CPU work outside async event loops.
- Preserve client cancellation through every transport layer.
- Do not log prompts, media, tool output, or credentials by default.
- Return actionable OpenAI-compatible errors rather than leaking tracebacks.
- Add comments only when the reason is not clear from the code.

Contributions are accepted under the [Apache License 2.0](LICENSE), except for
upstream-derived files that retain their existing license notices.
