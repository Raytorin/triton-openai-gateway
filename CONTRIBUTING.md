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
  --cov=backends/vllm_deepseek \
  --cov-fail-under=75
.venv/bin/python tests/runtime_media_smoke.py
.venv/bin/cffconvert --validate
.venv/bin/python scripts/audit_dependencies.py
```

GitHub Actions runs two Linux jobs for pull requests, pushes to `dev`/`main`,
and manual runs. Early checks validate tracked Python, JSON, and Bash sources;
actionlint and ShellCheck validate workflows and shell scripts. Test, runtime,
and SDK requirements are audited separately. The existing narrow runtime audit
exception is described in [SECURITY.md](SECURITY.md).

JUnit and coverage reports are retained for seven days. Only those two XML
files are uploaded; local notes, virtual environments, payloads, and runtime
dumps are excluded. The test requirements are intentionally smaller than the
GPU runtime and must not be used to build the production image.

Run the SDK compatibility suite in its separate client environment:

```bash
python3.12 -m venv .local/client-venv
.local/client-venv/bin/python -m pip install --requirement requirements-client.txt
.local/client-venv/bin/python tests/client_compatibility.py \
  --server-python "$PWD/.venv/bin/python"
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

Run the real vLLM API contract smoke test on CPU:

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

This checks sampling, structured output, LoRA, pooling, metrics, and engine
lifecycle contracts against the installed vLLM APIs without running GPU
inference. Missing runtime dependencies fail the check instead of skipping it.

Validate shell and Helm files when those areas change:

```bash
python3.12 scripts/check_repository.py
helm lint ./helm/triton-gateway
helm template test ./helm/triton-gateway >/dev/null
helm template test ./helm/triton-gateway \
  --values helm/triton-gateway/values.production.example.yaml >/dev/null
helm lint ./helm/dcgm-exporter
```

Ordinary CI does not download model weights or build the large Triton images.
The real vLLM contract checks above require a prepared image; native DeepSeek
and GPU inference qualification remain separate from the hosted CPU suite.

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
