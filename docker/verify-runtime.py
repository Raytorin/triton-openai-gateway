# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIREMENTS_FILE = Path(__file__).with_name("triton-chat-gateway-requirements.txt")

# These packages are supplied by the NVIDIA image and must move with it.
BASE_IMAGE_VERSIONS = {
    "compressed-tensors": "0.17.0",
    "flashinfer-python": "0.6.14+d0510b70.nv26.7.cu.59527636",
    "torch": "2.13.0a0+9186a08b2c.nv26.7.59513937",
    "transformers": "5.6.1",
    "tritonserver": "2.71.0",
    "vllm": "0.24.0+092c4842.nv26.7.59534043",
}

# Compatibility-sensitive overlay packages are intentionally duplicated here.
# Dependabot must not advance one of them without a complete runtime review.
# FastAPI 0.136.1 is the newest release in vLLM's supported >=0.133,<0.137
# range. Pydantic stays at 2.10.6 because Triton frontend 2.71 pins it exactly.
LOCKED_OVERLAY_VERSIONS = {
    "fastapi": "0.136.1",
    "grpcio": "1.67.1",
    "httpx": "0.27.2",
    "numpy": "1.26.4",
    "opentelemetry-api": "1.44.0",
    "opentelemetry-exporter-otlp-proto-http": "1.44.0",
    "opentelemetry-sdk": "1.44.0",
    "pillow": "12.3.0",
    "prometheus-client": "0.26.0",
    "protobuf": "6.33.6",
    "pydantic": "2.10.6",
    "sentencepiece": "0.2.2",
    "starlette": "1.3.1",
    "tritonclient": "2.71.0",
    "uvicorn": "0.51.0",
    "uvloop": "0.22.1",
}


def read_pinned_requirements(requirements_file: Path) -> dict[str, str]:
    pins = {}
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise ValueError(f"Runtime dependency must use an exact pin: {raw_line!r}")

        package, expected = line.split("==", 1)
        package = package.split("[", 1)[0].strip().lower().replace("_", "-")
        expected = expected.strip()
        if not package or not expected:
            raise ValueError(f"Invalid runtime dependency pin: {raw_line!r}")
        if package in pins and pins[package] != expected:
            raise ValueError(f"Conflicting runtime pins for {package}")
        pins[package] = expected
    return pins


def expected_versions(requirements_file: Path = REQUIREMENTS_FILE) -> dict[str, str]:
    requirements = read_pinned_requirements(requirements_file)
    runtime_policy = BASE_IMAGE_VERSIONS | LOCKED_OVERLAY_VERSIONS
    conflicts = {
        package: (requirements[package], expected)
        for package, expected in runtime_policy.items()
        if package in requirements and requirements[package] != expected
    }
    if conflicts:
        raise ValueError(
            "Runtime pins conflict with the tested runtime policy: "
            f"{conflicts}"
        )
    return requirements | runtime_policy


EXPECTED_VERSIONS = expected_versions()


def main() -> None:
    errors = []
    for package, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            errors.append(f"{package}: missing, expected {expected}")
            continue

        if actual != expected:
            errors.append(f"{package}: found {actual}, expected {expected}")

    if errors:
        details = "\n".join(f"- {error}" for error in errors)
        raise SystemExit(f"Runtime dependency verification failed:\n{details}")

    print(f"Verified {len(EXPECTED_VERSIONS)} runtime package versions")


if __name__ == "__main__":
    main()
