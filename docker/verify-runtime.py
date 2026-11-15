# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIREMENTS_FILE = Path(__file__).with_name("triton-chat-gateway-requirements.txt")

# These packages are supplied by the NVIDIA image and must move with it.
BASE_IMAGE_VERSIONS = {
    "compressed-tensors": "0.17.0",
    "flashinfer-python": "0.7.1rc5+c6a6b324.nvinternal.26.10.fi.rc5.5943.pr6394.cu.73382939",
    "torch": "2.15.0a0+875d815502.nv26.10.72655024",
    "transformers": "5.17.0",
    "tritonserver": "2.74.0",
    "vllm": "0.31.0+88641915.nv26.10.75644909",
    "openai": "2.54.0",
}

# Compatibility-sensitive overlay packages are intentionally duplicated here.
# Dependabot must not advance one of them without a complete runtime review.
# FastAPI 0.136.3 is the newest release in vLLM's supported >=0.133,<0.137
# range. Triton frontend 2.74 accepts Pydantic >=2.12,<3, matching vLLM.
LOCKED_OVERLAY_VERSIONS = {
    "fastapi": "0.136.3",
    "grpcio": "1.84.0",
    "httpx": "0.27.2",
    "ml-dtypes": "0.5.4",
    "numpy": "1.26.4",
    "opentelemetry-api": "1.45.2",
    "opentelemetry-exporter-otlp": "1.45.2",
    "opentelemetry-exporter-otlp-proto-http": "1.45.2",
    "opentelemetry-sdk": "1.45.2",
    "pillow": "13.0.0",
    "prometheus-client": "0.26.0",
    "protobuf": "6.33.6",
    "pydantic": "2.14.1",
    "sentencepiece": "0.2.2",
    "starlette": "1.9.0",
    "tritonclient": "2.74.0",
    "uvicorn": "0.54.0",
    "uvloop": "0.23.0",
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
