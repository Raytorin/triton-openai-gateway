# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version


EXPECTED_VERSIONS = {
    "accelerate": "1.14.0",
    "av": "17.1.0",
    "compressed-tensors": "0.15.0.1",
    "decord": "0.6.0",
    "fastapi": "0.139.2",
    "grpcio": "1.67.1",
    "httpx": "0.27.2",
    "numpy": "1.26.4",
    "pillow": "12.3.0",
    "protobuf": "6.33.6",
    "pydantic": "2.10.6",
    "pymupdf": "1.28.0",
    "prometheus-client": "0.25.0",
    "python-rapidjson": "1.23",
    "qwen-vl-utils": "0.0.14",
    "sentencepiece": "0.2.1",
    "starlette": "1.3.1",
    "torch": "2.13.0a0+8145d630e8.nv26.6.54250401",
    "transformers": "5.6.0",
    "tritonclient": "2.70.0",
    "uvicorn": "0.49.0",
    "uvloop": "0.22.1",
    "vllm": "0.22.1+7b9cb5b7.nv26.6.55098374",
}


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
