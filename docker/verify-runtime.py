# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REQUIREMENTS_FILE = Path(__file__).with_name("triton-chat-gateway-requirements.txt")

# These packages are part of one tested NVIDIA/vLLM runtime and must move with
# the base image, not through independent dependency updates.
BASE_IMAGE_VERSIONS = {
    "compressed-tensors": "0.15.0.1",
    "torch": "2.13.0a0+8145d630e8.nv26.6.54250401",
    "transformers": "5.6.0",
    "vllm": "0.22.1+7b9cb5b7.nv26.6.55098374",
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
    conflicts = {
        package: (requirements[package], expected)
        for package, expected in BASE_IMAGE_VERSIONS.items()
        if package in requirements and requirements[package] != expected
    }
    if conflicts:
        raise ValueError(
            "Runtime pins conflict with the NVIDIA base image: "
            f"{conflicts}"
        )
    return requirements | BASE_IMAGE_VERSIONS


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
