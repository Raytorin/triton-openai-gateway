# SPDX-License-Identifier: Apache-2.0
"""Verify the unified Python runtime, its gateway overlay and dependency graph."""
import argparse
from importlib import import_module, metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

from packaging.version import InvalidVersion, Version


DEFAULT_LOCK = Path(__file__).resolve().parent.parent / "runtime-locks/unified-vllm032.requirements.txt"
# Only the Triton client compatibility constraints replace inherited pins.
BASE_OVERRIDES = {"ml-dtypes": "0.5.4", "protobuf": "6.33.6"}
REQUIRED_OVERLAY = {"accelerate", "deepseek-recipe", "ml-dtypes", "protobuf",
                    "python-rapidjson", "qwen-vl-utils", "soundfile", "soxr", "tritonclient"}
FORBIDDEN = {"tritonserver", "tritonfrontend", "nvidia-tritonserver", "nvidia-tritonfrontend"}
IMPORTS = (
    "fastapi", "pydantic", "httpx", "tritonclient.grpc", "rapidjson",
    "deepseek_recipe", "accelerate", "qwen_vl_utils", "numpy", "av", "soundfile", "soxr",
    "decord", "fitz", "opentelemetry.exporter.otlp.proto.http.trace_exporter",
)


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_lock(path):
    """Read exact, hashed wheel pins without accepting pip options or ranges."""
    pins = {}
    text = Path(path).read_text(encoding="utf-8").replace("\\\n", " ")
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[A-Za-z0-9_,.-]+\])?"
            r"==([^\s;]+)(?:\s+--hash=sha256:[a-f0-9]{64})+", line,
        )
        if not match:
            raise ValueError(f"Runtime dependency requires an exact pin and SHA256: {raw!r}")
        name, version = normalize(match[1]), match[2]
        try:
            Version(version)
        except InvalidVersion as error:
            raise ValueError(f"Runtime dependency requires an exact pin and SHA256: {raw!r}") from error
        if name in pins:
            raise ValueError(f"Duplicate runtime pin: {name}")
        pins[name] = version
    if not pins:
        raise ValueError("Runtime lock has no exact pin and SHA256 entries")
    return pins


def expected_versions(base_lock, overlay_lock):
    """Preserve the entire manifested inference environment except two client pins."""
    base, overlay = read_lock(base_lock), read_lock(overlay_lock)
    for name, version in overlay.items():
        if name in base and version != base[name] and BASE_OVERRIDES.get(name) != version:
            raise ValueError(f"Overlay cannot replace base runtime pin: {name}=={base[name]}")
    missing = REQUIRED_OVERLAY - overlay.keys()
    if missing:
        raise ValueError(f"Runtime is missing gateway overlay pins: {sorted(missing)}")
    return base | overlay


def distribution_errors(distributions, expected):
    """Inspect real distribution records, including duplicate and legacy installs."""
    packages, errors = {}, []
    for distribution in distributions:
        name = normalize(distribution.metadata["Name"])
        if name in packages:
            errors.append(f"{name}: duplicate installed distribution metadata")
        packages[name] = distribution.version
    for name in sorted(FORBIDDEN & packages.keys()):
        errors.append(f"{name}: legacy NVIDIA Python distribution is not supported")
    for name, wanted in expected.items():
        actual = packages.get(name)
        if actual is None:
            errors.append(f"{name}: missing, expected {wanted}")
        elif actual != wanted:
            errors.append(f"{name}: found {actual}, expected {wanted}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--base-lock", type=Path)
    parser.add_argument("--imports", action="store_true", help="import gateway and media modules without model weights")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
        raise SystemExit("Unified runtime requires CPython 3.12 on linux/amd64")
    base_lock = args.base_lock or args.lock.with_name("vllm032-cu129.requirements.txt")
    expected = expected_versions(base_lock, args.lock)
    errors = distribution_errors(metadata.distributions(), expected)
    if errors:
        raise SystemExit("Unified dependency verification failed:\n" + "\n".join(f"- {error}" for error in errors))
    # Every dependency, including those retained from the base, must satisfy
    # its installed METADATA. No known-conflict exceptions are allowed here.
    checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    if checked.returncode:
        raise SystemExit("Unified pip check failed:\n" + checked.stdout + checked.stderr)
    imported = {}
    if args.imports:
        for name in IMPORTS:
            module = import_module(name)
            imported[name] = getattr(module, "__file__", None)
    print(json.dumps({"python": platform.python_version(), "verified_packages": len(expected),
                      "pip_check": "passed", "imports": imported}, sort_keys=True))


if __name__ == "__main__":
    main()
