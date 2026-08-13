#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import subprocess
import sys


DECORD_VERSION = "0.6.0"
KNOWN_DECORD_WHEEL_ISSUE = (
    f"decord {DECORD_VERSION} is not supported on this platform"
)


def has_only_known_decord_issue(returncode: int, output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return returncode == 1 and lines == [KNOWN_DECORD_WHEEL_ISSUE]


def verify_decord_import() -> None:
    try:
        installed_version = version("decord")
        module = importlib.import_module("decord")
    except (ImportError, PackageNotFoundError) as exc:
        raise SystemExit(f"Decord import check failed: {exc}") from exc

    module_version = getattr(module, "__version__", None)
    if installed_version != DECORD_VERSION or module_version != DECORD_VERSION:
        raise SystemExit(
            "Unexpected Decord version: "
            f"distribution={installed_version}, module={module_version}, "
            f"expected={DECORD_VERSION}"
        )


def main() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    clean_environment = result.returncode == 0
    known_decord_issue = has_only_known_decord_issue(
        result.returncode,
        result.stdout,
    )
    if not clean_environment and not known_decord_issue:
        print(result.stdout, file=sys.stderr, end="")
        raise SystemExit(result.returncode or 1)

    verify_decord_import()
    if clean_environment:
        print(result.stdout, end="")
    else:
        print(
            "Accepted the known Decord 0.6.0 wheel-tag mismatch after a "
            "successful import check."
        )


if __name__ == "__main__":
    main()
