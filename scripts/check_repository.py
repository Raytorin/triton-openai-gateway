#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Check tracked source syntax without importing code or test dependencies."""

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys


MAX_SOURCE_BYTES = 16 * 1024 * 1024


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--stage", "-z"],
        check=True, capture_output=True,
    )
    files = set()
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        header, name = entry.split(b"\t", 1)
        mode, _, stage = header.split()
        if stage != b"0":
            raise ValueError("Unresolved Git merge entries")
        if mode != b"160000":  # A submodule's files belong to its own checkout.
            files.add(Path(os.fsdecode(name)))
    return sorted(files)


def _source_bytes(root: Path, relative: Path, tracked: set[Path]) -> bytes | None:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("path is outside the repository")
    resolved = (root / relative).resolve(strict=True)
    try:
        target = resolved.relative_to(root)
    except ValueError:
        raise ValueError("symlink points outside the repository") from None
    if target not in tracked:
        raise ValueError("symlink target is not tracked")
    # Nonblocking open plus fstat prevents a replaced tracked file from hanging
    # the checker on a FIFO, and final-component symlinks are never followed here.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(resolved, flags)
    with os.fdopen(fd, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("source is not a regular file")
        prefix = source.read(256)
        if _kind(relative, prefix) is None:
            return None
        source.seek(0)
        data = source.read(MAX_SOURCE_BYTES + 1)
        if len(data) > MAX_SOURCE_BYTES:
            raise ValueError("source exceeds the 16 MiB syntax-check limit")
        return data


def _kind(path: Path, prefix: bytes) -> str | None:
    if path.suffix == ".py":
        return "python"
    if path.suffix == ".json":
        return "json"
    if path.suffix == ".sh":
        return "bash"
    first_line = prefix.split(b"\n", 1)[0]
    if first_line.startswith(b"#!"):
        words = first_line.lower().replace(b"/", b" ").split()
        if any(word.startswith(b"python") for word in words):
            return "python"
        if b"bash" in words:
            return "bash"
    return None


def _non_finite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value}")


def check_repository(root: Path) -> list[str]:
    root = root.resolve()
    try:
        files = tracked_files(root)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return ["Unable to list tracked repository files or unresolved Git entries."]
    tracked = set(files)
    errors = []
    bash_environment = dict(os.environ)
    bash_environment.pop("BASH_ENV", None)
    bash_environment.pop("ENV", None)
    for relative in files:
        label = repr(relative.as_posix())
        try:
            data = _source_bytes(root, relative, tracked)
            if data is None:
                continue
            kind = _kind(relative, data[:256])
            if kind == "python":
                compile(data, relative.as_posix(), "exec", dont_inherit=True)
            elif kind == "json":
                json.loads(data, parse_constant=_non_finite_json)
            elif kind == "bash":
                # Feed the already validated bytes, so filenames cannot become
                # options and Bash never reopens a replaced path or symlink.
                result = subprocess.run(
                    ["bash", "-n"], input=data, capture_output=True,
                    env=bash_environment, timeout=15, check=False,
                )
                if result.returncode:
                    detail = result.stderr.decode("utf-8", "replace").strip()[:1000]
                    errors.append(f"{label}: Bash syntax check failed: {detail}")
        except SyntaxError as exc:
            errors.append(f"{label}:{exc.lineno}: Python syntax error: {exc.msg}")
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{label}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_repository(args.root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Tracked Python, JSON and Bash syntax checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
