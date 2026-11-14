#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export release-matched native Triton files, without the donor's Python stack."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


SEEDS = (
    "opt/tritonserver/bin", "opt/tritonserver/lib",
    "opt/tritonserver/backends/python", "opt/tritonserver/backends/vllm",
    "opt/tritonserver/repoagents", "opt/tritonserver/caches",
    "opt/tritonserver/LICENSE", "opt/tritonserver/TRITON_VERSION",
    "opt/tritonserver/NVIDIA_Deep_Learning_Container_License.pdf", "opt/nvidia",
)


def _dependencies(path):
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = "/opt/tritonserver/lib:" + environment.get("LD_LIBRARY_PATH", "")
    result = subprocess.run(["ldd", str(path)], env=environment, text=True,
                            capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Cannot inspect native dependencies for {path}: {result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def assemble_native(source, destination, *, dependency_reader=None):
    """Copy native seeds and their resolved dependency closure; fail on missing libs."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not (source / "opt/tritonserver/bin/tritonserver").is_file():
        raise RuntimeError("Native donor does not contain tritonserver")
    dependency_reader = dependency_reader or _dependencies
    destination.mkdir(parents=True, exist_ok=True)
    for relative in SEEDS:
        original, target = source / relative, destination / relative
        if original.is_dir():
            shutil.copytree(original, target, symlinks=True, dirs_exist_ok=True)
        elif original.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)

    copied, system_libraries = set(), set()

    def copy_library(absolute):
        relative = Path(os.path.normpath(absolute)).relative_to("/")
        if any(part in {"site-packages", "dist-packages"} for part in relative.parts):
            raise RuntimeError(f"Native core unexpectedly requires Python distribution library: {absolute}")
        if (str(relative).startswith(("lib/", "lib64/", "usr/lib/", "usr/lib64/"))
                and not relative.name.startswith("libdcgm.so")):
            # The final Ubuntu stage installs current OS packages. Never replace
            # their libc/OpenSSL/etc. with older files and stale package metadata.
            # Embedded libpython is supplied by the selected common environment.
            system_libraries.add("/" + str(relative))
            return
        if relative in copied:
            return
        copied.add(relative)
        original, target = source / relative, destination / relative
        if relative.parts[0] in {"lib", "lib64"} and original.parent.resolve().is_relative_to(source):
            canonical_parent = original.parent.resolve().relative_to(source)
            if canonical_parent != relative.parent:
                copy_library("/" + str(canonical_parent / relative.name))
                return
        target.parent.mkdir(parents=True, exist_ok=True)
        if original.is_symlink():
            link = original.readlink()
            if not target.is_symlink():
                target.symlink_to(link)
            elif target.readlink() != link:
                raise RuntimeError(f"Conflicting native library link: {absolute}")
            linked = link if link.is_absolute() else Path("/") / relative.parent / link
            copy_library(str(linked))
        elif original.is_file():
            shutil.copy2(original, target)
        else:
            raise RuntimeError(f"Missing native dependency: {absolute}")
        # Preserve the real absolute path too when the donor uses merged-/usr
        # or a symlinked CUDA directory; ABI policy may name that canonical file.
        resolved = original.resolve()
        if resolved.is_relative_to(source):
            canonical = "/" + str(resolved.relative_to(source))
            if Path(canonical).relative_to("/") != relative:
                copy_library(canonical)

    for relative in SEEDS:
        seed = source / relative
        files = seed.rglob("*") if seed.is_dir() else (seed,)
        for path in files:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            listing = dependency_reader(path)
            if "not found" in listing:
                raise RuntimeError(f"Unresolved native dependencies for {path}: {listing}")
            for line in listing.splitlines():
                match = re.match(r"\s*(?:\S+\s+=>\s+)?(/\S+)\s+\(", line)
                if match:
                    copy_library(match.group(1))

    files = {}
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            files["/" + str(path.relative_to(destination))] = "symlink:" + str(path.readlink())
        elif path.is_file():
            with path.open("rb") as stream:
                files["/" + str(path.relative_to(destination))] = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"schema": 1, "native_files": files, "system_libraries": sorted(system_libraries)}
    manifest = destination / "opt/tritonserver/unified/native-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = assemble_native(args.source, args.output)
    print(f"Exported {len(result['native_files'])} release-matched native files")


if __name__ == "__main__":
    main()
