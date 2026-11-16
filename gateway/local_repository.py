# SPDX-License-Identifier: Apache-2.0
"""Update Gateway links for ready versions in Triton's local repositories."""

import json
import os
from contextlib import suppress
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen


def ready_versions(base_url):
    request = Request(
        base_url.rstrip("/") + "/v2/repository/index",
        data=b'{"ready":true}',
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=1) as response:
            payload = json.load(response)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, list):
        return None

    versions = {}
    for model in payload:
        if not isinstance(model, dict):
            return None
        if model.get("state") != "READY":
            continue
        name, version = model.get("name"), model.get("version")
        if (
            not isinstance(name, str) or name in {"", ".", ".."} or "/" in name or "\0" in name
            or not isinstance(version, str) or not version.isascii() or not version.isdecimal()
        ):
            return None
        versions.setdefault(name, []).append(version)
    return versions


def select_local_versions(versions, repositories):
    selected = {}
    for name, ready in versions.items():
        # Unversioned inference goes to Triton's highest READY version. A
        # missing directory must not make Gateway use an older version's config.
        version = max(ready, key=int)
        for root in repositories:
            target = root / name / version
            if target.is_dir():
                selected[name] = target
                break
    return selected


def readable_entries(directory):
    try:
        yield from directory.iterdir()
    except OSError:
        # A missing or unreadable local directory must not hide siblings.
        return


def overlaps_staging(staging, repositories):
    def overlaps(path):
        if staging.is_relative_to(path):
            return True
        if path.is_relative_to(staging):
            return path.relative_to(staging).parts[0].startswith("folder")
        return False

    def link_overlaps(path):
        with suppress(OSError, RuntimeError):
            return path.is_symlink() and overlaps(path.resolve())
        return False

    for root in repositories:
        if overlaps(root):
            return True
        for model in readable_entries(root):
            if link_overlaps(model):
                return True
            for version in readable_entries(model):
                if version.name.isdecimal() and link_overlaps(version):
                    return True
    return False


def sync_links(active, selected, repositories):
    for link in active.iterdir():
        if link.name in selected or not link.is_symlink():
            continue
        # The link target stays lexical so a removed version still identifies
        # its repository, even if the target no longer exists.
        target = Path(os.path.abspath(link.parent / link.readlink()))
        if target.parent.parent in repositories and target.parent.name == link.name and target.name.isdecimal():
            link.unlink(missing_ok=True)

    for name, target in selected.items():
        link = active / name
        if link.is_symlink():
            if link.readlink() == target:
                continue
            with TemporaryDirectory(dir=active, prefix=".local-link-") as directory:
                new_link = Path(directory) / "link"
                new_link.symlink_to(target)
                os.replace(new_link, link)
        elif not link.exists():
            link.symlink_to(target)


def main():
    base_url, active_directory, staging_directory, *repository_paths = sys.argv[1:]
    repositories = tuple(map(Path, repository_paths))
    versions = ready_versions(base_url)
    if versions is not None:
        try:
            selected = select_local_versions(versions, repositories)
        except OSError:
            # Keep current links if a version lookup fails during this scan.
            pass
        else:
            sync_links(Path(active_directory), selected, repositories)
    # Report overlap even when the Triton index is temporarily unavailable.
    # The watcher then skips staging discovery, path repair and cleanup.
    if overlaps_staging(Path(staging_directory).resolve(), repositories):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
