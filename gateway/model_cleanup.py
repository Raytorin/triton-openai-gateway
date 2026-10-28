#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retire and collect only watcher-owned local model versions (stdlib only)."""
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
import time

RETIRED = ".watcher-retired.json"


def _identity(path):
    info = path.lstat()
    return info.st_dev, info.st_ino


def _owned(root, active, target):
    if (target.is_symlink() or target.parent.is_symlink()
            or not target.parent.is_dir() or (target.exists() and not target.is_dir())):
        return False
    return (target.parent.parent == root
            and target.parent.name.startswith("folder")
            and re.fullmatch(r"[0-9]+", target.name) is not None
            and target.parent != active and target.parent not in active.parents)


def _active(active, target):
    for link in active.iterdir():
        if link.is_symlink():
            resolved = link.resolve()
            if resolved == target or target in resolved.parents:
                return True
    return False


def _snapshot(target):
    """Fingerprint metadata, never read weights or follow directory symlinks."""
    def fail(error):
        raise error

    entries = []
    for directory, dirs, files in os.walk(target, followlinks=False, onerror=fail):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            if path == target / RETIRED:
                continue
            info = path.lstat()
            entries.append((str(path.relative_to(target)), info.st_dev, info.st_ino,
                            info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return hashlib.sha256(json.dumps(sorted(entries)).encode()).hexdigest()


def _write_marker(marker, state):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=marker.parent,
                                         prefix=".watcher-retired-", delete=False) as stream:
            temporary = stream.name
            json.dump(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if marker.is_symlink():
            raise ValueError("retirement marker is a symlink")
        os.replace(temporary, marker)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def _in_use(target, proc_root):
    """An unreadable process inventory defers removal instead of guessing."""
    def refers(value):
        if not value.startswith("/"):
            return False
        resolved = Path(value.removesuffix(" (deleted)")).resolve()
        return resolved == target or target in resolved.parents

    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            for kind in ("cwd", "root", "exe"):
                try:
                    if refers(os.readlink(proc / kind)):
                        return True
                except FileNotFoundError:
                    pass
            for fd in (proc / "fd").iterdir():
                try:
                    if refers(os.readlink(fd)):
                        return True
                except FileNotFoundError:
                    pass
            for line in (proc / "maps").read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and refers(fields[5]):
                    return True
        except (FileNotFoundError, ProcessLookupError):
            if proc.exists():
                return True  # Missing inventory for a still-existing process.
            continue
        except OSError:
            return True
    return False


def _recover_cleanup_stage(root_fd, stage_name, root, active):
    """Finish one persisted metadata move without following unknown directories."""
    stage_fd = os.open(stage_name, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    files_fd = checkout_fd = None
    try:
        stage_entries = set(os.listdir(stage_fd))
        if not stage_entries:
            # A crash after journal removal can leave only the empty container.
            os.rmdir(stage_name, dir_fd=root_fd)
            os.fsync(root_fd)
            return
        if stage_entries - {"journal.json", "files", "removed"}:
            raise ValueError("unknown cleanup staging entries")
        names = []
        if "files" in stage_entries:
            files_fd = os.open("files", os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=stage_fd)
            names = os.listdir(files_fd)
        if not names:
            # Before the first move (or after the last restore) there is no
            # entrusted metadata, so a partial journal must not stall discovery.
            for name in stage_entries - {"files"}:
                if stat.S_ISDIR(os.stat(name, dir_fd=stage_fd,
                                       follow_symlinks=False).st_mode):
                    raise ValueError("unexpected directory in empty cleanup staging")
            if files_fd is not None:
                os.rmdir("files", dir_fd=stage_fd)
            for name in stage_entries - {"files"}:
                os.unlink(name, dir_fd=stage_fd)
            os.fsync(stage_fd)
            os.rmdir(stage_name, dir_fd=root_fd)
            os.fsync(root_fd)
            return
        journal_fd = os.open("journal.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=stage_fd)
        with os.fdopen(journal_fd, "r") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise ValueError("cleanup journal must be a regular file")
            raw = stream.read(2 * 1024**2 + 1)
        if len(raw) > 2 * 1024**2:
            raise ValueError("cleanup journal exceeds local byte limit")
        journal = json.loads(raw)
        if not isinstance(journal, dict) or journal.get("version") != 1:
            raise ValueError("invalid cleanup journal")
        checkout_name, identity, admitted = (
            journal.get("checkout"), journal.get("identity"), journal.get("entries"))

        def basename(name):
            return (isinstance(name, str) and name not in ("", ".", "..")
                    and "/" not in name and "\x00" not in name)

        if (not basename(checkout_name) or not checkout_name.startswith("folder")
                or root / checkout_name == active
                or not isinstance(identity, list) or len(identity) != 2
                or any(type(value) is not int or value < 0 for value in identity)
                or not isinstance(admitted, list) or not all(map(basename, admitted))
                or len(set(admitted)) != len(admitted)):
            raise ValueError("invalid cleanup journal identity or metadata names")

        removed = "removed" in stage_entries
        if removed:
            info = os.stat("removed", dir_fd=stage_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_size:
                raise ValueError("invalid checkout removal marker")
        else:
            try:
                checkout_fd = os.open(checkout_name, os.O_DIRECTORY | os.O_NOFOLLOW,
                                      dir_fd=root_fd)
            except FileNotFoundError:
                # Normal crash gap: rmdir succeeded before its marker was saved.
                removed = True
            else:
                info = os.fstat(checkout_fd)
                if [info.st_dev, info.st_ino] != identity:
                    raise ValueError("cleanup checkout identity changed")

        if files_fd is not None:
            if set(names) - set(admitted):
                raise ValueError("unknown staged metadata")
            if any(stat.S_ISDIR(os.stat(name, dir_fd=files_fd,
                                       follow_symlinks=False).st_mode) for name in names):
                raise ValueError("staged metadata contains a directory; retained for recovery")
            for name in names:
                if not removed:
                    try:
                        os.link(name, name, src_dir_fd=files_fd,
                                dst_dir_fd=checkout_fd, follow_symlinks=False)
                    except FileExistsError:
                        pass  # A producer's newer metadata always wins.
                    os.fsync(checkout_fd)
                os.unlink(name, dir_fd=files_fd)
                os.fsync(files_fd)
            os.rmdir("files", dir_fd=stage_fd)
        if "removed" in stage_entries:
            os.unlink("removed", dir_fd=stage_fd)
        os.unlink("journal.json", dir_fd=stage_fd)
        os.fsync(stage_fd)
        os.rmdir(stage_name, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        for descriptor in (files_fd, checkout_fd, stage_fd):
            if descriptor is not None:
                os.close(descriptor)


def recover_pending_cleanups(root, active):
    """Recover crash-interrupted checkout metadata moves before model discovery."""
    root, active = Path(root).resolve(), Path(active).resolve()
    root_fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW)
    complete = True
    try:
        for name in sorted(os.listdir(root_fd)):
            if not name.startswith(".watcher-cleanup-"):
                continue
            try:
                _recover_cleanup_stage(root_fd, name, root, active)
            except (OSError, ValueError, TypeError) as error:
                print(f"model cleanup recovery deferred for {root / name}: {error}", file=sys.stderr)
                complete = False
    finally:
        os.close(root_fd)
    return complete


def _remove_empty_checkout(root_fd, checkout, expected, active):
    descriptor = os.open(checkout.name, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
    staging = None
    staging_fd = None
    files_fd = None
    journal_ready = False
    try:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) != expected or _active(active, checkout):
            return False
        entries = {name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                   for name in os.listdir(descriptor)}
        if any(stat.S_ISDIR(info.st_mode) for info in entries.values()):
            return False
        for name, expected_info in entries.items():
            if os.stat(name, dir_fd=descriptor, follow_symlinks=False) != expected_info:
                return False
        # Stage shared metadata reversibly: a producer may create a version
        # between the inventory and rmdir. Never leave that version without its
        # config when the final, non-recursive directory removal fails.
        staging = Path(tempfile.mkdtemp(prefix=".watcher-cleanup-", dir=checkout.parent))
        staging_fd = os.open(staging, os.O_DIRECTORY | os.O_NOFOLLOW)
        os.mkdir("files", mode=0o700, dir_fd=staging_fd)
        files_fd = os.open("files", os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=staging_fd)
        journal_fd = os.open("journal.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=staging_fd)
        with os.fdopen(journal_fd, "w") as stream:
            json.dump({"version": 1, "checkout": checkout.name,
                       "identity": list(expected), "entries": list(entries)}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(staging_fd)
        os.fsync(root_fd)
        journal_ready = True
        for name in entries:
            os.rename(name, name, src_dir_fd=descriptor, dst_dir_fd=files_fd)
            os.fsync(files_fd)
            os.fsync(descriptor)
            info = os.stat(name, dir_fd=files_fd, follow_symlinks=False)
            if (stat.S_ISDIR(info.st_mode)
                    or (info.st_dev, info.st_ino) != (
                        entries[name].st_dev, entries[name].st_ino)):
                return False
        if _active(active, checkout):
            return False
        try:
            os.rmdir(checkout.name, dir_fd=root_fd)
        except OSError as error:
            if error.errno in (errno.ENOTEMPTY, errno.EEXIST):
                return False  # A version/download appeared; never recurse into it.
            raise
        os.fsync(root_fd)
        removed_fd = os.open("removed", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                             0o600, dir_fd=staging_fd)
        try:
            os.fsync(removed_fd)
        finally:
            os.close(removed_fd)
        os.fsync(staging_fd)
        print(f"removed inactive model checkout {checkout}", flush=True)
        return True
    finally:
        try:
            if staging_fd is not None:
                if journal_ready:
                    _recover_cleanup_stage(root_fd, staging.name, checkout.parent, active)
                else:
                    # No metadata move is allowed before the journal is durable.
                    try:
                        os.unlink("journal.json", dir_fd=staging_fd)
                    except FileNotFoundError:
                        pass
                    if files_fd is not None:
                        os.rmdir("files", dir_fd=staging_fd)
                    os.rmdir(staging)
        except OSError as error:
            raise OSError(error.errno,
                          f"metadata retained for recovery in {staging}: {error}") from error
        finally:
            if staging_fd is not None:
                os.close(staging_fd)
            if files_fd is not None:
                os.close(files_fd)
            os.close(descriptor)


def retire_model_copy(root, active, target, backend, grace_seconds=300, proc_root="/proc",
                      empty_only=False):
    """Return False while retirement is pending; True when removed/protected."""
    root, active = Path(root).resolve(), Path(active).resolve()
    target = Path(os.path.abspath(target))
    if not _owned(root, active, target) or _active(active, target):
        return True
    checkout = target.parent
    empty_checkout = not target.exists()
    if empty_only and not empty_checkout:
        return True  # Metadata collection must never retire a newly arrived version.
    if empty_checkout:
        if any(child.is_dir() for child in checkout.iterdir()) or _active(active, checkout):
            return True
        target = checkout
    target_id, checkout_id = _identity(target), _identity(checkout)
    # These checkouts may point to a shared checkpoint outside the watcher root.
    if backend == "vllm_deepseek" or (target / ".deepseek-retired").exists() or (target / ".deepseek-retired").is_symlink():
        try:
            descriptor = os.open(target / ".deepseek-retired",
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        return True

    marker = target / RETIRED
    fingerprint = _snapshot(target)
    now = time.time()
    if marker.is_symlink():
        raise ValueError("retirement marker is a symlink")
    if marker.exists():
        state = json.loads(marker.read_text())
        if not isinstance(state, dict):
            raise ValueError("invalid retirement marker")
        since = state.get("since")
        if isinstance(since, bool) or not isinstance(since, (int, float)) or not math.isfinite(since) or since < 0:
            raise ValueError("invalid retirement timestamp")
        if state.get("identity") != list(target_id):
            raise ValueError("retired model directory identity changed")
        if state.get("fingerprint") != fingerprint:
            state.update(fingerprint=fingerprint, since=now)
            _write_marker(marker, state)
    else:
        state = {"identity": list(target_id), "fingerprint": fingerprint, "since": now}
        _write_marker(marker, state)
    if now - float(state["since"]) < grace_seconds or _in_use(target, Path(proc_root)):
        return False
    if (_identity(target) != target_id or _identity(checkout) != checkout_id
            or _snapshot(target) != fingerprint or _active(active, target)):
        return False

    # Bind deletion to the admitted parent inode, not a replaceable pathname.
    root_fd = os.open(root, os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if empty_checkout:
            return _remove_empty_checkout(root_fd, checkout, checkout_id, active)
        checkout_fd = os.open(checkout.name, os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        try:
            info = os.fstat(checkout_fd)
            if (info.st_dev, info.st_ino) != checkout_id:
                return False
            info = os.stat(target.name, dir_fd=checkout_fd, follow_symlinks=False)
            if (info.st_dev, info.st_ino) != target_id or not stat.S_ISDIR(info.st_mode):
                return False
            shutil.rmtree(target.name, dir_fd=checkout_fd)
            print(f"removed inactive model version {target}", flush=True)
        finally:
            os.close(checkout_fd)
        if _identity(checkout) == checkout_id:
            _remove_empty_checkout(root_fd, checkout, checkout_id, active)
    finally:
        os.close(root_fd)
    return True


if __name__ == "__main__":
    try:
        if sys.argv[1:2] == ["--recover"]:
            if len(sys.argv) != 4:
                raise ValueError("expected --recover ROOT ACTIVE")
            raise SystemExit(0 if recover_pending_cleanups(sys.argv[2], sys.argv[3]) else 1)
        root, active, target, backend, grace, proc, mode = sys.argv[1:]
        if mode not in ("version", "empty"):
            raise ValueError("unknown cleanup mode")
        grace = float(grace)
        if not 0 <= grace < float("inf"):
            raise ValueError("cleanup grace must be finite and non-negative")
        complete = retire_model_copy(root, active, target, backend, grace, proc,
                                     empty_only=(mode == "empty"))
    except (OSError, ValueError, TypeError) as error:
        print(f"model cleanup deferred: {error}", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0 if complete else 75)
