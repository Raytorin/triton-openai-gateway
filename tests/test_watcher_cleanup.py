"""Execute watcher regression scenarios against local disposable model copies."""
import json
import os
from pathlib import Path
import subprocess
import sys
from test_deepseek_checkpoint import snapshot
from test_deepseek_watcher import checkout, run


def test_older_unpublished_duplicate_is_collected_without_replacing_active(tmp_path, snapshot):
    old = checkout(tmp_path, snapshot, folder="folderA", deepseek=False)
    active = checkout(tmp_path, snapshot, folder="folderZ", deepseek=False)
    os.utime(old / "model.json", (1000, 1000))
    os.utime(active / "model.json", (2000, 2000))
    links = tmp_path / "models-active"
    links.mkdir()
    (links / "selected").symlink_to(active)
    result = run(tmp_path, "remember_existing_links; scan_once; scan_once")
    assert result.returncode == 0, result.stderr
    assert (links / "selected").resolve() == active
    assert active.is_dir()
    assert not old.exists()


def test_unchanged_config_error_is_logged_once_per_watcher(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot, deepseek=False)
    (target.parent / "config.pbtxt").write_text('name: "selected" backend: 123')
    result = run(tmp_path, "scan_once; scan_once; scan_once")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("backend extraction failed:") == 1
    assert target.is_dir()
    assert not (tmp_path / "models-active/selected").exists()


def test_config_failure_cache_retries_when_file_changes(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot, deepseek=False)
    (target.parent / "config.pbtxt").write_text('name: "selected" backend: 123')
    result = run(tmp_path, '''scan_once; scan_once
printf '%s\n' 'name: "selected" backend: "vllm_multimodal"' > "$WATCHER_MODEL_DIR/folderA/config.pbtxt"
scan_once''')
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("backend extraction failed:") == 1
    assert (tmp_path / "models-active/selected").resolve() == target
    assert json.loads((target / "model.json").read_text())["model"] == str(target)


def process_view(tmp_path):
    """Expose this real test process through an isolated procfs view."""
    from pathlib import Path
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / str(os.getpid())).symlink_to(Path("/proc") / str(os.getpid()))
    return proc


def test_unlink_grace_survives_watcher_restart(tmp_path, snapshot, monkeypatch):
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "300")
    target = checkout(tmp_path, snapshot, deepseek=False)
    result = run(tmp_path, 'scan_once; rm "$MODELS_ACTIVE_DIR/selected"; scan_once; scan_once')
    assert result.returncode == 0, result.stderr
    assert target.is_dir(), "A just-retired model must not disappear during initialization"
    assert not (tmp_path / "models-active/selected").is_symlink()
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    assert target.is_dir()
    assert not (tmp_path / "models-active/selected").is_symlink()


def test_open_weight_defers_cleanup_until_process_releases_it(tmp_path, snapshot, monkeypatch):
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "0")
    monkeypatch.setenv("WATCHER_PROC_ROOT", str(process_view(tmp_path)))
    target = checkout(tmp_path, snapshot, deepseek=False)
    weight = target / "weights.safetensors"
    weight.write_bytes(b"test weights")
    with weight.open("rb"):
        result = run(tmp_path, 'scan_once; rm "$MODELS_ACTIVE_DIR/selected"; scan_once')
        assert result.returncode == 0, result.stderr
        assert weight.is_file(), "An unlinked model can still be used by a loader"
        assert not (tmp_path / "models-active/selected").is_symlink()
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    assert not target.exists()


def test_mapped_weight_keeps_retired_copy(tmp_path, snapshot, monkeypatch):
    import mmap
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "0")
    proc = tmp_path / "proc"
    (proc / "1/fd").mkdir(parents=True)
    monkeypatch.setenv("WATCHER_PROC_ROOT", str(proc))
    target = checkout(tmp_path, snapshot, deepseek=False)
    weight = target / "weights.safetensors"
    weight.write_bytes(b"test weights")
    with weight.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ):
        # Keep the real kernel maps snapshot but omit the duplicate mmap FD.
        (proc / "1/maps").write_text(Path("/proc/self/maps").read_text())
        result = run(tmp_path, 'scan_once; rm "$MODELS_ACTIVE_DIR/selected"; scan_once')
        assert result.returncode == 0, result.stderr
        assert weight.is_file()
    (proc / "1/maps").write_text("")
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    assert not target.exists()


def test_first_scan_selects_newest_original_copy_before_repair(tmp_path, snapshot):
    old = checkout(tmp_path, snapshot, folder="folderA", deepseek=False)
    new = checkout(tmp_path, snapshot, folder="folderZ", deepseek=False)
    os.utime(old / "model.json", ns=(1000000000000, 1000000000000))
    os.utime(new / "model.json", ns=(2000000000000, 2000000000000))
    result = run(tmp_path, "scan_once; scan_once")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models-active/selected").resolve() == new
    assert new.is_dir()
    assert not old.exists()


def test_newer_source_copy_survives_watcher_restart(tmp_path, snapshot):
    old = checkout(tmp_path, snapshot, folder="folderA", deepseek=False)
    os.utime(old / "model.json", ns=(1000000000000, 1000000000000))
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models-active/selected").resolve() == old
    new = checkout(tmp_path, snapshot, folder="folderZ", deepseek=False)
    os.utime(new / "model.json", ns=(2000000000000, 2000000000000))
    result = run(tmp_path, "remember_existing_links; scan_once; scan_once")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "models-active/selected").resolve() == new
    assert new.is_dir()
    assert not old.exists()


def test_cleanup_interval_uses_decimal_seconds(tmp_path, snapshot, monkeypatch):
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "300")
    target = checkout(tmp_path, snapshot, deepseek=False)
    result = run(tmp_path, '''WATCHER_CLEANUP_INTERVAL_SECONDS=08
scan_once
rm "$MODELS_ACTIVE_DIR/selected"
scan_once; scan_once''')
    assert result.returncode == 0, result.stderr
    assert "value too great for base" not in result.stderr
    assert target.is_dir()
    assert not (tmp_path / "models-active/selected").is_symlink()


def test_pending_empty_checkout_does_not_retire_new_version_zero(tmp_path, snapshot, monkeypatch):
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "300")
    old = checkout(tmp_path, snapshot, deepseek=False)
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    import shutil
    shutil.rmtree(old)
    result = run(tmp_path, "remember_existing_links; scan_once")
    assert result.returncode == 0, result.stderr
    assert (old.parent / '.watcher-retired.json').is_file()
    replacement = old.parent / '0'
    replacement.mkdir()
    (replacement / 'model.json').write_text('{"model":"/stale"}')
    result = run(tmp_path, "scan_once")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'models-active/selected').resolve() == replacement
    assert not (replacement / '.watcher-retired.json').exists()


def test_completed_reference_check_does_not_throttle_later_unload(tmp_path, snapshot, monkeypatch):
    monkeypatch.setenv("WATCHER_CLEANUP_GRACE_SECONDS", "0")
    target = checkout(tmp_path, snapshot, deepseek=False)
    result = run(tmp_path, '''WATCHER_CLEANUP_INTERVAL_SECONDS=30
scan_once
ln -s "$WATCHER_MODEL_DIR/folderA/1" "$MODELS_ACTIVE_DIR/alias"
remember_existing_links
rm "$MODELS_ACTIVE_DIR/selected"
cleanup_removed_links
rm "$MODELS_ACTIVE_DIR/alias"
scan_once''')
    assert result.returncode == 0, result.stderr
    assert not target.exists()
    assert not (tmp_path / 'models-active/selected').is_symlink()


def test_watcher_recovers_interrupted_metadata_cleanup_before_publication(tmp_path, snapshot):
    import shutil

    target = checkout(tmp_path, snapshot, deepseek=False)
    config = target.parent / 'config.pbtxt'
    original_config = config.read_bytes()
    shutil.rmtree(target)
    active = tmp_path / 'models-active'
    active.mkdir()
    proc = tmp_path / 'proc'
    proc.mkdir()
    script = '''
import os
from pathlib import Path
import sys
from gateway import model_cleanup as cleanup
root, active, target, proc = map(Path, sys.argv[1:])
original_rmdir = cleanup.os.rmdir
def crash_during_removal(path, *, dir_fd=None):
    if path == target.parent.name and dir_fd is not None:
        replacement = target.parent / '2'
        replacement.mkdir()
        (replacement / 'model.json').write_text('{"model":"/stale"}')
        (replacement / 'weights').write_bytes(b'new weights')
        os._exit(79)
    return original_rmdir(path, dir_fd=dir_fd)
cleanup.os.rmdir = crash_during_removal
cleanup.retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc)
'''
    crashed = subprocess.run(
        [sys.executable, '-c', script, str(tmp_path), str(active), str(target), str(proc)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
    )
    assert crashed.returncode == 79, crashed.stdout + crashed.stderr
    assert not config.exists()
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    assert config.read_bytes() == original_config
    replacement = target.parent / '2'
    assert (active / 'selected').resolve() == replacement
    assert (replacement / 'weights').read_bytes() == b'new weights'
    assert not list(tmp_path.glob('.watcher-cleanup-*'))


def test_deferred_recovery_keeps_other_models_available_without_publishing_missing_config(
        tmp_path, snapshot):
    from test_model_cleanup_safety import crash_checkout_cleanup

    bad = checkout(tmp_path, snapshot, deepseek=False)
    active = tmp_path / 'models-active'
    active.mkdir()
    proc = tmp_path / 'proc'
    proc.mkdir()
    crash_checkout_cleanup((tmp_path, active, bad, proc))
    bad.parent.rename(tmp_path / 'saved-checkout')
    replacement = bad.parent / '2'
    replacement.mkdir(parents=True)
    (replacement / 'model.json').write_text('{"model":"/stale"}')
    good = checkout(tmp_path, snapshot, folder='folderZ', deepseek=False)
    (good.parent / 'config.pbtxt').write_text('name: "good" backend: "vllm_multimodal"')

    result = run(tmp_path, 'scan_once; scan_once')
    assert result.returncode == 0, result.stderr
    assert (active / 'good').resolve() == good
    assert sorted(path.name for path in active.iterdir()) == ['good']
    assert replacement.is_dir()
    assert not (bad.parent / 'config.pbtxt').exists()
    assert (tmp_path / 'saved-checkout/2/weights').read_bytes() == b'new download'
    assert list(tmp_path.glob('.watcher-cleanup-*/files/config.pbtxt'))
    assert result.stderr.count('model cleanup recovery deferred') == 1
