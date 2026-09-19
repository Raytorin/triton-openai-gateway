import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def model(root, folder, version="1", name="model"):
    checkout = root / folder
    target = checkout / version
    target.mkdir(parents=True)
    (checkout / "config.pbtxt").write_text(f'name: "{name}"\n')
    (target / "model.json").write_text(json.dumps({"model": str(target)}))
    (target / "weights.bin").write_bytes(b"weights")
    return target


def run(root, script):
    active = root / "models-active"
    active.mkdir(exist_ok=True)
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; ' + script, "bash", str(ROOT / "watch_triton.sh")],
        env={**os.environ, "WATCHER_MODEL_DIR": str(root), "MODELS_ACTIVE_DIR": str(active)},
        text=True, capture_output=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_unlink_removes_checkout_without_republishing(tmp_path):
    old = model(tmp_path, "folderOld")
    run(tmp_path, 'scan_once; rm -- "$MODELS_ACTIVE_DIR/model"; scan_once; scan_once')
    assert not old.parent.exists()
    assert not (tmp_path / "models-active" / "model").is_symlink()


def test_replacement_removes_previous_checkout_and_keeps_active(tmp_path):
    old = model(tmp_path, "folderOld", "1")
    new = model(tmp_path, "folderNew", "2")
    active = tmp_path / "models-active"
    active.mkdir()
    (active / "model").symlink_to(old)
    run(tmp_path, 'remember_existing_links; scan_once')
    assert not old.parent.exists()
    assert new.is_dir()
    assert (active / "model").resolve() == new


def test_external_retarget_cleans_previous_version(tmp_path):
    old = model(tmp_path, "folderOld")
    new = model(tmp_path, "folderNew", "2")
    active = tmp_path / "models-active"
    active.mkdir()
    (active / "model").symlink_to(old)
    run(tmp_path, 'remember_existing_links; ln -sfnT "$WATCHER_MODEL_DIR/folderNew/2" "$MODELS_ACTIVE_DIR/model"; scan_once')
    assert not old.parent.exists()
    assert (active / "model").resolve() == new


def test_stale_link_removes_checkout_metadata(tmp_path):
    target = model(tmp_path, "folderStale")
    active = tmp_path / "models-active"
    active.mkdir()
    (active / "model").symlink_to(target)
    import shutil
    shutil.rmtree(target)
    run(tmp_path, 'remember_existing_links; scan_once')
    assert not target.parent.exists()
    assert not (active / "model").is_symlink()


def test_other_active_reference_protects_shared_checkout(tmp_path):
    target = model(tmp_path, "folderShared")
    active = tmp_path / "models-active"
    active.mkdir()
    (active / "model").symlink_to(target)
    (active / "alias").symlink_to(target)
    run(tmp_path, 'remember_existing_links; rm "$MODELS_ACTIVE_DIR/alias"; cleanup_removed_links')
    assert (target / "weights.bin").exists()


def test_other_version_and_unmanaged_directory_are_preserved(tmp_path):
    old = model(tmp_path, "folderShared", "1")
    new = model(tmp_path, "folderShared", "2")
    unrelated = model(tmp_path, "source-model", "1", "source")
    run(tmp_path, 'cleanup_unlinked_target "$WATCHER_MODEL_DIR/folderShared/1"; cleanup_unlinked_target "$WATCHER_MODEL_DIR/source-model/1"')
    assert not old.exists()
    assert new.is_dir()
    assert (new.parent / "config.pbtxt").is_file()
    assert unrelated.is_dir()


def test_cleanup_does_not_follow_checkout_or_weight_symlinks(tmp_path):
    watched = tmp_path / "watched"
    watched.mkdir()
    external = model(tmp_path, "folderExternal")
    (watched / "folderLink").symlink_to(external.parent, target_is_directory=True)
    target = model(watched, "folderOwned")
    (target / "shared").symlink_to(external, target_is_directory=True)
    run(watched, 'cleanup_unlinked_target "$WATCHER_MODEL_DIR/folderLink/1"; cleanup_unlinked_target "$WATCHER_MODEL_DIR/folderOwned/1"')
    assert not target.parent.exists()
    assert (external / "weights.bin").exists()


def test_model_name_cannot_escape_active_registry(tmp_path):
    target = model(tmp_path, "folderBad", name="../escaped")
    run(tmp_path, 'scan_once')
    assert target.exists()
    assert not (tmp_path / "escaped").exists()
