"""Cleanup safety and grace tests use real disposable directories and process views."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.model_cleanup import RETIRED, retire_model_copy


@pytest.fixture
def copies(tmp_path):
    root = tmp_path / 'downloads'
    active = root / 'models-active'
    target = root / 'folderOld/1'
    target.mkdir(parents=True)
    active.mkdir()
    (target / 'model.json').write_text('{}')
    (target / 'weights').write_bytes(b'weights')
    proc = tmp_path / 'proc'
    proc.mkdir()
    return root, active, target, proc


def collect(copies, grace=300):
    root, active, target, proc = copies
    return retire_model_copy(root, active, target, 'vllm_multimodal', grace, proc)


def test_changed_download_restarts_retirement_grace(copies, monkeypatch):
    import gateway.model_cleanup as cleanup
    clock = [1000.0]
    monkeypatch.setattr(cleanup.time, 'time', lambda: clock[0])
    target = copies[2]
    assert collect(copies) is False
    clock[0] = 1299
    assert collect(copies) is False
    (target / 'new-shard').write_bytes(b'download resumed')
    clock[0] = 1300
    assert collect(copies) is False
    clock[0] = 1599
    assert collect(copies) is False
    clock[0] = 1600
    assert collect(copies) is True
    assert not target.exists()


def test_active_version_is_never_marked_or_deleted(copies):
    _, active, target, _ = copies
    (active / 'selected').symlink_to(target)
    assert collect(copies, 0) is True
    assert target.is_dir()
    assert not (target / RETIRED).exists()


def test_external_directory_and_symlink_are_preserved(copies, tmp_path):
    root, active, _, proc = copies
    external = tmp_path / 'external/1'
    external.mkdir(parents=True)
    (external / 'weights').write_bytes(b'original')
    (root / 'folderExternal').symlink_to(external.parent)
    for target in [external, root / 'folderExternal/1']:
        assert retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc) is True
    assert (external / 'weights').read_bytes() == b'original'
    assert not (external / RETIRED).exists()


def test_other_version_and_external_weight_symlink_survive(copies, tmp_path):
    root, active, target, _ = copies
    sibling = target.parent / '2'
    sibling.mkdir()
    (sibling / 'weights').write_bytes(b'second version')
    external = tmp_path / 'shared-checkpoint'
    external.write_bytes(b'external weights')
    (target / 'shared').symlink_to(external)
    assert collect(copies, 0) is True
    assert sibling.is_dir()
    assert external.read_bytes() == b'external weights'
    assert target.parent.is_dir()


def test_retirement_marker_symlink_cannot_overwrite_external_file(copies, tmp_path):
    target = copies[2]
    external = tmp_path / 'external-marker'
    external.write_text('do not change')
    (target / RETIRED).symlink_to(external)
    with pytest.raises(ValueError, match='symlink'):
        collect(copies, 0)
    assert target.is_dir()
    assert external.read_text() == 'do not change'


def test_missing_inventory_for_an_existing_process_defers_cleanup(copies):
    target, proc = copies[2:]
    (proc / '123/fd').mkdir(parents=True)
    assert collect(copies, 0) is False
    assert target.is_dir()


@pytest.mark.parametrize('since', [float('nan'), float('-inf')])
def test_nonfinite_retirement_timestamp_cannot_expire_grace(copies, since):
    target = copies[2]
    assert collect(copies) is False
    marker = target / RETIRED
    state = json.loads(marker.read_text())
    state['since'] = since
    marker.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='timestamp'):
        collect(copies)
    assert target.is_dir()


@pytest.mark.parametrize('producer_replaces_config', [False, True])
def test_new_version_during_checkout_removal_preserves_shared_metadata(
        copies, monkeypatch, producer_replaces_config):
    import shutil
    import gateway.model_cleanup as cleanup

    root, active, target, proc = copies
    checkout = target.parent
    config = checkout / 'config.pbtxt'
    config.write_text('original shared config')
    shutil.rmtree(target)
    original_rmdir = cleanup.os.rmdir

    def download_arrives(path, *, dir_fd=None):
        if path == checkout.name and dir_fd is not None:
            new_version = checkout / '2'
            new_version.mkdir()
            (new_version / 'weights').write_bytes(b'new download')
            if producer_replaces_config:
                config.write_text('new producer config')
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(cleanup.os, 'rmdir', download_arrives)
    assert retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc) is False
    assert (checkout / '2/weights').read_bytes() == b'new download'
    assert config.read_text() == (
        'new producer config' if producer_replaces_config else 'original shared config')
    assert sorted(path.name for path in root.iterdir()) == ['folderOld', 'models-active']


def test_failed_checkout_removal_restores_shared_metadata(copies, monkeypatch):
    import errno
    import shutil
    import gateway.model_cleanup as cleanup

    root, active, target, proc = copies
    checkout = target.parent
    config = checkout / 'config.pbtxt'
    config.write_text('original shared config')
    shutil.rmtree(target)
    original_rmdir = cleanup.os.rmdir

    def removal_denied(path, *, dir_fd=None):
        if path == checkout.name and dir_fd is not None:
            raise PermissionError(errno.EACCES, 'test checkout removal denied')
        return original_rmdir(path, dir_fd=dir_fd)

    monkeypatch.setattr(cleanup.os, 'rmdir', removal_denied)
    with pytest.raises(PermissionError):
        retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc)
    assert config.read_text() == 'original shared config'
    assert sorted(path.name for path in root.iterdir()) == ['folderOld', 'models-active']


def crash_checkout_cleanup(copies, producer_replaces_config=False):
    import shutil

    root, active, target, proc = copies
    (target.parent / 'config.pbtxt').write_text('original shared config')
    shutil.rmtree(target)
    source_root = Path(__file__).resolve().parents[1]
    script = '''
import os
from pathlib import Path
import sys
from gateway import model_cleanup as cleanup
root, active, target, proc = map(Path, sys.argv[1:5])
original_rename = cleanup.os.rename
def crash_after_move(source, destination, **kwargs):
    result = original_rename(source, destination, **kwargs)
    if source == 'config.pbtxt':
        new_version = target.parent / '2'
        new_version.mkdir()
        (new_version / 'weights').write_bytes(b'new download')
        if sys.argv[5] == 'replace':
            (target.parent / 'config.pbtxt').write_text('new producer config')
        os._exit(71)
    return result
cleanup.os.rename = crash_after_move
cleanup.retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc)
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(root), str(active), str(target), str(proc),
         'replace' if producer_replaces_config else 'keep'],
        cwd=source_root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 71, result.stdout + result.stderr
    assert (target.parent / '2/weights').read_bytes() == b'new download'
    return source_root


@pytest.mark.parametrize('producer_replaces_config', [False, True])
def test_restart_recovers_staged_config_after_process_crash(copies, producer_replaces_config):
    source_root = crash_checkout_cleanup(copies, producer_replaces_config)
    root, active, target, _ = copies
    result = subprocess.run(
        [sys.executable, str(source_root / 'gateway/model_cleanup.py'),
         '--recover', str(root), str(active)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (target.parent / 'config.pbtxt').read_text() == (
        'new producer config' if producer_replaces_config else 'original shared config')
    assert (target.parent / '2/weights').read_bytes() == b'new download'
    assert sorted(path.name for path in root.iterdir()) == ['folderOld', 'models-active']


def test_recovery_refuses_replaced_checkout_identity(copies):
    source_root = crash_checkout_cleanup(copies)
    root, active, target, _ = copies
    checkout = target.parent
    checkout.rename(root / 'saved-checkout')
    checkout.mkdir()
    (checkout / 'config.pbtxt').write_text('replacement checkout config')
    result = subprocess.run(
        [sys.executable, str(source_root / 'gateway/model_cleanup.py'),
         '--recover', str(root), str(active)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert (checkout / 'config.pbtxt').read_text() == 'replacement checkout config'
    assert (root / 'saved-checkout/2/weights').read_bytes() == b'new download'
    assert any(path.read_text() == 'original shared config'
               for stage in root.glob('.watcher-cleanup-*')
               for path in stage.rglob('config.pbtxt'))


def test_recovery_discards_incomplete_journal_with_no_staged_metadata(copies):
    from gateway.model_cleanup import recover_pending_cleanups

    root, active, target, _ = copies
    stage = root / '.watcher-cleanup-interrupted-initialization'
    (stage / 'files').mkdir(parents=True)
    (stage / 'journal.json').write_text('{"version":')
    assert recover_pending_cleanups(root, active) is True
    assert not stage.exists()
    assert (target / 'weights').read_bytes() == b'weights'


@pytest.mark.parametrize('unexpected', ['extra-file', 'directory'])
def test_recovery_retains_unknown_or_directory_payload(copies, unexpected):
    from gateway.model_cleanup import recover_pending_cleanups

    crash_checkout_cleanup(copies)
    root, active, target, _ = copies
    stage, = root.glob('.watcher-cleanup-*')
    if unexpected == 'extra-file':
        (stage / 'files/unadmitted').write_text('retain unknown metadata')
    else:
        config = stage / 'files/config.pbtxt'
        config.unlink()
        config.mkdir()
        (config / 'new-download').write_bytes(b'do not recurse')
    assert recover_pending_cleanups(root, active) is False
    assert stage.is_dir()
    assert (target.parent / '2/weights').read_bytes() == b'new download'
    if unexpected == 'extra-file':
        assert (stage / 'files/config.pbtxt').read_text() == 'original shared config'
    else:
        assert (stage / 'files/config.pbtxt/new-download').read_bytes() == b'do not recurse'


def test_restart_finishes_cleanup_after_checkout_rmdir(copies):
    import shutil

    root, active, target, proc = copies
    (target.parent / 'config.pbtxt').write_text('obsolete shared config')
    shutil.rmtree(target)
    source_root = Path(__file__).resolve().parents[1]
    script = '''
import os
from pathlib import Path
import sys
from gateway import model_cleanup as cleanup
root, active, target, proc = map(Path, sys.argv[1:])
original_rmdir = cleanup.os.rmdir
def crash_after_rmdir(path, **kwargs):
    result = original_rmdir(path, **kwargs)
    if path == target.parent.name:
        os._exit(73)
    return result
cleanup.os.rmdir = crash_after_rmdir
cleanup.retire_model_copy(root, active, target, 'vllm_multimodal', 0, proc)
'''
    result = subprocess.run(
        [sys.executable, '-c', script, str(root), str(active), str(target), str(proc)],
        cwd=source_root, capture_output=True, text=True, timeout=10)
    assert result.returncode == 73, result.stdout + result.stderr
    assert not target.parent.exists()
    result = subprocess.run(
        [sys.executable, str(source_root / 'gateway/model_cleanup.py'),
         '--recover', str(root), str(active)],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(path.name for path in root.iterdir()) == ['models-active']
