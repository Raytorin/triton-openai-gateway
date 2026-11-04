"""The opt-in stock adapter patches only the exact reviewed metrics source."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT/'docker/unified/patch-stock.py'
ORIGINAL = b'''# Minimal source around the reviewed stock import boundary.
from vllm.v1.metrics.loggers import StatLoggerBase, build_1_2_5_buckets
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats


UNCHANGED_EXPORT = "stock"
'''
EXPECTED = b'''# Minimal source around the reviewed stock import boundary.
from vllm.v1.metrics.loggers import StatLoggerBase
from vllm.v1.metrics.buckets import build_1_2_5_buckets
from vllm.v1.metrics.stats import IterationStats, MultiModalCacheStats, SchedulerStats


UNCHANGED_EXPORT = "stock"
'''


def load_tool():
    assert TOOL.is_file(), 'the pinned stock compatibility patch tool is missing'
    spec = importlib.util.spec_from_file_location('patch_stock_test', TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_tool(backend):
    assert TOOL.is_file(), 'the pinned stock compatibility patch tool is missing'
    return subprocess.run([sys.executable, '-I', str(TOOL), '--backend', str(backend)],
                          capture_output=True, text=True, timeout=5)


def test_missing_backend_fails_without_creating_paths(tmp_path):
    result = run_tool(tmp_path/'missing')
    assert result.returncode != 0
    assert 'metrics.py' in result.stderr
    assert not list(tmp_path.iterdir())


def test_unknown_stock_hash_fails_without_changing_source(tmp_path):
    target = tmp_path/'utils/metrics.py'
    target.parent.mkdir()
    target.write_bytes(ORIGINAL)
    result = run_tool(tmp_path)
    assert result.returncode != 0
    assert 'SHA256' in result.stderr
    assert target.read_bytes() == ORIGINAL
    assert list(target.parent.iterdir()) == [target]


def fixture_tool(monkeypatch):
    module = load_tool()
    # Keep fixtures small; actual release hashes are exercised in the image smoke.
    monkeypatch.setattr(module, 'ORIGINAL_SHA256', hashlib.sha256(ORIGINAL).hexdigest())
    monkeypatch.setattr(module, 'PATCHED_SHA256', hashlib.sha256(EXPECTED).hexdigest())
    return module


def test_exact_patch_preserves_surrounding_source_and_noop_reinstall(tmp_path, monkeypatch):
    module = fixture_tool(monkeypatch)
    target = tmp_path/'utils/metrics.py'
    target.parent.mkdir()
    target.write_bytes(ORIGINAL)
    target.chmod(0o640)
    assert module.patch_backend(tmp_path) is True
    assert target.read_bytes() == EXPECTED
    assert target.stat().st_mode & 0o777 == 0o640
    before = target.stat()
    assert module.patch_backend(tmp_path) is False
    after = target.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize('context', ['absent', 'duplicated'])
def test_import_context_must_occur_exactly_once(tmp_path, monkeypatch, context):
    module = fixture_tool(monkeypatch)
    source = (ORIGINAL.replace(b'build_1_2_5_buckets', b'unreviewed_helper')
              if context == 'absent' else ORIGINAL + module.ORIGINAL_IMPORTS)
    monkeypatch.setattr(module, 'ORIGINAL_SHA256', hashlib.sha256(source).hexdigest())
    target = tmp_path/'utils/metrics.py'
    target.parent.mkdir()
    target.write_bytes(source)
    with pytest.raises(RuntimeError, match='context'):
        module.patch_backend(tmp_path)
    assert target.read_bytes() == source


def test_unexpected_patch_result_hash_fails_before_writing(tmp_path, monkeypatch):
    module = fixture_tool(monkeypatch)
    monkeypatch.setattr(module, 'PATCHED_SHA256', '0' * 64)
    target = tmp_path/'utils/metrics.py'
    target.parent.mkdir()
    target.write_bytes(ORIGINAL)
    with pytest.raises(RuntimeError, match='SHA256'):
        module.patch_backend(tmp_path)
    assert target.read_bytes() == ORIGINAL


def test_atomic_replace_failure_leaves_original_and_no_temporary_files(tmp_path, monkeypatch):
    module = fixture_tool(monkeypatch)
    target = tmp_path/'utils/metrics.py'
    target.parent.mkdir()
    target.write_bytes(ORIGINAL)
    original_replace = Path.replace
    def reject_replace(path, destination):
        if destination == target:
            raise OSError('controlled replace failure')
        return original_replace(path, destination)
    monkeypatch.setattr(Path, 'replace', reject_replace)
    with pytest.raises(OSError, match='controlled replace failure'):
        module.patch_backend(tmp_path)
    assert target.read_bytes() == ORIGINAL
    assert list(target.parent.iterdir()) == [target]
