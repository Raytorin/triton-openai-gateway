"""Use the downloaded checkpoint before publishing its stable repository alias."""
import ast
import copy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from test_deepseek_checkpoint import snapshot
from test_deepseek_runtime import load_policy
from test_deepseek_watcher import checkout, run

ROOT = Path(__file__).resolve().parents[1]


def materialized_checkout(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot)
    shutil.copytree(snapshot, target, dirs_exist_ok=True)
    active = tmp_path / 'models-active'
    config = {
        'model': str(active / 'selected/1'),
        'tokenizer': str(active / 'selected/1'),
        'language_model_only': True,
        'enable_mm_embeds': False,
        'safetensors_load_strategy': 'lazy',
    }
    (target / 'model.json').write_text(json.dumps(config))
    return target, active, config


def test_watcher_binds_downloaded_snapshot_before_publishing_alias(tmp_path, snapshot):
    target, active, config = materialized_checkout(tmp_path, snapshot)
    source = target / 'model.json'
    source.chmod(0o640)
    source_mtime = source.stat().st_mtime_ns

    result = run(tmp_path, 'scan_once')

    assert result.returncode == 0, result.stderr
    assert (active / 'selected').is_symlink(), result.stderr
    assert (active / 'selected').resolve() == target
    assert json.loads(source.read_text()) == {**config, 'model': str(target), 'tokenizer': str(target)}
    assert source.stat().st_mtime_ns == source_mtime
    assert source.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize('incomplete', ['missing', 'truncated'])
def test_partial_s3_snapshot_is_not_published_or_rewritten_and_can_retry(tmp_path, snapshot, incomplete):
    target, active, config = materialized_checkout(tmp_path, snapshot)
    source = target / 'model.json'
    before = source.read_bytes()
    shard = target / 'a.safetensors'
    complete = shard.read_bytes()
    if incomplete == 'missing':
        shard.unlink()
    else:
        shard.write_bytes(complete[:5])

    first = run(tmp_path, 'scan_once')

    assert first.returncode == 0, first.stderr
    assert not (active / 'selected').is_symlink()
    assert source.read_bytes() == before
    shard.write_bytes(complete)
    retry = run(tmp_path, 'scan_once')
    assert retry.returncode == 0, retry.stderr
    assert (active / 'selected').is_symlink(), retry.stderr
    assert (active / 'selected').resolve() == target
    assert json.loads(source.read_text())['model'] == str(target)


def test_stale_alias_does_not_validate_or_select_the_old_checkpoint(tmp_path, snapshot):
    target, active, _ = materialized_checkout(tmp_path, snapshot)
    previous = checkout(tmp_path, snapshot, folder='folderPrevious')
    shutil.copytree(snapshot, previous, dirs_exist_ok=True)
    (previous / 'a.safetensors').unlink()
    os.utime(previous / 'model.json', (1, 1))
    active.mkdir()
    (active / 'selected').symlink_to(previous, target_is_directory=True)

    result = run(tmp_path, 'scan_once')

    assert result.returncode == 0, result.stderr
    assert (active / 'selected').resolve() == target, result.stderr
    assert json.loads((target / 'model.json').read_text())['model'] == str(target)


def shared_argument_reader():
    """Execute the real shared config/argument methods without GPU imports."""
    tree = ast.parse((ROOT / 'backends/vllm_multimodal/model.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TritonPythonModel')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_init_engine_args', '_load_engine_config'}]
    namespace = {
        'os': os, 'json': json, '_VLLM_ENGINE_ARGS_FILENAME': 'model.json',
        'load_pooling_model_metadata': lambda config, **kwargs: {},
        'AsyncEngineArgs': lambda **kwargs: copy.deepcopy(kwargs),
        'pb_utils': sys.modules['triton_python_backend_utils'],
    }
    node = ast.ClassDef(name='SharedReader', bases=[], keywords=[], body=methods, decorator_list=[])
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<shared-config-reader>', 'exec'), namespace)
    return namespace['SharedReader']


@pytest.mark.parametrize('replace_source_after_admission', [False, True])
def test_backend_engine_uses_exactly_the_admitted_repository_paths(monkeypatch, tmp_path, snapshot, replace_source_after_admission):
    target, active, original = materialized_checkout(tmp_path, snapshot)
    source = target / 'model.json'
    before = source.read_bytes()
    policy = load_policy(monkeypatch)
    pb = sys.modules['triton_python_backend_utils']
    monkeypatch.setattr(pb, 'get_model_dir', lambda: str(target), raising=False)
    monkeypatch.setattr(pb, 'using_decoupled_model_transaction_policy', lambda config: True, raising=False)
    monkeypatch.setenv('MODELS_ACTIVE_DIR', str(active))
    checkpoint = policy._preflight_model.__globals__['_checkpoint']
    monkeypatch.setattr(checkpoint, 'check_runtime', lambda report: {'registry_membership_verified': True})
    reader = shared_argument_reader()

    class Engine(reader):
        def initialize(self, args):
            self.model_config = {}
            if replace_source_after_admission:
                source.write_text('{"model":"org/not-the-admitted-checkpoint"}')
            self._init_engine_args()
            self.supported_tasks = ('generate',)

        def _validate_device_config(self):
            pass

        def _setup_lora(self):
            pass

    class Model(policy, Engine):
        pass

    instance = Model()
    instance.initialize({'model_name': 'selected', 'model_version': '1'})

    assert instance.checkpoint_report['model_root'] == str(target)
    assert instance.checkpoint_report['tokenizer_root'] == str(target)
    assert instance._aync_engine_args['model'] == str(target)
    assert instance._aync_engine_args['tokenizer'] == str(target)
    assert instance._aync_engine_args['language_model_only'] is True
    if not replace_source_after_admission:
        assert source.read_bytes() == before
    instance.finalize()


def test_default_shared_argument_reader_preserves_non_deepseek_config(monkeypatch, tmp_path):
    source = tmp_path / 'model.json'
    config = {'model': '/weights/qwen', 'tensor_parallel_size': 2}
    source.write_text(json.dumps(config))
    before = source.read_bytes()
    load_policy(monkeypatch)
    pb = sys.modules['triton_python_backend_utils']
    monkeypatch.setattr(pb, 'get_model_dir', lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(pb, 'using_decoupled_model_transaction_policy', lambda config: True, raising=False)
    reader = shared_argument_reader()

    class Model(reader):
        model_config = {}

        def _validate_device_config(self):
            pass

        def _setup_lora(self):
            pass

    model = Model()
    model._init_engine_args()
    assert model._aync_engine_args == config
    assert source.read_bytes() == before
