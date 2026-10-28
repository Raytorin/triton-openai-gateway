"""Execute the real watcher against local snapshots and temporary checkouts."""
import json
import os
from pathlib import Path
import stat
import subprocess
import pytest
from test_deepseek_checkpoint import snapshot
ROOT=Path(__file__).resolve().parents[1]
BACKEND_FORMATS = [
    'backend: "vllm_deepseek" name: "selected"',
    "backend: 'vllm_deepseek' name: 'selected'",
    r'''input { name: "nested-input" data_type: TYPE_STRING dims: [1] } parameters { key: "name" value { string_value: "wrong" } } backend: "vllm_deepseek" name: '\163elec' "ted"''',
    'name: "selected" backend: "vllm_deepseek"',
    'name: "selected" backend : \'vllm_deepseek\'',
    'name: "selected" backend\n:\t\'vllm_\' "deepseek"',
    'name: "selected" backend: "\\166llm_\\x64eepseek"',
    'name: "selected" # backend: "vllm_multimodal"\nparameters { key: "notes" value { string_value: \'backend: "vllm_multimodal"\' } } backend: "vllm_deepseek"',
]


@pytest.fixture(params=BACKEND_FORMATS)
def config_text(request):
    # Qualify the fixtures with the actual protobuf parser used by Triton clients.
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2
    parsed=text_format.Parse(request.param,model_config_pb2.ModelConfig())
    assert parsed.backend=='vllm_deepseek'
    assert parsed.name=='selected'
    return request.param


def checkout(tmp_path,snapshot,version='1',folder='folderA',deepseek=True):
    target=tmp_path/folder/version;target.mkdir(parents=True)
    (target.parent/'config.pbtxt').write_text('name: "selected"\nbackend: "'+('vllm_deepseek' if deepseek else 'vllm_multimodal')+'"\n')
    (target/'model.json').write_text(json.dumps({'model':str(snapshot),'language_model_only':True,'enable_mm_embeds':False}))
    return target

def run(tmp_path,commands):
    active=tmp_path/'models-active';active.mkdir(exist_ok=True)
    proc=tmp_path/'proc';proc.mkdir(exist_ok=True)
    return subprocess.run(['bash','-c','source "$1"; '+commands,'bash',str(ROOT/'watch_triton.sh')],env={**os.environ,'WATCHER_MODEL_DIR':str(tmp_path),'MODELS_ACTIVE_DIR':str(active),'WATCHER_CLEANUP_GRACE_SECONDS':os.environ.get('WATCHER_CLEANUP_GRACE_SECONDS','0'),'WATCHER_CLEANUP_INTERVAL_SECONDS':'0','WATCHER_PROC_ROOT':os.environ.get('WATCHER_PROC_ROOT',str(proc))},capture_output=True,text=True,timeout=8)

def test_external_root_preserved_and_partial_checkpoint_retried(tmp_path,snapshot,config_text):
    target=checkout(tmp_path,snapshot)
    (target.parent/'config.pbtxt').write_text(config_text)
    shard=snapshot/'a.safetensors';saved=shard.read_bytes();shard.unlink()
    r=run(tmp_path,'scan_once');assert r.returncode==0,r.stderr
    link=tmp_path/'models-active/selected';assert not link.is_symlink()
    assert json.loads((target/'model.json').read_text())['model']==str(snapshot)
    shard.write_bytes(saved);r=run(tmp_path,'scan_once');assert r.returncode==0,r.stderr
    assert link.resolve()==target

def test_deepseek_retired_versions_survive_without_resurrection(tmp_path,snapshot,config_text):
    old=checkout(tmp_path,snapshot)
    (old.parent/'config.pbtxt').write_text(config_text)
    r=run(tmp_path,'scan_once');assert r.returncode==0,r.stderr
    new=checkout(tmp_path,snapshot,'2','folderB')
    (new.parent/'config.pbtxt').write_text(config_text)
    r=run(tmp_path,'remember_existing_links; scan_once; rm "$MODELS_ACTIVE_DIR/selected"; scan_once; scan_once')
    assert r.returncode==0,r.stderr
    assert old.is_dir() and new.is_dir(),'a loading stub can still hold the prior checkout'
    assert not (tmp_path/'models-active/selected').is_symlink()
    r=run(tmp_path,'remember_existing_links; scan_once');assert r.returncode==0,r.stderr
    assert not (tmp_path/'models-active/selected').is_symlink()

def test_atomic_json_preserves_permissions_and_qwen_rewrite(tmp_path,snapshot):
    target=checkout(tmp_path,snapshot,deepseek=False);file=target/'model.json';file.chmod(0o640);inode=file.stat().st_ino
    r=run(tmp_path,'scan_once');assert r.returncode==0,r.stderr
    assert json.loads(file.read_text())['model']==str(target)
    assert file.stat().st_ino!=inode and stat.S_IMODE(file.stat().st_mode)==0o640

def test_final_config_symlink_is_not_followed(tmp_path,snapshot):
    target=checkout(tmp_path,snapshot);file=target/'model.json';original=file.read_bytes();file.unlink()
    destination=tmp_path/'outside.json';destination.write_bytes(original);file.symlink_to(destination)
    r=run(tmp_path,'scan_once');assert r.returncode==0,r.stderr
    assert file.is_symlink() and destination.read_bytes()==original
    assert not (tmp_path/'models-active/selected').is_symlink()

@pytest.mark.parametrize('invalid',[
    'name: "selected" backend: "vllm_deepseek" backend: "vllm_multimodal"',
    'name: "selected" backend: "vllm_deepseek',
    'name: "selected" parameters { key: "x"',
    'name: "selected" backend: 123',
    None,
])
def test_backend_extraction_failure_blocks_publication_and_cleanup(tmp_path,snapshot,invalid):
    target=checkout(tmp_path,snapshot)
    config=target.parent/'config.pbtxt'
    if invalid is None:
        config.unlink();config.symlink_to(config)
    else:config.write_text(invalid)
    result=run(tmp_path,'scan_once; cleanup_unlinked_target "$WATCHER_MODEL_DIR/folderA/1" || true')
    assert result.returncode==0,result.stderr
    assert target.is_dir() and (target/'model.json').is_file()
    assert not (tmp_path/'models-active/selected').is_symlink()
    assert 'backend extraction failed' in result.stderr


def test_backend_lookalikes_do_not_change_qwen_cleanup(tmp_path,snapshot):
    target=checkout(tmp_path,snapshot,deepseek=False)
    (target.parent/'config.pbtxt').write_text('''name: "selected"
# backend: "vllm_deepseek"
parameters { key: "backend" value { string_value: "vllm_deepseek" } }
backend: "vllm_multimodal"
''')
    result=run(tmp_path,'scan_once; rm "$MODELS_ACTIVE_DIR/selected"; scan_once')
    assert result.returncode==0,result.stderr
    assert not target.exists()


@pytest.mark.parametrize('invalid', [
    'backend: "vllm_deepseek" name: 12',
    'backend: "vllm_deepseek" name: ""',
    'backend: "vllm_deepseek" name: "selected" name: "other"',
    'backend: "vllm_deepseek" name: "unterminated',
])
def test_invalid_name_cannot_use_legacy_fallback(tmp_path, snapshot, invalid):
    target = checkout(tmp_path, snapshot)
    (target.parent/'config.pbtxt').write_text(invalid)
    (target/'model.json').write_text(json.dumps({'model':str(snapshot), 'served_model_name':'fallback'}))
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path/'models-active').iterdir())


@pytest.mark.parametrize('config', [
    'backend: "vllm_multimodal"',
    'backend: "vllm_multimodal" input { name: "nested-input" data_type: TYPE_STRING dims: [1] } parameters { key: "name" value { string_value: "nested" } }',
])
@pytest.mark.parametrize('fallback', ['metadata', 'hf', 'folder'])
def test_absent_name_keeps_legacy_fallback(tmp_path, snapshot, config, fallback):
    target = checkout(tmp_path, snapshot, deepseek=False)
    (target.parent/'config.pbtxt').write_text(config)
    if fallback == 'metadata':
        (target/'model.json').write_text('{"served_model_name":"legacy"}')
    elif fallback == 'hf':
        (target/'config.json').write_text('{"_name_or_path":"org/legacy"}')
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    expected = 'folderA' if fallback == 'folder' else 'legacy'
    assert (tmp_path/'models-active'/expected).resolve() == target


@pytest.mark.parametrize('kind', ['json', 'python'])
@pytest.mark.parametrize('fallback', ['metadata', 'hf', 'folder'])
def test_missing_config_keeps_legacy_publication(tmp_path, snapshot, kind, fallback):
    target = checkout(tmp_path, snapshot, deepseek=False)
    (target.parent/'config.pbtxt').unlink()
    if kind == 'python':
        (target/'model.json').unlink()
        (target/'model.py').write_text('class TritonPythonModel: pass\n')
    if fallback == 'metadata' and kind == 'json':
        (target/'model.json').write_text('{"served_model_name":"legacy"}')
    elif fallback in {'metadata', 'hf'}:
        (target/'config.json').write_text('{"_name_or_path":"org/legacy"}')
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    expected = 'folderA' if fallback == 'folder' else 'legacy'
    link = tmp_path/'models-active'/expected
    assert link.is_symlink(), result.stderr
    assert link.resolve() == target
    if kind == 'json':
        assert json.loads((target/'model.json').read_text())['model'] == str(target)


def test_missing_config_keeps_legacy_unlink_cleanup(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot, deepseek=False)
    (target.parent/'config.pbtxt').unlink()
    (target/'model.json').write_text('{"served_model_name":"legacy"}')
    result = run(tmp_path, 'scan_once; rm -- "$MODELS_ACTIVE_DIR/legacy"; scan_once')
    assert result.returncode == 0, result.stderr
    assert not target.parent.exists()
    assert not (tmp_path/'models-active/legacy').is_symlink()


def test_dangling_config_cannot_use_legacy_fallback(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot, deepseek=False)
    config = target.parent/'config.pbtxt'
    config.unlink()
    config.symlink_to(target.parent/'missing-config.pbtxt')
    (target/'model.json').write_text('{"served_model_name":"legacy"}')
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    assert not list((tmp_path/'models-active').iterdir())


def test_retired_deepseek_without_config_is_preserved(tmp_path, snapshot):
    target = checkout(tmp_path, snapshot)
    (target.parent/'config.pbtxt').unlink()
    (target/'.deepseek-retired').touch()
    original = (target/'model.json').read_bytes()
    result = run(tmp_path, 'scan_once; cleanup_unlinked_target "$WATCHER_MODEL_DIR/folderA/1" || true')
    assert result.returncode == 0, result.stderr
    assert target.is_dir()
    assert (target/'model.json').read_bytes() == original
    assert not list((tmp_path/'models-active').iterdir())
