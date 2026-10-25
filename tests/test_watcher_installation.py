"""Run the installed watcher COPY layout without repository helper fallbacks."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import pytest
from test_deepseek_checkpoint import snapshot
from test_deepseek_watcher import checkout
ROOT=Path(__file__).resolve().parents[1]


def install_watcher(tmp_path,dockerfile):
    image=tmp_path/'image'
    # Materialize the actual COPY contract for the watcher and its common helper.
    # Unrelated image contents (including optional DeepSeek sources) stay absent.
    for line in (ROOT/dockerfile).read_text().splitlines():
        if not line.startswith('COPY '):continue
        words=shlex.split(line)
        if len(words)!=3 or words[0]!='COPY' or words[1] not in ('watch_triton.sh','gateway/model_config.py','gateway'):
            continue
        destination=image/words[2].lstrip('/')
        if words[2].endswith('/'):destination/=Path(words[1]).name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if (ROOT/words[1]).is_dir():
            shutil.copytree(ROOT/words[1], destination, dirs_exist_ok=True)
        else:
            shutil.copyfile(ROOT/words[1],destination)
    launcher=tmp_path/'bin';launcher.mkdir()
    # Reject helper execution outside this installation, even if an unrelated
    # /opt tree happens to exist on the test host. Inline stdlib data operations
    # still execute normally, and the watcher itself remains unchanged.
    python=launcher/'python3'
    python.write_text('#!/bin/sh\ncase "$1" in\n'+shlex.quote(str(image))+'/*|-) exec '+shlex.quote(sys.executable)+' "$@" ;;\n*) echo "helper outside installed image: $1" >&2; exit 99 ;;\nesac\n')
    python.chmod(0o755)
    return image/'usr/local/bin/watch_triton.sh',launcher


def scan(watcher,launcher,root,commands='scan_once'):
    active=root/'models-active';active.mkdir(exist_ok=True)
    return subprocess.run(['bash','-c','source "$1"; '+commands,'bash',str(watcher)],env={**os.environ,'PATH':str(launcher)+os.pathsep+os.environ['PATH'],'WATCHER_MODEL_DIR':str(root),'MODELS_ACTIVE_DIR':str(active)},text=True,capture_output=True,timeout=8)


@pytest.mark.parametrize('dockerfile',['Dockerfile.triton-gateway','Dockerfile.triton-gateway-deepseek'])
def test_installed_qwen_watcher_publishes_and_cleans_without_deepseek_runtime(tmp_path,snapshot,dockerfile):
    watcher,launcher=install_watcher(tmp_path,dockerfile)
    root=tmp_path/'downloads';target=checkout(root,snapshot,deepseek=False)
    result=scan(watcher,launcher,root)
    assert result.returncode==0,result.stderr
    assert (root/'models-active/selected').resolve()==target
    assert json.loads((target/'model.json').read_text())['model']==str(target)
    result=scan(watcher,launcher,root,'remember_existing_links; rm "$MODELS_ACTIVE_DIR/selected"; scan_once')
    assert result.returncode==0,result.stderr
    assert not target.exists()


@pytest.mark.parametrize('dockerfile',['Dockerfile.triton-gateway','Dockerfile.triton-gateway-deepseek'])
def test_installed_shared_watcher_fails_closed_without_deepseek_validator(tmp_path,snapshot,dockerfile):
    watcher,launcher=install_watcher(tmp_path,dockerfile)
    root=tmp_path/'downloads';target=checkout(root,snapshot)
    (snapshot/'a.safetensors').unlink()
    result=scan(watcher,launcher,root)
    assert result.returncode==0,result.stderr
    assert not (root/'models-active/selected').is_symlink()
    assert json.loads((target/'model.json').read_text())['model']==str(snapshot)
    assert 'checkpoint/config admission failed' in result.stderr
    assert 'backend extraction failed' not in result.stderr


@pytest.mark.parametrize('dockerfile',['Dockerfile.triton-gateway','Dockerfile.triton-gateway-deepseek'])
@pytest.mark.parametrize('config', [
    'backend: "vllm_multimodal" name: "selected"',
    "backend: 'vllm_multimodal' name: 'selected'",
    r"backend: 'vllm_multi' 'modal' name: '\163elected'",
])
def test_installed_structural_reader_matches_watcher_identity(tmp_path, snapshot, dockerfile, config):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2
    parsed = text_format.Parse(config, model_config_pb2.ModelConfig())
    watcher, launcher = install_watcher(tmp_path,dockerfile)
    root=tmp_path/'downloads';target=checkout(root,snapshot,deepseek=False)
    config_path=target.parent/'config.pbtxt';config_path.write_text(config)
    result=scan(watcher,launcher,root)
    assert result.returncode==0,result.stderr
    assert sorted(p.name for p in (root/'models-active').iterdir())==[parsed.name]
    image=tmp_path/'image'
    # Import only the installed package using a clean cwd and no source-tree
    # PYTHONPATH. The common reader requires no optional engine/dependencies.
    probe="import sys; sys.path.insert(0,sys.argv[1]); from gateway.model_config import read_model_config; print(read_model_config(sys.argv[2])['backend']); assert 'vllm' not in sys.modules"
    result=subprocess.run([sys.executable,'-I','-S','-c',probe,str(image/'opt/tritonserver'),str(config_path)],cwd=image,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()==parsed.backend
    probe="import sys; sys.path.insert(0,sys.argv[1]); from gateway.registry import _read_backend; import gateway.registry as registry; assert registry.__file__.startswith(sys.argv[1]); print(_read_backend(sys.argv[2],0)); assert 'vllm' not in sys.modules"
    result=subprocess.run([sys.executable,'-I','-c',probe,str(image/'opt/tritonserver'),str(config_path)],cwd=image,text=True,capture_output=True,timeout=5)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()==parsed.backend
