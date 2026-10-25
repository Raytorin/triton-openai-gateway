"""The custom stub must select the native loader before exec, not after import."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = ROOT / 'backends/vllm_deepseek/launch.py'


@pytest.fixture
def launch_env(tmp_path):
    env = tmp_path / 'env'; cuda = tmp_path / 'cuda'
    (env / 'bin').mkdir(parents=True); (env / 'lib').mkdir()
    (env / 'bin/python3.12').symlink_to(sys.executable)
    (cuda / 'bin').mkdir(parents=True)
    for name in ('nvcc','ptxas','cuobjdump','nvdisasm'):
        (cuda / 'bin' / name).write_text('#!/bin/sh\necho fixture\n')
        (cuda / 'bin' / name).chmod(0o755)
    def run(*args, inherited=None):
        assert LAUNCH.exists(), 'pre-exec launcher has not been implemented'
        return subprocess.run([sys.executable, '-I', str(LAUNCH), '--root', str(env),
                               '--cuda', str(cuda), '--exec', *map(str,args)],
            env={**os.environ, 'DEEPSEEK_CACHE_DIR':str(tmp_path / 'cache'),
                 'DEEPSEEK_CACHE_ARCH':'sm90', **(inherited or {})},
            text=True, capture_output=True, timeout=30)
    return env, cuda, run


def test_pip_install_ignores_inherited_base_constraints_and_configuration(launch_env, tmp_path):
    _, _, run = launch_env
    wheel = tmp_path / 'deepseek_isolation_proof-1.0-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        prefix='deepseek_isolation_proof-1.0.dist-info/'
        archive.writestr(prefix+'METADATA', 'Metadata-Version: 2.1\nName: deepseek-isolation-proof\nVersion: 1.0\n')
        archive.writestr(prefix+'WHEEL', 'Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr(prefix+'RECORD','')
    constraint=tmp_path/'constraints.txt';constraint.write_text('deepseek-isolation-proof<1\n')
    config=tmp_path/'pip.conf';config.write_text('[global]\nconstraint='+str(constraint)+'\n')
    requirement=tmp_path/'requirements.txt'
    requirement.write_text('deepseek-isolation-proof==1.0 --hash=sha256:'+hashlib.sha256(wheel.read_bytes()).hexdigest()+'\n')
    destination=tmp_path/'installed'
    result=run(sys.executable,'-m','pip','install','--no-index','--find-links',tmp_path,
               '--target',destination,'--require-hashes','-r',requirement,
               inherited={'PIP_CONSTRAINT':str(constraint),'PIP_CONFIG_FILE':str(config),
                          'PIP_REQUIREMENT':str(tmp_path/'must-not-read.txt')})
    assert result.returncode == 0, result.stdout+result.stderr
    assert (destination/'deepseek_isolation_proof-1.0.dist-info/METADATA').exists()


def test_loader_uses_isolated_library_after_exec(launch_env, tmp_path):
    compiler=shutil.which('cc')
    if compiler is None: pytest.skip('C compiler required for native loader sentinel')
    env, _, run=launch_env
    legacy=tmp_path/'legacy';legacy.mkdir()
    for directory, value in ((legacy,29),(env/'lib',32)):
        source=directory/'marker.c';source.write_text(f'int deepseek_marker(void) {{ return {value}; }}')
        built=subprocess.run([compiler,'-shared','-fPIC',str(source),'-o',str(directory/'libdeepseek_loader_marker.so')],capture_output=True,text=True)
        assert built.returncode == 0, built.stderr
    body='import ctypes; print(ctypes.CDLL("libdeepseek_loader_marker.so").deepseek_marker())'
    baseline=subprocess.run([sys.executable,'-c',body],env={**os.environ,'LD_LIBRARY_PATH':str(legacy)},capture_output=True,text=True)
    assert baseline.returncode == 0 and baseline.stdout.strip()=='29', baseline.stderr
    result=run(sys.executable,'-c',body,inherited={'LD_LIBRARY_PATH':str(legacy)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '32'


def test_launcher_replaces_compiler_headers_and_all_mutable_cache_namespaces(launch_env, tmp_path):
    env,cuda,run=launch_env
    foreign={
        'TRITON_PTXAS_PATH':'/usr/local/cuda/bin/ptxas',
        'TRITON_CUOBJDUMP_PATH':'/usr/local/cuda/bin/cuobjdump',
        'TRITON_NVDISASM_PATH':'/usr/local/cuda/bin/nvdisasm',
        'TRITON_CUDACRT_PATH':'/usr/local/cuda/include',
        'TRITON_CUDART_PATH':'/usr/local/cuda/include',
        'TRITON_CUPTI_PATH':'/usr/local/cuda/include',
        'CPATH':'/usr/local/cuda/include:', 'LIBRARY_PATH':'/usr/local/cuda/lib64/stubs:',
        'CPLUS_INCLUDE_PATH':'/qwen/include', 'C_INCLUDE_PATH':'/qwen/include',
        'VLLM_CACHE_ROOT':'/qwen/cache', 'TORCHINDUCTOR_CACHE_DIR':'/qwen/inductor',
        'TRITON_CACHE_DIR':'/qwen/triton', 'FLASHINFER_WORKSPACE_BASE':'/qwen/flashinfer',
        'DG_JIT_CACHE_DIR':'/qwen/deepgemm', 'TORCH_EXTENSIONS_DIR':'/qwen/extensions',
        'CUDA_CACHE_PATH':'/qwen/cuda', 'XDG_CACHE_HOME':'/qwen/xdg'}
    result=run(sys.executable,'-c','import json,os; print(json.dumps(dict(os.environ)))', inherited=foreign)
    assert result.returncode == 0, result.stderr
    selected=json.loads(result.stdout)
    for key in ('TRITON_PTXAS_PATH','TRITON_CUOBJDUMP_PATH','TRITON_NVDISASM_PATH'):
        assert selected[key].startswith(str(cuda/'bin'))
    for key in ('TRITON_CUDACRT_PATH','TRITON_CUDART_PATH','TRITON_CUPTI_PATH','CPATH'):
        assert selected[key] == str(cuda/'include')
    assert selected['LIBRARY_PATH'] == str(cuda/'lib64')
    for key in ('CPLUS_INCLUDE_PATH','C_INCLUDE_PATH'):
        assert '/qwen/' not in selected.get(key,'')
    for key in ('VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                'FLASHINFER_WORKSPACE_BASE','DG_JIT_CACHE_DIR','TORCH_EXTENSIONS_DIR',
                'CUDA_CACHE_PATH','XDG_CACHE_HOME'):
        path=Path(selected[key])
        assert path.is_relative_to(tmp_path/'cache/sm90') and path.is_dir()
    assert selected['PIP_CONFIG_FILE'] == os.devnull


def test_launcher_preserves_packaged_flashinfer_lookup(launch_env):
    _,_,run=launch_env
    result=run(sys.executable,'-c','import json,os; print(json.dumps(dict(os.environ)))',
               inherited={'FLASHINFER_CUBIN_DIR':'/qwen/cubins','FLASHINFER_DISABLE_VERSION_CHECK':'1'})
    assert result.returncode == 0, result.stderr
    environment=json.loads(result.stdout)
    assert 'FLASHINFER_CUBIN_DIR' not in environment
    assert 'FLASHINFER_DISABLE_VERSION_CHECK' not in environment


def test_launcher_prevents_post_admission_model_redirection(launch_env):
    _, _, run = launch_env
    result = run(sys.executable, '-c',
                 'import os; assert "VLLM_MODEL_REDIRECT_PATH" not in os.environ',
                 inherited={'VLLM_MODEL_REDIRECT_PATH':'/qwen/model-redirect.json'})
    assert result.returncode == 0, result.stderr


def test_launcher_namespaces_flashinfer_autotune_cache(launch_env, tmp_path):
    _, _, run = launch_env
    result = run(sys.executable, '-c',
                 'import os; from pathlib import Path; '
                 'p=Path(os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"]); '
                 'print(p); assert p.is_dir(); '
                 '(p/"autotune_configs.json").write_text("{}")',
                 inherited={'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'/qwen/shared-tuning'})
    assert result.returncode == 0, result.stderr
    cache = Path(result.stdout.strip())
    assert cache.is_relative_to(tmp_path/'cache/sm90')
    assert (cache/'autotune_configs.json').read_text() == '{}'


def test_facade_rejects_foreign_native_mapping_introduced_by_shared_import(tmp_path):
    compiler=shutil.which('cc')
    if compiler is None: pytest.skip('C compiler required for native import regression')
    source=tmp_path/'foreign.c';source.write_text('int foreign_marker(void) { return 29; }')
    library=tmp_path/'libtorch_review.so'
    built=subprocess.run([compiler,'-shared','-fPIC',str(source),'-o',str(library)],capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    backend=tmp_path/'backends/vllm_deepseek'; backend.mkdir(parents=True)
    shared=backend.parent/'vllm_multimodal';shared.mkdir()
    shutil.copy(ROOT/'backends/vllm_deepseek/model.py',backend/'model.py')
    # Isolate the no-GPU preconditions; keep the real mapping guard and dlopen.
    (backend/'runtime.py').write_text(f'''import importlib.util
from pathlib import Path
import sys
spec=importlib.util.spec_from_file_location('real_runtime',{str(ROOT/'backends/vllm_deepseek/runtime.py')!r})
real=importlib.util.module_from_spec(spec);spec.loader.exec_module(real)
ENV_ROOT=Path(sys.base_prefix)
CUDA_ROOT=Path({str(tmp_path/'cuda')!r})
real.validate_native_maps(ENV_ROOT,CUDA_ROOT)
def bootstrap(**kwargs): return {{}}
def validate_module_origins(root): return {{}}
def validate_native_maps(): return real.validate_native_maps(ENV_ROOT,CUDA_ROOT)
''')
    (shared/'model.py').write_text(f'import ctypes\nhandle=ctypes.CDLL({str(library)!r})\nclass TritonPythonModel: pass\n')
    (backend/'policy.py').write_text('class GenerationOnlyModel: pass\n')
    body=f'''import importlib.util
spec=importlib.util.spec_from_file_location('facade',{str(backend/'model.py')!r})
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
'''
    result=subprocess.run([sys.executable,'-c',body],capture_output=True,text=True)
    assert result.returncode != 0
    assert 'Foreign native engine mapping' in result.stderr and 'libtorch_review.so' in result.stderr
