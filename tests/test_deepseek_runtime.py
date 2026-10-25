"""CPU contracts: fail closed before loading a foreign engine or child process."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / 'backends/vllm_deepseek/runtime.py'
PY312 = os.environ.get('DEEPSEEK_TEST_PYTHON')


def load_runtime():
    assert BOOTSTRAP.exists(), 'isolated runtime bootstrap is not implemented'
    spec = importlib.util.spec_from_file_location('deepseek_runtime_test', BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrong_python_fails_before_runtime_import(monkeypatch):
    runtime = load_runtime()
    monkeypatch.setattr(runtime.sys, 'version_info', (3, 11))
    with pytest.raises(RuntimeError, match='Python 3.12'):
        runtime.bootstrap('/nonexistent', '/nonexistent')


@pytest.fixture
def runtime_process(tmp_path):
    if not PY312 or not Path(PY312).exists():
        pytest.skip('set DEEPSEEK_TEST_PYTHON to an isolated standalone Python 3.12')
    python = Path(PY312)
    env_root = python.parent.parent
    cuda = tmp_path / 'cuda'
    (cuda / 'bin').mkdir(parents=True)
    (cuda / 'bin/nvcc').write_text('#!/bin/sh\necho release 12.9\n')
    (cuda / 'bin/nvcc').chmod(0o755)
    def run(body, **environment):
        source = f'''import importlib.util, sys, os, json
spec = importlib.util.spec_from_file_location("runtime", {str(BOOTSTRAP)!r})
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
root = {str(env_root)!r}
cuda = {str(cuda)!r}
''' + body
        return subprocess.run([str(python), '-I', str(BOOTSTRAP.parent/'launch.py'),
                               '--root',str(env_root),'--cuda',str(cuda),
                               '--exec',str(python), '-c', source], text=True, capture_output=True,
                              env={**os.environ, 'VIRTUAL_ENV': str(env_root),
                                   'DEEPSEEK_CACHE_DIR':str(tmp_path/'cache'), 'DEEPSEEK_CACHE_ARCH':'cpu',
                                   **environment}, timeout=15)
    return run


def test_missing_environment_is_rejected(runtime_process):
    result = runtime_process("os.environ['VIRTUAL_ENV']='/wrong-env'; runtime.bootstrap(root, cuda)")
    assert result.returncode != 0
    assert 'VIRTUAL_ENV' in result.stderr


def test_bootstrap_removes_base_paths_and_sets_child_interpreter(runtime_process):
    result = runtime_process('''
sys.path.insert(0, '/usr/local/lib/python3.12/site-packages')
runtime.bootstrap(root, cuda)
from multiprocessing.spawn import get_executable
print(json.dumps({'paths':sys.path, 'child':os.fsdecode(get_executable()),
 'pythonpath':os.environ['PYTHONPATH'], 'cuda':os.environ['CUDA_HOME'],
 'ld':os.environ['LD_LIBRARY_PATH'], 'preload':os.environ.get('LD_PRELOAD')}))
''', PYTHONPATH='/base/site-packages', CUDA_HOME='/usr/local/cuda-13.4',
        LD_LIBRARY_PATH='/usr/local/cuda-13.4/lib64:/base/torch/lib', LD_PRELOAD='')
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert all('/usr/local/' not in path and '/base/' not in path for path in data['paths'])
    assert data['child'].endswith('/bin/python3.12') or data['child'].endswith('/bin/python')
    assert '/base/' not in data['pythonpath']
    assert data['cuda'].endswith('/cuda')
    assert '13.4' not in data['ld']
    assert data['preload'] is None


def test_runtime_rejects_preimported_foreign_torch(runtime_process):
    result = runtime_process('''
import types
foreign = types.ModuleType('torch'); foreign.__file__='/base/torch/__init__.py'
sys.modules['torch']=foreign
runtime.bootstrap(root, cuda)
''')
    assert result.returncode != 0
    assert 'torch' in result.stderr and 'outside' in result.stderr


def test_modules_must_resolve_inside_execution_environment(runtime_process, tmp_path):
    foreign = tmp_path / 'foreign'
    (foreign / 'torch').mkdir(parents=True)
    (foreign / 'torch/__init__.py').write_text('raise AssertionError("must not import")')
    result = runtime_process(f'''
sys.path.insert(0, {str(foreign)!r})
runtime.validate_module_origins(root, ('torch',))
''')
    assert result.returncode != 0
    assert 'outside' in result.stderr
    assert 'must not import' not in result.stderr


@pytest.mark.parametrize('modern', [False, True])
def test_metrics_loads_with_both_supported_bucket_apis(tmp_path, modern):
    # Missing relocated helper used to prevent model.py import even with metrics disabled.
    source = '''import importlib.util, sys, types
for name in ['triton_python_backend_utils','vllm','vllm.config','vllm.v1','vllm.v1.metrics','vllm.v1.metrics.loggers','vllm.v1.metrics.stats']:
    m=types.ModuleType(name); m.__path__=[]; sys.modules[name]=m
sys.modules['vllm.config'].VllmConfig=object
sys.modules['vllm.v1.metrics.loggers'].StatLoggerBase=object
for name in ['IterationStats','MultiModalCacheStats','SchedulerStats']:
    setattr(sys.modules['vllm.v1.metrics.stats'],name,object)
'''
    if modern:
        source += '''m=types.ModuleType('vllm.v1.metrics.buckets')
def buckets(family, maximum):
    assert family=='request_tokens' and maximum==10
    return [1,2,5,10]
m.histogram_buckets=buckets
sys.modules[m.__name__]=m
'''
    else:
        source += "sys.modules['vllm.v1.metrics.loggers'].build_1_2_5_buckets=lambda maximum:[1,2,5,10]\n"
    source += f'''spec=importlib.util.spec_from_file_location('metrics', {str(ROOT / 'backends/vllm_multimodal/utils/metrics.py')!r})
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
assert m.build_1_2_5_buckets(10)==[1,2,5,10]
'''
    result = subprocess.run([sys.executable, '-c', source], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_spawned_worker_uses_environment_and_clean_paths(runtime_process, tmp_path):
    (tmp_path / 'worker_probe.py').write_text('''import sys, os
from pathlib import Path
def report(queue):
    queue.put({'python':sys.executable, 'prefix':sys.prefix, 'paths':sys.path,
               'cuda':os.environ.get('CUDA_HOME'),
               'selected':{key:os.environ.get(key) for key in ('TRITON_PTXAS_PATH',
                   'TRITON_CUOBJDUMP_PATH','TRITON_NVDISASM_PATH','TRITON_CUDACRT_PATH',
                   'TRITON_CUDART_PATH','TRITON_CUPTI_PATH','CPATH','LIBRARY_PATH',
                   'VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
                   'FLASHINFER_WORKSPACE_BASE','DG_JIT_CACHE_DIR')}})
''')
    result = runtime_process(f'''
runtime.bootstrap(root, cuda, trusted_paths=({str(tmp_path)!r},))
import multiprocessing
from worker_probe import report
context = multiprocessing.get_context('spawn')
queue = context.Queue()
worker = context.Process(target=report, args=(queue,)); worker.start()
value = queue.get(timeout=8); worker.join(timeout=8)
assert worker.exitcode == 0
assert value['prefix'] == root and value['python'].startswith(root + '/bin/')
assert not any('/base/' in path for path in value['paths'])
assert value['cuda'] == cuda
assert all('/qwen/' not in str(path) and '/usr/local/cuda' not in str(path)
           for path in value['selected'].values())
for key in ('TRITON_PTXAS_PATH','TRITON_CUOBJDUMP_PATH','TRITON_NVDISASM_PATH'):
    assert value['selected'][key].startswith(cuda + '/bin/')
for key in ('TRITON_CUDACRT_PATH','TRITON_CUDART_PATH','TRITON_CUPTI_PATH','CPATH'):
    assert value['selected'][key] == cuda + '/include'
for key in ('VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR',
            'FLASHINFER_WORKSPACE_BASE','DG_JIT_CACHE_DIR'):
    assert '/cache/cpu/' in value['selected'][key]
print('worker-ok')
''', PYTHONPATH='/base/site-packages',
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
        TRITON_CUOBJDUMP_PATH='/usr/local/cuda/bin/cuobjdump',
        TRITON_NVDISASM_PATH='/usr/local/cuda/bin/nvdisasm',
        TRITON_CUDACRT_PATH='/usr/local/cuda/include',TRITON_CUDART_PATH='/usr/local/cuda/include',
        TRITON_CUPTI_PATH='/usr/local/cuda/include',CPATH='/usr/local/cuda/include:',
        LIBRARY_PATH='/usr/local/cuda/lib64/stubs:',VLLM_CACHE_ROOT='/qwen/cache',
        TORCHINDUCTOR_CACHE_DIR='/qwen/inductor',TRITON_CACHE_DIR='/qwen/triton',
        FLASHINFER_WORKSPACE_BASE='/qwen/flashinfer',DG_JIT_CACHE_DIR='/qwen/deepgemm')
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'worker-ok'


def load_policy(monkeypatch):
    monkeypatch.setenv('DEEPSEEK_CACHE_ARCH','sm90')
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, 'triton_python_backend_utils',
                        SimpleNamespace(get_input_tensor_by_name=lambda request, name: request.get(name)))
    path = ROOT / 'backends/vllm_deepseek/policy.py'
    assert path.exists(), 'generation-only runtime policy is missing'
    spec = importlib.util.spec_from_file_location('deepseek_policy_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.GenerationOnlyModel


@pytest.mark.parametrize('name', ['embedding_request', 'image', 'video', 'audio', 'pdf', 'media_parameters'])
def test_generation_only_backend_rejects_unsupported_input_before_engine(monkeypatch, name):
    policy = load_policy(monkeypatch)
    with pytest.raises(ValueError, match='generation-only'):
        policy()._validate_request_task_name({name: object()})
    assert policy()._validate_request_task_name({'text_input': object()}) == 'generate'


def test_failed_initialization_cleans_engine_and_metrics_and_preserves_error(monkeypatch):
    import threading
    policy = load_policy(monkeypatch)
    released = threading.Event()
    metrics_finalized = []
    class MetricFactory:
        def finalize(self):
            metrics_finalized.append(True)
    class FailingEngine:
        def initialize(self, args):
            self._vllm_metrics = [MetricFactory()]
            self._event_thread = threading.Thread(target=released.wait)
            self._event_thread.start()
            self._event_loop = type('Loop', (), {'is_running':lambda _:True,
                'call_soon_threadsafe':lambda _, fn:fn()})()
            self._llm_engine_shutdown_event = released
            raise ValueError('engine startup failed')
    class Model(policy, FailingEngine):
        def _preflight_model(self, args): pass
    model = Model()
    with pytest.raises(ValueError, match='engine startup failed'):
        model.initialize({})
    assert released.is_set()
    assert model._event_thread is None
    assert metrics_finalized == [True]
    model.finalize()  # Triton can finalize an instance whose initialize raised.
    assert metrics_finalized == [True]


def test_preflight_failure_never_starts_engine(monkeypatch):
    policy = load_policy(monkeypatch)
    started = []
    class Engine:
        def initialize(self, args):
            started.append(args)
    class Model(policy, Engine):
        def _preflight_model(self, args):
            raise ValueError('incomplete checkpoint')
    with pytest.raises(ValueError, match='incomplete checkpoint'):
        Model().initialize({})
    assert not started


@pytest.mark.parametrize('library', ['libtorch_cuda.so','libc10_cuda.so','libnvrtc.so.12',
    'libcudart.so.12','libcublas.so.12','libcudnn.so.9','libnccl.so.2','libnvshmem_host.so.3',
    'libpython3.12.so.1.0'])
def test_foreign_engine_library_mapping_is_rejected(library, tmp_path):
    runtime=load_runtime()
    mapping=f'7f000000-7f001000 r-xp 00000000 00:01 1 /legacy/{library}\n'
    with pytest.raises(RuntimeError, match='outside derived roots'):
        runtime.validate_native_maps(tmp_path/'env',tmp_path/'cuda',mapping)


def test_native_guard_allows_derived_engine_and_system_driver_only(tmp_path):
    runtime=load_runtime()
    env=tmp_path/'env'; cuda=tmp_path/'cuda'
    paths=[str(env/'lib/libtorch_cuda.so'),str(cuda/'lib64/libnvrtc.so.12'),
           '/usr/lib/x86_64-linux-gnu/libcuda.so.615', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6']
    maps='\n'.join(f'7f000000-7f001000 r-xp 00000000 00:01 1 {path}' for path in paths)
    assert set(runtime.validate_native_maps(env,cuda,maps))==set(paths)
    with pytest.raises(RuntimeError,match='driver mapping'):
        runtime.validate_native_maps(env,cuda,'7f00-7f10 r-xp 00000000 00:01 1 /usr/local/cuda/lib64/stubs/libcuda.so.1')
    with pytest.raises(RuntimeError,match='outside derived roots'):
        runtime.validate_native_maps(env,cuda,'7f00-7f10 r-xp 00000000 00:01 1 /legacy/libcudnn.so.9 (deleted)')


@pytest.mark.parametrize('arch',['cpu','unqualified'])
def test_gpu_engine_never_starts_with_unqualified_cache_architecture(monkeypatch,arch):
    policy=load_policy(monkeypatch)
    monkeypatch.setenv('DEEPSEEK_CACHE_ARCH',arch)
    started=[]
    class Engine:
        def initialize(self,args):
            started.append(args); self.supported_tasks=('generate',)
    class Model(policy,Engine):
        def _preflight_model(self,args): pass
    with pytest.raises(RuntimeError,match='DEEPSEEK_CACHE_ARCH'):
        Model().initialize({})
    assert started==[]


def test_only_exact_stock_stub_can_retain_hash_verified_core_cuda13(tmp_path):
    import hashlib
    runtime=load_runtime()
    stub=tmp_path/'stock-stub';stub.write_bytes(b'pinned-stock-stub')
    library=tmp_path/'libcudart.so.13.4.49';library.write_bytes(b'pinned-core-runtime')
    policy={'stub':{'path':str(stub),'sha256':hashlib.sha256(stub.read_bytes()).hexdigest()},
            'libraries':[{'path':str(library),'soname':'libcudart.so.13',
                          'sha256':hashlib.sha256(library.read_bytes()).hexdigest()}]}
    assert runtime.is_core_abi_exception(str(library),policy,str(stub))
    assert not runtime.is_core_abi_exception(str(library),policy,sys.executable)
    library.write_bytes(b'foreign-runtime')
    assert not runtime.is_core_abi_exception(str(library),policy,str(stub))


def docker_bootstrap_commands():
    """Execute the product artifact's shell/RUN rather than assert its source text."""
    source=(ROOT/'Dockerfile.triton-gateway-deepseek').read_text()
    shell=json.loads(next(line[6:] for line in source.splitlines() if line.startswith('SHELL ')))
    command=source.split('\nRUN ',1)[1].split('\n\n',1)[0].replace('\\\n','')
    return shell,command


@pytest.mark.parametrize('marker_present',[False,True])
def test_owned_python_archive_marker_is_optional(tmp_path,marker_present):
    import tarfile
    extracted=tmp_path/'python/lib/python3.12';extracted.mkdir(parents=True)
    (tmp_path/'python/bin').mkdir()
    (extracted/'fixture-stdlib.py').write_text('fixture')
    if marker_present: (extracted/'EXTERNALLY-MANAGED').write_text('owned marker')
    archive=tmp_path/'python.tar.gz'
    with tarfile.open(archive,'w:gz') as bundle:
        bundle.add(tmp_path/'python',arcname='python')
    target=tmp_path/'environment'
    shell,command=docker_bootstrap_commands()
    command=command.replace('/opt/envs/vllm032-cu129',str(target)).replace('/tmp/deepseek-python.tar.gz',str(archive))
    result=subprocess.run([*shell,command],capture_output=True,text=True,
                          env={key:value for key,value in os.environ.items() if key not in ('BASH_ENV','ENV')})
    assert result.returncode == 0,result.stderr
    assert (target/'lib/python3.12/fixture-stdlib.py').exists()
    assert (target/'bin/activate').is_file()
    assert not (target/'lib/python3.12/EXTERNALLY-MANAGED').exists()


def test_builder_shell_does_not_source_inherited_nvidia_rc(tmp_path):
    marker=tmp_path/'sourced'
    rc=tmp_path/'legacy-rc';rc.write_text(f'printf sourced > {marker}\nprintf "%s" "$PS1"\n')
    shell,_=docker_bootstrap_commands()
    result=subprocess.run([*shell,'printf runtime-ready'],capture_output=True,text=True,
                          env={**os.environ,'BASH_ENV':str(rc),'ENV':str(rc)})
    assert result.returncode == 0,result.stderr
    assert result.stdout == 'runtime-ready'
    assert not marker.exists()
    assert result.stderr == ''


def test_runtime_jit_shell_does_not_source_inherited_nvidia_rc(tmp_path,monkeypatch):
    marker=tmp_path/'sourced'
    rc=tmp_path/'legacy-rc';rc.write_text(f'printf sourced > {marker}\nprintf "%s" "$PS1"\n')
    monkeypatch.setenv('BASH_ENV',str(rc));monkeypatch.setenv('ENV',str(rc))
    monkeypatch.setenv('DEEPSEEK_CACHE_DIR',str(tmp_path/'cache'))
    environment=load_runtime().prepare_environment(tmp_path/'env',tmp_path/'cuda')
    result=subprocess.run(['/bin/bash','-euo','pipefail','-c','printf runtime-ready'],
                          env=environment,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    assert result.stdout == 'runtime-ready'
    assert not marker.exists()
    assert 'BASH_ENV' not in environment and 'ENV' not in environment
