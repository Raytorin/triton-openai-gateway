# SPDX-License-Identifier: Apache-2.0
"""Stdlib-only, process-local bootstrap for Triton's embedded Python 3.12 stub.

Call before any Torch/vLLM imports, including from auto-completion. Changing
LD_LIBRARY_PATH cannot unload a library already mapped by the stub: reject a
foreign libpython and qualify native CUDA maps separately in the real stub.
"""
import hashlib
import importlib.util
import json
import multiprocessing
import os
import re
from pathlib import Path
import sys

ENV_ROOT = Path('/opt/envs/vllm032-cu129')
CUDA_ROOT = Path('/opt/cuda/12.9')
MODULES = ('torch', 'vllm', 'transformers', 'pydantic', 'flashinfer')


def _within(path, root):
    return Path(path).resolve().is_relative_to(Path(root).resolve())


def validate_module_origins(root, modules=MODULES):
    """Find distributions without importing them or running their CUDA probes."""
    origins = {}
    for name in modules:
        loaded = sys.modules.get(name)
        spec = None if loaded is not None else importlib.util.find_spec(name)
        origin = getattr(loaded, '__file__', None) if loaded is not None else getattr(spec, 'origin', None)
        if not origin or not _within(origin, root):
            raise RuntimeError(f'{name} resolves outside DeepSeek execution environment: {origin}')
        origins[name] = str(Path(origin).resolve())
    return origins


CACHE_NAMES = {
    'XDG_CACHE_HOME':'xdg', 'VLLM_CACHE_ROOT':'vllm', 'TORCH_HOME':'torch',
    'TORCHINDUCTOR_CACHE_DIR':'inductor', 'TORCH_EXTENSIONS_DIR':'torch-extensions',
    'TRITON_CACHE_DIR':'triton', 'TRITON_DUMP_DIR':'triton-dump',
    'FLASHINFER_WORKSPACE_BASE':'flashinfer',
    'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'flashinfer-autotune',
    'DG_JIT_CACHE_DIR':'deepgemm', 'CUDA_CACHE_PATH':'cuda',
    'NUMBA_CACHE_DIR':'numba', 'TVM_FFI_CACHE_DIR':'tvm-ffi',
    'CUTE_DSL_CACHE_DIR':'cute-dsl', 'HF_MODULES_CACHE':'hf-modules',
    'PYTHONPYCACHEPREFIX':'python-pycache',
}
ENGINE_LIBRARY_PREFIXES = ('libtorch', 'libc10', 'libnvrtc', 'libnvJitLink',
    'libcudart', 'libcublas', 'libcudnn', 'libnccl', 'libnvshmem', 'libcusparse',
    'libcusolver', 'libcurand', 'libcufft', 'libcupti', 'libcutensor', 'libnvptxcompiler')
DRIVER_LIBRARY_PREFIXES = ('libcuda.so', 'libnvidia-ml.so', 'libnvidia-ptxjitcompiler.so', 'libnvidia-nvvm.so')
DRIVER_DIRS = (Path('/usr/local/nvidia/lib'), Path('/usr/local/nvidia/lib64'),
               Path('/usr/lib/x86_64-linux-gnu'), Path('/lib/x86_64-linux-gnu'),
               Path('/usr/lib64'), Path('/lib64'), Path('/usr/lib'), Path('/lib'))


def native_library_paths(root, cuda):
    root, cuda = Path(root).resolve(), Path(cuda).resolve()
    site = root / 'lib/python3.12/site-packages'
    paths = [root / 'lib', cuda / 'lib64', site / 'torch/lib']
    paths += sorted((site / 'nvidia').glob('*/lib'))
    for path in paths:
        if not (_within(path, root) or _within(path, cuda)):
            raise RuntimeError(f'Native library directory resolves outside runtime: {path}')
    return paths + [Path('/usr/local/nvidia/lib'), Path('/usr/local/nvidia/lib64')]


def prepare_environment(root=ENV_ROOT, cuda=CUDA_ROOT, *, stub=False):
    """Return the environment for a fresh exec; never mutate the server/Qwen."""
    root, cuda = Path(root).resolve(), Path(cuda).resolve()
    environment = dict(os.environ)
    for key in list(environment):
        if (key.startswith('PIP_') or (key.startswith('TRITON_') and key.endswith('_PATH'))
                or key in {
                'BASH_ENV','ENV','PYTHONHOME','PYTHONUSERBASE','PYTHONPATH','LD_PRELOAD','LD_AUDIT',
                'CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH','OBJC_INCLUDE_PATH',
                'LIBRARY_PATH','TRITON_OVERRIDE_DIR','TRITON_LIBCUDA_PATH',
                'FLASHINFER_CUBIN_DIR','FLASHINFER_DISABLE_VERSION_CHECK',
                'FLASHINFER_CUBIN_CHECKSUM_DISABLED','FLASHINFER_CUBINS_REPOSITORY',
                'PKG_CONFIG_PATH','CMAKE_PREFIX_PATH','CMAKE_LIBRARY_PATH','CMAKE_INCLUDE_PATH',
                'LD_RUN_PATH','LD_ORIGIN_PATH','NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS',
                'CUDAToolkit_ROOT','CUDA_BIN_PATH','VLLM_MODEL_REDIRECT_PATH'}):
            environment.pop(key, None)
    environment.update(VIRTUAL_ENV=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='', PIP_CONFIG_FILE=os.devnull, FLASHINFER_NO_DOWNLOAD='1', CUDA_HOME=str(cuda), CUDA_PATH=str(cuda),
        CUDACXX=str(cuda/'bin/nvcc'), NVCC=str(cuda/'bin/nvcc'),
        PATH=os.pathsep.join((str(root/'bin'),str(cuda/'bin'),'/usr/bin','/bin')),
        CPATH=str(cuda/'include'), LIBRARY_PATH=str(cuda/'lib64'))
    for key, executable in {'TRITON_PTXAS_PATH':'ptxas', 'TRITON_CUOBJDUMP_PATH':'cuobjdump',
                            'TRITON_NVDISASM_PATH':'nvdisasm'}.items():
        environment[key] = str(cuda/'bin'/executable)
    for key in ('TRITON_CUDACRT_PATH','TRITON_CUDART_PATH','TRITON_CUPTI_PATH'):
        environment[key] = str(cuda/'include')
    canonical = os.pathsep.join(map(str,native_library_paths(root,cuda)))
    # The stock Triton stub erases the first environment library element.
    environment['LD_LIBRARY_PATH'] = (str(root/'lib')+os.pathsep if stub else '') + canonical
    environment['DEEPSEEK_PREEXEC_NATIVE_PATH'] = hashlib.sha256(canonical.encode()).hexdigest()
    environment['VLLM_NCCL_SO_PATH'] = str(root/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2')
    environment['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    arch = environment.get('DEEPSEEK_CACHE_ARCH','unqualified')
    if not re.fullmatch(r'(?:sm[0-9]{2,3}[af]?|cpu|unqualified)', arch):
        raise RuntimeError(f'Invalid DEEPSEEK_CACHE_ARCH: {arch}')
    cache_base = Path(environment.get('DEEPSEEK_CACHE_DIR','/tmp/triton-runtime-cache/vllm032-cu129')).resolve()
    if _within(cache_base, root) or _within(cache_base, cuda):
        raise RuntimeError('Mutable caches must be outside immutable environment/toolkit')
    namespace = cache_base / arch
    for key, name in CACHE_NAMES.items():
        directory = namespace / name
        directory.mkdir(parents=True,exist_ok=True)
        if not directory.resolve().is_relative_to(namespace.resolve()):
            raise RuntimeError(f'Cache points outside architecture namespace: {directory}')
        if not os.access(directory,os.W_OK):
            raise RuntimeError(f'Cache is not writable: {directory}')
        environment[key] = str(directory)
    environment['DEEPSEEK_CACHE_ARCH'] = arch
    environment['DEEPSEEK_CACHE_DIR'] = str(cache_base)
    return environment


def is_core_abi_exception(path, policy, executable):
    """Only the pinned C++ stub retains its exact CUDA13 DT_NEEDED library."""
    if not Path(path).name.startswith('libcudart.so.13'):
        return False
    stub = policy.get('stub',{})
    if str(Path(executable).resolve()) != stub.get('path'):
        return False
    entries = [entry for entry in policy.get('libraries',())
               if entry.get('path') == path and entry.get('soname') == 'libcudart.so.13']
    if not entries:
        return False
    try:
        with Path(executable).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest() != stub.get('sha256'):
                return False
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest() == entries[0].get('sha256')
    except OSError:
        return False


def validate_native_maps(root=ENV_ROOT, cuda=CUDA_ROOT, maps=None):
    """Check every mapped engine library, allowing only explicit driver exceptions."""
    if maps is None:
        maps = Path('/proc/self/maps').read_text()
    selected = set()
    policy_path = Path(__file__).with_name('core-native-libraries.json')
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    executable = str(Path('/proc/self/exe').resolve())
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        path = fields[5].removesuffix(' (deleted)')
        name = Path(path).name
        if name.startswith(DRIVER_LIBRARY_PREFIXES):
            if '/stubs/' in path or not any(_within(path,base) for base in DRIVER_DIRS):
                raise RuntimeError(f'Unexpected CUDA driver mapping: {path}')
        elif name.startswith(ENGINE_LIBRARY_PREFIXES) or name.startswith('libpython3.12'):
            if (not (_within(path,root) or _within(path,cuda))
                    and not is_core_abi_exception(path,policy,executable)):
                raise RuntimeError(f'Foreign native engine mapping outside derived roots: {path}')
        elif not name.startswith('libstdc++'):
            continue
        selected.add(path)
    return sorted(selected)


def bootstrap(root=ENV_ROOT, cuda=CUDA_ROOT, trusted_paths=()):
    """Validate pre-exec selection before importing any engine/native modules."""
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f'DeepSeek requires Python 3.12, got {sys.version}')
    root, cuda = Path(root).resolve(), Path(cuda).resolve()
    if Path(os.environ.get('VIRTUAL_ENV', '/missing')).resolve() != root:
        raise RuntimeError(f'VIRTUAL_ENV must select {root}')
    if Path(sys.prefix).resolve() != root or Path(sys.exec_prefix).resolve() != root:
        raise RuntimeError(f'Python prefix/exec_prefix must select {root}')
    python = root/'bin/python3.12'
    if not python.is_file() or not (cuda/'bin/nvcc').is_file():
        raise RuntimeError('DeepSeek environment Python and CUDA 12.9 compiler must exist')
    canonical = os.pathsep.join(map(str,native_library_paths(root,cuda)))
    if (os.environ.get('DEEPSEEK_PREEXEC_NATIVE_PATH') != hashlib.sha256(canonical.encode()).hexdigest()
            or os.environ.get('LD_LIBRARY_PATH') != canonical):
        raise RuntimeError('DeepSeek requires its pre-exec native launcher; late LD_LIBRARY_PATH changes are unsafe')
    native = validate_native_maps(root,cuda)
    libpython = [path for path in native if 'libpython3.12' in path]
    if (not (root/'lib/libpython3.12.so.1.0').is_file()
            or (not libpython and not _within(sys.executable,root))):
        raise RuntimeError(f'libpython3.12 must be loaded from {root}: {libpython}')
    for name in MODULES:
        if name in sys.modules:
            validate_module_origins(root,(name,))
    trusted = [str(Path(path).resolve()) for path in trusted_paths]
    sys.path[:] = list(dict.fromkeys([str(Path(p).resolve()) for p in sys.path
                                    if p and _within(p,root)] + trusted))
    environment = prepare_environment(root,cuda)
    os.environ.clear(); os.environ.update(environment)
    os.environ['PYTHONPATH'] = os.pathsep.join(trusted)
    sys.executable = str(python)
    multiprocessing.set_executable(str(python))
    return {'python':str(python),'prefix':str(root),'cuda':str(cuda),'libpython':libpython,
            'native_maps':native, 'cache_arch':os.environ['DEEPSEEK_CACHE_ARCH'],
            'caches':{key:os.environ[key] for key in CACHE_NAMES},
            'compiler_paths':{key:value for key,value in os.environ.items() if key.startswith('TRITON_') and key.endswith('_PATH')}}
