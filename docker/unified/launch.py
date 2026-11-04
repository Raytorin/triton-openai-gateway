# SPDX-License-Identifier: Apache-2.0
"""Select the common Python/CUDA runtime before starting a fresh process.

This launcher does not import models or enforce their task/checkpoint policies.
The DeepSeek aliases preserve its separate bootstrap/admission contract.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import sys


ENV_ROOT = Path('/opt/envs/vllm032-cu129')
CUDA_ROOT = Path('/opt/cuda/12.9')
CACHE_NAMES = {
    'XDG_CACHE_HOME': 'xdg', 'VLLM_CACHE_ROOT': 'vllm', 'TORCH_HOME': 'torch',
    'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'TORCH_EXTENSIONS_DIR': 'torch-extensions',
    'TRITON_CACHE_DIR': 'triton', 'TRITON_DUMP_DIR': 'triton-dump',
    'FLASHINFER_WORKSPACE_BASE': 'flashinfer',
    'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR': 'flashinfer-autotune',
    'DG_JIT_CACHE_DIR': 'deepgemm', 'CUDA_CACHE_PATH': 'cuda',
    'NUMBA_CACHE_DIR': 'numba', 'TVM_FFI_CACHE_DIR': 'tvm-ffi',
    'CUTE_DSL_CACHE_DIR': 'cute-dsl', 'HF_MODULES_CACHE': 'hf-modules',
    'PYTHONPYCACHEPREFIX': 'python-pycache',
}
FOREIGN_SETTINGS = {
    'BASH_ENV', 'ENV', 'PYTHONHOME', 'PYTHONUSERBASE', 'PYTHONPATH',
    'PYTHONPLATLIBDIR', 'PYTHONEXECUTABLE', '__PYVENV_LAUNCHER__',
    '_PYTHON_SYSCONFIGDATA_NAME', 'LD_PRELOAD', 'LD_AUDIT',
    'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'OBJC_INCLUDE_PATH',
    'LIBRARY_PATH', 'TRITON_OVERRIDE_DIR', 'TRITON_LIBCUDA_PATH',
    'FLASHINFER_CUBIN_DIR', 'FLASHINFER_DISABLE_VERSION_CHECK',
    'FLASHINFER_CUBIN_CHECKSUM_DISABLED', 'FLASHINFER_CUBINS_REPOSITORY',
    'PKG_CONFIG_PATH', 'CMAKE_PREFIX_PATH', 'CMAKE_LIBRARY_PATH', 'CMAKE_INCLUDE_PATH',
    'LD_RUN_PATH', 'LD_ORIGIN_PATH', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS',
    'CUDAToolkit_ROOT', 'CUDA_BIN_PATH', 'VLLM_MODEL_REDIRECT_PATH',
    'GCC_EXEC_PREFIX', 'COMPILER_PATH',
}


def _within(path, root):
    return Path(path).resolve().is_relative_to(root)


def native_library_paths(root, cuda):
    site = root / 'lib/python3.12/site-packages'
    paths = [root/'lib', cuda/'lib64', site/'torch/lib']
    paths += sorted((site/'nvidia').glob('*/lib'))
    for path in paths:
        if not (_within(path, root) or _within(path, cuda)):
            raise RuntimeError(f'Native library directory resolves outside runtime: {path}')
    return paths + [Path('/usr/local/nvidia/lib'), Path('/usr/local/nvidia/lib64')]


def prepare_environment(root=ENV_ROOT, cuda=CUDA_ROOT, *, stub=False):
    """Build a process-local environment without changing the caller's state."""
    root, cuda = Path(root).resolve(), Path(cuda).resolve()
    environment = dict(os.environ)
    for key in list(environment):
        if (key.startswith('PIP_') or (key.startswith('TRITON_') and key.endswith('_PATH'))
                or key in FOREIGN_SETTINGS):
            environment.pop(key, None)
    environment.update(
        VIRTUAL_ENV=str(root), PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
        PYTHONPATH='', PIP_CONFIG_FILE=os.devnull, FLASHINFER_NO_DOWNLOAD='1',
        CUDA_HOME=str(cuda), CUDA_PATH=str(cuda), CUDACXX=str(cuda/'bin/nvcc'),
        NVCC=str(cuda/'bin/nvcc'),
        PATH=os.pathsep.join((str(root/'bin'), str(cuda/'bin'), '/opt/tritonserver/bin',
                            '/opt/ffmpeg-safe/bin', '/usr/local/sbin', '/usr/local/bin',
                            '/usr/sbin', '/usr/bin', '/sbin', '/bin')),
        CPATH=str(cuda/'include'), LIBRARY_PATH=str(cuda/'lib64'),
        VLLM_NCCL_SO_PATH=str(root/'lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2'),
        VLLM_WORKER_MULTIPROC_METHOD='spawn',
    )
    for key, executable in {'TRITON_PTXAS_PATH': 'ptxas', 'TRITON_CUOBJDUMP_PATH': 'cuobjdump',
                            'TRITON_NVDISASM_PATH': 'nvdisasm'}.items():
        environment[key] = str(cuda/'bin'/executable)
    for key in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH', 'TRITON_CUPTI_PATH'):
        environment[key] = str(cuda/'include')
    canonical = os.pathsep.join(map(str, native_library_paths(root, cuda)))
    # Triton's stock stub removes the first LD_LIBRARY_PATH element itself.
    environment['LD_LIBRARY_PATH'] = (str(root/'lib')+os.pathsep if stub else '') + canonical
    environment['DEEPSEEK_PREEXEC_NATIVE_PATH'] = hashlib.sha256(canonical.encode()).hexdigest()
    if stub:
        # Embedded CPython does not infer a venv from the C++ stub's location.
        environment['PYTHONHOME'] = str(root)
        environment['PYTHONPATH'] = str(Path(__file__).resolve().parent)
        environment['TRITON_RUNTIME_PYTHON'] = str(root/'bin/python3.12')
    else:
        environment.pop('TRITON_RUNTIME_PYTHON', None)
    arch = environment.get('TRITON_RUNTIME_CACHE_ARCH', environment.get('DEEPSEEK_CACHE_ARCH', 'unqualified'))
    if not re.fullmatch(r'(?:sm[0-9]{2,3}[af]?|cpu|unqualified)', arch):
        raise RuntimeError(f'Invalid runtime cache architecture: {arch}')
    cache_base = Path(environment.get('TRITON_RUNTIME_CACHE_DIR', environment.get(
        'DEEPSEEK_CACHE_DIR', '/tmp/triton-runtime-cache/vllm032-cu129'))).resolve()
    if _within(cache_base, root) or _within(cache_base, cuda):
        raise RuntimeError('Mutable caches must be outside immutable environment/toolkit')
    namespace = cache_base / arch
    if not namespace.resolve().is_relative_to(cache_base):
        raise RuntimeError(f'Cache namespace resolves outside cache root: {namespace}')
    for key, name in CACHE_NAMES.items():
        directory = namespace / name
        if not directory.resolve().is_relative_to(namespace.resolve()):
            raise RuntimeError(f'Cache points outside architecture namespace: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.W_OK):
            raise RuntimeError(f'Cache is not writable: {directory}')
        environment[key] = str(directory)
    for prefix in ('TRITON_RUNTIME', 'DEEPSEEK'):
        environment[f'{prefix}_CACHE_ARCH'] = arch
        environment[f'{prefix}_CACHE_DIR'] = str(cache_base)
    return environment


def execute(command, root=ENV_ROOT, cuda=CUDA_ROOT, *, stub=False):
    if not command or not Path(command[0]).is_absolute():
        raise ValueError('An absolute executable path is required')
    os.execve(command[0], list(command), prepare_environment(root, cuda, stub=stub))


def stock_stub():
    native = Path(__file__).resolve().parent.parent / 'backends/python/triton_python_backend_stub.native'
    if native.is_symlink():
        raise RuntimeError(f'Native stub must be a preserved ELF, not a symlink: {native}')
    with native.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise RuntimeError(f'Native stub is not an ELF executable: {native}')
    execute([str(native), *sys.argv[1:]], stub=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        usage='%(prog)s [-h] [--root ROOT] [--cuda CUDA] --exec EXECUTABLE [ARG ...]')
    parser.add_argument('--root', default=ENV_ROOT, type=Path)
    parser.add_argument('--cuda', default=CUDA_ROOT, type=Path)
    arguments = sys.argv[1:]
    try:
        separator = arguments.index('--exec')
    except ValueError:
        # Parse help and invalid controls normally, before reporting missing exec.
        parser.parse_args(arguments)
        parser.error('--exec requires an absolute executable path')
    args = parser.parse_args(arguments[:separator])
    # argparse treats a standalone -- specially, even with nargs=REMAINDER.
    # The first --exec is our boundary; every later token belongs to the child.
    command = arguments[separator+1:]
    if not command or not Path(command[0]).is_absolute():
        parser.error('--exec requires an absolute executable path')
    execute(command, args.root, args.cuda)


if __name__ == '__main__':
    main()
