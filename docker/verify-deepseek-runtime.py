# SPDX-License-Identifier: Apache-2.0
"""No-weight CPU/native import + spawned-child checks for a materialized runtime."""
import argparse
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import re
import subprocess
import sys

REQUIRED = {'vllm':'0.32.0+cu129', 'torch':'2.13.0+cu129',
            'flashinfer-python':'0.7.0.post1', 'flashinfer-cubin':'0.7.0.post1',
            'flashinfer-jit-cache':'0.7.0.post1+cu129',
            'compressed-tensors':'0.17.0', 'xgrammar':'0.2.8', 'nvidia-nccl-cu12':'2.29.7'}


def check_versions(packages):
    normalized = {name.lower().replace('_', '-'):version for name, version in packages.items()}
    for name, expected in REQUIRED.items():
        if normalized.get(name) != expected:
            raise RuntimeError(f'{name}: expected {expected}, got {normalized.get(name)}')


def check_backend_import(backend):
    """Run inside a real Triton stub, where the C++ pb_utils module exists."""
    spec = importlib.util.spec_from_file_location('deepseek_verified_backend', Path(backend) / 'model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cutlass_flavor_report(root):
    """Observe existing imports and installed candidates without importing CUDA code."""
    def flavor(path):
        match = re.search(r'_cutlass_ir\.cu(\d+)\.', str(path or ''))
        return int(match.group(1)) if match else None

    module = sys.modules.get('cutlass._mlir._mlir_libs._cutlass_ir')
    namespace = vars(module) if module is not None else {}
    origin = namespace.get('__file__') or getattr(namespace.get('__spec__'),'origin',None)
    folder = (Path(root) / 'lib/python3.12/site-packages/nvidia_cutlass_dsl'
              / 'dsl_packages/cutlass/_mlir/_mlir_libs')
    return {
        'status':'not_loaded' if module is None else 'loaded',
        'path':origin, 'cuda_major':flavor(origin),
        'available':[{'cuda_major':flavor(path.name),'path':str(path)}
                     for path in sorted(folder.glob('_cutlass_ir.cu*.so'))],
    }


FLASHINFER_PROVIDERS = {
    f'flashinfer-jit-cache-{arch}':'0.7.0.post1+cu129'
    for arch in ('sm80','sm89','sm90a','sm100a','sm103a','sm120f')
}


def check_flashinfer_artifacts(root, env):
    """Check 0.7's required providers separately from its optional builtin AOT."""
    root = Path(root).resolve()
    site = root / 'lib/python3.12/site-packages'

    def inside(path, *, directory=False):
        path = Path(path).resolve()
        if not path.is_relative_to(root) or not (path.is_dir() if directory else path.is_file()):
            raise RuntimeError(f'FlashInfer artifact is missing or outside immutable env: {path}')
        return path

    source = inside(env.__file__)
    if source != (site/'flashinfer/jit/env.py').resolve():
        raise RuntimeError(f'FlashInfer environment module has an unexpected origin: {source}')
    cubins = inside(env.FLASHINFER_CUBIN_DIR, directory=True)
    if cubins != (site/'flashinfer_cubin/cubins').resolve() or not any(cubins.iterdir()):
        raise RuntimeError(f'FlashInfer cubins must use the populated installed package: {cubins}')
    builtin = Path(env.FLASHINFER_AOT_DIR)
    if builtin.resolve() != (site/'flashinfer/data/aot').resolve():
        raise RuntimeError(f'FlashInfer builtin AOT has an unexpected path: {builtin}')
    if not builtin.resolve().is_relative_to(root) or (builtin.is_symlink() and not builtin.exists()):
        raise RuntimeError(f'FlashInfer builtin AOT escapes the environment or is broken: {builtin}')
    if builtin.exists() and not builtin.is_dir():
        raise RuntimeError(f'FlashInfer builtin AOT is not a directory: {builtin}')

    # Use installed metadata and the entry points that the official shim loads.
    # A builtin directory cannot substitute for a missing or ignored provider.
    def normalize(name):
        return name.lower().replace('_','-')

    installed = {}
    for distribution in importlib.metadata.distributions(path=[str(site)]):
        name = normalize(distribution.metadata['Name'])
        entries = [entry for entry in distribution.entry_points
                   if entry.group == 'flashinfer.jit_cache.providers']
        if entries and name not in FLASHINFER_PROVIDERS:
            raise RuntimeError(f'FlashInfer has an unknown provider distribution: {name}')
        if name in FLASHINFER_PROVIDERS:
            if name in installed or len(entries) != 1:
                raise RuntimeError(f'FlashInfer provider metadata is duplicate or invalid: {name}')
            if distribution.version != FLASHINFER_PROVIDERS[name]:
                raise RuntimeError(f'FlashInfer provider version mismatch: {name} {distribution.version}')
            metadata = [path for path in distribution.files or ()
                        if path.name == 'METADATA' and path.parent.name.endswith('.dist-info')]
            if len(metadata) != 1:
                raise RuntimeError(f'FlashInfer provider has no unique installed metadata record: {name}')
            inside(distribution.locate_file(metadata[0]))
            installed[name] = (distribution,entries[0])
    if set(installed) != set(FLASHINFER_PROVIDERS):
        raise RuntimeError(f'FlashInfer missing pinned provider distributions: {sorted(set(FLASHINFER_PROVIDERS)-set(installed))}')

    providers = []
    seen = set()
    for provider in env.FLASHINFER_AOT_PROVIDERS:
        name = normalize(provider.distribution)
        if name not in installed or name in seen:
            raise RuntimeError(f'FlashInfer has an unknown or duplicate active provider: {name}')
        distribution, entry = installed[name]
        if provider.version != FLASHINFER_PROVIDERS[name] or provider.provider_id != entry.name:
            raise RuntimeError(f'FlashInfer active provider differs from its pinned metadata: {name}')
        loaded = sys.modules.get(entry.module)
        origin = inside(getattr(loaded,'__file__',''))
        owned = {Path(distribution.locate_file(path)).resolve() for path in distribution.files or ()}
        if origin not in owned:
            raise RuntimeError(f'FlashInfer provider module is not owned by its distribution: {name} {origin}')
        cache = inside(provider.jit_cache_dir, directory=True)
        modules = sorted(provider.modules)
        if not modules or not provider.cuda_architectures:
            raise RuntimeError(f'FlashInfer provider has no modules or architectures: {name}')
        for module in modules:
            if not isinstance(module,str) or Path(module).name != module or module in ('.','..') or '\\' in module:
                raise RuntimeError(f'FlashInfer provider has an unsafe module name: {name} {module}')
            binary = inside(cache/module/(module+'.so'))
            if not binary.is_relative_to(cache) or binary not in owned:
                raise RuntimeError(f'FlashInfer provider binary is not owned by its distribution: {name} {binary}')
        providers.append({'id':provider.provider_id,'distribution':name,'version':provider.version,
            'origin':str(origin),'path':str(cache),'cuda_architectures':sorted(provider.cuda_architectures),
            'modules':modules})
        seen.add(name)
    if seen != set(FLASHINFER_PROVIDERS):
        raise RuntimeError(f'FlashInfer missing active providers: {sorted(set(FLASHINFER_PROVIDERS)-seen)}')
    expected_dirs = [builtin.resolve(),*(Path(provider['path']) for provider in providers)]
    if [Path(path).resolve() for path in env.FLASHINFER_AOT_DIRS] != expected_dirs:
        raise RuntimeError('FlashInfer AOT search paths differ from the builtin and pinned providers')
    return {'cubins':str(cubins),'builtin_aot':{'path':str(builtin.resolve()),'present':builtin.is_dir()},
            'aot':[provider['path'] for provider in providers],'providers':providers,
            'downloads_disabled':os.environ.get('FLASHINFER_NO_DOWNLOAD')}


def _load_runtime(backend):
    spec = importlib.util.spec_from_file_location('deepseek_verify_bootstrap', Path(backend) / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    return runtime


def report(root, cuda, backend, imports=False):
    runtime = _load_runtime(backend)
    here = Path(__file__).resolve().parent
    identity = runtime.bootstrap(root, cuda, trusted_paths=(here, backend, Path(backend).parent / 'vllm_multimodal',
                                                          Path(backend).parent / 'python'))
    identity['modules'] = runtime.validate_module_origins(root)
    site = Path(root) / 'lib/python3.12/site-packages'
    identity['distributions'] = {d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(site)])}
    check_versions(identity['distributions'])
    identity['tool_versions'] = {}
    for name in ('nvcc','ptxas','cuobjdump','nvdisasm'):
        tool = Path(cuda)/'bin'/name
        if not tool.is_file() or not os.access(tool,os.X_OK):
            raise RuntimeError(f'Missing selected CUDA tool: {tool}')
        identity['tool_versions'][name] = subprocess.check_output([str(tool),'--version'],text=True)
    compiler = identity['tool_versions']['nvcc']
    if 'release 12.9' not in compiler:
        raise RuntimeError(f'Unexpected CUDA compiler: {compiler}')
    identity['compiler'] = compiler
    if imports:
        import torch
        import vllm
        import flashinfer
        runtime.validate_module_origins(root)
        if torch.version.cuda != '12.9' or torch.cuda.is_initialized():
            raise RuntimeError('Expected CUDA 12.9 Torch without an initialized CUDA context')
        from flashinfer.jit import env as flashinfer_env
        identity['flashinfer_artifacts'] = check_flashinfer_artifacts(root, flashinfer_env)
        identity['imported'] = {'torch':torch.__version__, 'vllm':vllm.__version__,
                                'flashinfer':flashinfer.__version__, 'torch_cuda':torch.version.cuda}
    identity['selected_cutlass_ir'] = cutlass_flavor_report(root)
    identity['native_maps'] = runtime.validate_native_maps(root,cuda)
    return identity


def _child_report(queue, root, cuda, backend, imports):
    try:
        queue.put({'ok':report(root, cuda, backend, imports)})
    except BaseException as error:
        queue.put({'error':f'{type(error).__name__}: {error}'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/opt/envs/vllm032-cu129')
    parser.add_argument('--cuda', default='/opt/cuda/12.9')
    parser.add_argument('--backend', default='/opt/tritonserver/backends/vllm_deepseek')
    parser.add_argument('--imports', action='store_true')
    parser.add_argument('--spawn', action='store_true')
    args = parser.parse_args()
    result = {'parent':report(args.root, args.cuda, args.backend, args.imports)}
    subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, stdout=sys.stderr)
    if args.spawn:
        context = multiprocessing.get_context('spawn')
        queue = context.Queue()
        worker = context.Process(target=_child_report, args=(queue,args.root,args.cuda,args.backend,args.imports))
        worker.start()
        try:
            child = queue.get(timeout=120)
            worker.join(timeout=10)
            if worker.exitcode != 0 or 'error' in child:
                raise RuntimeError(f'Spawned runtime validation failed: {child}, exit={worker.exitcode}')
            result['child'] = child['ok']
            if result['child']['modules'] != result['parent']['modules']:
                raise RuntimeError('Spawned worker module origins differ from parent')
        finally:
            if worker.is_alive():
                worker.terminate(); worker.join(timeout=10)
            queue.close()
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
