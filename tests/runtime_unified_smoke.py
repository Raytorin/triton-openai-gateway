# SPDX-License-Identifier: Apache-2.0
"""Real Triton CPU/stub/IPC qualification; no weights, GPU inference or remote services.

Run inside the unified image with its prepared Python launch environment. Both
ordinary Python models and existing explicit EXECUTION_ENV_PATH configs are
loaded, inferred, unloaded and loaded again against the native server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid


RUNTIME = Path('/opt/envs/vllm032-cu129')
STUB = Path('/opt/tritonserver/backends/python/triton_python_backend_stub.native')
CORE_POLICY = Path('/opt/tritonserver/backends/vllm_deepseek/core-native-libraries.json')
CORE_CUDART = Path('/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13.4.92')
VERSIONS = {'torch': '2.13.0+cu129', 'vllm': '0.32.0+cu129',
            'transformers': '5.19.0', 'numpy': '2.3.5'}
MODULE_VERSIONS = dict(VERSIONS, vllm='0.32.0')
CACHE_NAMES = {'VLLM_CACHE_ROOT': 'vllm', 'TORCH_HOME': 'torch',
               'TRITON_CACHE_DIR': 'triton', 'HF_MODULES_CACHE': 'hf-modules',
               'PYTHONPYCACHEPREFIX': 'python-pycache'}


def inside(path, root):
    try:
        return bool(path) and Path(path).is_absolute() and Path(path).resolve().is_relative_to(Path(root).resolve())
    except (TypeError, ValueError, OSError):
        return False


def verify_core_abi(policy_path, executable):
    """Allow only the reviewed native stub's exact CUDA 13 runtime dependency."""
    try:
        policy = json.loads(Path(policy_path).read_text())
        stub = policy['stub']
        libraries = policy['libraries']
        if (policy['schema'] != 1 or stub['path'] != str(STUB)
                or executable != str(STUB) or STUB.is_symlink()
                or len(libraries) != 1):
            raise ValueError('stub identity or explicit library set changed')
        library = libraries[0]
        if library['path'] != str(CORE_CUDART) or library['soname'] != 'libcudart.so.13':
            raise ValueError('only the explicit libcudart13 entry is supported')
        for path, wanted in ((STUB, stub['sha256']), (CORE_CUDART, library['sha256'])):
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != wanted:
                raise ValueError(f'hash mismatch: {path}')
        return {str(CORE_CUDART)}
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise RuntimeError(f'Unverified native core ABI exception: {error}') from error


def validate_loaded_runtime(report, root, label, *, core_policy=None, require_libpython=True):
    modules = report.get('modules', {})
    if set(modules) != set(VERSIONS):
        raise RuntimeError(f'{label}: missing primary module origins')
    origins = report.get('engine_origins', {})
    if not origins:
        raise RuntimeError(f'{label}: missing loaded engine origins')
    for name, path in (*origins.items(), *modules.items()):
        if not inside(path, root):
            raise RuntimeError(f'{label}: foreign engine origin: {name} {path}')
    native = report.get('native_libraries', [])
    if not any(Path(path).name.startswith('libpython3.12.so') for path in native):
        # The standalone Python executable can statically link CPython. The
        # embedded C++ stub must still map the selected shared libpython.
        if require_libpython or Path(report.get('executable', '')).resolve() != (root / 'bin/python3.12').resolve():
            raise RuntimeError(f'{label}: no loaded libpython3.12 evidence')
    if not any(Path(path).name == 'libtorch_cpu.so' for path in native):
        raise RuntimeError(f'{label}: no loaded Torch native library evidence')
    for path in native:
        engine_library = Path(path).name.startswith(('libpython', 'libtorch', 'libc10'))
        if inside(path, root) or (not engine_library and inside(path, '/opt/cuda/12.9')):
            continue
        if core_policy is not None and path == str(CORE_CUDART):
            if path in verify_core_abi(core_policy, report.get('executable')):
                continue
        raise RuntimeError(f'{label}: foreign native runtime library: {path}')
    for path in report.get('sys_path', []):
        if {'site-packages', 'dist-packages'} & set(Path(path).parts) and not inside(path, root):
            raise RuntimeError(f'{label}: foreign Python package path: {path}')


def validate_report(report, root, cache, model, nonce, *, core_policy=CORE_POLICY):
    """Check native evidence after it has crossed real Triton tensor IPC."""
    root, cache = Path(root).resolve(), Path(cache).resolve()
    expected = {
        'model': model, 'echo': nonce, 'prefix': str(root), 'python': '3.12.15',
        'versions': VERSIONS, 'pb_utils_tensor_module': 'c_python_backend_utils',
        'cuda_initialized': False, 'torch_cuda': '12.9', 'cache_arch': 'cpu',
        'cache_root': str(cache), 'pythonhome': str(root), 'virtual_env': str(root),
        'worker_method': 'spawn', 'downloads_disabled': '1', 'executable': str(STUB),
        'sys_executable': str(root / 'bin/python3.12'), 'module_versions': MODULE_VERSIONS,
    }
    for name, wanted in expected.items():
        if report.get(name) != wanted:
            raise RuntimeError(f'{model}: unexpected {name}: {report.get(name)!r}; expected {wanted!r}')
    if not isinstance(report.get('pid'), int) or report['pid'] <= 0 or not report.get('instance'):
        raise RuntimeError(f'{model}: missing native instance identity')
    argv = report.get('native_argv', [])
    if len(argv) < 2 or argv[0] != str(STUB):
        raise RuntimeError(f'{model}: native stub argv was not preserved: {argv!r}')
    validate_loaded_runtime(report, root, model, core_policy=core_policy)
    if inside(cache, root) or inside(cache, '/opt/cuda/12.9'):
        raise RuntimeError(f'{model}: mutable cache is inside immutable runtime')
    for name, suffix in CACHE_NAMES.items():
        if report.get('caches', {}).get(name) != str(cache / 'cpu' / suffix):
            raise RuntimeError(f'{model}: wrong cache selection for {name}')

    child = report.get('child', {})
    child_expected = {key: expected[key] for key in (
        'prefix', 'python', 'versions', 'module_versions', 'cuda_initialized',
        'torch_cuda', 'cache_arch', 'cache_root', 'pythonhome', 'virtual_env',
        'worker_method', 'downloads_disabled', 'sys_executable')}
    child_expected.update(executable=str(root / 'bin/python3.12'),
                          parent_pid=report['pid'], start_method='spawn', native_backend_module=False)
    for name, wanted in child_expected.items():
        if child.get(name) != wanted:
            raise RuntimeError(f'{model} child: unexpected {name}: {child.get(name)!r}')
    if not isinstance(child.get('pid'), int) or child['pid'] <= 0 or child['pid'] == report['pid']:
        raise RuntimeError(f'{model} child: not a fresh Python process')
    for name, suffix in CACHE_NAMES.items():
        if child.get('caches', {}).get(name) != str(cache / 'cpu' / suffix):
            raise RuntimeError(f'{model} child: wrong cache selection for {name}')
    # A spawned engine is a fresh Python executable, never the C++ core stub.
    # Its loader therefore gets no native core CUDA 13 ABI exception.
    validate_loaded_runtime(child, root, f'{model} child', require_libpython=False)


def wait_for(predicate, process, timeout, description):
    deadline = time.monotonic() + timeout
    while True:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f'Triton exited with code {code} while waiting for {description}')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Timed out waiting for {description}')
        if predicate(min(1.0, remaining)):
            return
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))


def stop_server(process, timeout):
    """Give the server a graceful exit, then reap only our owned process group."""
    forced = False
    try:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            forced = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
    finally:
        # A failed server may have exited before reaping its Python stubs.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {'returncode': process.returncode, 'forced': forced}


def close_client_and_server(client, process, timeout):
    try:
        if client is not None:
            client.close()
    finally:
        if process is not None:
            shutdown = stop_server(process, timeout)
    return shutdown if process is not None else None


SPAWN_SOURCE = r'''
import importlib
from importlib import metadata
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time


def runtime_report():
    modules = {name: importlib.import_module(name)
               for name in ('torch', 'vllm', 'transformers', 'numpy')}
    prefixes = ('torch', 'vllm', 'transformers', 'numpy', 'flashinfer',
                'compressed_tensors', 'xgrammar', 'triton', 'cutlass', 'nvidia')
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = vars(module).get('__file__') if module is not None else None
        if origin and any(name == prefix or name.startswith(prefix + '.') for prefix in prefixes):
            origins[name] = str(Path(origin).resolve())
    native = set()
    prefixes = ('libpython', 'libtorch', 'libc10', 'libcudart', 'libnvrtc',
                'libcublas', 'libnccl', 'libnvJitLink', 'libcudnn')
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = fields[5]
            if Path(path).name.startswith(prefixes):
                native.add(str(Path(path).resolve()))
    cache_keys = ('VLLM_CACHE_ROOT', 'TORCH_HOME', 'TRITON_CACHE_DIR',
                  'HF_MODULES_CACHE', 'PYTHONPYCACHEPREFIX')
    return {
        'pid': os.getpid(), 'parent_pid': os.getppid(),
        'prefix': sys.prefix, 'python': platform.python_version(),
        'executable': os.readlink('/proc/self/exe'), 'sys_executable': sys.executable,
        'native_argv': Path('/proc/self/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
        'versions': {name: metadata.version(name) for name in modules},
        'module_versions': {name: str(module.__version__) for name, module in modules.items()},
        'modules': {name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
        'engine_origins': origins, 'native_libraries': sorted(native), 'sys_path': sys.path,
        'cuda_initialized': modules['torch'].cuda.is_initialized(),
        'torch_cuda': modules['torch'].version.cuda,
        'cache_arch': os.environ.get('TRITON_RUNTIME_CACHE_ARCH'),
        'cache_root': os.environ.get('TRITON_RUNTIME_CACHE_DIR'),
        'caches': {name: os.environ.get(name) for name in cache_keys},
        'pythonhome': os.environ.get('PYTHONHOME'), 'virtual_env': os.environ.get('VIRTUAL_ENV'),
        'worker_method': os.environ.get('VLLM_WORKER_MULTIPROC_METHOD'),
        'downloads_disabled': os.environ.get('FLASHINFER_NO_DOWNLOAD'),
        'start_method': multiprocessing.get_start_method(allow_none=True),
        'native_backend_module': 'c_python_backend_utils' in sys.modules,
    }


def _send_report(connection, collector):
    try:
        connection.send({'report': collector()})
    except BaseException as error:
        connection.send({'error': f'{type(error).__name__}: {str(error)[:1024]}'})
    finally:
        connection.close()


def spawn_probe(collector, timeout=60, shutdown_timeout=10):
    if timeout <= 0 or not 0 < shutdown_timeout <= 10:
        raise ValueError('Spawn receive deadline must be positive; shutdown grace must be in (0, 10]')
    context = multiprocessing.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_send_report, args=(writer, collector))
    started = False
    started_at = time.monotonic()
    received_at = None
    try:
        process.start()
        started = True
        writer.close()
        if not reader.poll(max(0, timeout - (time.monotonic() - started_at))):
            raise TimeoutError(f'Embedded model spawn probe timed out: phase=receive pid={process.pid} '
                               f'elapsed={time.monotonic() - started_at:.3f}s budget={timeout}s')
        result = reader.recv()
        received_at = time.monotonic()
        if 'error' in result:
            raise RuntimeError(f'Embedded model spawn probe failed: phase=collect pid={process.pid} '
                               f'elapsed={received_at - started_at:.3f}s error={str(result["error"])[:1024]}')
        # Python/Torch finalizers run after sending the report. Their bounded
        # grace is independent of the import/receive deadline above.
        process.join(timeout=shutdown_timeout)
        ended_at = time.monotonic()
        timing = {'receive_seconds': round(received_at - started_at, 4),
                  'shutdown_seconds': round(ended_at - received_at, 4),
                  'receive_budget_seconds': timeout, 'shutdown_budget_seconds': shutdown_timeout}
        if process.is_alive() or process.exitcode != 0:
            raise RuntimeError(f'Embedded model spawn probe failed: phase=shutdown report_received=yes '
                               f'pid={process.pid} exit={process.exitcode} timings={timing}')
        report = result['report']
        report['spawn_timings'] = timing
        return report
    finally:
        reader.close()
        writer.close()
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
            if process.is_alive():
                raise RuntimeError(f'Embedded model spawn probe could not reap child: '
                                   f'phase=cleanup pid={process.pid} elapsed={time.monotonic() - started_at:.3f}s')
            process.close()

'''

MODEL_SOURCE = r'''
import json
import uuid
import numpy as np
import triton_python_backend_utils as pb_utils
from unified_spawn_probe import runtime_report, spawn_probe

class TritonPythonModel:
    def initialize(self, args):
        self.report = runtime_report()
        self.report.update(model=args['model_name'], instance=uuid.uuid4().hex,
                           pb_utils_tensor_module=pb_utils.Tensor.__module__)
        self.report['child'] = spawn_probe(runtime_report, timeout=60)

    def execute(self, requests):
        responses = []
        for request in requests:
            value = pb_utils.get_input_tensor_by_name(request, 'REQUEST').as_numpy()[0]
            if not isinstance(value, bytes):
                raise RuntimeError('The real Triton BYTES input did not arrive as bytes')
            payload = json.dumps(dict(self.report, echo=value.decode('utf-8')), sort_keys=True).encode()
            tensor = pb_utils.Tensor('REPORT', np.array([payload], dtype=np.object_))
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))
        return responses
'''


def create_repository(repository, root):
    names = ('unified_probe', 'unified_explicit_env_probe')
    for name in names:
        directory = repository / name
        (directory / '1').mkdir(parents=True)
        config = (
            f'name: "{name}"\nbackend: "python"\nmax_batch_size: 0\n'
            'input [{ name: "REQUEST" data_type: TYPE_STRING dims: [1] }]\n'
            'output [{ name: "REPORT" data_type: TYPE_STRING dims: [1] }]\n'
            'instance_group [{ kind: KIND_CPU count: 1 }]\n'
        )
        if name == 'unified_explicit_env_probe':
            config += ('parameters { key: "EXECUTION_ENV_PATH" value { string_value: '
                       + json.dumps(str(root)) + ' } }\n')
        (directory / 'config.pbtxt').write_text(config)
        (directory / '1/model.py').write_text(MODEL_SOURCE)
        (directory / '1/unified_spawn_probe.py').write_text(SPAWN_SOURCE)
    return names


def log_excerpt(log, limit=8192):
    log.flush()
    size = os.fstat(log.fileno()).st_size
    log.seek(0)
    first = log.read(min(size, limit))
    if size <= limit:
        return first.decode(errors='replace')
    log.seek(max(limit, size - limit))
    return first.decode(errors='replace') + '\n[... log middle omitted ...]\n' + log.read(limit).decode(errors='replace')


def run_smoke(tritonserver, root, timeout, shutdown_timeout, *, model_load_thread_count=1):
    import numpy as np
    import tritonclient.grpc as grpc
    from tritonclient.utils import InferenceServerException

    if Path(sys.prefix).resolve() != root.resolve():
        raise RuntimeError(f'Smoke driver must use runtime Python: {sys.prefix}')
    with tempfile.TemporaryDirectory(prefix='unified-native-smoke-') as temporary, tempfile.TemporaryFile() as log:
        work = Path(temporary)
        repository = work / 'models'
        names = create_repository(repository, root)
        cache = work / 'cache'
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        command = [str(tritonserver), f'--model-repository={repository}',
                   '--model-control-mode=explicit', '--allow-http=false', '--allow-metrics=false',
                   '--allow-grpc=true', '--grpc-address=127.0.0.1', f'--grpc-port={port}',
                   '--exit-on-error=true', '--strict-model-config=true', '--exit-timeout-secs=5',
                   f'--model-load-thread-count={model_load_thread_count}',
                   '--pinned-memory-pool-byte-size=0', *[f'--load-model={name}' for name in names]]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                           VLLM_NO_USAGE_STATS='1', DO_NOT_TRACK='1',
                           TRITON_RUNTIME_CACHE_ARCH='cpu', TRITON_RUNTIME_CACHE_DIR=str(cache))
        process = None
        client = None
        result = {'qualification': 'CPU native Triton Python stub and tensor IPC', 'models': {}}
        try:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            client = grpc.InferenceServerClient(f'127.0.0.1:{port}')

            def readiness(method, *arguments, negate=False):
                def check(budget):
                    try:
                        value = method(*arguments, client_timeout=budget)
                        return not value if negate else value
                    except InferenceServerException:
                        return False
                return check

            wait_for(readiness(client.is_server_ready), process, timeout, 'native server readiness')
            for name in names:
                wait_for(readiness(client.is_model_ready, name), process, timeout, f'{name} readiness')
                reports = []
                for generation in range(2):
                    nonce = uuid.uuid4().hex
                    request = grpc.InferInput('REQUEST', [1], 'BYTES')
                    request.set_data_from_numpy(np.asarray([nonce.encode()], dtype=np.object_))
                    response = client.infer(name, [request], outputs=[grpc.InferRequestedOutput('REPORT')],
                                            client_timeout=timeout)
                    values = response.as_numpy('REPORT')
                    if values is None or values.shape != (1,) or not isinstance(values[0], bytes):
                        raise RuntimeError(f'{name}: invalid real Triton BYTES response')
                    report = json.loads(values[0])
                    validate_report(report, root, cache, name, nonce)
                    reports.append(report)
                    client.unload_model(name, client_timeout=timeout)
                    wait_for(readiness(client.is_model_ready, name, negate=True), process, timeout,
                             f'{name} unload')
                    if generation == 0:
                        client.load_model(name, client_timeout=timeout)
                        wait_for(readiness(client.is_model_ready, name), process, timeout, f'{name} reload')
                if reports[0]['pid'] == reports[1]['pid'] or reports[0]['instance'] == reports[1]['instance']:
                    raise RuntimeError(f'{name}: reload reused the old stub instance')
                result['models'][name] = {
                    key: reports[1][key] for key in ('prefix', 'versions', 'modules', 'native_libraries',
                                                    'executable', 'sys_executable', 'module_versions', 'cache_arch', 'caches')}
                result['models'][name]['child'] = {key: reports[1]['child'][key] for key in (
                    'prefix', 'sys_executable', 'executable', 'versions', 'module_versions',
                    'modules', 'native_libraries', 'pid', 'parent_pid', 'start_method', 'spawn_timings')}
                result['models'][name].update(
                    pids=[item['pid'] for item in reports],
                    engine_origins_checked=len(reports[1]['engine_origins']),
                    inference_requests=2, unloads=2, reloads=1,
                )
        except BaseException as error:
            raise RuntimeError(f'{error}\nTriton log:\n{log_excerpt(log)}') from error
        finally:
            shutdown = close_client_and_server(client, process, shutdown_timeout)
        if shutdown['forced'] or shutdown['returncode'] != 0:
            raise RuntimeError(f'Native Triton shutdown failed: {shutdown}\n{log_excerpt(log)}')
        result['shutdown'] = shutdown
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tritonserver', type=Path, default=Path('/opt/tritonserver/bin/tritonserver'))
    parser.add_argument('--root', type=Path, default=RUNTIME)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--shutdown-timeout', type=float, default=10)
    parser.add_argument('--model-load-thread-count', type=int, default=1,
                        help='Triton loader threads for this CPU smoke (default: 1; use 2 for parallel startup)')
    args = parser.parse_args()
    if args.timeout <= 0 or args.shutdown_timeout <= 0:
        parser.error('timeouts must be positive')
    if args.model_load_thread_count < 1:
        parser.error('--model-load-thread-count must be positive')
    try:
        result = run_smoke(args.tritonserver, args.root, args.timeout, args.shutdown_timeout,
                           model_load_thread_count=args.model_load_thread_count)
        print(json.dumps(result, sort_keys=True))
    except (Exception, KeyboardInterrupt) as error:
        print(f'Unified native smoke failed: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
