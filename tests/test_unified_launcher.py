"""The opt-in launcher selects a runtime before exec, without model policy."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'docker/unified'
LAUNCHER = TOOLS / 'launch.py'
BACKENDS = ('python', 'vllm', 'vllm_multimodal', 'vllm_deepseek')
STUB = 'triton_python_backend_stub'
UTILS = 'triton_python_backend_utils.py'


def launch(tmp_path, source, **inherited):
    assert LAUNCHER.is_file(), 'the common pre-exec launcher is missing'
    return subprocess.run(
        [sys.executable, '-I', str(LAUNCHER), '--root', str(tmp_path/'env'),
         '--cuda', str(tmp_path/'cuda'), '--exec', sys.executable, '-c', source],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache'), **inherited},
        capture_output=True, text=True, timeout=10,
    )


def test_preexec_removes_foreign_python_native_compiler_and_package_settings(tmp_path):
    result = launch(tmp_path, 'import os, json; print(json.dumps(dict(os.environ)))',
        PYTHONHOME='/foreign/python', PYTHONUSERBASE='/foreign/user',
        PYTHONPATH='/foreign/site-packages', LD_LIBRARY_PATH='/foreign/lib',
        LD_PRELOAD='', LD_AUDIT='', PATH='/foreign/bin',
        PIP_INDEX_URL='https://invalid.example/simple', PIP_CONSTRAINT='/foreign/constraints',
        CPATH='/foreign/include', LIBRARY_PATH='/foreign/lib',
        CMAKE_PREFIX_PATH='/foreign/cmake', CUDAToolkit_ROOT='/foreign/cuda',
        TRITON_PTXAS_PATH='/foreign/ptxas', TRITON_CUDART_PATH='/foreign/cudart',
        VLLM_NCCL_SO_PATH='/foreign/nccl', VLLM_MODEL_REDIRECT_PATH='/foreign/redirect')
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    for key in ('PYTHONHOME', 'PYTHONUSERBASE', 'LD_PRELOAD', 'LD_AUDIT',
                'PIP_INDEX_URL', 'PIP_CONSTRAINT', 'CMAKE_PREFIX_PATH',
                'CUDAToolkit_ROOT', 'VLLM_MODEL_REDIRECT_PATH'):
        assert key not in env
    assert env['PYTHONPATH'] == ''
    assert env['VIRTUAL_ENV'] == str(tmp_path/'env')
    assert env['PYTHONNOUSERSITE'] == '1'
    assert env['PIP_CONFIG_FILE'] == os.devnull
    assert env['PATH'].split(os.pathsep) == [
        str(tmp_path/'env/bin'), str(tmp_path/'cuda/bin'), '/opt/tritonserver/bin',
        '/opt/ffmpeg-safe/bin', '/usr/local/sbin', '/usr/local/bin', '/usr/sbin',
        '/usr/bin', '/sbin', '/bin']
    assert env['LD_LIBRARY_PATH'].split(os.pathsep) == [
        str(tmp_path/'env/lib'), str(tmp_path/'cuda/lib64'),
        str(tmp_path/'env/lib/python3.12/site-packages/torch/lib'),
        '/usr/local/nvidia/lib', '/usr/local/nvidia/lib64']
    assert '/usr/lib' not in env['LD_LIBRARY_PATH']
    assert env['CPATH'] == str(tmp_path/'cuda/include')
    assert env['LIBRARY_PATH'] == str(tmp_path/'cuda/lib64')
    assert env['TRITON_PTXAS_PATH'] == str(tmp_path/'cuda/bin/ptxas')
    assert env['TRITON_CUDART_PATH'] == str(tmp_path/'cuda/include')
    assert env['VLLM_NCCL_SO_PATH'] == str(
        tmp_path/'env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2')
    assert env['VLLM_WORKER_MULTIPROC_METHOD'] == 'spawn'


def test_literal_ffmpeg_selects_safe_donor_over_system_and_inherited_tools(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'unified_launcher_ffmpeg_test', LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def executable(directory, message):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/'ffmpeg'
        path.write_text(f'#!/bin/sh\nprintf "%s\\n" "{message}"\n')
        path.chmod(0o755)
    foreign = tmp_path/'foreign/bin'
    executable(foreign, 'inherited-ffmpeg')
    # Project the fixed image directories into a local fixture filesystem;
    # keep their order and retain any unknown inherited directories verbatim.
    fixed = ['/opt/tritonserver/bin', '/opt/ffmpeg-safe/bin', '/usr/local/sbin',
             '/usr/local/bin', '/usr/sbin', '/usr/bin', '/sbin', '/bin']
    image_paths = {path: tmp_path/'image'/path.lstrip('/') for path in fixed}
    executable(image_paths['/opt/ffmpeg-safe/bin'], 'safe-donor-ffmpeg')
    executable(image_paths['/usr/local/bin'], 'other-local-ffmpeg')
    executable(image_paths['/usr/bin'], 'other-system-ffmpeg')
    monkeypatch.setenv('PATH', str(foreign) + os.pathsep + '/usr/local/bin:/usr/bin')
    monkeypatch.setenv('TRITON_RUNTIME_CACHE_DIR', str(tmp_path/'cache'))
    environment = module.prepare_environment(tmp_path/'env', tmp_path/'cuda')
    environment['PATH'] = os.pathsep.join(str(image_paths.get(path, path))
        for path in environment['PATH'].split(os.pathsep))
    result = subprocess.run(['ffmpeg', '-version'], env=environment,
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout == 'safe-donor-ffmpeg\n'


@pytest.mark.parametrize('arguments', [
    ['--', '/absolute/client.py', '--serve'],
    ['--exec', '--exec', '--root', '/child-root', '--cuda', '/child-cuda'],
    ['--help', 'argument with spaces', 'Юникод', '', '--', '--'],
])
def test_exec_tail_is_opaque_and_preserves_child_argv(tmp_path, arguments):
    result = subprocess.run([sys.executable, '-I', str(LAUNCHER),
        '--root', str(tmp_path/'env'), '--cuda', str(tmp_path/'cuda'),
        '--exec', sys.executable, '-c',
        'import json, sys; print(json.dumps(sys.argv[1:]))', *arguments],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == arguments


def test_bash_placeholder_separator_reaches_child_script(tmp_path):
    script = tmp_path/'child script.py'
    script.write_text('import json, sys; print(json.dumps(sys.argv[1:]))\n')
    result = subprocess.run([sys.executable, '-I', str(LAUNCHER),
        '--root', str(tmp_path/'env'), '--cuda', str(tmp_path/'cuda'),
        '--exec', '/bin/bash', '-euc', 'exec "$@"', '--',
        sys.executable, str(script), '--serve', '--root', 'child value'],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == ['--serve', '--root', 'child value']


@pytest.mark.parametrize('arguments,diagnostic', [
    ([], '--exec'),
    (['--root', '/chosen-root'], '--exec'),
    (['--exec'], 'absolute executable'),
    (['--exec', 'python3'], 'absolute executable'),
    (['--root', '--exec', '/bin/true'], '--root'),
    (['--unknown', '--exec', '/bin/true'], 'unrecognized arguments'),
    (['--roo', '/chosen-root', '--exec', '/bin/true'], 'unrecognized arguments'),
    (['--exec=/bin/true'], 'unrecognized arguments'),
])
def test_missing_command_and_invalid_launcher_prefix_fail_before_exec(tmp_path, arguments, diagnostic):
    result = subprocess.run([sys.executable, '-I', str(LAUNCHER), *arguments],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert diagnostic in result.stderr
    assert result.stdout == ''
    assert not (tmp_path/'cache').exists()


def test_launcher_help_does_not_require_a_child_command(tmp_path):
    result = subprocess.run([sys.executable, '-I', str(LAUNCHER), '--help'],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache')},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert '--root' in result.stdout and '--cuda' in result.stdout
    assert not (tmp_path/'cache').exists()


@pytest.mark.parametrize('generic,legacy,expected', [
    ({}, {}, 'unqualified'),
    ({}, {'DEEPSEEK_CACHE_ARCH': 'sm90'}, 'sm90'),
    ({'TRITON_RUNTIME_CACHE_ARCH': 'cpu'}, {'DEEPSEEK_CACHE_ARCH': 'sm90'}, 'cpu'),
])
def test_cache_flags_keep_deepseek_compatibility_without_requiring_gpu_arch(
        tmp_path, monkeypatch, generic, legacy, expected):
    for prefix in ('TRITON_RUNTIME', 'DEEPSEEK'):
        for suffix in ('ARCH', 'DIR'):
            monkeypatch.delenv(f'{prefix}_CACHE_{suffix}', raising=False)
    result = launch(tmp_path, 'import os,json; print(json.dumps(dict(os.environ)))',
                    DEEPSEEK_CACHE_DIR=str(tmp_path/'old-cache'), **legacy, **generic)
    assert result.returncode == 0, result.stderr
    env = json.loads(result.stdout)
    assert env['TRITON_RUNTIME_CACHE_ARCH'] == env['DEEPSEEK_CACHE_ARCH'] == expected
    assert env['TRITON_RUNTIME_CACHE_DIR'] == env['DEEPSEEK_CACHE_DIR'] == str(tmp_path/'cache')
    for key, suffix in [('VLLM_CACHE_ROOT', 'vllm'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                        ('TRITON_CACHE_DIR', 'triton'), ('DG_JIT_CACHE_DIR', 'deepgemm')]:
        assert env[key] == str(tmp_path/'cache'/expected/suffix)
        assert Path(env[key]).is_dir()


@pytest.mark.parametrize('bad_flag,bad_value', [
    ('TRITON_RUNTIME_CACHE_ARCH', '../foreign'),
    ('TRITON_RUNTIME_CACHE_DIR', 'env/cache'),
])
def test_bad_cache_selection_fails_before_exec(tmp_path, bad_flag, bad_value):
    value = str(tmp_path/bad_value) if bad_flag.endswith('_DIR') else bad_value
    result = launch(tmp_path, 'print("EXECUTED")', **{bad_flag: value})
    assert result.returncode != 0
    assert 'EXECUTED' not in result.stdout
    assert 'cache' in result.stderr.lower()


def test_native_library_symlink_escape_fails_before_exec(tmp_path):
    package = tmp_path/'env/lib/python3.12/site-packages/nvidia/nccl'
    package.mkdir(parents=True)
    (tmp_path/'foreign').mkdir()
    (package/'lib').symlink_to(tmp_path/'foreign', target_is_directory=True)
    result = launch(tmp_path, 'print("EXECUTED")')
    assert result.returncode != 0
    assert 'EXECUTED' not in result.stdout
    assert 'outside runtime' in result.stderr


def test_cache_namespace_cannot_redirect_writes_into_the_runtime(tmp_path):
    cache = tmp_path/'cache'
    cache.mkdir()
    environment = tmp_path/'env'
    environment.mkdir()
    (cache/'cpu').symlink_to(environment, target_is_directory=True)
    result = launch(tmp_path, 'print("EXECUTED")', TRITON_RUNTIME_CACHE_ARCH='cpu')
    assert result.returncode != 0
    assert 'EXECUTED' not in result.stdout
    assert 'cache' in result.stderr.lower()
    assert not list(environment.iterdir())


@pytest.mark.parametrize('task', ['embed', 'classify', 'image', 'audio', 'generate'])
def test_generic_python_model_can_accept_its_own_tasks_and_import_local_helpers(tmp_path, task):
    (tmp_path/'model_helpers.py').write_text('def accept(task): return task\n')
    model = tmp_path/'model.py'
    model.write_text('from model_helpers import accept\nimport sys\nprint(accept(sys.argv[1]))\n')
    assert LAUNCHER.is_file(), 'the common pre-exec launcher is missing'
    result = subprocess.run(
        [sys.executable, '-I', str(LAUNCHER), '--root', str(tmp_path/'env'),
         '--cuda', str(tmp_path/'cuda'), '--exec', sys.executable, str(model), task],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tmp_path/'cache')},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == task


@pytest.fixture(scope='session')
def native_probe(tmp_path_factory):
    compiler = shutil.which('cc')
    if not compiler:
        pytest.skip('a C compiler is required for the native exec boundary probe')
    directory = tmp_path_factory.mktemp('unified-native-probe')
    source = directory/'probe.c'
    source.write_text('''#include <stdio.h>
#include <string.h>
extern char **environ;
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--environment") == 0) {
        for (char **item = environ; *item; ++item) fwrite(*item, 1, strlen(*item) + 1, stdout);
    } else {
        for (int index = 1; index < argc; ++index) printf("%s%s", index == 1 ? "" : " ", argv[index]);
        puts("");
    }
    return 0;
}
''')
    executable = directory/'probe'
    result = subprocess.run([compiler, '-o', str(executable), str(source)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return executable


@pytest.fixture
def installed_tree(tmp_path, native_probe):
    assert (TOOLS/'install.py').is_file(), 'the safe stub installer is missing'
    tree = tmp_path/'triton'
    shutil.copytree(TOOLS, tree/'unified')
    for name in BACKENDS:
        backend = tree/'backends'/name
        backend.mkdir(parents=True)
        (backend/STUB).write_text('original per-backend wrapper\n')
    # Use a real ELF so preservation and exec are tested without a Triton/GPU dependency.
    shutil.copy2(native_probe, tree/'backends/python'/STUB)
    (tree/'backends/python'/UTILS).write_text('# stock utility fixture\nVALUE = "base"\n')
    original = (tree/'backends/python'/STUB).read_bytes()
    policy = tree/'backends/vllm_deepseek/core-native-libraries.json'
    policy.write_text(json.dumps({'stub': {'path': str(tree/'backends/python'/STUB),
        'sha256': hashlib.sha256(original).hexdigest()}, 'libraries': [{'fixture': 'retained'}]}))
    return tree, original, policy


def install(tree):
    return subprocess.run([sys.executable, '-I', str(tree/'unified/install.py'),
        '--triton-root', str(tree)], capture_output=True, text=True, timeout=10)


def test_installer_preserves_native_elf_and_each_backend_forwards_stock_argv(installed_tree):
    tree, original, policy = installed_tree
    result = install(tree)
    assert result.returncode == 0, result.stderr
    native = tree/'backends/python'/f'{STUB}.native'
    assert not native.is_symlink()
    assert native.read_bytes() == original
    updated = json.loads(policy.read_text())
    assert updated['stub'] == {'path': str(native), 'sha256': hashlib.sha256(original).hexdigest()}
    assert updated['libraries'] == [{'fixture': 'retained'}]
    arguments = ['model.py', 'shared-memory', '12', '34', '56', 'a path with spaces']
    for name in BACKENDS:
        wrapper = tree/'backends'/name/STUB
        assert wrapper.is_symlink()
        assert wrapper.resolve() == tree/'unified'/STUB
        probe = subprocess.run([sys.executable, '-I', str(wrapper), *arguments],
            env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tree/'cache')},
            capture_output=True, text=True, timeout=5)
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout == ' '.join(arguments) + '\n'


def test_reinstall_keeps_native_bytes_and_policy_pinning(installed_tree):
    tree, original, policy = installed_tree
    assert install(tree).returncode == 0
    second = install(tree)
    assert second.returncode == 0, second.stderr
    assert (tree/'backends/python'/f'{STUB}.native').read_bytes() == original
    assert json.loads(policy.read_text())['stub']['sha256'] == hashlib.sha256(original).hexdigest()


def test_installer_provides_sibling_utils_for_custom_stub_selection(installed_tree):
    tree, _, _ = installed_tree
    base = tree/'backends/python'/UTILS
    original = base.read_bytes()
    before = base.stat()
    result = install(tree)
    assert result.returncode == 0, result.stderr
    for backend in ('vllm', 'vllm_multimodal', 'vllm_deepseek'):
        utility = tree/'backends'/backend/UTILS
        assert utility.is_symlink()
        assert utility.readlink() == Path('../python/triton_python_backend_utils.py')
        assert utility.resolve() == base
        assert utility.read_bytes() == original
        assert (utility.parent/STUB).is_file()
    assert base.read_bytes() == original
    assert not base.is_symlink()
    assert (base.stat().st_ino, base.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_reinstall_keeps_existing_managed_utility_links(installed_tree):
    tree, _, _ = installed_tree
    assert install(tree).returncode == 0
    utilities = [tree/'backends'/backend/UTILS
                 for backend in ('vllm', 'vllm_multimodal', 'vllm_deepseek')]
    before = [path.lstat().st_ino for path in utilities]
    result = install(tree)
    assert result.returncode == 0, result.stderr
    assert [path.lstat().st_ino for path in utilities] == before
    assert all(path.resolve() == tree/'backends/python'/UTILS for path in utilities)


@pytest.mark.parametrize('kind', ['missing', 'directory', 'dangling-link', 'foreign-link'])
def test_installer_requires_base_utility_before_any_mutation(installed_tree, kind):
    tree, original, policy = installed_tree
    utility = tree/'backends/python'/UTILS
    utility.unlink()
    if kind == 'directory':
        utility.mkdir()
    elif kind == 'dangling-link':
        utility.symlink_to('missing-utility.py')
    elif kind == 'foreign-link':
        other = tree/'other-utility.py'
        other.write_text('unrelated utility\n')
        utility.symlink_to(other)
    policy_before = policy.read_bytes()
    stubs_before = {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS}
    result = install(tree)
    assert result.returncode != 0
    assert 'utility' in result.stderr.lower()
    assert {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS} == stubs_before
    assert (tree/'backends/python'/STUB).read_bytes() == original
    assert not (tree/'backends/python'/f'{STUB}.native').exists()
    assert policy.read_bytes() == policy_before
    assert all(not (tree/'backends'/name/UTILS).is_symlink() for name in BACKENDS[1:])


@pytest.mark.parametrize('kind', ['file', 'directory', 'wrong-link', 'absolute-link'])
def test_installer_rejects_conflicting_utility_paths_before_any_mutation(installed_tree, kind):
    tree, _, policy = installed_tree
    utility = tree/'backends/vllm_multimodal'/UTILS
    if kind == 'file':
        utility.write_text('retain custom utility\n')
    elif kind == 'directory':
        utility.mkdir()
        (utility/'keep').write_text('retain directory\n')
    elif kind == 'wrong-link':
        utility.symlink_to('../../unrelated-utility.py')
    else:
        utility.symlink_to(tree/'backends/python'/UTILS)
    before = utility.lstat()
    policy_before = policy.read_bytes()
    stubs_before = {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS}
    result = install(tree)
    assert result.returncode != 0
    assert 'utility' in result.stderr.lower()
    assert utility.lstat().st_ino == before.st_ino
    assert {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS} == stubs_before
    assert policy.read_bytes() == policy_before
    assert not (tree/'backends/python'/f'{STUB}.native').exists()
    assert not (tree/'backends/vllm'/UTILS).exists()
    assert not (tree/'backends/vllm_deepseek'/UTILS).exists()


def test_installer_recovers_interruption_immediately_after_preserving_native(installed_tree):
    tree, original, policy = installed_tree
    stock = tree/'backends/python'/STUB
    native = stock.with_name(f'{STUB}.native')
    stock.rename(native)
    result = install(tree)
    assert result.returncode == 0, result.stderr
    assert native.read_bytes() == original
    assert stock.resolve() == tree/'unified'/STUB
    assert json.loads(policy.read_text())['stub']['path'] == str(native)


def test_stub_selects_embedded_python_home_and_compensates_stock_library_pop(installed_tree):
    tree, _, policy = installed_tree
    stock = tree/'backends/python'/STUB
    result = install(tree)
    assert result.returncode == 0, result.stderr
    probe = subprocess.run([sys.executable, '-I', str(stock), '--environment'],
        env={**os.environ, 'TRITON_RUNTIME_CACHE_DIR': str(tree/'cache')},
        capture_output=True, timeout=5)
    assert probe.returncode == 0, probe.stderr
    env = dict(item.split('=', 1) for item in probe.stdout.decode().split('\0') if item)
    assert env['PYTHONHOME'] == '/opt/envs/vllm032-cu129'
    assert env['PYTHONPATH'] == str(tree/'unified')
    assert env['TRITON_RUNTIME_PYTHON'] == '/opt/envs/vllm032-cu129/bin/python3.12'
    paths = env['LD_LIBRARY_PATH'].split(os.pathsep)
    assert paths[:2] == ['/opt/envs/vllm032-cu129/lib'] * 2
    assert env['DEEPSEEK_PREEXEC_NATIVE_PATH'] == hashlib.sha256(
        os.pathsep.join(paths[1:]).encode()).hexdigest()


def test_embedded_bootstrap_selects_real_python_for_spawn_without_restricting_model_imports(tmp_path):
    hook = TOOLS/'sitecustomize.py'
    assert hook.is_file(), 'the embedded Python child-process bootstrap is missing'
    python = tmp_path/'env/bin/python3.12'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    (tmp_path/'local_helper.py').write_text('MESSAGE = "generic-model-import-ok"\n')
    probe = tmp_path/'probe.py'
    probe.write_text(f'''import multiprocessing, os, runpy, sys
sys.path.insert(0, {str(tmp_path)!r})
def worker(queue):
    from local_helper import MESSAGE
    queue.put((sys.executable, MESSAGE))
if __name__ == '__main__':
    sys.executable = '/foreign/native-stub'
    runpy.run_path({str(hook)!r})
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    process = context.Process(target=worker, args=(queue,))
    process.start()
    print(queue.get(timeout=5))
    process.join(timeout=5)
    assert process.exitcode == 0
''')
    result = subprocess.run([sys.executable, '-I', str(probe)], env={**os.environ,
        'VIRTUAL_ENV': str(tmp_path/'env'), 'TRITON_RUNTIME_PYTHON': str(python)},
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert str(python) in result.stdout
    assert 'generic-model-import-ok' in result.stdout


@pytest.mark.parametrize('failure', ['hash', 'script', 'native-symlink'])
def test_installer_rejects_untrusted_native_stub_before_replacing_any_backend(installed_tree, failure):
    tree, _, policy = installed_tree
    stock = tree/'backends/python'/STUB
    if failure == 'hash':
        value = json.loads(policy.read_text())
        value['stub']['sha256'] = '0' * 64
        policy.write_text(json.dumps(value))
    elif failure == 'script':
        stock.write_text('#!/bin/sh\nexit 0\n')
    else:
        (tree/'backends/python'/f'{STUB}.native').symlink_to(stock)
    before = {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS}
    original_policy = policy.read_bytes()
    result = install(tree)
    assert result.returncode != 0
    assert {name: (tree/'backends'/name/STUB).read_bytes() for name in BACKENDS} == before
    assert all(not (tree/'backends'/name/STUB).is_symlink() for name in BACKENDS)
    assert policy.read_bytes() == original_policy
