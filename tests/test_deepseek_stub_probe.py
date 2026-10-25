"""Probe exit handling is testable without falsely qualifying a native stub."""
import os
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('exit_code,message,expected',[(1,'DeepSeek-preflight-probe config.json',0),(0,'DeepSeek-preflight-probe config.json',1),(139,'DeepSeek-preflight-probe config.json segmentation fault',1),(1,'unrelated loader failure',1)])
def test_probe_requires_intended_preflight_failure_and_clean_process_exit(tmp_path,exit_code,message,expected):
    executable=tmp_path/'tritonserver'
    executable.write_text(f'#!/bin/sh\necho "{message}"\nexit {exit_code}\n');executable.chmod(0o755)
    result=subprocess.run([sys.executable,str(ROOT/'docker/smoke-deepseek-stub.py'),'--tritonserver',str(executable)],text=True,capture_output=True,timeout=5)
    assert result.returncode==expected,result.stderr


def write_python_executable(path, source):
    path.write_text(f'#!{sys.executable}\n' + source)
    path.chmod(0o755)
    return path


def run_probe(executable, *arguments):
    return subprocess.run(
        [sys.executable, str(ROOT/'docker/smoke-deepseek-stub.py'),
         '--tritonserver', str(executable), *arguments],
        text=True, capture_output=True, timeout=8,
    )


@pytest.mark.parametrize('timeout', [False, True])
def test_probe_preserves_bounded_first_cause_and_log_tail(tmp_path, timeout):
    executable = write_python_executable(tmp_path/'tritonserver',
        'import sys, time\n'
        'print("FIRST_CAUSE: native bootstrap failed", flush=True)\n'
        'print("x" * 200000, flush=True)\n'
        'print("FINAL_DIAGNOSTIC", flush=True)\n'
        + ('time.sleep(60)\n' if timeout else 'sys.exit(2)\n'))
    result = run_probe(executable, '--timeout', '0.3')
    assert result.returncode == 1
    assert 'FIRST_CAUSE: native bootstrap failed' in result.stderr
    assert 'FINAL_DIAGNOSTIC' in result.stderr
    assert 'omitted' in result.stderr.lower()
    assert len(result.stderr.encode()) < 68000


def test_probe_reports_launch_oserror(tmp_path):
    result = run_probe(tmp_path/'missing-tritonserver')
    assert result.returncode == 1
    assert 'No such file or directory' in result.stderr
    assert 'Triton log' in result.stderr


def process_is_running(pid):
    try:
        # A killed orphan may await reaping by the host's PID 1.
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except FileNotFoundError:
        return False


@pytest.mark.skipif(os.name != 'posix' or not Path('/proc').exists(), reason='POSIX process groups and procfs')
@pytest.mark.parametrize('interrupt', [False, True])
def test_probe_cleans_its_process_group_without_touching_unrelated_process(tmp_path, interrupt):
    import signal
    import time

    pid_file = tmp_path/'owned-pids'
    executable = write_python_executable(tmp_path/'tritonserver',
        'import os, signal, time\n'
        'signal.signal(signal.SIGTERM, signal.SIG_IGN)\n'
        'signal.signal(signal.SIGINT, signal.SIG_IGN)\n'
        'child = os.fork()\n'
        'if child == 0:\n'
        '    while True: time.sleep(1)\n'
        'print("FIRST_CAUSE: waiting for native exit", flush=True)\n'
        f'open({str(pid_file)+".ready"!r}, "w").write(f"{{os.getpid()}} {{child}}")\n'
        f'os.replace({str(pid_file)+".ready"!r}, {str(pid_file)!r})\n'
        'while True: time.sleep(1)\n')
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    probe = subprocess.Popen(
        [sys.executable, str(ROOT/'docker/smoke-deepseek-stub.py'),
         '--tritonserver', str(executable), '--timeout', '5' if interrupt else '0.5'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    owned = []
    try:
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists(), 'controlled executable did not start'
        owned = [int(value) for value in pid_file.read_text().split()]
        if interrupt:
            probe.send_signal(signal.SIGINT)
        _, stderr = probe.communicate(timeout=5)
        assert probe.returncode == 1
        deadline = time.monotonic() + 1
        while any(process_is_running(pid) for pid in owned) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not any(process_is_running(pid) for pid in owned), f'owned processes remain: {owned}'
        assert unrelated.poll() is None
        assert 'FIRST_CAUSE: waiting for native exit' in stderr
    finally:
        if probe.poll() is None:
            probe.kill()
            probe.communicate()
        for pid in owned:
            if process_is_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_probe_bounds_each_log_read_and_keeps_both_ends(tmp_path, monkeypatch, capsys):
    import contextlib
    import runpy
    import tempfile

    executable = write_python_executable(tmp_path/'tritonserver',
        'print("FIRST_CAUSE", flush=True)\n'
        'print("x" * 1000000, flush=True)\n'
        'print("FINAL_DIAGNOSTIC", flush=True)\n'
        'raise SystemExit(2)\n')
    probe = runpy.run_path(str(ROOT/'docker/smoke-deepseek-stub.py'))
    real_temporary_file = tempfile.TemporaryFile

    @contextlib.contextmanager
    def guarded_temporary_file():
        with real_temporary_file() as log:
            class GuardedLog:
                def __getattr__(self, name):
                    return getattr(log, name)

                def read(self, size=-1):
                    assert 0 <= size <= 32768, 'probe attempted an unbounded log read'
                    return log.read(size)
            yield GuardedLog()

    monkeypatch.setattr(tempfile, 'TemporaryFile', guarded_temporary_file)
    monkeypatch.setattr(sys, 'argv', ['probe', '--tritonserver', str(executable)])
    assert probe['main']() == 1
    stderr = capsys.readouterr().err
    assert 'FIRST_CAUSE' in stderr
    assert 'FINAL_DIAGNOSTIC' in stderr
    assert len(stderr.encode()) < 68000


def test_probe_preserves_logs_when_wait_raises_oserror(tmp_path, monkeypatch, capsys):
    import runpy
    import time

    executable = write_python_executable(tmp_path/'tritonserver',
        'import time\n'
        'print("FIRST_CAUSE: before wait failure", flush=True)\n'
        'time.sleep(0.2)\n'
        'raise SystemExit(2)\n')
    probe = runpy.run_path(str(ROOT/'docker/smoke-deepseek-stub.py'))
    real_popen = subprocess.Popen

    def popen_with_wait_error(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        real_wait = process.wait
        fail_once = True

        def wait(timeout=None):
            nonlocal fail_once
            if fail_once:
                fail_once = False
                deadline = time.monotonic() + 2
                while os.fstat(kwargs['stdout'].fileno()).st_size == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                raise OSError('controlled wait failure')
            return real_wait(timeout=timeout)

        process.wait = wait
        return process

    monkeypatch.setattr(subprocess, 'Popen', popen_with_wait_error)
    monkeypatch.setattr(sys, 'argv', ['probe', '--tritonserver', str(executable)])
    assert probe['main']() == 1
    stderr = capsys.readouterr().err
    assert 'controlled wait failure' in stderr
    assert 'FIRST_CAUSE: before wait failure' in stderr


@pytest.mark.parametrize('marker', ['SIGSEGV', 'sEgMeNtAtIoN fAuLt'])
@pytest.mark.parametrize('offset', [40000, 65533])
def test_probe_rejects_crash_markers_outside_excerpt_and_across_scan_chunks(tmp_path, marker, offset):
    prefix = 'DeepSeek-preflight-probe config.json\n'
    payload = prefix + 'x' * (offset - len(prefix)) + marker + 'y' * 40000
    executable = write_python_executable(tmp_path/'tritonserver',
        f'import sys\nsys.stdout.write({payload!r})\nraise SystemExit(1)\n')
    result = run_probe(executable)
    assert result.returncode == 1
    assert 'reported' in result.stderr.lower() and 'sigsegv' in result.stderr.lower()
    assert 'omitted' in result.stderr
    assert len(result.stderr.encode()) < 68000
    assert 'Native no-weight preflight failure surfaced' not in result.stdout


@pytest.mark.parametrize('marker', ['DeepSeek-preflight-probe', 'config.json'])
def test_probe_accepts_complete_clean_log_with_expected_marker_across_scan_boundary(tmp_path, marker):
    other = 'config.json' if marker == 'DeepSeek-preflight-probe' else 'DeepSeek-preflight-probe'
    prefix = other + '\n'
    payload = prefix + 'x' * (65533 - len(prefix)) + marker + 'y' * 40000
    executable = write_python_executable(tmp_path/'tritonserver',
        f'import sys\nsys.stdout.write({payload!r})\nraise SystemExit(1)\n')
    result = run_probe(executable)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('fault', ['read-error', 'short-read', 'truncate', 'grow'])
@pytest.mark.parametrize('phase', ['excerpt', 'scan'])
def test_probe_fails_closed_when_complete_log_inspection_is_unreliable(tmp_path, monkeypatch, capsys, fault, phase):
    import contextlib
    import runpy
    import tempfile

    # The visible excerpt has every success identifier; only reliable full
    # inspection can justify acceptance of this otherwise successful fixture.
    payload = 'DeepSeek-preflight-probe config.json\n' + 'x' * 100000
    executable = write_python_executable(tmp_path/'tritonserver',
        f'import sys\nsys.stdout.write({payload!r})\nraise SystemExit(1)\n')
    probe = runpy.run_path(str(ROOT/'docker/smoke-deepseek-stub.py'))
    real_temporary_file = tempfile.TemporaryFile

    @contextlib.contextmanager
    def faulty_temporary_file():
        with real_temporary_file() as log:
            class FaultyLog:
                injected = False
                reads = 0

                def __getattr__(self, name):
                    return getattr(log, name)

                def read(self, size=-1):
                    assert 0 <= size <= 32768, 'probe attempted an unbounded log read'
                    self.reads += 1
                    if not self.injected and (phase == 'excerpt' or self.reads >= 3):
                        self.injected = True
                        if fault == 'read-error':
                            raise OSError('controlled log read failure')
                        if fault == 'short-read':
                            return log.read(size - 1)
                        if fault == 'truncate':
                            log.truncate(50)
                        if fault == 'grow':
                            os.pwrite(log.fileno(), b'new bytes', os.fstat(log.fileno()).st_size)
                    return log.read(size)
            yield FaultyLog()

    monkeypatch.setattr(tempfile, 'TemporaryFile', faulty_temporary_file)
    monkeypatch.setattr(sys, 'argv', ['probe', '--tritonserver', str(executable)])
    assert probe['main']() == 1
    captured = capsys.readouterr()
    assert 'log inspection' in captured.err.lower()
    assert 'Native no-weight preflight failure surfaced' not in captured.out


def test_probe_fails_closed_for_log_beyond_complete_inspection_size_limit(tmp_path):
    executable = write_python_executable(tmp_path/'tritonserver',
        'import os, sys\n'
        'print("DeepSeek-preflight-probe config.json", flush=True)\n'
        'os.ftruncate(1, 64 * 1024 * 1024 + 1)\n'
        'raise SystemExit(1)\n')
    result = run_probe(executable)
    assert result.returncode == 1
    assert '64 MiB' in result.stderr
    assert 'log inspection' in result.stderr.lower()


def test_probe_fails_closed_if_final_scan_chunk_exceeds_time_budget(tmp_path, monkeypatch, capsys):
    import runpy
    import time

    executable = write_python_executable(tmp_path/'tritonserver',
        'print("DeepSeek-preflight-probe config.json", flush=True)\n'
        'raise SystemExit(1)\n')
    probe = runpy.run_path(str(ROOT/'docker/smoke-deepseek-stub.py'))
    ticks = iter([0, 0, 6])
    monkeypatch.setattr(time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(sys, 'argv', ['probe', '--tritonserver', str(executable)])
    assert probe['main']() == 1
    assert '5 second limit' in capsys.readouterr().err
