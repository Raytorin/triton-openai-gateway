import contextlib
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "docker" / "triton-gateway"


@contextlib.contextmanager
def supervisor(tmp_path, *, mode="healthy", grace="0", threshold="2"):
    binary = tmp_path / "bin"
    binary.mkdir()
    uvicorn = binary / "uvicorn"
    uvicorn.write_text('''#!/usr/bin/env python3
import os, pathlib, signal, time
root = pathlib.Path(os.environ["TEST_STATE"])
starts = root / "starts"
first = not starts.exists()
with starts.open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")
if first and os.environ["TEST_MODE"] == "crash":
    raise SystemExit(1)
if first and os.environ["TEST_MODE"] in {"hung", "manual-hung"}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(0.05)
''')
    curl = binary / "curl"
    curl.write_text('''#!/usr/bin/env python3
import fcntl, os, pathlib, sys
root = pathlib.Path(os.environ["TEST_STATE"])
assert "--max-time" in sys.argv
if sys.argv[-1].endswith("/v2/health/live"):
    raise SystemExit(1 if (root / "triton-down").exists() else 0)
if sys.argv[-1].endswith("/v2/health/ready"):
    raise SystemExit(1 if (root / "triton-not-ready").exists() else 0)
assert sys.argv[-1].startswith("http://127.0.0.1:")
assert sys.argv[-1].endswith("/health")
count_file = root / "probes"
with (root / "probes.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    temporary = root / "probes.next"
    temporary.write_text(str(count))
    temporary.replace(count_file)
mode = os.environ["TEST_MODE"]
starts = root / "starts"
first = not starts.exists() or len(starts.read_text().splitlines()) < 2
failed = ((mode == "hung" and first) or (mode == "transient" and count == 1)
          or (root / "health-down").exists())
raise SystemExit(1 if failed else 0)
''')
    for script in (uvicorn, curl):
        script.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
           "TEST_STATE": str(tmp_path), "TEST_MODE": mode,
           "GATEWAY_HEALTH_INTERVAL_SECONDS": "1", "GATEWAY_HEALTH_TIMEOUT_SECONDS": "1",
           "GATEWAY_HEALTH_FAILURE_THRESHOLD": threshold,
           "GATEWAY_STARTUP_GRACE_SECONDS": grace, "GATEWAY_STOP_GRACE_SECONDS": "1",
           "GATEWAY_RESTART_DELAY_SECONDS": "1", "TRITON_BASE_URL": "http://triton.test",
           "GATEWAY_RUN_DIR": str(tmp_path / "run")}
    process = subprocess.Popen(
        ["bash", "-c", 'source "$1"; wait "$gateway_supervisor_pid"', "bash",
         str(ROOT / "docker" / "90-triton-chat-gateway.sh")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
    process.test_env = env
    try:
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=4)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=3)


def wait_for(predicate, process, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f"supervisor exited early: {process.communicate()[0]}")
        time.sleep(0.05)
    pytest.fail("watchdog did not reach expected state")


def starts(root):
    path = root / "starts"
    return path.read_text().splitlines() if path.exists() else []


def probes(root):
    path = root / "probes"
    return int(path.read_text() or "0") if path.exists() else 0


def control(process, *args, timeout=15):
    return subprocess.run(
        [sys.executable, str(CONTROL), *args], env=process.test_env,
        capture_output=True, text=True, timeout=timeout,
    )


def output(result):
    return (result.stdout + result.stderr).lower()


def test_unresponsive_gateway_is_killed_and_restarted(tmp_path):
    with supervisor(tmp_path, mode="hung") as process:
        wait_for(lambda: len(starts(tmp_path)) == 2, process)
        old_pid = int(starts(tmp_path)[0])
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)
        wait_for(lambda: probes(tmp_path) >= 4, process)
        assert len(starts(tmp_path)) == 2


def test_crashed_gateway_is_restarted(tmp_path):
    with supervisor(tmp_path, mode="crash") as process:
        wait_for(lambda: len(starts(tmp_path)) == 2, process)
        wait_for(lambda: probes(tmp_path) >= 2, process)
        assert len(starts(tmp_path)) == 2


def test_single_failed_probe_does_not_restart_gateway(tmp_path):
    with supervisor(tmp_path, mode="transient") as process:
        wait_for(lambda: probes(tmp_path) >= 3, process)
        assert len(starts(tmp_path)) == 1


def test_startup_grace_allows_slow_gateway(tmp_path):
    with supervisor(tmp_path, mode="hung", grace="30", threshold="1") as process:
        wait_for(lambda: probes(tmp_path) >= 3, process)
        assert len(starts(tmp_path)) == 1


def test_triton_shutdown_stops_gateway_without_restart(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        (tmp_path / "triton-down").touch()
        output, _ = process.communicate(timeout=8)
        assert process.returncode == 0, output
        assert "Triton stopped responding" in output
        assert len(starts(tmp_path)) == 1
        with pytest.raises(ProcessLookupError):
            os.kill(int(starts(tmp_path)[0]), 0)


def test_gateway_waits_for_initial_triton_startup(tmp_path):
    (tmp_path / "triton-down").touch()
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 3, process)
        assert len(starts(tmp_path)) == 1
        (tmp_path / "triton-down").unlink()
        previous = probes(tmp_path)
        wait_for(lambda: probes(tmp_path) >= previous + 2, process)
        assert len(starts(tmp_path)) == 1


def test_status_reports_healthy_gateway_and_triton(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        result = control(process, "status")
        assert result.returncode == 0, output(result)
        assert "supervisor" in output(result)
        assert "gateway" in output(result)
        assert "triton" in output(result)
        assert len(starts(tmp_path)) == 1


def test_status_distinguishes_triton_readiness_from_gateway_health(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        (tmp_path / "triton-not-ready").touch()
        result = control(process, "status")
        assert result.returncode == 0, output(result)
        assert "ready" in output(result)
        assert "triton" in output(result)
        assert len(starts(tmp_path)) == 1


def test_manual_restart_replaces_only_gateway_when_health_still_responds(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        old_pid = int(starts(tmp_path)[0])
        result = control(process, "restart", "--timeout", "10")
        assert result.returncode == 0, output(result)
        assert len(starts(tmp_path)) == 2
        assert starts(tmp_path)[1] != str(old_pid)
        assert process.poll() is None
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)
        assert control(process, "status").returncode == 0


def test_manual_restart_kills_gateway_ignoring_sigterm(tmp_path):
    with supervisor(tmp_path, mode="manual-hung") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        old_pid = int(starts(tmp_path)[0])
        result = control(process, "restart", "--timeout", "10")
        assert result.returncode == 0, output(result)
        assert len(starts(tmp_path)) == 2
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)


def test_simultaneous_restarts_share_one_replacement(tmp_path):
    with supervisor(tmp_path, mode="manual-hung") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        callers = [
            subprocess.Popen(
                [sys.executable, str(CONTROL), "restart", "--timeout", "10"],
                env=process.test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            for _ in range(3)
        ]
        results = [caller.communicate(timeout=15)[0] for caller in callers]
        assert all(caller.returncode == 0 for caller in callers), results
        previous = probes(tmp_path)
        wait_for(lambda: probes(tmp_path) >= previous + 2, process)
        assert len(starts(tmp_path)) == 2


def test_duplicate_hook_does_not_start_second_supervisor(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        duplicate = subprocess.run(
            ["bash", "-c", 'source "$1"', "bash",
             str(ROOT / "docker" / "90-triton-chat-gateway.sh")],
            env=process.test_env, capture_output=True, text=True, timeout=5,
        )
        assert duplicate.returncode == 0, duplicate.stdout + duplicate.stderr
        previous = probes(tmp_path)
        wait_for(lambda: probes(tmp_path) >= previous + 2, process)
        assert len(starts(tmp_path)) == 1


def test_restart_times_out_until_new_gateway_is_healthy(tmp_path):
    with supervisor(tmp_path, grace="30", threshold="100") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        (tmp_path / "health-down").touch()
        result = control(process, "restart", "--timeout", "4")
        assert result.returncode != 0
        assert "tim" in output(result), output(result)
        wait_for(lambda: len(starts(tmp_path)) == 2, process)
        assert control(process, "status").returncode != 0
        (tmp_path / "health-down").unlink()
        wait_for(lambda: control(process, "status").returncode == 0, process)


@pytest.mark.parametrize("command", ["status", "restart"])
def test_control_without_supervisor_fails_without_starting_gateway(tmp_path, command):
    result = subprocess.run(
        [sys.executable, str(CONTROL), command],
        env={**os.environ, "GATEWAY_RUN_DIR": str(tmp_path / "run")},
        capture_output=True, text=True, timeout=3,
    )
    assert result.returncode != 0
    assert "supervisor" in output(result), output(result)
    assert not (tmp_path / "starts").exists()


def test_status_fails_when_gateway_health_is_unresponsive(tmp_path):
    with supervisor(tmp_path, threshold="100") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        (tmp_path / "health-down").touch()
        result = control(process, "status")
        assert result.returncode != 0
        assert "gateway" in output(result), output(result)
        assert "health" in output(result), output(result)
        assert len(starts(tmp_path)) == 1


@pytest.mark.parametrize("command", ["status", "restart"])
def test_stale_pid_record_does_not_signal_unrelated_process(tmp_path, command):
    run_dir = tmp_path / "run"
    run_dir.mkdir(mode=0o700)
    # Reused PID: a real live process, but not the recorded supervisor generation.
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        state = run_dir / "state"
        state.write_text(
            f"supervisor_pid={unrelated.pid}\nsupervisor_start=0\n"
            "gateway_pid=0\ngateway_start=0\ngeneration=1\nphase=running\n"
            "completed_request=\ngateway_port=8080\ntriton_url=http://triton.test\n"
        )
        state.chmod(0o600)
        result = subprocess.run(
            [sys.executable, str(CONTROL), command],
            env={**os.environ, "GATEWAY_RUN_DIR": str(run_dir)},
            capture_output=True, text=True, timeout=3,
        )
        assert result.returncode != 0
        assert "supervisor" in output(result), output(result)
        assert unrelated.poll() is None
        assert not (run_dir / "restart-request").exists()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_triton_shutdown_during_manual_restart_prevents_replacement(tmp_path):
    with supervisor(tmp_path, mode="manual-hung") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        old_pid = int(starts(tmp_path)[0])
        caller = subprocess.Popen(
            [sys.executable, str(CONTROL), "restart", "--timeout", "10"],
            env=process.test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            def old_gateway_stopped():
                try:
                    os.kill(old_pid, 0)
                except ProcessLookupError:
                    return True
                return False

            wait_for(old_gateway_stopped, process)
            (tmp_path / "triton-down").touch()
            result = caller.communicate(timeout=8)[0]
            assert caller.returncode != 0, result
            process.communicate(timeout=8)
            assert process.returncode == 0
            assert len(starts(tmp_path)) == 1
        finally:
            if caller.poll() is None:
                caller.terminate()
                caller.communicate(timeout=3)


@pytest.mark.parametrize("directory_kind", ["public", "symlink"])
def test_control_rejects_unsafe_runtime_directory(tmp_path, directory_kind):
    run_dir = tmp_path / "run"
    if directory_kind == "public":
        run_dir.mkdir()
        run_dir.chmod(0o777)
    else:
        target = tmp_path / "target"
        target.mkdir(mode=0o700)
        run_dir.symlink_to(target, target_is_directory=True)
    result = subprocess.run(
        [sys.executable, str(CONTROL), "restart"],
        env={**os.environ, "GATEWAY_RUN_DIR": str(run_dir)},
        capture_output=True, text=True, timeout=3,
    )
    assert result.returncode != 0
    assert any(word in output(result) for word in ("unsafe", "private", "permission")), output(result)
    assert not (run_dir / "restart-request").exists()


def test_supervisor_term_during_manual_restart_prevents_replacement(tmp_path):
    with supervisor(tmp_path, mode="manual-hung") as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        state_path = tmp_path / "run" / "state"
        assert state_path.exists(), "supervisor did not publish its control state"
        state = dict(
            line.split("=", 1)
            for line in state_path.read_text().splitlines()
        )
        supervisor_pid = int(state["supervisor_pid"])
        caller = subprocess.Popen(
            [sys.executable, str(CONTROL), "restart", "--timeout", "10"],
            env=process.test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for(lambda: (tmp_path / "run" / "restart-request").exists(), process)
            os.kill(supervisor_pid, signal.SIGTERM)
            result = caller.communicate(timeout=8)[0]
            assert caller.returncode != 0, result
            process.communicate(timeout=8)
            assert process.returncode == 0
            assert len(starts(tmp_path)) == 1
            with pytest.raises(ProcessLookupError):
                os.kill(int(starts(tmp_path)[0]), 0)
        finally:
            if caller.poll() is None:
                caller.terminate()
                caller.communicate(timeout=3)


def test_supervisor_term_stops_gateway_after_runtime_directory_disappears(tmp_path):
    with supervisor(tmp_path) as process:
        wait_for(lambda: probes(tmp_path) >= 1, process)
        state = dict(
            line.split("=", 1)
            for line in (tmp_path / "run" / "state").read_text().splitlines()
        )
        supervisor_pid = int(state["supervisor_pid"])
        gateway_pid = int(starts(tmp_path)[0])

        def gateway_running():
            try:
                record = Path(f"/proc/{gateway_pid}/stat").read_text()
            except FileNotFoundError:
                return False
            return record.rsplit(") ", 1)[1].split()[0] not in {"Z", "X"}

        try:
            shutil.rmtree(tmp_path / "run")
            os.kill(supervisor_pid, signal.SIGTERM)
            deadline = time.monotonic() + 4
            while gateway_running() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not gateway_running(), "supervisor left its gateway process running"
        finally:
            # An orphan keeps the capture pipe open, so clean it before the fixture
            # calls communicate() even when the supervisor has already exited.
            if gateway_running():
                os.kill(gateway_pid, signal.SIGKILL)


def test_manual_restart_joins_automatic_replacement_awaiting_health(tmp_path):
    (tmp_path / "health-down").touch()
    with supervisor(tmp_path, mode="crash", grace="30", threshold="1") as process:
        wait_for(lambda: len(starts(tmp_path)) == 2, process)
        second_pid = starts(tmp_path)[1]

        def replacement_starting():
            state = dict(
                line.split("=", 1)
                for line in (tmp_path / "run" / "state").read_text().splitlines()
            )
            return state["generation"] == "2" and state["phase"] == "starting"

        wait_for(replacement_starting, process)
        caller = subprocess.Popen(
            [sys.executable, str(CONTROL), "restart", "--timeout", "10"],
            env=process.test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for(lambda: (tmp_path / "run" / "restart-request").exists(), process)
            (tmp_path / "health-down").unlink()
            result = caller.communicate(timeout=12)[0]
            assert caller.returncode == 0, result
            previous = probes(tmp_path)
            wait_for(lambda: probes(tmp_path) >= previous + 2, process)
            assert len(starts(tmp_path)) == 2
            assert starts(tmp_path)[1] == second_pid
        finally:
            if caller.poll() is None:
                caller.terminate()
                caller.communicate(timeout=3)
