import contextlib
import os
from pathlib import Path
import signal
import subprocess
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


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
if first and os.environ["TEST_MODE"] == "hung":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True:
    time.sleep(0.05)
''')
    curl = binary / "curl"
    curl.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
root = pathlib.Path(os.environ["TEST_STATE"])
assert "--max-time" in sys.argv
if sys.argv[-1].endswith("/v2/health/live"):
    raise SystemExit(1 if (root / "triton-down").exists() else 0)
assert sys.argv[-1].startswith("http://127.0.0.1:")
assert sys.argv[-1].endswith("/health")
count_file = root / "probes"
count = int(count_file.read_text()) + 1 if count_file.exists() else 1
count_file.write_text(str(count))
mode = os.environ["TEST_MODE"]
starts = root / "starts"
first = not starts.exists() or len(starts.read_text().splitlines()) < 2
failed = (mode == "hung" and first) or (mode == "transient" and count == 1)
raise SystemExit(1 if failed else 0)
''')
    for script in (uvicorn, curl):
        script.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}",
           "TEST_STATE": str(tmp_path), "TEST_MODE": mode,
           "GATEWAY_HEALTH_INTERVAL_SECONDS": "1", "GATEWAY_HEALTH_TIMEOUT_SECONDS": "1",
           "GATEWAY_HEALTH_FAILURE_THRESHOLD": threshold,
           "GATEWAY_STARTUP_GRACE_SECONDS": grace, "GATEWAY_STOP_GRACE_SECONDS": "1",
           "GATEWAY_RESTART_DELAY_SECONDS": "1", "TRITON_BASE_URL": "http://triton.test"}
    process = subprocess.Popen(
        ["bash", "-c", 'source "$1"; wait "$gateway_supervisor_pid"', "bash",
         str(ROOT / "docker" / "90-triton-chat-gateway.sh")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,
    )
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
