# SPDX-License-Identifier: Apache-2.0
"""Fail-closed native smoke validation and bounded cleanup without GPU/model weights."""
from copy import deepcopy
from importlib import util
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).with_name("runtime_unified_smoke.py")
RUNTIME = Path("/opt/envs/vllm032-cu129")
CACHE = Path("/tmp/unified-smoke/cache")
STUB = "/opt/tritonserver/backends/python/triton_python_backend_stub.native"


def smoke():
    assert SCRIPT.is_file(), "repeatable real native Triton smoke is required"
    spec = util.spec_from_file_location("runtime_unified_smoke", SCRIPT)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_report():
    report = {
        "model": "unified_probe", "echo": "contract-test", "pid": 4321,
        "instance": "fresh-instance", "prefix": "/opt/envs/vllm032-cu129",
        "python": "3.12.15", "executable": STUB,
        "sys_executable": "/opt/envs/vllm032-cu129/bin/python3.12",
        "module_versions": {"torch": "2.13.0+cu129", "vllm": "0.32.0",
                            "transformers": "5.19.0", "numpy": "2.3.5"},
        "native_argv": [STUB, "/tmp/models/unified_probe/1/model.py", "shared-memory"],
        "versions": {"torch": "2.13.0+cu129", "vllm": "0.32.0+cu129",
                     "transformers": "5.19.0", "numpy": "2.3.5"},
        "modules": {
            "torch": "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/__init__.py",
            "vllm": "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/vllm/__init__.py",
            "transformers": "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/transformers/__init__.py",
            "numpy": "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/numpy/__init__.py",
        },
        "engine_origins": {"torch._C": "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/_C.so"},
        "native_libraries": ["/opt/envs/vllm032-cu129/lib/libpython3.12.so.1.0",
                             "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"],
        "sys_path": ["/opt/envs/vllm032-cu129/lib/python3.12/site-packages"],
        "pb_utils_tensor_module": "c_python_backend_utils", "cuda_initialized": False,
        "torch_cuda": "12.9", "cache_arch": "cpu", "cache_root": "/tmp/unified-smoke/cache",
        "caches": {"VLLM_CACHE_ROOT": "/tmp/unified-smoke/cache/cpu/vllm",
                   "TORCH_HOME": "/tmp/unified-smoke/cache/cpu/torch",
                   "TRITON_CACHE_DIR": "/tmp/unified-smoke/cache/cpu/triton",
                   "HF_MODULES_CACHE": "/tmp/unified-smoke/cache/cpu/hf-modules",
                   "PYTHONPYCACHEPREFIX": "/tmp/unified-smoke/cache/cpu/python-pycache"},
        "pythonhome": "/opt/envs/vllm032-cu129", "virtual_env": "/opt/envs/vllm032-cu129",
        "worker_method": "spawn", "downloads_disabled": "1",
    }
    child = {key: deepcopy(report[key]) for key in ("prefix", "python", "versions",
             "module_versions", "modules", "engine_origins", "native_libraries", "sys_path",
             "cuda_initialized", "torch_cuda", "cache_arch", "cache_root", "caches",
             "pythonhome", "virtual_env", "worker_method", "downloads_disabled")}
    child.update(pid=4322, parent_pid=4321, executable="/opt/envs/vllm032-cu129/bin/python3.12",
                 sys_executable="/opt/envs/vllm032-cu129/bin/python3.12",
                 start_method="spawn", native_backend_module=False)
    report["child"] = child
    return report


def validate(report):
    return smoke().validate_report(report, RUNTIME, CACHE, "unified_probe", "contract-test")


def test_native_report_accepts_consistent_actual_runtime_evidence():
    validate(valid_report())


@pytest.mark.parametrize("field,value", [
    ("prefix", "/usr/local"), ("pythonhome", "/foreign/python"),
    ("pb_utils_tensor_module", "mock_backend"), ("cuda_initialized", True),
    ("echo", "stale-request"), ("cache_arch", "sm90"),
    ("cache_root", "/opt/envs/vllm032-cu129/cache"),
])
def test_native_report_rejects_wrong_runtime_ipc_or_request(field, value):
    report = valid_report()
    report[field] = value
    with pytest.raises(RuntimeError):
        validate(report)


@pytest.mark.parametrize("field", ["modules", "engine_origins"])
def test_native_report_rejects_foreign_engine_imports(field):
    report = valid_report()
    report[field]["torch"] = "/usr/local/lib/python3.12/dist-packages/torch/__init__.py"
    with pytest.raises(RuntimeError, match="origin"):
        validate(report)


@pytest.mark.parametrize("change", ["libpython", "sys_path", "cache", "torch_version", "native_argv"])
def test_native_report_rejects_loader_cache_and_abi_mismatches(change):
    report = deepcopy(valid_report())
    if change == "libpython":
        report["native_libraries"][0] = "/usr/lib/libpython3.12.so.1.0"
    elif change == "sys_path":
        report["sys_path"].append("/usr/local/lib/python3.12/dist-packages")
    elif change == "cache":
        report["caches"]["TORCH_HOME"] = "/opt/envs/vllm032-cu129/cache"
    elif change == "torch_version":
        report["versions"]["torch"] = "2.14.0a0"
    else:
        report["native_argv"] = ["/wrong/stub"]
    with pytest.raises(RuntimeError):
        validate(report)


def test_model_readiness_wait_has_a_deadline():
    module = smoke()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="probe model"):
            module.wait_for(lambda budget: False, process, 0.03, "probe model")
        assert time.monotonic() - started < 1
    finally:
        module.stop_server(process, 0.2)


def test_model_readiness_stops_when_server_has_exited():
    module = smoke()
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], start_new_session=True)
    process.wait(timeout=3)
    with pytest.raises(RuntimeError, match="exited.*3"):
        module.wait_for(lambda budget: False, process, 10, "probe model")


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").exists(), reason="POSIX groups and procfs")
def test_cleanup_kills_stubborn_owned_child_without_touching_other_process(tmp_path):
    module = smoke()
    ready = tmp_path / "child-pid"
    source = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    while True: time.sleep(1)\n"
        f"open({str(ready)!r}, 'w').write(str(child))\n"
        "while True: time.sleep(1)\n"
    )
    owned = subprocess.Popen([sys.executable, "-c", source], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    child = None
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "controlled child did not start"
        child = int(ready.read_text())
        result = module.stop_server(owned, 0.05)
        assert result["forced"]
        assert owned.poll() is not None
        deadline = time.monotonic() + 1
        while Path(f"/proc/{child}/stat").exists() and time.monotonic() < deadline:
            if Path(f"/proc/{child}/stat").read_text().split(") ", 1)[1][0] == "Z":
                break
            time.sleep(0.01)
        if Path(f"/proc/{child}/stat").exists():
            assert Path(f"/proc/{child}/stat").read_text().split(") ", 1)[1][0] == "Z"
        assert unrelated.poll() is None
    finally:
        if owned.poll() is None:
            os.killpg(owned.pid, 9)
            owned.wait(timeout=3)
        unrelated.terminate()
        unrelated.wait(timeout=3)


@pytest.mark.parametrize("index,path", [
    (0, "/opt/cuda/12.9/lib64/libpython3.12.so.1.0"),
    (1, "/opt/cuda/12.9/lib64/libtorch_cpu.so"),
])
def test_engine_libraries_must_come_from_python_runtime_not_toolkit(index, path):
    report = valid_report()
    report["native_libraries"][index] = path
    with pytest.raises(RuntimeError, match="native runtime library"):
        validate(report)


def test_client_close_failure_still_reaps_server():
    module = smoke()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)

    class ClosingClient:
        def close(self):
            raise OSError("controlled gRPC close failure")

    try:
        with pytest.raises(OSError, match="controlled gRPC close failure"):
            module.close_client_and_server(ClosingClient(), process, 0.2)
        assert process.poll() is not None
    finally:
        if process.poll() is None:
            module.stop_server(process, 0.2)


def test_embedded_model_reports_selected_executable_for_multiprocessing():
    report = valid_report()
    report["sys_executable"] = STUB
    with pytest.raises(RuntimeError, match="sys_executable"):
        validate(report)


@pytest.mark.parametrize("field,value", [
    ("prefix", "/usr/local"), ("sys_executable", STUB), ("executable", STUB),
    ("parent_pid", 9999), ("start_method", "fork"), ("native_backend_module", True),
])
def test_embedded_spawn_child_must_be_fresh_selected_python(field, value):
    report = valid_report()
    report["child"][field] = value
    with pytest.raises(RuntimeError, match="child"):
        validate(report)


def test_spawned_engine_rejects_core_stub_abi_exception():
    report = valid_report()
    report["child"]["native_libraries"].append(
        "/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13.4.49")
    with pytest.raises(RuntimeError, match="child.*native"):
        validate(report)


def make_core_policy(tmp_path, module, monkeypatch):
    stub = tmp_path / "triton_python_backend_stub.native"
    library = tmp_path / "cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13.4.49"
    library.parent.mkdir(parents=True)
    stub.write_bytes(b"reviewed-native-stub")
    library.write_bytes(b"reviewed-native-runtime")
    monkeypatch.setattr(module, "STUB", stub)
    monkeypatch.setattr(module, "CORE_CUDART", library, raising=False)
    policy = {
        "schema": 1,
        "stub": {"path": str(stub), "sha256": hashlib.sha256(stub.read_bytes()).hexdigest()},
        "libraries": [{"path": str(library), "soname": "libcudart.so.13",
                       "sha256": hashlib.sha256(library.read_bytes()).hexdigest()}],
    }
    path = tmp_path / "core-native-libraries.json"
    path.write_text(json.dumps(policy))
    return stub, library, path, policy


def test_only_hash_verified_core_cudart_is_accepted_for_native_stub(tmp_path, monkeypatch):
    module = smoke()
    stub, library, path, policy = make_core_policy(tmp_path, module, monkeypatch)
    report = valid_report()
    report.update(executable=str(stub), native_argv=[str(stub), "model.py", "shm"])
    report["native_libraries"].append(str(library))
    module.validate_report(report, RUNTIME, CACHE, "unified_probe", "contract-test", core_policy=path)


@pytest.mark.parametrize("corruption", ["stub-hash", "library-hash", "stub-path", "library-path", "soname"])
def test_core_abi_exception_rejects_changed_hash_path_or_kind(tmp_path, monkeypatch, corruption):
    module = smoke()
    stub, library, path, policy = make_core_policy(tmp_path, module, monkeypatch)
    if corruption == "stub-hash":
        stub.write_bytes(b"changed stub")
    elif corruption == "library-hash":
        library.write_bytes(b"changed CUDA runtime")
    elif corruption == "stub-path":
        policy["stub"]["path"] = str(tmp_path / "another-stub")
    elif corruption == "library-path":
        policy["libraries"][0]["path"] = str(tmp_path / "another-library")
    else:
        policy["libraries"][0]["soname"] = "libpython3.12.so.1.0"
    path.write_text(json.dumps(policy))
    report = valid_report()
    report.update(executable=str(stub), native_argv=[str(stub), "model.py", "shm"])
    report["native_libraries"].append(str(library))
    with pytest.raises(RuntimeError, match="core ABI"):
        module.validate_report(report, RUNTIME, CACHE, "unified_probe", "contract-test", core_policy=path)


def run_spawn_helper(tmp_path, collector, timeout=3, shutdown_timeout=None):
    module = smoke()
    assert hasattr(module, "SPAWN_SOURCE"), "the model-version spawn helper is required"
    (tmp_path / "unified_spawn_probe.py").write_text(module.SPAWN_SOURCE + "\n" + collector)
    driver = tmp_path / "driver.py"
    shutdown_argument = '' if shutdown_timeout is None else f', shutdown_timeout={shutdown_timeout}'
    driver.write_text(
        "import json\nfrom unified_spawn_probe import spawn_probe, collect\n"
        "if __name__ == '__main__':\n"
        f"    print(json.dumps(spawn_probe(collect, timeout={timeout}{shutdown_argument})))\n")
    shutdown_budget = 10 if shutdown_timeout is None else shutdown_timeout
    outer_timeout = max(8, timeout + shutdown_budget + 3)
    return subprocess.run([sys.executable, str(driver)], capture_output=True, text=True, timeout=outer_timeout)


def test_generated_helper_uses_actual_spawn_and_picklable_version_module(tmp_path):
    result = run_spawn_helper(tmp_path, "def collect():\n    return {'method': __import__('multiprocessing').get_start_method(), 'pid': __import__('os').getpid()}\n")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["method"] == "spawn"
    assert report["pid"] > 0


def test_generated_spawn_helper_joins_and_reaps_failed_child(tmp_path):
    result = run_spawn_helper(tmp_path, "def collect():\n    raise RuntimeError('controlled child import failure')\n")
    assert result.returncode != 0
    assert "controlled child import failure" in result.stderr


def test_generated_spawn_helper_bounds_a_stuck_child(tmp_path):
    result = run_spawn_helper(tmp_path, "def collect():\n    __import__('time').sleep(60)\n", timeout=0.2)
    assert result.returncode != 0
    assert "timed out" in result.stderr.lower()


def test_real_spawned_static_python_needs_no_separate_libpython_mapping():
    report = valid_report()
    report["child"]["native_libraries"] = [
        "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"]
    validate(report)


def test_static_child_cannot_forge_selected_interpreter_with_foreign_executable():
    report = valid_report()
    report["child"]["native_libraries"] = [
        "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"]
    report["child"]["executable"] = "/usr/bin/python3.12"
    with pytest.raises(RuntimeError, match="child.*executable"):
        validate(report)


def test_embedded_cpp_parent_still_requires_selected_shared_libpython():
    report = valid_report()
    report["native_libraries"] = [
        "/opt/envs/vllm032-cu129/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so"]
    with pytest.raises(RuntimeError, match="libpython3.12"):
        validate(report)


def test_static_child_still_rejects_any_foreign_mapped_libpython():
    report = valid_report()
    report["child"]["native_libraries"][0] = "/usr/lib/libpython3.12.so.1.0"
    with pytest.raises(RuntimeError, match="child.*native"):
        validate(report)


def test_spawn_report_has_independent_grace_for_slow_python_teardown(tmp_path):
    result = run_spawn_helper(tmp_path,
        "def collect():\n"
        "    from multiprocessing.util import Finalize\n"
        "    import time, os\n"
        "    Finalize(None, time.sleep, args=(3.2,), exitpriority=1)\n"
        "    return {'pid': os.getpid(), 'payload': 'x' * 200000}\n", timeout=2)
    assert result.returncode == 0, result.stderr[:2000]
    report = json.loads(result.stdout)
    assert len(report["payload"]) == 200000
    assert report["spawn_timings"]["shutdown_seconds"] >= 3
    assert report["spawn_timings"]["receive_budget_seconds"] == 2


def test_hung_teardown_reaps_child_and_never_dumps_large_report_into_metadata(tmp_path):
    pid_file = tmp_path / "child-pid"
    result = run_spawn_helper(tmp_path,
        "def collect():\n"
        "    from multiprocessing.util import Finalize\n"
        "    from pathlib import Path\n"
        "    import time, os, signal\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"    Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "    Finalize(None, time.sleep, args=(60,), exitpriority=1)\n"
        "    return {'pid': os.getpid(), 'payload': 'PAYLOAD_MUST_NOT_LEAK' * 10000}\n", timeout=10, shutdown_timeout=0.2)
    assert result.returncode != 0
    assert "PAYLOAD_MUST_NOT_LEAK" not in result.stderr
    assert len(result.stderr.encode()) < 4096
    assert "shutdown" in result.stderr.lower() and "received" in result.stderr.lower()
    pid = int(pid_file.read_text())
    if Path(f"/proc/{pid}/stat").exists():
        assert Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1][0] == "Z"
