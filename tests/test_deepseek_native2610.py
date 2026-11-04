"""Keep the isolated engine's native exception bound to the classic image ABI."""
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "backends/vllm_deepseek/core-native-libraries.json"
BASE_IMAGE = (
    "nvcr.io/nvidia/tritonserver:26.10-vllm-python-py3@"
    "sha256:06803bff18e43379a7ecb4ce46feed537ca717c06c5472b354aec94bf20466e2"
)


def test_native_exception_matches_the_inspected_triton_2610_artifacts():
    policy = json.loads(POLICY.read_text())
    assert policy == {
        "schema": 1,
        "base_image": BASE_IMAGE,
        "stub": {
            "path": "/opt/tritonserver/backends/python/triton_python_backend_stub",
            "sha256": "2a7a3b8d9f04cc6acd75f9494d83b03a85ed285acbcf253fca854af41b0f36a4",
        },
        "libraries": [{
            "path": "/usr/local/cuda-13.4/targets/x86_64-linux/lib/libcudart.so.13.4.92",
            "soname": "libcudart.so.13",
            "sha256": "7fb00519a24512bf677b59e87921e5c05aac8246ceb6a57435c0d9dfeef364f6",
        }],
    }


def test_native_exception_follows_the_default_classic_image():
    recipe = (ROOT / "Dockerfile.triton-gateway").read_text()
    base = re.search(r"^ARG BASE_IMAGE=(\S+)$", recipe, re.MULTILINE)
    assert base is not None
    assert json.loads(POLICY.read_text())["base_image"] == base[1]


def test_engine_provenance_binds_the_exact_native_exception_policy():
    inputs = json.loads((ROOT / "docker/runtime-locks/vllm032-cu129.inputs.json").read_text())
    assert inputs["core_native_abi_policy_sha256"] == hashlib.sha256(POLICY.read_bytes()).hexdigest()
