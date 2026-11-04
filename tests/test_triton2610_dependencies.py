"""Constraints observed in the immutable NVIDIA 26.10 wheel METADATA."""
from pathlib import Path
import runpy

from packaging.requirements import Requirement
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("consumer,requirement", [
    ("NVIDIA vLLM 0.31", "pydantic>=2.12.0"),
    ("NVIDIA Triton frontend 2.74", "pydantic>=2.12.0,<3"),
    ("NVIDIA vLLM 0.31", "fastapi>=0.133.0,<0.137.0"),
    ("NVIDIA Triton server 2.74", "numpy<2"),
    ("Triton client 2.74 grpc", "protobuf>=6.30.0,<7.0"),
    ("Triton client 2.74 grpc", "grpcio>=1.81.1"),
    ("Triton client 2.74", "ml-dtypes<=0.5.4"),
    ("ml-dtypes 0.5.4 Python 3.12", "numpy>=1.21"),
])
def test_gateway_overlay_respects_updated_native_constraints(consumer, requirement):
    policy = runpy.run_path(str(ROOT / "docker/verify-runtime.py"))
    wanted = Requirement(requirement)
    actual = policy["EXPECTED_VERSIONS"][wanted.name]
    assert wanted.specifier.contains(actual), f"{consumer}: {wanted}, found {actual}"


def test_release_base_keeps_its_native_components_as_one_matrix():
    policy = runpy.run_path(str(ROOT / "docker/verify-runtime.py"))
    pins = policy["read_pinned_requirements"](ROOT / "docker/triton-chat-gateway-requirements.txt")
    core = policy["BASE_IMAGE_VERSIONS"]
    # Installing independently resolved Torch/vLLM/FlashInfer would bypass the
    # NVIDIA runtime qualification, even when a version string looks newer.
    assert not ({"torch", "vllm", "transformers", "flashinfer-python"} & pins.keys())
    assert core["vllm"].startswith("0.31.0+")
    assert ".nv26.10." in core["vllm"]
    assert core["tritonserver"] == "2.74.0"
