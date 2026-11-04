# SPDX-License-Identifier: Apache-2.0
"""The unified verifier rejects missing, inconsistent and legacy dependencies."""
from importlib import metadata, util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "docker/unified/verify-environment.py"


def verifier():
    assert VERIFIER.is_file(), "unified runtime environment verifier is required"
    spec = util.spec_from_file_location("unified_environment_verifier", VERIFIER)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def distributions(tmp_path, entries):
    for name, version in entries:
        info = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
            encoding="utf-8",
        )
    return metadata.distributions(path=[str(tmp_path)])


def test_missing_gateway_client_is_rejected(tmp_path):
    errors = verifier().distribution_errors(
        distributions(tmp_path, [("numpy", "2.3.5")]),
        {"tritonclient": "2.73.0", "numpy": "2.3.5"},
    )
    assert errors == ["tritonclient: missing, expected 2.73.0"]


def test_old_pydantic_and_new_protobuf_are_rejected(tmp_path):
    errors = verifier().distribution_errors(
        distributions(tmp_path, [("pydantic", "2.10.6"), ("protobuf", "7.36.2")]),
        {"pydantic": "2.14.1", "protobuf": "6.33.6"},
    )
    assert errors == [
        "pydantic: found 2.10.6, expected 2.14.1",
        "protobuf: found 7.36.2, expected 6.33.6",
    ]


def test_manual_nvidia_frontends_are_rejected(tmp_path):
    errors = verifier().distribution_errors(
        distributions(tmp_path, [("nvidia_tritonserver", "2.73.0"),
                                  ("tritonfrontend", "2.73.0"),
                                  ("tritonclient", "2.73.0")]),
        {"tritonclient": "2.73.0"},
    )
    assert errors == [
        "nvidia-tritonserver: legacy NVIDIA Python distribution is not supported",
        "tritonfrontend: legacy NVIDIA Python distribution is not supported",
    ]


def test_duplicate_distribution_metadata_is_rejected(tmp_path):
    errors = verifier().distribution_errors(
        distributions(tmp_path, [("ml-dtypes", "0.5.4"), ("ml_dtypes", "0.6.0")]),
        {"ml-dtypes": "0.5.4"},
    )
    assert any("duplicate" in error and "ml-dtypes" in error for error in errors)


def test_coherent_distribution_metadata_is_accepted(tmp_path):
    assert verifier().distribution_errors(
        distributions(tmp_path, [("pydantic", "2.14.1"), ("numpy", "2.3.5"),
                                  ("ml_dtypes", "0.5.4")]),
        {"pydantic": "2.14.1", "numpy": "2.3.5", "ml-dtypes": "0.5.4"},
    ) == []


@pytest.mark.parametrize("line", ["example>=1.0", "example==1.0", "example==1.0 --hash=sha256:bad",
                                          "example==1.* --hash=sha256:" + "a" * 64,
                                          "example===1.0 --hash=sha256:" + "a" * 64])
def test_unpinned_or_unhashed_overlay_is_rejected(tmp_path, line):
    lock = tmp_path / "overlay.txt"
    lock.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact pin and SHA256"):
        verifier().read_lock(lock)


def test_hashed_overlay_preserves_client_extra(tmp_path):
    lock = tmp_path / "overlay.txt"
    lock.write_text("# public wheel\ntritonclient[grpc]==2.73.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    assert verifier().read_lock(lock) == {"tritonclient": "2.73.0"}


def test_overlay_cannot_replace_inference_dependencies(tmp_path):
    base = tmp_path / "base.txt"
    overlay = tmp_path / "overlay.txt"
    base.write_text("torch==2.13.0+cu129 --hash=sha256:" + "a" * 64 + "\n")
    overlay.write_text("torch==2.14.0 --hash=sha256:" + "b" * 64 + "\n")
    with pytest.raises(ValueError, match="cannot replace base runtime pin.*torch"):
        verifier().expected_versions(base, overlay)


def test_missing_gateway_pin_is_rejected(tmp_path):
    base = tmp_path / "base.txt"
    overlay = tmp_path / "overlay.txt"
    base.write_text("torch==2.13.0+cu129 --hash=sha256:" + "a" * 64 + "\n")
    overlay.write_text("accelerate==1.14.0 --hash=sha256:" + "b" * 64 + "\n")
    with pytest.raises(ValueError, match="missing gateway overlay pins.*tritonclient"):
        verifier().expected_versions(base, overlay)


@pytest.mark.parametrize("missing", ["soundfile", "soxr"])
def test_native_audio_dependencies_cannot_be_omitted_from_overlay(tmp_path, missing):
    lock = ROOT / "docker/runtime-locks/unified-vllm032.requirements.txt"
    text = lock.read_text().replace("\\\n", " ")
    overlay = tmp_path / "missing-audio.txt"
    overlay.write_text("\n".join(line for line in text.splitlines()
                                  if not line.startswith(missing + "==")) + "\n")
    with pytest.raises(ValueError, match="missing gateway overlay pins.*" + missing):
        verifier().expected_versions(ROOT / "docker/runtime-locks/vllm032-cu129.requirements.txt", overlay)
