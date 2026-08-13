# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import re

from backends.vllm_multimodal.utils.observability import (
    BACKEND_VERSION,
    CEF_VERSION as BACKEND_CEF_VERSION,
)
from gateway import __version__
from gateway.app import app
from gateway.observability import CEF_VERSION as GATEWAY_CEF_VERSION
from gateway.tracing import DEFAULT_OTEL_SERVICE_VERSION


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _top_level_scalar(path: Path, key: str) -> str:
    pattern = re.compile(
        rf"^{re.escape(key)}:\s*[\"']?([^\"'\s]+)",
        re.MULTILINE,
    )
    match = pattern.search(path.read_text(encoding="utf-8"))
    assert match is not None, f"Missing {key} in {path.relative_to(ROOT)}"
    return match.group(1)


def test_python_service_versions_match() -> None:
    assert SEMANTIC_VERSION.fullmatch(__version__)
    assert app.version == __version__
    assert GATEWAY_CEF_VERSION == __version__
    assert DEFAULT_OTEL_SERVICE_VERSION == __version__
    assert BACKEND_VERSION == __version__
    assert BACKEND_CEF_VERSION == __version__


def test_packaging_release_versions_match() -> None:
    dockerfile = (ROOT / "Dockerfile.triton-gateway").read_text(encoding="utf-8")
    chart = ROOT / "helm" / "triton-gateway" / "Chart.yaml"
    citation = ROOT / "CITATION.cff"

    assert f"ARG GATEWAY_VERSION={__version__}" in dockerfile
    assert 'org.opencontainers.image.version="${GATEWAY_VERSION}"' in dockerfile
    assert _top_level_scalar(chart, "version") == __version__
    assert _top_level_scalar(citation, "version") == __version__


def test_chart_runtime_matches_the_pinned_base_image() -> None:
    dockerfile = (ROOT / "Dockerfile.triton-gateway").read_text(encoding="utf-8")
    base_image = re.search(
        r"^ARG BASE_IMAGE=.*tritonserver:([0-9]+\.[0-9]+)-",
        dockerfile,
        re.MULTILINE,
    )
    assert base_image is not None, "Unable to determine Triton base image version"

    chart = ROOT / "helm" / "triton-gateway" / "Chart.yaml"
    assert _top_level_scalar(chart, "appVersion") == base_image.group(1)
