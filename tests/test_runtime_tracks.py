# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Keep official updates available while the newer runtime remains opt-in."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _recipe(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_official_track_keeps_its_own_pinned_runtime_contract() -> None:
    recipe = _recipe("Dockerfile.triton-gateway")
    assert re.search(
        r"^ARG BASE_IMAGE=nvcr\.io/nvidia/tritonserver:[^\s@]+@sha256:[a-f0-9]{64}$",
        recipe,
        re.MULTILINE,
    )
    assert "FROM ${BASE_IMAGE}" in recipe
    assert "COPY docker/triton-chat-gateway-requirements.txt " in recipe
    assert "python3 /tmp/verify-runtime.py" in recipe
    assert "/opt/envs/vllm032-cu129" not in recipe


def test_new_runtime_is_explicit_and_all_tracks_include_current_gateway() -> None:
    official = _recipe("Dockerfile.triton-gateway")
    additive = _recipe("Dockerfile.triton-gateway-deepseek")
    unified = _recipe("Dockerfile.triton-gateway-unified")
    assert "FROM ${GATEWAY_IMAGE} AS deepseek" in additive
    assert re.search(
        r"^ARG RUNTIME_IMAGE=triton-openai-gateway:deepseek-local$",
        unified,
        re.MULTILINE,
    )
    assert re.search(r"^FROM \$\{RUNTIME_IMAGE\} AS ", unified, re.MULTILINE)
    assert "docker/runtime-locks/unified-vllm032.requirements.txt" in unified
    for recipe in (official, additive, unified):
        assert "COPY gateway /opt/tritonserver/gateway" in recipe


def test_official_docker_recipe_keeps_recurring_dependency_update_coverage() -> None:
    config = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^  - package-ecosystem:\s*", config)[1:]
    docker_blocks = [block for block in blocks if block.splitlines()[0] == "docker"]
    assert docker_blocks, "Docker updates must remain configured for the official track"
    root_updates = [
        block for block in docker_blocks
        if re.search(r"(?m)^    directory:\s*[\"']?/[\"']?\s*$", block)
    ]
    assert root_updates, "The official root Dockerfile must remain in update scope"
    assert any(
        re.search(r"(?m)^      interval:\s*(daily|weekly|monthly)\s*$", block)
        and not re.search(r"(?m)^    ignore:\s*$", block)
        for block in root_updates
    ), "Keep recurring Docker updates without ignoring the pinned NVIDIA base"
