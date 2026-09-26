# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import re
import runpy
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_PAIRS = (
    ("docs/responses.md", "docs/responses.ru.md"),
    ("docs/release-responses.md", "docs/release-responses.ru.md"),
    ("README.md", "README.ru.md"),
    ("AUTHORS.md", "AUTHORS.ru.md"),
    ("CONTRIBUTING.md", "CONTRIBUTING.ru.md"),
    ("SECURITY.md", "SECURITY.ru.md"),
    ("docs/architecture.md", "docs/architecture.ru.md"),
    ("docs/configuration.md", "docs/configuration.ru.md"),
    ("docs/migration-26.07.md", "docs/migration-26.07.ru.md"),
    ("docs/operations.md", "docs/operations.ru.md"),
    ("backends/vllm_multimodal/README.md", "backends/vllm_multimodal/README.ru.md"),
    ("helm/triton-gateway/README.md", "helm/triton-gateway/README.ru.md"),
    ("helm/dcgm-exporter/README.md", "helm/dcgm-exporter/README.ru.md"),
    ("examples/bge-m3-hybrid/README.md", "examples/bge-m3-hybrid/README.ru.md"),
    (
        "examples/bge-m3-vllm-multimodal/README.md",
        "examples/bge-m3-vllm-multimodal/README.ru.md",
    ),
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def test_local_markdown_links_resolve():
    errors = []
    for document in ROOT.rglob("*.md"):
        relative_document = document.relative_to(ROOT)
        if any(part.startswith(".") for part in relative_document.parts):
            continue

        for raw_target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            target = raw_target.strip().split()[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue

            relative_target = unquote(target.split("#", 1)[0])
            resolved = (document.parent / relative_target).resolve()
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                errors.append(f"{relative_document}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                errors.append(f"{relative_document}: missing link target: {target}")

    assert not errors, "\n".join(errors)


def test_bilingual_documents_keep_matching_heading_structure():
    for english_name, russian_name in LANGUAGE_PAIRS:
        english = (ROOT / english_name).read_text(encoding="utf-8")
        russian = (ROOT / russian_name).read_text(encoding="utf-8")
        english_levels = [len(line) - len(line.lstrip("#")) for line in english.splitlines() if line.startswith("#")]
        russian_levels = [len(line) - len(line.lstrip("#")) for line in russian.splitlines() if line.startswith("#")]
        assert english_levels == russian_levels, f"Heading structure differs: {english_name} and {russian_name}"


def test_runtime_requirement_pins_match_verifier():
    verifier = runpy.run_path(str(ROOT / "docker" / "verify-runtime.py"))
    expected = verifier["EXPECTED_VERSIONS"]
    read_pins = verifier["read_pinned_requirements"]
    requirements = read_pins(ROOT / "docker" / "triton-chat-gateway-requirements.txt")

    mismatches = {
        package: (version, expected.get(package))
        for package, version in requirements.items()
        if expected.get(package) != version
    }
    assert not mismatches, f"Runtime pins differ from verifier: {mismatches}"


def test_shared_test_and_runtime_requirement_pins_match():
    verifier = runpy.run_path(str(ROOT / "docker" / "verify-runtime.py"))
    read_pins = verifier["read_pinned_requirements"]
    test_requirements = read_pins(ROOT / "requirements-test.txt")
    runtime_requirements = read_pins(
        ROOT / "docker" / "triton-chat-gateway-requirements.txt"
    )

    mismatches = {
        package: (test_requirements[package], runtime_requirements[package])
        for package in test_requirements.keys() & runtime_requirements.keys()
        if test_requirements[package] != runtime_requirements[package]
    }
    assert not mismatches, f"Test and runtime pins differ: {mismatches}"


def test_external_github_actions_are_pinned_to_commit_shas():
    unpinned = []
    for workflow in (ROOT / ".github" / "workflows").glob("*.y*ml"):
        for line_number, line in enumerate(
            workflow.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped.startswith("uses:"):
                continue

            action = stripped.split("#", 1)[0].split(":", 1)[1].strip()
            if action.startswith("./"):
                continue
            _, separator, reference = action.rpartition("@")
            if not separator or not COMMIT_SHA.fullmatch(reference):
                unpinned.append(f"{workflow.relative_to(ROOT)}:{line_number}: {action}")

    assert not unpinned, "GitHub Actions must use immutable commit SHAs:\n" + "\n".join(
        unpinned
    )
