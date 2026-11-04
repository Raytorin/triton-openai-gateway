# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_repository.py"
PIN_POLICY = ROOT / "tests" / "test_repository_quality.py"
SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"


def _repo(root, files):
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)
    subprocess.run(["git", "-C", str(root), "add", "--all"], check=True)
    return root


def _run_checker(root):
    # Isolated mode and no site-packages prove that early checks need no test deps.
    return subprocess.run(
        [sys.executable, "-I", "-S", str(CHECKER), "--root", str(root)],
        capture_output=True, text=True, timeout=10,
    )


def test_valid_tracked_sources_need_no_site_packages(tmp_path):
    root = _repo(tmp_path, {
        "module.py": "answer = 42\n",
        "tools/cli": "#!/usr/bin/env python3\nprint('ok')\n",
        "first.sh": "#!/bin/bash\necho ok\n",
        "config.json": '{"enabled": true}\n',
        "binary.bin": b"\x00\xff\x00",
    })
    result = _run_checker(root)
    assert result.returncode == 0, result.stderr


def test_syntax_error_in_second_shell_script_is_not_missed(tmp_path):
    root = _repo(tmp_path, {
        "a-first.sh": "#!/bin/bash\necho ok\n",
        "b-second.sh": "#!/bin/bash\nif true; then\n",
    })
    result = _run_checker(root)
    assert result.returncode != 0
    assert "b-second.sh" in result.stderr


@pytest.mark.parametrize(("name", "content"), [
    ("module.py", "if True print('broken')\n"),
    ("module.py", "return 1\n"),
    ("tools/cli", "#!/usr/bin/env python3\nif True print('broken')\n"),
    ("tools/cli", "#!/usr/bin/env python3\nbreak\n"),
    ("settings.json", "{broken}\n"),
    ("settings.json", '{"value": NaN}\n'),
    ("tools/runner", "#!/usr/bin/env bash\nif true; then\n"),
])
def test_invalid_tracked_source_fails_with_its_name(tmp_path, name, content):
    result = _run_checker(_repo(tmp_path, {name: content}))
    assert result.returncode != 0
    assert name in result.stderr


def test_untracked_and_ignored_notes_are_not_read(tmp_path):
    root = _repo(tmp_path, {"module.py": "answer = 42\n", ".gitignore": ".local/\n"})
    (root / "untracked.py").write_text("invalid python!\n")
    (root / ".local").mkdir()
    (root / ".local" / "notes.py").write_text("invalid python!\n")
    result = _run_checker(root)
    assert result.returncode == 0, result.stderr


def test_shell_paths_with_spaces_and_leading_dash_are_not_options(tmp_path):
    result = _run_checker(_repo(tmp_path, {
        "-first.sh": "#!/bin/bash\necho ok\n",
        "script with spaces.sh": "#!/bin/bash\necho ok\n",
    }))
    assert result.returncode == 0, result.stderr


def test_python_encoding_cookie_is_respected(tmp_path):
    result = _run_checker(_repo(tmp_path, {
        "module.py": b"# coding: latin-1\nname = 'caf\xe9'\n",
    }))
    assert result.returncode == 0, result.stderr


def test_symlink_to_external_source_is_rejected(tmp_path):
    root = _repo(tmp_path / "repo", {"module.py": "answer = 42\n"})
    outside = tmp_path / "outside.py"
    outside.write_text("if True print('broken')\n")
    (root / "link.py").symlink_to(outside)
    subprocess.run(["git", "-C", str(root), "add", "link.py"], check=True)
    result = _run_checker(root)
    assert result.returncode != 0
    assert "link.py" in result.stderr
    assert "outside the repository" in result.stderr


def test_symlink_to_ignored_note_is_rejected(tmp_path):
    root = _repo(tmp_path, {".gitignore": ".local/\n"})
    (root / ".local").mkdir()
    (root / ".local" / "notes.py").write_text("if True print('broken')\n")
    (root / "link.py").symlink_to(".local/notes.py")
    subprocess.run(["git", "-C", str(root), "add", "link.py"], check=True)
    result = _run_checker(root)
    assert result.returncode != 0
    assert "not tracked" in result.stderr


def test_symlink_to_another_tracked_source_is_supported(tmp_path):
    root = _repo(tmp_path, {"module.py": "answer = 42\n"})
    (root / "link.py").symlink_to("module.py")
    subprocess.run(["git", "-C", str(root), "add", "link.py"], check=True)
    result = _run_checker(root)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO requires POSIX")
def test_replaced_source_cannot_block_the_checker_on_a_fifo(tmp_path):
    root = _repo(tmp_path, {"module.py": "answer = 42\n"})
    (root / "module.py").unlink()
    os.mkfifo(root / "module.py")
    result = _run_checker(root)
    assert result.returncode != 0
    assert "regular file" in result.stderr


def _check_actions(root, monkeypatch, yaml):
    directory = root / ".github" / "workflows"
    directory.mkdir(parents=True)
    (directory / "ci.yml").write_text(yaml)
    policy = runpy.run_path(str(PIN_POLICY))
    test = policy["test_external_github_actions_are_pinned_to_commit_shas"]
    monkeypatch.setitem(test.__globals__, "ROOT", root)
    test()


@pytest.mark.parametrize("yaml", [
    "jobs:\n  test:\n    steps:\n      - uses: owner/action@v1\n",
    "jobs:\n  test:\n    uses: owner/repo/.github/workflows/test.yml@main\n",
    "jobs: {test: {steps: [{uses: owner/action@v1}]}}\n",
    'jobs:\n  test:\n    steps:\n      - "uses": "owner/action@main"\n',
    "jobs:\n  test:\n    steps:\n      - uses: docker://example/image:latest\n",
])
def test_unpinned_actions_in_supported_yaml_forms_fail(tmp_path, monkeypatch, yaml):
    with pytest.raises(AssertionError, match="immutable"):
        _check_actions(tmp_path, monkeypatch, yaml)


@pytest.mark.parametrize("yaml", [
    f"jobs:\n  test:\n    steps:\n      - uses: owner/action@{SHA}\n",
    f'jobs: {{test: {{steps: [{{uses: "owner/action@{SHA}"}}]}}}}\n',
    f"jobs:\n  test:\n    steps:\n      - uses: 'owner/action@{SHA}' # pinned\n",
    f"jobs:\n  test:\n    uses: owner/repo/.github/workflows/test.yml@{SHA}\n",
    "jobs:\n  test:\n    steps:\n      - uses: ./local-action\n",
    f"jobs:\n  test:\n    steps:\n      - uses: docker://example/image@sha256:{DIGEST}\n",
    f"jobs:\n  test:\n    steps:\n      - &shared\n        uses: owner/action@{SHA}\n      - <<: *shared\n        name: reused\n",
    "jobs:\n  test:\n    steps:\n      - run: |\n          echo ok\n          uses: a-script-command\n        env:\n          uses: ordinary-value\n",
])
def test_immutable_and_local_actions_pass_without_scanning_script_text(tmp_path, monkeypatch, yaml):
    _check_actions(tmp_path, monkeypatch, yaml)
