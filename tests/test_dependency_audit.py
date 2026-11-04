# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from datetime import date
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch

import pytest


POLICY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_dependencies.py"))


def test_exception_expires_at_utc_date_boundary():
    args = POLICY["runtime_exception_args"]
    assert args("accelerate==1.14.0", date(2026, 10, 12)) == ["--ignore-vuln", "PYSEC-2026-3804"]
    assert args("accelerate==1.14.0", date(2026, 11, 9)) == ["--ignore-vuln", "PYSEC-2026-3804"]
    assert args("accelerate==1.14.0", date(2026, 11, 10)) == []
    assert args("accelerate==1.14.0", date(2026, 11, 11)) == []


def test_exception_does_not_follow_package_upgrades_or_other_packages():
    args = POLICY["runtime_exception_args"]
    for pins in ("accelerate==1.15.0", "other==1.14.0", "# accelerate==1.14.0", ""):
        assert args(pins, date(2026, 11, 9)) == []


@pytest.mark.parametrize("requirements", [
    "accelerate==1.14.0 --hash=sha256:" + "a" * 64,
    "accelerate==1.14.0 \\\n    --hash=sha256:" + "a" * 64 + " \\\n    --hash=sha256:" + "b" * 64,
])
def test_hashed_exact_runtime_pin_keeps_the_same_exception_boundary(requirements):
    args = POLICY["runtime_exception_args"]
    assert args(requirements, date(2026, 11, 9)) == ["--ignore-vuln", "PYSEC-2026-3804"]
    assert args(requirements, date(2026, 11, 10)) == []


@pytest.mark.parametrize("requirements", [
    "accelerate==1.15.0 --hash=sha256:" + "a" * 64,
    "# accelerate==1.14.0 \\\n    --hash=sha256:" + "a" * 64,
    "other==1.14.0 --hash=sha256:" + "a" * 64,
    "accelerate==1.14.0.post1 --hash=sha256:" + "a" * 64,
    "accelerate==1.14.0 --hash=sha256:invalid",
    "accelerate==1.14.0 --hash=sha256:" + "a" * 64 + " --unexpected-option",
])
def test_hashed_pin_recognition_does_not_expand_exception_scope(requirements):
    assert POLICY["runtime_exception_args"](requirements, date(2026, 11, 9)) == []


@pytest.mark.parametrize("today,expected_overlay_exception", [
    (date(2026, 11, 9), ["--ignore-vuln", "PYSEC-2026-3804"]),
    (date(2026, 11, 10), []),
])
@pytest.mark.parametrize("overlay_returncode", [1, 2])
def test_unified_overlay_is_audited_and_other_failures_remain_blocking(
    tmp_path, today, expected_overlay_exception, overlay_returncode,
):
    paths = (
        "requirements-test.txt",
        "docker/triton-chat-gateway-requirements.txt",
        "docker/runtime-locks/unified-vllm032.requirements.txt",
    )
    for relative_path in paths:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("accelerate==1.14.0 \\\n    --hash=sha256:" + "a" * 64 + "\n", encoding="utf-8")
    commands = []

    def audit(command, *, check):
        commands.append(command)
        return SimpleNamespace(returncode=overlay_returncode if len(commands) == 3 else 0)

    clock = SimpleNamespace(now=lambda zone: SimpleNamespace(date=lambda: today))
    with patch.dict(POLICY["main"].__globals__, {"ROOT": tmp_path, "datetime": clock}):
        with patch("subprocess.run", side_effect=audit):
            assert POLICY["main"]() == overlay_returncode

    assert [Path(command[4]).relative_to(tmp_path).as_posix() for command in commands] == list(paths)
    assert commands[0][7:] == [], "Test dependencies must not inherit a runtime exception"
    assert commands[1][7:] == expected_overlay_exception
    assert commands[2][7:] == ["--disable-pip", "--no-deps", *expected_overlay_exception]


def test_actual_overlay_collects_pinned_audit_inputs_without_pip_resolution():
    from pip_audit._cli import _parser
    from pip_audit._dependency_source.requirement import RequirementSource

    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
        assert POLICY["main"]() == 0
    commands = [call.args[0] for call in run.call_args_list]
    parsed = [_parser().parse_args(command[3:]) for command in commands]
    for legacy in parsed[:2]:
        assert not legacy.disable_pip
        assert not legacy.no_deps

    overlay = parsed[2]
    source = RequirementSource(
        overlay.requirements,
        require_hashes=overlay.require_hashes,
        no_deps=overlay.no_deps,
        disable_pip=overlay.disable_pip,
    )
    with patch(
        "pip_audit._dependency_source.requirement.VirtualEnv.create",
        side_effect=AssertionError("The manifested overlay must not be freshly resolved"),
    ):
        dependencies = list(source.collect())
    assert not any(dependency.is_skipped() for dependency in dependencies)
    assert {dependency.name: str(dependency.version) for dependency in dependencies} == {
        "accelerate": "1.14.0",
        "deepseek-recipe": "0.2.0",
        "ml-dtypes": "0.5.4",
        "protobuf": "6.33.6",
        "python-rapidjson": "1.25",
        "qwen-vl-utils": "0.0.14",
        "soundfile": "0.14.0",
        "soxr": "1.1.0",
        "tritonclient": "2.73.0",
    }


def test_unrelated_audit_failure_is_propagated():
    with patch("subprocess.run", side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(returncode=1)]) as run:
        assert POLICY["main"]() == 1
    assert len(run.call_args_list) == 2
    assert "--ignore-vuln" not in run.call_args_list[0].args[0]
    assert "docker/triton-chat-gateway-requirements.txt" in run.call_args_list[1].args[0][4]


def test_scanner_error_is_not_swallowed():
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=2)) as run:
        assert POLICY["main"]() == 2
    run.assert_called_once()
