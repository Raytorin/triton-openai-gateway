# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from datetime import date, datetime
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


def test_client_requirements_are_audited_separately():
    with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
        assert POLICY["main"]() == 0

    commands = [call.args[0] for call in run.call_args_list]
    assert [Path(command[4]).name for command in commands] == [
        "requirements-test.txt",
        "triton-chat-gateway-requirements.txt",
        "requirements-client.txt",
    ]
    assert all(command.count("--requirement") == 1 for command in commands)


@pytest.mark.parametrize("returncode", [1, 2])
def test_client_audit_failure_or_scanner_error_is_propagated(returncode):
    def audit(command, *, check):
        return SimpleNamespace(
            returncode=returncode
            if Path(command[4]).name == "requirements-client.txt" else 0
        )

    with patch("subprocess.run", side_effect=audit):
        assert POLICY["main"]() == returncode


def test_runtime_advisory_exception_is_never_inherited_by_client(tmp_path):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 4, tzinfo=tz)

    for name in (
        "requirements-test.txt",
        "docker/triton-chat-gateway-requirements.txt",
        "requirements-client.txt",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("accelerate==1.14.0\n", encoding="utf-8")

    with patch.dict(POLICY["main"].__globals__, ROOT=tmp_path, datetime=FixedDatetime):
        with patch("subprocess.run", return_value=SimpleNamespace(returncode=0)) as run:
            assert POLICY["main"]() == 0

    commands = [call.args[0] for call in run.call_args_list]
    assert [Path(command[4]).name for command in commands] == [
        "requirements-test.txt",
        "triton-chat-gateway-requirements.txt",
        "requirements-client.txt",
    ]
    assert ["--ignore-vuln" in command for command in commands] == [False, True, False]
