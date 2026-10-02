# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from datetime import date
from pathlib import Path
import runpy
from types import SimpleNamespace
from unittest.mock import patch


POLICY = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_dependencies.py"))


def test_exception_expires_at_utc_date_boundary():
    args = POLICY["runtime_exception_args"]
    assert args("accelerate==1.14.0", date(2026, 10, 9)) == ["--ignore-vuln", "PYSEC-2026-3804"]
    assert args("accelerate==1.14.0", date(2026, 10, 10)) == []
    assert args("accelerate==1.14.0", date(2026, 10, 11)) == []


def test_exception_does_not_follow_package_upgrades_or_other_packages():
    args = POLICY["runtime_exception_args"]
    for pins in ("accelerate==1.15.0", "other==1.14.0", "# accelerate==1.14.0", ""):
        assert args(pins, date(2026, 9, 26)) == []


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
