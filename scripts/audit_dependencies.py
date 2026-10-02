# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Audit all dependency lists, with one expiring runtime advisory exception."""
from datetime import date, datetime, timezone
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ACCELERATE_ADVISORY = "PYSEC-2026-3804"
# Also known as GHSA-4j2p-28q2-5m79 / CVE-2026-69112.
# Temporary acceptance for issue #28; this is not a vulnerability fix.
# The unmodified audit resumes at 00:00 UTC on this date.
ACCELERATE_EXCEPTION_EXPIRES = date(2026, 10, 10)


def runtime_exception_args(requirements: str, today: date) -> list[str]:
    pins = {line.split("#", 1)[0].strip() for line in requirements.splitlines()}
    if "accelerate==1.14.0" in pins and today < ACCELERATE_EXCEPTION_EXPIRES:
        return ["--ignore-vuln", ACCELERATE_ADVISORY]
    return []


def main() -> int:
    today = datetime.now(timezone.utc).date()
    for relative_path in (
        "requirements-test.txt",
        "docker/triton-chat-gateway-requirements.txt",
    ):
        path = ROOT / relative_path
        exceptions = (
            runtime_exception_args(path.read_text(encoding="utf-8"), today)
            if relative_path.startswith("docker/") else []
        )
        if exceptions:
            print(
                f"Temporary audit exception: accelerate==1.14.0 / {ACCELERATE_ADVISORY}; "
                f"expires {ACCELERATE_EXCEPTION_EXPIRES} UTC. The vulnerability remains; "
                "all other advisories remain blocking. See SECURITY.md.",
                file=sys.stderr, flush=True,
            )
        result = subprocess.run([
            sys.executable, "-m", "pip_audit", "--requirement", str(path),
            "--progress-spinner", "off", *exceptions,
        ], check=False)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
