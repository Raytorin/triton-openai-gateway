import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "docker" / "80-watch-triton.sh"


class WatchEntrypointTests(unittest.TestCase):
    def _fake_watcher(self, directory: Path) -> Path:
        watcher = directory / "watch_triton.sh"
        watcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        watcher.chmod(0o755)
        return watcher

    def _run(self, tmp_root: Path, watcher: Path) -> subprocess.CompletedProcess[str]:
        env = {
            **os.environ,
            "TMP_ROOT": str(tmp_root),
            "WATCH_TRITON_BIN": str(watcher),
        }
        return subprocess.run(
            ["bash", "-c", 'umask 007; source "$1"', "bash", str(SCRIPT)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_missing_tmp_root_is_created_without_forced_chmod(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tmp_root = base / "nested" / "models"
            result = self._run(tmp_root, self._fake_watcher(base))

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(tmp_root.is_dir())
            self.assertEqual(0o770, stat.S_IMODE(tmp_root.stat().st_mode))
            self.assertIn("created TMP_ROOT directory", result.stdout)

    def test_regular_file_cannot_be_used_as_tmp_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            tmp_root = base / "not-a-directory"
            tmp_root.write_text("data", encoding="utf-8")
            result = self._run(tmp_root, self._fake_watcher(base))

            self.assertNotEqual(0, result.returncode)
            self.assertIn("TMP_ROOT is not a directory", result.stderr)

    def test_missing_watcher_binary_fails_before_triton_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(base / "models", base / "missing-watcher")

            self.assertNotEqual(0, result.returncode)
            self.assertIn("watcher is not executable", result.stderr)


if __name__ == "__main__":
    unittest.main()
