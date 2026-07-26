import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT_SCRIPT = ROOT / "docker" / "80-watch-triton.sh"
WATCHER_SCRIPT = ROOT / "watch_triton.sh"
WATCHER_ENV_KEYS = {
    "MODELS_ACTIVE_DIR",
    "MODELS_ACTIVE_DIR_SOURCE",
    "TMP_ROOT",
    "TMPDIR",
    "TRITON_FRONTEND_LOCAL_MODEL_REPOSITORY",
    "WATCHER_MODEL_DIR",
    "WATCHER_MODEL_DIR_SOURCE",
}


class WatchEntrypointTests(unittest.TestCase):
    def _fake_watcher(self, directory: Path) -> Path:
        watcher = directory / "watch_triton.sh"
        watcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        watcher.chmod(0o755)
        return watcher

    def _run(
        self,
        watcher: Path,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        for key in WATCHER_ENV_KEYS:
            env.pop(key, None)
        env["WATCH_TRITON_BIN"] = str(watcher)
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            [
                "bash",
                "-c",
                (
                    'source "$1"; '
                    "printf 'exported:%s|%s\\n' "
                    '"${MODELS_ACTIVE_DIR}" '
                    '"${TRITON_FRONTEND_LOCAL_MODEL_REPOSITORY}"'
                ),
                "bash",
                str(ENTRYPOINT_SCRIPT),
            ],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_explicit_watcher_directory_has_highest_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            explicit = base / "explicit"
            legacy = base / "legacy"
            triton_tmp = base / "triton-tmp"
            explicit.mkdir()
            legacy.mkdir()
            triton_tmp.mkdir()

            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "WATCHER_MODEL_DIR": str(explicit),
                    "TMP_ROOT": str(legacy),
                    "TMPDIR": str(triton_tmp),
                },
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                f"selected watcher model directory {explicit} "
                "(source=WATCHER_MODEL_DIR)",
                result.stdout,
            )
            self.assertIn(
                f"selected models-active directory "
                f"{explicit / 'models-active'} (source=derived)",
                result.stdout,
            )

    def test_legacy_tmp_root_wins_over_triton_tmpdir(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            legacy = base / "legacy"
            triton_tmp = base / "triton-tmp"
            legacy.mkdir()
            triton_tmp.mkdir()

            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "TMP_ROOT": str(legacy),
                    "TMPDIR": str(triton_tmp),
                },
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                f"selected watcher model directory {legacy} "
                "(source=TMP_ROOT)",
                result.stdout,
            )

    def test_triton_tmpdir_is_monitored_and_hosts_active_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            triton_tmp = base / "triton-tmp"
            triton_tmp.mkdir()

            result = self._run(
                self._fake_watcher(base),
                extra_env={"TMPDIR": str(triton_tmp)},
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue((triton_tmp / "models-active").is_dir())
            self.assertIn(
                f"selected watcher model directory {triton_tmp} "
                "(source=TMPDIR)",
                result.stdout,
            )
            self.assertIn(
                f"selected models-active directory "
                f"{triton_tmp / 'models-active'} (source=derived)",
                result.stdout,
            )
            expected_active = triton_tmp / "models-active"
            self.assertIn(
                f"exported:{expected_active}|{expected_active}",
                result.stdout,
            )

    def test_default_matches_active_deployment_tmp_mount(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(self._fake_watcher(base))

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                "selected watcher model directory /tmp (source=default)",
                result.stdout,
            )
            self.assertIn(
                "selected models-active directory /tmp/models-active "
                "(source=derived)",
                result.stdout,
            )

    def test_explicit_active_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model_dir = base / "triton-tmp"
            active_dir = base / "custom-active"
            model_dir.mkdir()

            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "TMPDIR": str(model_dir),
                    "MODELS_ACTIVE_DIR": str(active_dir),
                },
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(active_dir.is_dir())
            self.assertIn(
                f"selected models-active directory {active_dir} "
                "(source=MODELS_ACTIVE_DIR)",
                result.stdout,
            )

    def test_missing_explicit_root_and_active_directory_are_created(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model_dir = base / "nested" / "triton-tmp"
            result = self._run(
                self._fake_watcher(base),
                extra_env={"WATCHER_MODEL_DIR": str(model_dir)},
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(model_dir.is_dir())
            self.assertTrue((model_dir / "models-active").is_dir())
            self.assertEqual(
                0o700,
                stat.S_IMODE(model_dir.stat().st_mode) & 0o700,
            )

    def test_empty_watcher_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                self._fake_watcher(base),
                extra_env={"WATCHER_MODEL_DIR": ""},
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("WATCHER_MODEL_DIR is set but empty", result.stderr)

    def test_empty_legacy_tmp_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                self._fake_watcher(base),
                extra_env={"TMP_ROOT": ""},
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("TMP_ROOT is set but empty", result.stderr)

    def test_empty_tmpdir_uses_default(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                self._fake_watcher(base),
                extra_env={"TMPDIR": ""},
            )

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                "selected watcher model directory /tmp (source=default)",
                result.stdout,
            )

    def test_regular_file_cannot_be_used_as_watcher_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            invalid_path = base / "not-a-directory"
            invalid_path.write_text("data", encoding="utf-8")
            result = self._run(
                self._fake_watcher(base),
                extra_env={"WATCHER_MODEL_DIR": str(invalid_path)},
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "watcher model directory is not a directory",
                result.stderr,
            )

    def test_regular_file_cannot_be_used_as_active_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            model_dir = base / "triton-tmp"
            invalid_path = base / "not-a-directory"
            model_dir.mkdir()
            invalid_path.write_text("data", encoding="utf-8")
            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "TMPDIR": str(model_dir),
                    "MODELS_ACTIVE_DIR": str(invalid_path),
                },
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "models-active directory is not a directory",
                result.stderr,
            )

    def test_active_directory_cannot_equal_watcher_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "TMPDIR": str(base),
                    "MODELS_ACTIVE_DIR": str(base),
                },
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "MODELS_ACTIVE_DIR must differ",
                result.stderr,
            )

    def test_remote_path_is_rejected_without_logging_uri(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                self._fake_watcher(base),
                extra_env={
                    "WATCHER_MODEL_DIR": (
                        "s3://secret:password@minio.example/models"
                    ),
                },
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn(
                "must be a local filesystem path",
                result.stderr,
            )
            self.assertNotIn("password", result.stderr)
            self.assertNotIn("minio", result.stderr)

    def test_missing_watcher_binary_fails_before_triton_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            result = self._run(
                base / "missing-watcher",
                extra_env={"TMPDIR": str(base / "triton-tmp")},
            )

            self.assertNotEqual(0, result.returncode)
            self.assertIn("watcher is not executable", result.stderr)


class WatcherDirectoryTests(unittest.TestCase):
    def _wait_for_symlink(self, link: Path) -> None:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if link.is_symlink():
                return
            time.sleep(0.05)
        self.fail(f"watcher did not create symlink {link}")

    def test_transient_checkout_and_active_registry_share_selected_root(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            checkout = model_dir / "folderABC"
            version = checkout / "1"
            active = model_dir / "models-active"
            version.mkdir(parents=True)
            (checkout / "config.pbtxt").write_text(
                'name: "downloaded-model"\nbackend: "vllm"\n',
                encoding="utf-8",
            )
            model_json = version / "model.json"
            model_json.write_text(
                json.dumps(
                    {
                        "model": "/stale/path",
                        "swap_space": 4,
                    }
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            for key in WATCHER_ENV_KEYS:
                env.pop(key, None)
            env["WATCHER_MODEL_DIR"] = str(model_dir)

            process = subprocess.Popen(
                ["bash", str(WATCHER_SCRIPT)],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                link = active / "downloaded-model"
                self._wait_for_symlink(link)
                rewritten = json.loads(model_json.read_text(encoding="utf-8"))
                self.assertEqual(str(version), rewritten["model"])
                self.assertNotIn("swap_space", rewritten)
                self.assertEqual(version.resolve(), link.resolve())
            finally:
                process.terminate()
                process.communicate(timeout=3)


if __name__ == "__main__":
    unittest.main()
