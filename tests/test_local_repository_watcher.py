# SPDX-License-Identifier: Apache-2.0
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
from threading import Thread

import pytest


WATCHER = Path(__file__).resolve().parents[1] / "watch_triton.sh"


@pytest.fixture
def triton_index():
    state = {"models": [], "status": 200}

    class IndexHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            assert self.path == "/v2/repository/index"
            assert json.loads(body) == {"ready": True}
            self.send_response(state["status"])
            self.end_headers()
            self.wfile.write(json.dumps(state["models"]).encode())

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), IndexHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", state
        finally:
            server.shutdown()
            thread.join()


def model_version(repo, name, version):
    path = repo / name / str(version)
    path.mkdir(parents=True)
    (path.parent / "config.pbtxt").write_text(f'name: "{name}"\nbackend: "python"\n')
    (path / "model.json").write_text('{"model": "/packaged/weights"}\n')
    return path


def run_watcher(tmp_path, url, *repositories, staging=None):
    active = tmp_path / "models-active"
    active.mkdir(exist_ok=True)
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "WATCHER_MODEL_DIR": str(staging or tmp_path / "staging"),
        "MODELS_ACTIVE_DIR": str(active),
        "WATCHER_CLEANUP_GRACE_SECONDS": "0",
        "WATCHER_CLEANUP_INTERVAL_SECONDS": "0",
        "WATCHER_PROC_ROOT": str(proc),
        "TRITON_BASE_URL": url,
    }
    result = subprocess.run(
        ["bash", "-c", 'source "$1" "${@:2}"; remember_existing_links; scan_once',
         "bash", str(WATCHER), "tritonserver", *repositories],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return active


def test_discovers_ready_versions_and_keeps_s3_checkout(tmp_path, triton_index):
    url, state = triton_index
    staging = tmp_path / "staging"
    checkout = model_version(staging, "folderS3", 1)
    (checkout.parent / "config.pbtxt").write_text('name: "remote"\nbackend: "python"\n')
    repository = tmp_path / "local repository"
    version_2 = model_version(repository, "local", 2)
    version_10 = model_version(repository, "local", 10)
    model_version(repository, "local", 20)
    other = model_version(tmp_path / "another", "private", 1)
    original = (version_2 / "model.json").read_bytes()
    arguments = ("--model-repository=s3://example-bucket/models", "--model-repository",
                 repository.name, "--model-repository=another")
    state["models"] = [
        {"name": "remote", "version": "1", "state": "READY"},
        {"name": "local", "version": "2", "state": "READY"},
        {"name": "local", "version": "10", "state": "READY"},
        {"name": "local", "version": "20", "state": "UNAVAILABLE"},
        {"name": "private", "version": "1", "state": "READY"},
    ]
    active = run_watcher(tmp_path, url, *arguments)
    assert (active / "local").resolve(strict=True) == version_10
    assert (active / "private").resolve(strict=True) == other
    assert (active / "remote").resolve(strict=True) == checkout

    state["status"] = 503
    run_watcher(tmp_path, url, *arguments)
    assert (active / "local").resolve(strict=True) == version_10

    state["status"] = 200
    detached = version_10.rename(tmp_path / "detached-version")
    run_watcher(tmp_path, url, *arguments)
    assert not (active / "local").is_symlink()
    detached.rename(version_10)

    state["models"] = [{"name": "local", "version": "2", "state": "READY"}]
    run_watcher(tmp_path, url, *arguments)
    assert (active / "local").resolve(strict=True) == version_2
    assert not (active / "private").is_symlink()
    assert (active / "remote").resolve(strict=True) == checkout
    assert other.is_dir() and version_10.is_dir()
    assert (version_2 / "model.json").read_bytes() == original


def test_local_repo_named_like_staging_preserves_source_on_unload(tmp_path, triton_index):
    url, state = triton_index
    repo = tmp_path / "models"
    version = model_version(repo, "folderModel", 1)
    original = (version / "model.json").read_bytes()
    state["models"] = [{"name": "folderModel", "version": "1", "state": "READY"}]
    active = run_watcher(tmp_path, url, f"--model-repository={repo}", staging=repo)
    assert (active / "folderModel").resolve(strict=True) == version
    assert (version / "model.json").read_bytes() == original

    state["models"] = []
    run_watcher(tmp_path, url, f"--model-repository={repo}", staging=repo)
    assert not (active / "folderModel").is_symlink()
    assert (version / "model.json").read_bytes() == original
    assert not (version / ".watcher-retired.json").exists()


@pytest.mark.parametrize("staging_alias", [False, True])
def test_unreadable_local_model_does_not_mask_staging_check(tmp_path, triton_index, staging_alias):
    url, state = triton_index
    staging = tmp_path / "staging"
    checkout = model_version(staging, "folderS3", 1)
    (checkout.parent / "config.pbtxt").write_text('name: "remote"\nbackend: "python"\n')
    original = (checkout / "model.json").read_bytes()
    repository = tmp_path / "local"
    unreadable = repository / "unreadable"
    unreadable.mkdir(parents=True)
    sibling = tmp_path / "other"
    local = model_version(sibling, "local", 1)
    if staging_alias:
        (sibling / "alias").symlink_to(checkout.parent, target_is_directory=True)
    state["models"] = [
        {"name": "remote", "version": "1", "state": "READY"},
        {"name": "local", "version": "1", "state": "READY"},
    ]
    unreadable.chmod(0)
    try:
        if os.access(unreadable, os.R_OK):
            pytest.skip("test requires a user subject to directory read permissions")
        active = run_watcher(tmp_path, url, f"--model-repository={repository}",
                             f"--model-repository={sibling}", "--model-repository=s3://example-bucket/models")
        assert (active / "local").resolve(strict=True) == local
        if staging_alias:
            assert not (active / "remote").is_symlink()
            assert (checkout / "model.json").read_bytes() == original
        else:
            assert (active / "remote").resolve(strict=True) == checkout
            assert json.loads((checkout / "model.json").read_text())["model"] == str(checkout)
    finally:
        unreadable.chmod(0o700)


@pytest.mark.parametrize("link_model_root", [True, False])
def test_staging_does_not_rewrite_symlinked_local_sources(tmp_path, triton_index, link_model_root):
    url, state = triton_index
    staging = tmp_path / "staging"
    version = model_version(staging, "folderSnapshot", 1)
    original = (version / "model.json").read_bytes()
    repo = tmp_path / "models"
    repo.mkdir()
    alias = repo / "alias"
    if link_model_root:
        alias.symlink_to(version.parent, target_is_directory=True)
    else:
        alias.mkdir()
        (alias / "1").symlink_to(version, target_is_directory=True)
    state["models"] = [{"name": "alias", "version": "1", "state": "READY"}]
    active = run_watcher(tmp_path, url, f"--model-repository={repo}")
    assert (active / "alias").resolve(strict=True) == version
    assert (version / "model.json").read_bytes() == original

    state["models"] = []
    run_watcher(tmp_path, url, f"--model-repository={repo}")
    assert not (active / "alias").is_symlink()
    assert (version / "model.json").read_bytes() == original
    assert not (version / ".watcher-retired.json").exists()
