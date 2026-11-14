"""A clean image carries the native Triton closure without a second ML stack."""
import importlib.util
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_unified_recipe_uses_release_native_donor_and_stock_thinking_adapter():
    default_recipe = (ROOT / "Dockerfile.triton-gateway").read_text()
    unified_recipe = (ROOT / "Dockerfile.triton-gateway-unified").read_text()
    default_base = re.search(r"^ARG BASE_IMAGE=(.*)$", default_recipe, re.MULTILINE)[1]
    assert f"ARG TRITON_IMAGE={default_base}\n" in unified_recipe
    assert "COPY backends/vllm_multimodal/utils/thinking_budget.py /opt/tritonserver/backends/vllm/utils/thinking_budget.py" in unified_recipe
    assert "python3 /tmp/patch-native-thinking-budget.py" in unified_recipe


def assembler():
    path = ROOT / "docker/unified/assemble-native.py"
    assert path.is_file(), "Unified native closure assembler is missing"
    spec = importlib.util.spec_from_file_location("assemble_native", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_closure_keeps_soname_and_real_files_but_excludes_python_packages(tmp_path):
    module = assembler()
    source, destination = tmp_path / "source", tmp_path / "image"
    binary = source / "opt/tritonserver/bin/tritonserver"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELFfixture")
    library = source / "usr/local/cuda-13.4/lib/libcudart.so.13.4.92"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"\x7fELFcuda")
    library.with_name("libcudart.so.13").symlink_to(library.name)
    private_stack = source / "usr/local/lib/python3.12/dist-packages/torch"
    private_stack.mkdir(parents=True)
    (private_stack / "__init__.py").write_text("raise RuntimeError('old stack')")
    ldd_output = "libcudart.so.13 => /usr/local/cuda-13.4/lib/libcudart.so.13 (0x1234)\n"
    result = module.assemble_native(source, destination, dependency_reader=lambda _: ldd_output)
    assert (destination / "opt/tritonserver/bin/tritonserver").read_bytes() == binary.read_bytes()
    copied = destination / "usr/local/cuda-13.4/lib/libcudart.so.13"
    assert copied.is_symlink()
    assert copied.readlink() == Path(library.name)
    assert copied.resolve().read_bytes() == library.read_bytes()
    assert not (destination / "usr/local/lib/python3.12").exists()
    assert result["native_files"]["/usr/local/cuda-13.4/lib/libcudart.so.13.4.92"]
    assert json.loads((destination / "opt/tritonserver/unified/native-manifest.json").read_text()) == result


def test_native_closure_rejects_unresolved_dependencies(tmp_path):
    module = assembler()
    binary = tmp_path / "source/opt/tritonserver/bin/tritonserver"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELFfixture")
    with pytest.raises(RuntimeError, match="libmissing.so"):
        module.assemble_native(tmp_path / "source", tmp_path / "image",
                               dependency_reader=lambda _: "libmissing.so => not found\n")


def test_native_closure_requires_server_and_preserves_repository_agents(tmp_path):
    module = assembler()
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="tritonserver"):
        module.assemble_native(source, tmp_path / "image", dependency_reader=lambda _: "")
    binary = source / "opt/tritonserver/bin/tritonserver"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELFfixture")
    agent = source / "opt/tritonserver/repoagents/checksum/libtritonrepoagent_checksum.so"
    agent.parent.mkdir(parents=True)
    agent.write_bytes(b"\x7fELFagent")
    output = tmp_path / "valid-image"
    module.assemble_native(source, output, dependency_reader=lambda _: "")
    assert (output / agent.relative_to(source)).read_bytes() == agent.read_bytes()


def test_native_export_leaves_os_libraries_to_updated_distribution_packages(tmp_path):
    module = assembler()
    source, output = tmp_path / "source", tmp_path / "image"
    binary = source / "opt/tritonserver/bin/tritonserver"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"\x7fELFfixture")
    libraries = source / "usr/lib/x86_64-linux-gnu"
    libraries.mkdir(parents=True)
    (libraries / "libssl.so.3").write_bytes(b"old OS package")
    (libraries / "libdcgm.so.4").write_bytes(b"vendor GPU monitoring")
    listing = ("libssl.so.3 => /usr/lib/x86_64-linux-gnu/libssl.so.3 (0x1)\n"
               "libdcgm.so.4 => /usr/lib/x86_64-linux-gnu/libdcgm.so.4 (0x2)\n")
    result = module.assemble_native(source, output, dependency_reader=lambda _: listing)
    assert not (output / "usr/lib/x86_64-linux-gnu/libssl.so.3").exists()
    assert (output / "usr/lib/x86_64-linux-gnu/libdcgm.so.4").read_bytes() == b"vendor GPU monitoring"
    assert "/usr/lib/x86_64-linux-gnu/libssl.so.3" in result["system_libraries"]


def test_composed_provenance_refuses_overlay_that_does_not_match_its_hash(tmp_path):
    path = ROOT / "docker/unified/compose-inputs.py"
    assert path.exists(), "Composed runtime provenance tool is missing"
    spec = importlib.util.spec_from_file_location("compose_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"inputs": {"requirements_sha256": "base-lock"},
                                "derived_content_sha256": "base-content"}))
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("changed overlay")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"requirements_sha256": "0" * 64,
                                  "base_requirements_sha256": "base-lock"}))
    with pytest.raises(RuntimeError, match="overlay.*SHA256"):
        module.compose_inputs(base, overlay, inputs, "triton", "runtime", "os",
                              base_lock=tmp_path / "missing", base_inputs=tmp_path / "missing-inputs")


def test_composed_provenance_binds_base_and_overlay_content(tmp_path):
    import hashlib
    path = ROOT / "docker/unified/compose-inputs.py"
    assert path.exists(), "Composed runtime provenance tool is missing"
    spec = importlib.util.spec_from_file_location("compose_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base_lock = tmp_path / "base.txt"
    base_lock.write_text("base wheel pins")
    base_digest = hashlib.sha256(base_lock.read_bytes()).hexdigest()
    base_inputs = tmp_path / "base-inputs.json"
    base_inputs.write_text(json.dumps({"requirements_sha256": base_digest}))
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"inputs": json.loads(base_inputs.read_text()),
                                "derived_content_sha256": "base-content"}))
    overlay = tmp_path / "overlay.txt"
    overlay.write_text("locked overlay")
    digest = hashlib.sha256(overlay.read_bytes()).hexdigest()
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"requirements_sha256": digest,
                                  "base_requirements_sha256": base_digest,
                                  "base_inputs_sha256": hashlib.sha256(base_inputs.read_bytes()).hexdigest()}))
    keywords = {"base_lock": base_lock, "base_inputs": base_inputs}
    report = module.compose_inputs(base, overlay, inputs, "triton", "runtime", "os", **keywords)
    assert report["base_content_sha256"] == "base-content"
    assert report["overlay_sha256"] == digest
    assert report["triton_image"] == "triton"
    assert report["runtime_image"] == "runtime"
    inputs.write_text(json.dumps({"requirements_sha256": digest,
                                  "base_requirements_sha256": "other-base"}))
    with pytest.raises(RuntimeError, match="base.*SHA256"):
        module.compose_inputs(base, overlay, inputs, "triton", "runtime", "os", **keywords)


def test_provenance_checks_actual_base_lock_and_input_files(tmp_path):
    import hashlib
    import inspect
    path = ROOT / "docker/unified/compose-inputs.py"
    spec = importlib.util.spec_from_file_location("compose_inputs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert {"base_lock", "base_inputs"} <= set(inspect.signature(module.compose_inputs).parameters)
    base_lock, base_inputs = tmp_path / "base.txt", tmp_path / "base-inputs.json"
    base_lock.write_text("original base wheel hashes")
    base_inputs.write_text(json.dumps({"requirements_sha256": hashlib.sha256(base_lock.read_bytes()).hexdigest(),
                                      "python": "pinned-python", "cuda": "pinned-cuda"}))
    overlay, overlay_inputs, manifest = tmp_path / "overlay.txt", tmp_path / "overlay.json", tmp_path / "manifest.json"
    overlay.write_text("overlay pins")
    overlay_inputs.write_text(json.dumps({"requirements_sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(),
        "base_requirements_sha256": hashlib.sha256(base_lock.read_bytes()).hexdigest(),
        "base_inputs_sha256": hashlib.sha256(base_inputs.read_bytes()).hexdigest()}))
    donor = {"inputs": json.loads(base_inputs.read_text()), "derived_content_sha256": "donor-content"}
    manifest.write_text(json.dumps(donor))
    arguments = (manifest, overlay, overlay_inputs, "triton", "runtime", "os")
    keywords = {"base_lock": base_lock, "base_inputs": base_inputs}
    module.compose_inputs(*arguments, **keywords)
    original = base_lock.read_text()
    base_lock.write_text("same versions, different wheel hashes")
    with pytest.raises(RuntimeError, match="base.*SHA256"):
        module.compose_inputs(*arguments, **keywords)
    base_lock.write_text(original)
    original = base_inputs.read_text()
    base_inputs.write_text(original + "\n")
    with pytest.raises(RuntimeError, match="base.*inputs.*SHA256"):
        module.compose_inputs(*arguments, **keywords)
    base_inputs.write_text(original)
    donor["inputs"]["cuda"] = "different-cuda"
    manifest.write_text(json.dumps(donor))
    with pytest.raises(RuntimeError, match="donor.*inputs"):
        module.compose_inputs(*arguments, **keywords)
