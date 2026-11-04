# SPDX-License-Identifier: Apache-2.0
"""Bind the gateway overlay to the verified donor and native image inputs."""
import argparse
import hashlib
import json
from pathlib import Path


def compose_inputs(base_manifest, overlay_lock, overlay_inputs, triton_image, runtime_image, os_image,
                   *, base_lock, base_inputs):
    base = json.loads(Path(base_manifest).read_text())
    overlay = json.loads(Path(overlay_inputs).read_text())
    digest = hashlib.sha256(Path(overlay_lock).read_bytes()).hexdigest()
    if digest != overlay["requirements_sha256"]:
        raise RuntimeError("Gateway overlay SHA256 does not match its recorded inputs")
    if base["inputs"]["requirements_sha256"] != overlay["base_requirements_sha256"]:
        raise RuntimeError("Runtime base lock SHA256 does not match the gateway overlay")
    if hashlib.sha256(Path(base_lock).read_bytes()).hexdigest() != overlay["base_requirements_sha256"]:
        raise RuntimeError("Copied base lock SHA256 does not match the gateway overlay")
    if hashlib.sha256(Path(base_inputs).read_bytes()).hexdigest() != overlay["base_inputs_sha256"]:
        raise RuntimeError("Copied base inputs SHA256 does not match the gateway overlay")
    if json.loads(Path(base_inputs).read_text()) != base["inputs"]:
        raise RuntimeError("Actual donor runtime inputs differ from the pinned base inputs")
    return {"base_runtime": base["inputs"], "base_content_sha256": base["derived_content_sha256"],
            "overlay_sha256": digest, "overlay": overlay, "triton_image": triton_image,
            "runtime_image": runtime_image, "os_image": os_image}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--base-inputs", type=Path, required=True)
    parser.add_argument("--overlay-lock", type=Path, required=True)
    parser.add_argument("--overlay-inputs", type=Path, required=True)
    parser.add_argument("--triton-image", required=True)
    parser.add_argument("--runtime-image", required=True)
    parser.add_argument("--os-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = compose_inputs(args.base_manifest, args.overlay_lock, args.overlay_inputs,
                          args.triton_image, args.runtime_image, args.os_image,
                          base_lock=args.base_lock, base_inputs=args.base_inputs)
    args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
