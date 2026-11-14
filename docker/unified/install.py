# SPDX-License-Identifier: Apache-2.0
"""Install common stub wrappers into the opt-in image, preserving native ABI."""
import argparse
import hashlib
import json
import os
from pathlib import Path


BACKENDS = ('python', 'vllm', 'vllm_multimodal', 'vllm_deepseek')
STUB = 'triton_python_backend_stub'
UTILS = 'triton_python_backend_utils.py'


def install(triton_root):
    root = Path(triton_root).resolve()
    wrapper = root/'unified'/STUB
    if not all(path.is_file() for path in (wrapper, wrapper.parent/'launch.py',
                                          wrapper.parent/'sitecustomize.py')):
        raise RuntimeError('Copy the common launcher and wrapper into triton-root/unified first')
    destinations = [root/'backends'/name/STUB for name in BACKENDS]
    if not all(path.parent.is_dir() for path in destinations):
        raise RuntimeError('All supported backends must be copied before installing wrappers')
    base_utility = root/'backends/python'/UTILS
    if base_utility.is_symlink() or not base_utility.is_file():
        raise RuntimeError(f'Python backend utility must be an existing regular file: {base_utility}')
    utility_target = Path('../python')/UTILS
    missing_utilities = []
    for name in BACKENDS[1:]:
        utility = root/'backends'/name/UTILS
        if utility.exists() or utility.is_symlink():
            if (not utility.is_symlink() or utility.readlink() != utility_target
                    or utility.resolve() != base_utility):
                raise RuntimeError(f'Conflicting backend utility path: {utility}')
        else:
            missing_utilities.append(utility)
    stock = root/'backends/python'/STUB
    native = stock.with_name(f'{STUB}.native')
    preserved = native.exists() or native.is_symlink()
    source = native if preserved else stock
    if source.is_symlink() or not source.is_file() or not os.access(source, os.X_OK):
        raise RuntimeError(f'Native stub must be an executable regular ELF: {source}')
    with source.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            raise RuntimeError(f'Native stub must be an ELF executable: {source}')
        stream.seek(0)
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (preserved and (stock.exists() or stock.is_symlink())
            and (not stock.is_symlink() or stock.resolve() != wrapper)):
        raise RuntimeError('Preserved native stub exists but the Python backend wrapper is unexpected')
    policy_path = root/'backends/vllm_deepseek/core-native-libraries.json'
    policy = json.loads(policy_path.read_text())
    if policy.get('stub', {}).get('sha256') != digest:
        raise RuntimeError('Native stub does not match the pinned release SHA256')
    if policy['stub'].get('path') not in (str(stock), str(native)):
        raise RuntimeError('Pinned stub path does not select this Triton installation')
    policy['stub'] = {'path': str(native), 'sha256': digest}
    # Everything that can invalidate the input is checked before the first rename.
    wrapper.chmod(0o755)
    if not preserved:
        stock.rename(native)
    # Triton selects a per-backend stub only when its utility is a sibling.
    for utility in missing_utilities:
        utility.symlink_to(utility_target)
    for destination in destinations:
        temporary = destination.with_name(f'.{STUB}.unified-{os.getpid()}')
        try:
            temporary.symlink_to(os.path.relpath(wrapper, destination.parent))
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    temporary_policy = policy_path.with_name(f'.{policy_path.name}.unified-{os.getpid()}')
    try:
        temporary_policy.write_text(json.dumps(policy, indent=2) + '\n')
        temporary_policy.replace(policy_path)
    finally:
        temporary_policy.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--triton-root', type=Path, default=Path('/opt/tritonserver'))
    args = parser.parse_args()
    install(args.triton_root)


if __name__ == '__main__':
    main()
