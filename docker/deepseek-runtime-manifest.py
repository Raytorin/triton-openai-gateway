# SPDX-License-Identifier: Apache-2.0
"""Content-addressed inventory of the derived environment, toolkit and kernel cache."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys


def inventory(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise RuntimeError(f'Runtime tree is missing: {root}')
    files = {}
    for path in sorted(root.rglob('*')):
        if '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        name = str(path.relative_to(root))
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise RuntimeError(f'Runtime symlink points outside {root}: {name}')
            if not path.exists():
                raise RuntimeError(f'Runtime symlink is broken: {name}')
            files[name] = {'symlink': str(path.readlink())}
        elif path.is_file():
            with path.open('rb') as stream:
                files[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return files


def verify_inventory(root, expected):
    actual = inventory(root)
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(name for name in expected.keys() & actual.keys() if expected[name] != actual[name])
    if missing or unexpected or changed:
        raise RuntimeError(f'Runtime integrity failure: missing={missing}, unexpected={unexpected}, changed={changed}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('create', 'verify'))
    parser.add_argument('--root', type=Path, default=Path('/opt/envs/vllm032-cu129'))
    parser.add_argument('--cuda', type=Path, default=Path('/opt/cuda/12.9'))
    parser.add_argument('--kernels', type=Path, default=Path('/opt/runtime-artifacts/vllm032-cu129'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--inputs', type=Path)
    args = parser.parse_args()
    trees = {'environment':args.root, 'cuda':args.cuda, 'kernels':args.kernels}
    if args.mode == 'create':
        if args.inputs is None:
            parser.error('create requires --inputs provenance JSON')
        site = args.root / 'lib/python3.12/site-packages'
        packages = {d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(site)])}
        data = {'schema':1, 'inputs':json.loads(args.inputs.read_text()),
                'python':sys.version, 'platform':platform.platform(), 'libc':platform.libc_ver(),
                'distributions':dict(sorted(packages.items())),
                'trees':{name:inventory(root) for name, root in trees.items()}}
        # Identity of the composed trees, never the reference container digest.
        data['derived_content_sha256'] = hashlib.sha256(json.dumps(data['trees'], sort_keys=True).encode()).hexdigest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
    else:
        data = json.loads(args.manifest.read_text())
        for name, root in trees.items():
            verify_inventory(root, data['trees'][name])
        print(data['derived_content_sha256'])


if __name__ == '__main__':
    main()
