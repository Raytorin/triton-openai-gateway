# SPDX-License-Identifier: Apache-2.0
"""Correct the exact qualified Decord wheel's filename/internal-tag mismatch.

Upstream v0.6.0 pypi.yml renames cp36-cp36m to py3-none without updating
WHEEL (and rewrites top_level.txt without rehashing RECORD). Compare installed
payloads to the hash-locked archive itself, never to its stale RECORD digests.
This is intentionally restricted to one artifact and qualified execution env.
"""
import base64
import csv
from email.parser import BytesParser
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import zipfile

ENV_ROOT = Path('/opt/envs/vllm032-cu129')
ARTIFACT_ROOT = Path('/opt/runtime-artifacts/vllm032-cu129')
WHEEL_NAME = 'decord-0.6.0-py3-none-manylinux2010_x86_64.whl'
SOURCE_SHA256 = '51997f20be8958e23b7c4061ba45d0efcd86bffd5fe81c695d0befee0d442976'
SOURCE_URL = 'https://files.pythonhosted.org/packages/11/79/936af42edf90a7bd4e41a6cac89c913d4b47fa48a26b042d5129a9242ee3/' + WHEEL_NAME
OLD_WHEEL = (b'Wheel-Version: 1.0\nGenerator: bdist_wheel (0.36.2)\n'
             b'Root-Is-Purelib: false\nTag: cp36-cp36m-manylinux2010_x86_64\n\n')
NEW_WHEEL = OLD_WHEEL.replace(b'cp36-cp36m', b'py3-none')
DIST = 'decord-0.6.0.dist-info'
WHEEL = DIST + '/WHEEL'
RECORD = DIST + '/RECORD'
OWNED_DIRS = ('decord', 'decord.libs', DIST)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    """Refuse symlinks in every component before reading or writing."""
    if path.resolve() != path.absolute() or not path.is_file():
        raise RuntimeError(f'Expected regular owned file: {path}')
    return path


def file_digest(path):
    with regular(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or str(path) != name or path.parts[0] not in OWNED_DIRS):
        raise RuntimeError(f'Unexpected Decord path: {name}')
    return name


def record_names(data, expected):
    names = []
    for row in csv.reader(io.StringIO(data.decode('utf-8'))):
        if len(row) != 3:
            raise RuntimeError('Malformed installed Decord RECORD')
        names.append(safe_name(row[0]))
    if len(names) != len(set(names)) or set(names) != expected:
        raise RuntimeError('Installed Decord RECORD file list differs from the verified installation')


def repair(wheelhouse):
    if (sys.prefix != str(ENV_ROOT) or sys.exec_prefix != str(ENV_ROOT)
            or ENV_ROOT.resolve() != ENV_ROOT):
        raise RuntimeError('Decord correction requires the owned execution environment')
    if (tuple(sys.version_info[:3]) != (3, 12, 15) or platform.system() != 'Linux'
            or platform.machine() != 'x86_64'):
        raise RuntimeError('Decord correction requires qualified Python 3.12.15 / Linux x86_64')
    libc, version = platform.libc_ver()
    if libc != 'glibc' or not re.fullmatch(r'\d+\.\d+(?:\.\d+)*', version) or tuple(map(int, version.split('.'))) < (2, 12):
        raise RuntimeError('Decord correction requires glibc >= 2.12 for manylinux2010')
    site = ENV_ROOT / 'lib/python3.12/site-packages'
    if sorted(p.name for p in site.glob('decord*.dist-info')) != [DIST]:
        raise RuntimeError('Unexpected Decord distribution/version layout')
    distributions = list(importlib.metadata.distributions(path=[str(site)]))
    decord = [d.version for d in distributions if d.metadata.get('Name', '').lower() == 'decord']
    if decord != ['0.6.0']:
        raise RuntimeError('Unexpected duplicate Decord metadata/name/version')
    numpy = [d.version for d in distributions
             if d.metadata.get('Name', '').lower() == 'numpy']
    if numpy != ['2.3.5']:
        raise RuntimeError('Decord correction requires qualified NumPy 2.3.5')
    wheel = Path(wheelhouse) / WHEEL_NAME
    if file_digest(wheel) != SOURCE_SHA256:
        raise RuntimeError('Decord source wheel SHA256 mismatch')
    expected = {}
    with zipfile.ZipFile(wheel) as archive:
        seen = set()
        for info in archive.infolist():
            name = safe_name(info.filename.rstrip('/') if info.is_dir() else info.filename)
            if name in seen or stat.S_ISLNK(info.external_attr >> 16):
                raise RuntimeError(f'Duplicate/symlink Decord archive path: {name}')
            seen.add(name)
            if info.is_dir():
                continue
            with archive.open(info) as stream:
                expected[name] = {'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(), 'size': info.file_size}
        if archive.read(WHEEL) != OLD_WHEEL:
            raise RuntimeError('Unrecognized original Decord WHEEL metadata')
        metadata = BytesParser().parsebytes(archive.read(DIST + '/METADATA'))
        if metadata['Name'] != 'decord' or metadata['Version'] != '0.6.0':
            raise RuntimeError('Unrecognized original Decord name/version')
    extras = {DIST + '/INSTALLER': b'pip\n', DIST + '/REQUESTED': b''}
    direct_url = site / DIST / 'direct_url.json'
    if direct_url.exists() or direct_url.is_symlink():
        data = json.loads(regular(direct_url).read_bytes())
        if data.get('archive_info', {}).get('hashes', {}).get('sha256') != SOURCE_SHA256:
            raise RuntimeError('Unexpected Decord direct URL provenance')
        extras[DIST + '/direct_url.json'] = direct_url.read_bytes()
    actual = {}
    for folder in OWNED_DIRS:
        root = site / folder
        if not root.is_dir() or root.resolve() != root:
            raise RuntimeError(f'Unexpected Decord installed directory: {root}')
        for path in root.rglob('*'):
            if path.is_symlink():
                raise RuntimeError(f'Decord installed symlink: {path}')
            if path.is_dir():
                continue
            name = safe_name(path.relative_to(site).as_posix())
            actual[name] = {'sha256': file_digest(path), 'size': path.stat().st_size}
    allowed = set(expected) | {DIST + '/INSTALLER'}
    allowed.update(name for name in extras if (site / name).exists())
    if set(actual) != allowed:
        raise RuntimeError('Unexpected or missing Decord installed files (install with --no-compile)')
    for name, data in extras.items():
        if name in actual and (site / name).read_bytes() != data:
            raise RuntimeError(f'Unexpected Decord installer file: {name}')
    payload = {name: value for name, value in expected.items() if name not in (WHEEL, RECORD)}
    for name, value in payload.items():
        if actual[name] != value:
            raise RuntimeError(f'Installed Decord payload differs from source wheel: {name}')
    current_wheel = (site / WHEEL).read_bytes()
    if current_wheel not in (OLD_WHEEL, NEW_WHEEL):
        raise RuntimeError('Unexpected installed Decord WHEEL tag/metadata')
    original_record = (site / RECORD).read_bytes()
    record_names(original_record, set(actual))
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    for name in sorted(actual):
        if name == RECORD:
            writer.writerow((name, '', ''))
        else:
            digest = sha(NEW_WHEEL) if name == WHEEL else actual[name]['sha256']
            size = len(NEW_WHEEL) if name == WHEEL else actual[name]['size']
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode().rstrip('=')
            writer.writerow((name, 'sha256=' + encoded, size))
    new_record = buffer.getvalue().encode('utf-8')
    report_path = ARTIFACT_ROOT / 'decord-metadata-repair.json'
    if report_path.resolve() != report_path:
        raise RuntimeError('Decord derivation path must be an owned regular path')
    prior = None
    if report_path.exists():
        prior = json.loads(regular(report_path).read_bytes())
        if current_wheel != NEW_WHEEL:
            raise RuntimeError('Prior Decord derivation disagrees with installed WHEEL')
        try:
            original_record = base64.b64decode(prior['original_record_base64'], validate=True)
            record_names(original_record, set(actual))
        except (KeyError, ValueError) as exc:
            raise RuntimeError('Invalid prior Decord derivation') from exc
    elif current_wheel == NEW_WHEEL:
        raise RuntimeError('Corrected Decord metadata lacks its original derivation')
    changes = {WHEEL: {'before_sha256': sha(OLD_WHEEL), 'after_sha256': sha(NEW_WHEEL)},
               RECORD: {'before_sha256': sha(original_record), 'after_sha256': sha(new_record)}}
    report = {'schema': 1, 'policy': 'decord-0.6.0-qualified-filename-tag-v1',
              'source': {'url': SOURCE_URL, 'filename': WHEEL_NAME, 'sha256': SOURCE_SHA256},
              'runtime': {'prefix': str(ENV_ROOT), 'python': '3.12.15', 'numpy': '2.3.5',
                          'system': 'Linux', 'machine': 'x86_64', 'libc': {'name': libc, 'version': version}},
              'tags': {'before': 'cp36-cp36m-manylinux2010_x86_64', 'after': 'py3-none-manylinux2010_x86_64'},
              'repair_script_sha256': file_digest(Path(__file__).resolve()), 'changes': changes,
              'original_record_base64': base64.b64encode(original_record).decode(),
              'unchanged_payload': {'files': len(payload), 'sha256': sha(json.dumps(payload, sort_keys=True).encode())}}
    if prior is not None:
        if prior != report or (site / RECORD).read_bytes() != new_record:
            raise RuntimeError('Decord repair provenance/RECORD integrity mismatch')
        return report
    # All validation and derivation are complete before changing installed files.
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    (site / WHEEL).write_bytes(NEW_WHEEL)
    (site / RECORD).write_bytes(new_record)
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
    return report


if __name__ == '__main__':
    try:
        repair(Path('/tmp/deepseek-wheelhouse'))
    except (RuntimeError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        sys.exit(f'Decord metadata repair refused: {exc}')
