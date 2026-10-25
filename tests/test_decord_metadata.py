"""The qualified Decord correction rejects drift and preserves wheel payload bytes."""
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import pytest
from pip._internal.metadata import get_environment
from pip._internal.operations.check import check_unsupported
from pip._vendor.packaging.tags import sys_tags

ROOT = Path(__file__).resolve().parents[1]
OLD_WHEEL = (b'Wheel-Version: 1.0\nGenerator: bdist_wheel (0.36.2)\n'
             b'Root-Is-Purelib: false\nTag: cp36-cp36m-manylinux2010_x86_64\n\n')
DIST = 'decord-0.6.0.dist-info'


@pytest.fixture
def installed(tmp_path, monkeypatch):
    helper = ROOT / 'docker/repair-decord-metadata.py'
    assert helper.exists(), 'qualified Decord metadata repair is missing'
    spec = importlib.util.spec_from_file_location('decord_repair', helper)
    repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)
    env = tmp_path / 'env'
    site = env / 'lib/python3.12/site-packages'
    payload = {'decord/__init__.py': b'__version__ = "0.6.0"\n',
               'decord/libdecord.so': b'\x7fELF-native-fixture',
               'decord.libs/codec.so': b'\x7fELF-codec-fixture',
               f'{DIST}/METADATA': b'Metadata-Version: 2.1\nName: decord\nVersion: 0.6.0\nRequires-Dist: numpy (>=1.14.0)\n',
               f'{DIST}/top_level.txt': b'decord\n', f'{DIST}/WHEEL': OLD_WHEEL,
               # Deliberately stale upstream hash, matching the actual defect.
               f'{DIST}/RECORD': f'{DIST}/top_level.txt,sha256=stale,17\n{DIST}/RECORD,,\n'.encode()}
    wheelhouse = tmp_path / 'wheels'; wheelhouse.mkdir()
    wheel = wheelhouse / 'decord-0.6.0-py3-none-manylinux2010_x86_64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
            path = site / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (site / DIST / 'INSTALLER').write_bytes(b'pip\n')
    (site / DIST / 'REQUESTED').write_bytes(b'')
    # pip's installed RECORD retains upstream stale hashes but lists every file.
    (site / DIST / 'RECORD').write_text(''.join(f'{name},sha256=stale,17\n' if name != f'{DIST}/RECORD' else f'{name},,\n'
        for name in sorted(payload)) + f'{DIST}/INSTALLER,,\n{DIST}/REQUESTED,,\n')
    numpy = site / 'numpy-2.3.5.dist-info'; numpy.mkdir()
    (numpy / 'METADATA').write_text('Metadata-Version: 2.1\nName: numpy\nVersion: 2.3.5\n')
    artifacts = tmp_path / 'artifacts'
    monkeypatch.setattr(repair, 'ENV_ROOT', env)
    monkeypatch.setattr(repair, 'ARTIFACT_ROOT', artifacts)
    monkeypatch.setattr(repair, 'SOURCE_SHA256', hashlib.sha256(wheel.read_bytes()).hexdigest())
    monkeypatch.setattr(repair.sys, 'prefix', str(env))
    monkeypatch.setattr(repair.sys, 'exec_prefix', str(env))
    monkeypatch.setattr(repair.sys, 'version_info', (3, 12, 15))
    monkeypatch.setattr(repair.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(repair.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(repair.platform, 'libc_ver', lambda: ('glibc', '2.39'))
    return repair, env, site, wheelhouse, wheel, artifacts


def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_repair_makes_actual_pip_tag_check_pass_and_rehashes_all_installed_files(installed):
    repair, env, site, wheelhouse, wheel, artifacts = installed
    before = snapshot(env)
    tags = list(sys_tags())
    assert [d.raw_name for d in check_unsupported(get_environment([str(site)]).iter_installed_distributions(), tags)] == ['decord']
    repair.repair(wheelhouse)
    assert list(check_unsupported(get_environment([str(site)]).iter_installed_distributions(), tags)) == []
    after = snapshot(env)
    changed = {name for name in before if before[name] != after[name]}
    assert changed == {f'lib/python3.12/site-packages/{DIST}/WHEEL', f'lib/python3.12/site-packages/{DIST}/RECORD'}
    assert before.keys() == after.keys()
    rows = list(csv.reader(io.StringIO((site / DIST / 'RECORD').read_text())))
    assert {row[0] for row in rows} == {str(p.relative_to(site)) for folder in ('decord', 'decord.libs', DIST) for p in (site / folder).rglob('*') if p.is_file()}
    for name, digest, size in rows:
        if name == f'{DIST}/RECORD':
            assert (digest, size) == ('', '')
        else:
            data = (site / name).read_bytes()
            assert digest == 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
            assert int(size) == len(data)
    report = json.loads((artifacts / 'decord-metadata-repair.json').read_text())
    assert report['source']['sha256'] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert report['unchanged_payload']['files'] == 5
    assert report['changes'][f'{DIST}/WHEEL']['before_sha256'] == 'd24f09731316657ed32488ba245812cbb5047342b148776a371264e69966d856'
    assert report['changes'][f'{DIST}/RECORD']['before_sha256'] == hashlib.sha256(before[f'lib/python3.12/site-packages/{DIST}/RECORD']).hexdigest()
    report_before = snapshot(artifacts)
    repair.repair(wheelhouse)
    assert snapshot(env) == after and snapshot(artifacts) == report_before


@pytest.mark.parametrize('fault', ['source_hash', 'payload', 'native', 'metadata', 'wrong_tag', 'extra', 'pyc', 'record_escape', 'record_duplicate', 'record_missing', 'symlink', 'extra_distribution', 'aliased_metadata', 'egg_metadata', 'installer', 'precorrected', 'foreign_prefix', 'foreign_exec_prefix', 'python', 'platform', 'machine', 'musl', 'old_glibc', 'numpy', 'artifact_symlink'])
def test_repair_rejects_drift_before_writing(installed, monkeypatch, fault):
    repair, env, site, wheelhouse, wheel, artifacts = installed
    if fault == 'source_hash': wheel.write_bytes(wheel.read_bytes() + b'tamper')
    elif fault == 'payload': (site / 'decord/__init__.py').write_bytes(b'changed')
    elif fault == 'native': (site / 'decord.libs/codec.so').write_bytes(b'changed')
    elif fault == 'metadata': (site / DIST / 'METADATA').write_text('Name: decord\nVersion: 0.7.0\n')
    elif fault == 'wrong_tag': (site / DIST / 'WHEEL').write_bytes(OLD_WHEEL.replace(b'cp36', b'cp37'))
    elif fault == 'extra': (site / 'decord/extra.py').write_bytes(b'unknown')
    elif fault == 'pyc':
        cache = site / 'decord/__pycache__'; cache.mkdir(); (cache / '__init__.cpython-312.pyc').write_bytes(b'unknown')
    elif fault.startswith('record_'):
        with (site / DIST / 'RECORD').open('a') as stream:
            stream.write({'record_escape': '../../foreign,,\n', 'record_duplicate': f'{DIST}/RECORD,,\n', 'record_missing': 'decord/missing.py,,\n'}[fault])
    elif fault == 'symlink':
        target = site / 'decord/libdecord.so'; target.unlink(); target.symlink_to(wheel)
    elif fault == 'extra_distribution': (site / 'decord-0.7.0.dist-info').mkdir()
    elif fault in ('aliased_metadata', 'egg_metadata'):
        folder = site / ('foreign-0.7.0.dist-info' if fault == 'aliased_metadata' else 'DECORD-0.7.0.egg-info')
        folder.mkdir()
        (folder / ('METADATA' if fault == 'aliased_metadata' else 'PKG-INFO')).write_text('Name: DECORD\nVersion: 0.7.0\n')
    elif fault == 'installer': (site / DIST / 'INSTALLER').write_bytes(b'other\n')
    elif fault == 'precorrected': (site / DIST / 'WHEEL').write_bytes(OLD_WHEEL.replace(b'cp36-cp36m', b'py3-none'))
    elif fault == 'foreign_prefix': monkeypatch.setattr(repair.sys, 'prefix', '/usr/local')
    elif fault == 'foreign_exec_prefix': monkeypatch.setattr(repair.sys, 'exec_prefix', '/usr/local')
    elif fault == 'python': monkeypatch.setattr(repair.sys, 'version_info', (3, 13, 0))
    elif fault == 'platform': monkeypatch.setattr(repair.platform, 'system', lambda: 'Darwin')
    elif fault == 'machine': monkeypatch.setattr(repair.platform, 'machine', lambda: 'aarch64')
    elif fault == 'musl': monkeypatch.setattr(repair.platform, 'libc_ver', lambda: ('musl', '1.2.5'))
    elif fault == 'old_glibc': monkeypatch.setattr(repair.platform, 'libc_ver', lambda: ('glibc', '2.11'))
    elif fault == 'numpy': (site / 'numpy-2.3.5.dist-info/METADATA').write_text('Name: numpy\nVersion: 2.4.0\n')
    elif fault == 'artifact_symlink': artifacts.symlink_to(wheelhouse, target_is_directory=True)
    before = snapshot(env)
    with pytest.raises(RuntimeError): repair.repair(wheelhouse)
    assert snapshot(env) == before
    assert not (artifacts / 'decord-metadata-repair.json').exists()


@pytest.mark.parametrize('fault', ['payload', 'record', 'report', 'missing_report', 'report_script'])
def test_idempotence_requires_unchanged_payload_record_and_prior_derivation(installed, fault):
    repair, env, site, wheelhouse, wheel, artifacts = installed
    repair.repair(wheelhouse)
    report = artifacts / 'decord-metadata-repair.json'
    if fault == 'payload': (site / 'decord/libdecord.so').write_bytes(b'changed')
    elif fault == 'record': (site / DIST / 'RECORD').write_text(f'{DIST}/RECORD,,\n')
    elif fault == 'missing_report': report.unlink()
    else:
        data = json.loads(report.read_text())
        if fault == 'report': data['source']['sha256'] = '0' * 64
        else: data['repair_script_sha256'] = '0' * 64
        report.write_text(json.dumps(data))
    before = snapshot(env)
    with pytest.raises(RuntimeError): repair.repair(wheelhouse)
    assert snapshot(env) == before


def test_cli_refuses_the_system_interpreter(tmp_path):
    import subprocess
    helper = ROOT / 'docker/repair-decord-metadata.py'
    assert helper.exists(), 'qualified Decord metadata repair is missing'
    result = subprocess.run([sys.executable, '-I', str(helper)], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'owned execution environment' in result.stderr
