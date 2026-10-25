"""A materialized runtime manifest detects changed files and missing artifacts."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / 'docker/deepseek-runtime-manifest.py'
    assert path.exists(), 'runtime artifact verification is missing'
    spec = importlib.util.spec_from_file_location('runtime_manifest_test', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def test_runtime_tree_hash_detects_mutated_library(tmp_path):
    manifest = module()
    (tmp_path / 'lib').mkdir()
    (tmp_path / 'lib/libengine.so').write_bytes(b'original-library')
    inventory = manifest.inventory(tmp_path)
    assert inventory['lib/libengine.so'] == 'bb521a325f7f92f47b1cc12952d1bbdeb7ee77c1e97d3e96ec3a934487701e9c'
    (tmp_path / 'lib/libengine.so').write_bytes(b'mutated-library')
    with pytest.raises(RuntimeError, match='libengine.so'):
        manifest.verify_inventory(tmp_path, inventory)


def test_runtime_tree_hash_detects_deleted_and_unexpected_files(tmp_path):
    manifest = module()
    (tmp_path / 'wheel').write_bytes(b'wheel')
    inventory = manifest.inventory(tmp_path)
    (tmp_path / 'wheel').unlink()
    (tmp_path / 'foreign-library').write_bytes(b'wrong')
    with pytest.raises(RuntimeError, match='missing.*wheel'):
        manifest.verify_inventory(tmp_path, inventory)


def test_runtime_inventory_tracks_symlinks_and_rejects_external_targets(tmp_path):
    manifest = module()
    (tmp_path / 'python').write_bytes(b'python')
    (tmp_path / 'python3').symlink_to('python')
    assert manifest.inventory(tmp_path)['python3'] == {'symlink': 'python'}
    (tmp_path / 'foreign').symlink_to('/usr/local/lib/python3.12/site-packages')
    with pytest.raises(RuntimeError, match='outside'):
        manifest.inventory(tmp_path)


def test_manifest_cli_round_trip_and_corruption(tmp_path):
    import json
    import subprocess
    import sys
    env = tmp_path / 'env'; cuda = tmp_path / 'cuda'; kernels = tmp_path / 'kernels'
    for root in (env, cuda, kernels):
        root.mkdir(); (root / 'artifact').write_text('artifact')
    inputs = tmp_path / 'inputs.json'; inputs.write_text('{"python":"test-fixture"}')
    manifest = tmp_path / 'manifest.json'
    command = [sys.executable, str(ROOT / 'docker/deepseek-runtime-manifest.py'),
        '--root',str(env),'--cuda',str(cuda),'--kernels',str(kernels),'--manifest',str(manifest)]
    created = subprocess.run(command + ['create','--inputs',str(inputs)], capture_output=True, text=True)
    assert created.returncode == 0, created.stderr
    data = json.loads(manifest.read_text())
    assert len(data['derived_content_sha256']) == 64
    assert data['inputs'] == {'python':'test-fixture'}
    verified = subprocess.run(command + ['verify'], capture_output=True, text=True)
    assert verified.returncode == 0, verified.stderr
    (kernels / 'artifact').write_text('corrupted')
    corrupted = subprocess.run(command + ['verify'], capture_output=True, text=True)
    assert corrupted.returncode != 0
    assert 'artifact' in corrupted.stderr


def test_runtime_version_guard_rejects_cuda_build_substitution():
    path = ROOT / 'docker/verify-deepseek-runtime.py'
    assert path.exists(), 'isolated runtime version verifier is missing'
    spec = importlib.util.spec_from_file_location('runtime_verify_test', path)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    packages = {'vllm':'0.32.0+cu129','torch':'2.13.0+cu129','flashinfer-python':'0.7.0.post1',
        'flashinfer-cubin':'0.7.0.post1','flashinfer-jit-cache':'0.7.0.post1+cu129',
        'compressed-tensors':'0.17.0','xgrammar':'0.2.8','nvidia-nccl-cu12':'2.29.7'}
    verifier.check_versions(packages)
    packages['torch'] = '2.13.0+cu130'
    with pytest.raises(RuntimeError, match='torch'):
        verifier.check_versions(packages)


def test_inventory_rejects_missing_runtime_tree(tmp_path):
    with pytest.raises(RuntimeError, match='missing'):
        module().inventory(tmp_path / 'missing')


def test_inventory_rejects_broken_runtime_symlink(tmp_path):
    (tmp_path / 'broken').symlink_to('missing-library')
    with pytest.raises(RuntimeError, match='broken'):
        module().inventory(tmp_path)


def test_backend_import_gate_surfaces_dependency_error(tmp_path):
    path = ROOT / 'docker/verify-deepseek-runtime.py'
    spec = importlib.util.spec_from_file_location('runtime_verify_test', path)
    verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
    (tmp_path / 'model.py').write_text('raise ImportError("missing engine API")')
    with pytest.raises(ImportError, match='missing engine API'):
        verifier.check_backend_import(tmp_path)


@pytest.mark.parametrize('cuda_major',[12,13,None])
def test_cutlass_report_observes_loaded_flavor_without_import(tmp_path,monkeypatch,cuda_major):
    import sys
    from types import ModuleType
    path=ROOT/'docker/verify-deepseek-runtime.py'
    spec=importlib.util.spec_from_file_location('runtime_verify_test',path)
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    folder=tmp_path/'lib/python3.12/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/_mlir/_mlir_libs'
    folder.mkdir(parents=True)
    cu12=folder/'_cutlass_ir.cu12.cpython-312-x86_64-linux-gnu.so';cu12.touch()
    cu13=folder/'_cutlass_ir.cu13.cpython-312-x86_64-linux-gnu.so';cu13.touch()
    # Importing this package is an unwanted observable side effect, even if the
    # extension's origin is already present in sys.modules.
    trap=tmp_path/'import-trap/cutlass';trap.mkdir(parents=True)
    marker=tmp_path/'cutlass-imported'
    (trap/'__init__.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\nraise AssertionError("forced cutlass import")\n')
    monkeypatch.syspath_prepend(str(trap.parent))
    monkeypatch.delitem(sys.modules,'cutlass',raising=False)
    name='cutlass._mlir._mlir_libs._cutlass_ir'
    monkeypatch.delitem(sys.modules,name,raising=False)
    origin=None if cuda_major is None else str(cu12 if cuda_major==12 else cu13)
    if cuda_major is not None:
        loaded=ModuleType(name);loaded.__file__=origin
        monkeypatch.setitem(sys.modules,name,loaded)
    result=verifier.cutlass_flavor_report(tmp_path)
    assert result['status']==('not_loaded' if cuda_major is None else 'loaded')
    assert result['path']==origin
    assert result['cuda_major']==cuda_major
    assert result['available']==[{'cuda_major':12,'path':str(cu12)},
                                 {'cuda_major':13,'path':str(cu13)}]
    assert not marker.exists()


@pytest.fixture
def flashinfer_artifacts(tmp_path,monkeypatch):
    """Official 0.7 provider API over real small installed-distribution records."""
    import csv
    import sys
    from types import ModuleType, SimpleNamespace
    spec=importlib.util.spec_from_file_location('runtime_verify_flashinfer_test',ROOT/'docker/verify-deepseek-runtime.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    root=tmp_path/'env';site=root/'lib/python3.12/site-packages'
    source=site/'flashinfer/jit/env.py';source.parent.mkdir(parents=True);source.touch()
    cubins=site/'flashinfer_cubin/cubins';cubins.mkdir(parents=True)
    (cubins/'kernel.cubin').write_bytes(b'packaged-cubin')
    providers=[];metadata={};modules={}
    for arch in ('sm80','sm89','sm90a','sm100a','sm103a','sm120f'):
        package='flashinfer_jit_cache_'+arch;distribution=package.replace('_','-')
        folder=site/package;folder.mkdir();origin=folder/'__init__.py';origin.touch()
        cache=folder/'jit_cache';binary=cache/'decode'/'decode.so'
        binary.parent.mkdir(parents=True);binary.write_bytes(b'packaged-aot')
        record=site/(package+'-0.7.0.post1+cu129.dist-info');record.mkdir()
        (record/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {distribution}\nVersion: 0.7.0.post1+cu129\n')
        (record/'entry_points.txt').write_text(f'[flashinfer.jit_cache.providers]\n{arch} = {package}:manifest\n')
        with (record/'RECORD').open('w',newline='') as stream:
            csv.writer(stream).writerows([(str(path.relative_to(site)),'','') for path in (origin,binary,record/'METADATA',record/'entry_points.txt')])
        loaded=ModuleType(package);loaded.__file__=str(origin);monkeypatch.setitem(sys.modules,package,loaded)
        provider=SimpleNamespace(provider_id=arch,distribution=distribution,version='0.7.0.post1+cu129',
            jit_cache_dir=cache,cuda_architectures=frozenset({arch}),modules=frozenset({'decode'}))
        providers.append(provider);metadata[arch]=record;modules[arch]=loaded
    builtin=site/'flashinfer/data/aot'
    env=SimpleNamespace(__file__=str(source),FLASHINFER_CUBIN_DIR=cubins,FLASHINFER_AOT_DIR=builtin,
        FLASHINFER_AOT_PROVIDERS=tuple(providers),FLASHINFER_AOT_DIRS=(builtin,*(p.jit_cache_dir for p in providers)))
    return SimpleNamespace(verifier=verifier,root=root,site=site,env=env,providers=providers,
                           metadata=metadata,modules=modules,builtin=builtin)


@pytest.mark.parametrize('builtin_present',[False,True])
def test_flashinfer_optional_builtin_does_not_replace_required_providers(flashinfer_artifacts,builtin_present):
    f=flashinfer_artifacts
    if builtin_present:f.builtin.mkdir(parents=True)
    result=f.verifier.check_flashinfer_artifacts(f.root,f.env)
    assert result['builtin_aot']=={'path':str(f.builtin),'present':builtin_present}
    assert len(result['providers'])==6
    assert {p['distribution'] for p in result['providers']}=={p.distribution for p in f.providers}
    assert result['aot']==[str(p.jit_cache_dir) for p in f.providers]


@pytest.mark.parametrize('failure',[
    'missing_provider','missing_distribution','wrong_version','missing_directory','foreign_directory',
    'foreign_module_origin','missing_binary','unowned_binary','unknown_provider','duplicate_provider',
    'unknown_search_path','fake_builtin','foreign_builtin','missing_cubins','foreign_cubins',
    'foreign_metadata','wrong_installed_version','wrong_provider_id','foreign_binary',
    'unknown_registered_provider','empty_cubins',
])
def test_flashinfer_rejects_broken_inventory_even_with_builtin(flashinfer_artifacts,tmp_path,failure):
    import shutil
    from types import SimpleNamespace
    f=flashinfer_artifacts;f.builtin.mkdir(parents=True)
    first=f.providers[0]
    if failure=='missing_provider':f.env.FLASHINFER_AOT_PROVIDERS=tuple(f.providers[1:])
    elif failure=='missing_distribution':shutil.rmtree(f.metadata['sm80'])
    elif failure=='wrong_version':first.version='0.7.0+cu129'
    elif failure=='missing_directory':shutil.rmtree(first.jit_cache_dir)
    elif failure=='foreign_directory':
        first.jit_cache_dir=tmp_path/'foreign';first.jit_cache_dir.mkdir()
    elif failure=='foreign_module_origin':
        foreign=tmp_path/'foreign.py';foreign.touch();f.modules['sm80'].__file__=str(foreign)
    elif failure=='missing_binary':(first.jit_cache_dir/'decode/decode.so').unlink()
    elif failure=='unowned_binary':
        record=f.metadata['sm80']/'RECORD'
        record.write_text('\n'.join(line for line in record.read_text().splitlines() if 'decode/decode.so' not in line)+'\n')
    elif failure=='unknown_provider':
        extra=SimpleNamespace(**vars(first));extra.distribution='unknown-cache';extra.provider_id='unknown'
        f.env.FLASHINFER_AOT_PROVIDERS=(*f.providers,extra)
    elif failure=='duplicate_provider':f.env.FLASHINFER_AOT_PROVIDERS=(*f.providers,first)
    elif failure=='unknown_search_path':f.env.FLASHINFER_AOT_DIRS=(*f.env.FLASHINFER_AOT_DIRS,f.builtin.parent)
    elif failure=='fake_builtin':f.env.FLASHINFER_AOT_DIR=f.builtin.parent
    elif failure=='foreign_builtin':
        f.builtin.rmdir();foreign=tmp_path/'foreign';foreign.mkdir();f.builtin.symlink_to(foreign,target_is_directory=True)
    elif failure=='missing_cubins':shutil.rmtree(f.env.FLASHINFER_CUBIN_DIR)
    elif failure=='foreign_cubins':
        f.env.FLASHINFER_CUBIN_DIR=tmp_path/'foreign';f.env.FLASHINFER_CUBIN_DIR.mkdir()
    elif failure=='foreign_metadata':
        record=f.metadata['sm80'];foreign=tmp_path/'foreign-dist-info';record.rename(foreign)
        record.symlink_to(foreign,target_is_directory=True)
    elif failure=='wrong_installed_version':
        p=f.metadata['sm80']/'METADATA';p.write_text(p.read_text().replace('0.7.0.post1+cu129','0.7.0+cu129'))
    elif failure=='wrong_provider_id':first.provider_id='different-entry-point'
    elif failure=='foreign_binary':
        binary=first.jit_cache_dir/'decode/decode.so';binary.unlink()
        foreign=tmp_path/'foreign.so';foreign.touch();binary.symlink_to(foreign)
    elif failure=='unknown_registered_provider':
        record=f.site/'unknown_cache-0.7.0.post1+cu129.dist-info';record.mkdir()
        (record/'METADATA').write_text('Metadata-Version: 2.1\nName: unknown-cache\nVersion: 0.7.0.post1+cu129\n')
        (record/'entry_points.txt').write_text('[flashinfer.jit_cache.providers]\nunknown = unknown_cache:manifest\n')
    elif failure=='empty_cubins':(f.env.FLASHINFER_CUBIN_DIR/'kernel.cubin').unlink()
    reason={'unowned_binary':'FlashInfer provider binary is not owned',
            'foreign_metadata':'FlashInfer artifact is missing or outside',
            'wrong_version':'FlashInfer active provider differs',
            'wrong_installed_version':'FlashInfer provider version mismatch'}.get(failure,'FlashInfer')
    with pytest.raises(RuntimeError,match=reason):
        f.verifier.check_flashinfer_artifacts(f.root,f.env)
