"""Tiny real snapshots exercise structural admission without tensor loading."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[1]
HELPER=ROOT/'backends/vllm_deepseek/checkpoint_preflight.py'
def helper():
    assert HELPER.exists(), 'bounded checkpoint admission is missing'
    spec=importlib.util.spec_from_file_location('checkpoint_test',HELPER)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def shard(path,header,payload=4):
    raw=header if isinstance(header,bytes) else json.dumps(header).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(raw)));f.write(raw);f.truncate(8+len(raw)+payload)

@pytest.fixture
def snapshot(tmp_path):
    root=tmp_path/'snapshot';root.mkdir()
    for name,value in [('config.json',{'model_type':'deepseek_v41','architectures':['DeepseekV41ForCausalLM']}),('tokenizer_config.json',{}),('tokenizer.json',{})]:
        (root/name).write_text(json.dumps(value))
    (root/'chat_template.jinja').write_text('{{ messages }}')
    shard(root/'a.safetensors',{'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]},'empty':{'dtype':'F32','shape':[0],'data_offsets':[4,4]}})
    shard(root/'b.safetensors',{'scalar':{'dtype':'F32','shape':[],'data_offsets':[0,4]}})
    (root/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':8},'weight_map':{'a':'a.safetensors','empty':'a.safetensors','scalar':'b.safetensors'}}))
    return root

def validate(root,**config):return helper().validate_checkpoint({'model':str(root),**config})

def test_valid_scalar_empty_repeated_mapping(snapshot):
    r=validate(snapshot)
    assert r['validation_level']=='structural_headers_only'
    assert (r['shard_count'],r['tensor_count'],r['tensor_payload_bytes'])==(2,3,8)
    assert r['serialized_bytes']>8

def test_sparse_huge_checkpoint_reads_headers_only(snapshot,monkeypatch):
    m=helper();size=100*1024**3
    shard(snapshot/'a.safetensors',{'a':{'dtype':'I8','shape':[size],'data_offsets':[0,size]}},size)
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'a.safetensors'}}))
    original=m.os.pread;reads=[]
    def bounded(fd,count,offset):
        assert count<=m.HEADER_LIMIT and offset<1000,'payload read attempted'
        reads.append(count);return original(fd,count,offset)
    monkeypatch.setattr(m.os,'pread',bounded)
    r=m.validate_checkpoint({'model':str(snapshot)})
    assert r['tensor_payload_bytes']==size and r['allocated_bytes']<1024**2 and sum(reads)<10000

@pytest.mark.parametrize('name',['config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','model.safetensors.index.json','a.safetensors'])
def test_missing_file(snapshot,name):
    (snapshot/name).unlink()
    with pytest.raises(ValueError,match=name):validate(snapshot)

@pytest.mark.parametrize('name',['config.json','model.safetensors.index.json','a.safetensors','tokenizer.json'])
def test_lfs_pointer(snapshot,name):
    (snapshot/name).write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 999\n')
    with pytest.raises(ValueError,match='LFS'):validate(snapshot)

@pytest.mark.parametrize('bad',['../a.safetensors','/a.safetensors','a\\b.safetensors','a//b.safetensors','./a.safetensors','C:a.safetensors','https:a.safetensors','a\x00.safetensors'])
def test_unsafe_paths(snapshot,bad):
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':bad}}))
    with pytest.raises(ValueError,match='shard path'):validate(snapshot)

@pytest.mark.parametrize('header,payload,error',[
    ({'a':{'dtype':'F4','shape':[8],'data_offsets':[0,4]}},4,'dtype'),
    ({'a':{'dtype':'BF16','shape':[True],'data_offsets':[0,4]}},4,'shape'),
    ({'a':{'dtype':'BF16','shape':[-1],'data_offsets':[0,4]}},4,'shape'),
    ({'a':{'dtype':'BF16','shape':[2.0],'data_offsets':[0,4]}},4,'shape'),
    ({'a':{'dtype':'BF16','shape':[2],'data_offsets':[False,4]}},4,'offset'),
    ({'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,3]}},4,'size'),
    ({'a':{'dtype':'BF16','shape':[2],'data_offsets':[1,5]}},5,'gap'),
    ({'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}},5,'coverage'),
    ({'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}},3,'payload'),
    ({'a':{'dtype':'BF16','shape':[2**63,2**63],'data_offsets':[0,4]}},4,'size'),
    (b'{"a":{},"a":{}}',4,'duplicate'),(b'{"a":NaN}',4,'finite'),(b'[]',4,'object'),(b'\xff',4,'JSON')])
def test_invalid_header(snapshot,header,payload,error):
    shard(snapshot/'a.safetensors',header,payload)
    with pytest.raises(ValueError,match=error):validate(snapshot)

@pytest.mark.parametrize('content,error',[(b'123','truncated'),(struct.pack('<Q',2**32),'header'),(struct.pack('<Q',100)+b'{}','truncated')])
def test_truncated_header(snapshot,content,error):
    (snapshot/'a.safetensors').write_bytes(content)
    with pytest.raises(ValueError,match=error):validate(snapshot)

@pytest.mark.parametrize('content,error',[('{"weight_map":{},"weight_map":{}}','duplicate'),('{"weight_map":{}}','nonempty'),('[]','object'),('{"weight_map":{"a":"a.safetensors"},"metadata":{"total_size":7}}','indexed'),('{"weight_map":{"a":"b.safetensors"}}','indexed')])
def test_index_closure(snapshot,content,error):
    (snapshot/'model.safetensors.index.json').write_text(content)
    with pytest.raises(ValueError,match=error):validate(snapshot)

def test_external_hf_blob_link_and_duplicate_alias(snapshot,tmp_path):
    blob=tmp_path/'blob';(snapshot/'b.safetensors').rename(blob);(snapshot/'b.safetensors').symlink_to(blob)
    assert validate(snapshot)['shard_count']==2
    (snapshot/'alias.safetensors').symlink_to(blob)
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'scalar':'alias.safetensors','other':'b.safetensors'}}))
    with pytest.raises(ValueError,match='duplicate shard alias'):validate(snapshot)

@pytest.mark.parametrize('kind',['directory','fifo','broken','loop','hardlink'])
def test_nonregular_and_duplicate_inode(snapshot,kind):
    p=snapshot/'b.safetensors';p.unlink()
    if kind=='directory':p.mkdir()
    elif kind=='fifo':os.mkfifo(p)
    elif kind=='broken':p.symlink_to('missing')
    elif kind=='loop':p.symlink_to(p)
    else:os.link(snapshot/'a.safetensors',p)
    with pytest.raises(ValueError):validate(snapshot)

def test_mutation(snapshot,monkeypatch):
    m=helper();original=m.os.pread
    def read(fd,count,offset):
        result=original(fd,count,offset)
        if offset==8:
            with (snapshot/'a.safetensors').open('ab') as f:f.write(b'!')
        return result
    monkeypatch.setattr(m.os,'pread',read)
    with pytest.raises(ValueError,match='changed'):m.validate_checkpoint({'model':str(snapshot)})

@pytest.mark.parametrize('config,error',[({'model':'org/model'},'absolute'),({'model':'https://host/model'},'absolute'),({'hf_overrides':{}},'override'),({'hf_config_path':'x'},'override')])
def test_engine_root_identity(snapshot,config,error):
    with pytest.raises(ValueError,match=error):helper().validate_checkpoint({'model':str(snapshot),**config})

def test_wrong_model_and_missing_runtime_registration(snapshot):
    m=helper()
    with pytest.raises(ValueError,match='DeepseekV41'):m.validate_runtime_registries([],{'deepseek_v41':object()},'0.32')
    with pytest.raises(ValueError,match='deepseek_v41'):m.validate_runtime_registries(['DeepseekV41ForCausalLM'],{},'0.32')
    (snapshot/'config.json').write_text('{"model_type":"wrong","architectures":["DeepseekV41ForCausalLM"]}')
    with pytest.raises(ValueError,match='model_type'):validate(snapshot)

def test_json_limits(snapshot,monkeypatch):
    m=helper();monkeypatch.setattr(m,'INDEX_LIMIT',16)
    with pytest.raises(ValueError,match='limit'):m.validate_checkpoint({'model':str(snapshot)})
    (snapshot/'config.json').write_text('['*100+'0'+']'*100)
    with pytest.raises(ValueError,match='nesting'):validate(snapshot)

def test_cli_structural_mode(snapshot,tmp_path):
    cfg=tmp_path/'model.json';cfg.write_text(json.dumps({'model':str(snapshot)}))
    cmd=[sys.executable,str(ROOT/'scripts/validate_deepseek_checkpoint.py'),str(cfg),'--json']
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
    assert r.returncode==0,r.stderr
    assert json.loads(r.stdout)['validation_level']=='structural_headers_only'
    (snapshot/'a.safetensors').unlink();r=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
    assert r.returncode!=0 and 'a.safetensors' in r.stderr

@pytest.mark.parametrize('raw',[b'{"weight_map":{},"x":1e999}',b'{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}'])
def test_recursive_duplicate_and_overflow_float(snapshot,raw):
    (snapshot/'model.safetensors.index.json').write_bytes(raw)
    with pytest.raises(ValueError,match='finite|duplicate'):validate(snapshot)

def test_overlap_duplicate_total_size_and_wrong_architecture(snapshot):
    m=helper();index=snapshot/'model.safetensors.index.json'
    data=json.loads(index.read_text());data['metadata']['total_size']=9;index.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='total_size'):validate(snapshot)
    data['metadata']['total_size']=8;index.write_text(json.dumps(data))
    shard(snapshot/'a.safetensors',{'a':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]},'empty':{'dtype':'I8','shape':[1],'data_offsets':[3,4]}})
    with pytest.raises(ValueError,match='overlap'):validate(snapshot)
    (snapshot/'config.json').write_text('{"model_type":"deepseek_v41","architectures":["Other"]}')
    with pytest.raises(ValueError,match='architectures'):validate(snapshot)


def test_tiny_shards_never_read_across_actual_payload_boundary(snapshot,monkeypatch):
    m=helper();original=m.os.pread;boundaries={}
    for name in ('a.safetensors','b.safetensors'):
        with (snapshot/name).open('rb') as stream:
            length,=struct.unpack('<Q',stream.read(8))
        boundaries[(snapshot/name).stat().st_ino]=8+length
    def headers_only(fd,count,offset):
        boundary=boundaries.get(os.fstat(fd).st_ino)
        if boundary is not None:
            assert offset+count<=boundary, 'read crosses safetensors payload boundary'
        return original(fd,count,offset)
    monkeypatch.setattr(m.os,'pread',headers_only)
    assert m.validate_checkpoint({'model':str(snapshot)})['tensor_payload_bytes']==8


def ue8m0_snapshot(snapshot):
    shard(snapshot/'a.safetensors',{'layer.scale':{'dtype':'F8_E8M0','shape':[2,2],'data_offsets':[0,4]}})
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({
        'metadata':{'total_size':8},
        'weight_map':{'layer.scale':'a.safetensors','scalar':'b.safetensors'}}))
    return snapshot


def test_ue8m0_headers_are_supported_without_claiming_runtime_compatibility(snapshot,monkeypatch):
    root=ue8m0_snapshot(snapshot)
    cfg=json.loads((root/'config.json').read_text())
    cfg['quantization_config']={'quant_method':'fp8','scale_fmt':'ue8m0','weight_block_size':[32,32]}
    (root/'config.json').write_text(json.dumps(cfg))
    m=helper();original=m.os.pread
    boundary=8+struct.unpack('<Q',(root/'a.safetensors').read_bytes()[:8])[0]
    inode=(root/'a.safetensors').stat().st_ino
    def headers_only(fd,count,offset):
        if os.fstat(fd).st_ino==inode:
            assert offset+count<=boundary,'UE8M0 tensor payload was read'
        return original(fd,count,offset)
    monkeypatch.setattr(m.os,'pread',headers_only)
    report=m.validate_checkpoint({'model':str(root)})
    assert report['serialized_dtypes']==['F32','F8_E8M0']
    assert report['tensor_dtype_counts']=={'F32':1,'F8_E8M0':1}
    assert report['quantization_profile']=={'quant_method':'fp8','scale_fmt':'ue8m0','weight_block_size':[32,32]}
    assert report['runtime_validation']=='not_checked'
    assert report['validation_level']=='structural_headers_only'
    assert report['runtime_format_checks_required']==['F8_E8M0']


@pytest.mark.parametrize('dtype,shape,offsets,payload,error',[
    ('F8_E8M0',[2,2],[0,3],4,'size'),
    ('F8_E8M0',[2,2],[0,4],3,'payload'),
    ('F8_E8M0',[2,2],[1,5],5,'gap'),
    ('F8_E8M0_BOGUS',[2,2],[0,4],4,'dtype'),
])
def test_ue8m0_keeps_header_bounds_and_dtype_validation(snapshot,dtype,shape,offsets,payload,error):
    ue8m0_snapshot(snapshot)
    shard(snapshot/'a.safetensors',{'layer.scale':{'dtype':dtype,'shape':shape,'data_offsets':offsets}},payload)
    with pytest.raises(ValueError,match=error):validate(snapshot)


def runtime_modules(monkeypatch, *, has_dtype=True, broken_load=False, wrong_dtype=False, probe_paths=None, eager_works=False):
    from contextlib import contextmanager
    from types import ModuleType,SimpleNamespace
    vllm=ModuleType('vllm');vllm.__version__='0.32.0'
    models=ModuleType('vllm.model_executor.models')
    models.ModelRegistry=SimpleNamespace(get_supported_archs=lambda:['DeepseekV41ForCausalLM'])
    registry=ModuleType('vllm.tokenizers.registry')
    registry.TokenizerRegistry=SimpleNamespace(tokenizers={'deepseek_v41':object()})
    torch=ModuleType('torch');torch.uint8=object()
    if has_dtype:torch.float8_e8m0fnu=object()
    safe=ModuleType('safetensors');safe_torch=ModuleType('safetensors.torch');calls=[]
    def decode(raw):
        length,=struct.unpack('<Q',raw[:8]);header=json.loads(raw[8:8+length]);payload=raw[8+length:]
        assert header=={'scale':{'dtype':'F8_E8M0','shape':[4],'data_offsets':[0,4]}}
        assert payload==bytes([0,126,127,254])
        calls.append(payload)
        if broken_load:raise RuntimeError('unsupported safetensors dtype')
        return SimpleNamespace(dtype='wrong' if wrong_dtype else torch.float8_e8m0fnu,
            shape=(4,),device=SimpleNamespace(type='cpu'),
            view=lambda dtype:SimpleNamespace(tolist=lambda:list(payload)))
    @contextmanager
    def safe_open(path, *, framework, device):
        assert framework=='pt' and device=='cpu'
        path=Path(path)
        if probe_paths is not None:probe_paths.append(path)
        raw=path.read_bytes()
        def get_tensor(name):
            assert name=='scale'
            return decode(raw)
        yield SimpleNamespace(get_tensor=get_tensor)
    def eager_load(raw):
        if not eager_works:raise KeyError('F8_E8M0') # eager can fail while safe_open works
        return {'scale':decode(raw)}
    safe_torch.load=eager_load;safe_torch.safe_open=safe_open
    for name,value in [('vllm',vllm),('vllm.model_executor.models',models),
            ('vllm.tokenizers.registry',registry),('torch',torch),
            ('safetensors',safe),('safetensors.torch',safe_torch)]:
        monkeypatch.setitem(sys.modules,name,value)
    return calls


def test_runtime_ue8m0_probe_checks_actual_serialization_and_keeps_qualification_explicit(snapshot,monkeypatch):
    report=validate(ue8m0_snapshot(snapshot),safetensors_load_strategy='lazy')
    paths=[];calls=runtime_modules(monkeypatch,probe_paths=paths)
    runtime=helper().check_runtime(report)
    assert calls==[bytes([0,126,127,254])]
    assert len(paths)==1 and not paths[0].exists()
    assert not paths[0].parent.exists()
    assert runtime['serialized_formats']=={'F8_E8M0':'cpu_deserialization_verified'}
    assert runtime['verified_safetensors_load_strategy']=='lazy'
    assert report['safetensors_load_strategy']=='lazy'
    assert runtime['gpu_qualified'] is False
    assert report['runtime_validation']=='not_checked' # structural report not silently promoted


@pytest.mark.parametrize('kwargs',[{'has_dtype':False},{'broken_load':True},{'wrong_dtype':True}])
def test_runtime_ue8m0_probe_fails_closed(snapshot,monkeypatch,kwargs):
    report=validate(ue8m0_snapshot(snapshot),safetensors_load_strategy='lazy')
    paths=[];runtime_modules(monkeypatch,probe_paths=paths,**kwargs)
    with pytest.raises(RuntimeError,match='F8_E8M0'):
        helper().check_runtime(report)
    assert all(not path.exists() and not path.parent.exists() for path in paths)


def test_runtime_ue8m0_probe_requires_installed_decoder(snapshot,monkeypatch):
    report=validate(ue8m0_snapshot(snapshot),safetensors_load_strategy='lazy')
    runtime_modules(monkeypatch)
    monkeypatch.setitem(sys.modules,'safetensors.torch',None)
    with pytest.raises(RuntimeError,match='F8_E8M0'):
        helper().check_runtime(report)


@pytest.mark.parametrize('strategy',[None,'auto','unknown','torchao'])
def test_runtime_ue8m0_rejects_unqualified_strategy(snapshot,monkeypatch,strategy):
    report=validate(ue8m0_snapshot(snapshot),safetensors_load_strategy=strategy)
    calls=runtime_modules(monkeypatch)
    with pytest.raises(RuntimeError,match='safetensors_load_strategy.*lazy'):
        helper().check_runtime(report)
    assert calls==[]
    assert report['safetensors_load_strategy']==strategy


@pytest.mark.parametrize('eager_works',[False,True])
def test_runtime_ue8m0_eager_checks_the_eager_decoder(snapshot,monkeypatch,eager_works):
    report=validate(ue8m0_snapshot(snapshot),safetensors_load_strategy='eager')
    paths=[];calls=runtime_modules(monkeypatch,eager_works=eager_works,probe_paths=paths)
    if eager_works:
        runtime=helper().check_runtime(report)
        assert runtime['verified_safetensors_load_strategy']=='eager'
        assert calls==[bytes([0,126,127,254])]
    else:
        with pytest.raises(RuntimeError,match='F8_E8M0'):
            helper().check_runtime(report)
    assert paths==[] # an eager failure must not fall back to lazy deserialization


def test_legacy_runtime_registry_check_needs_no_format_probe(snapshot,monkeypatch):
    calls=runtime_modules(monkeypatch,has_dtype=False)
    runtime=helper().check_runtime(validate(snapshot))
    assert calls==[]
    assert runtime['serialized_formats']=={}


def test_structural_ue8m0_cli_never_imports_runtime_or_reads_tensor_payload(snapshot,tmp_path):
    root=ue8m0_snapshot(snapshot)
    cfg=tmp_path/'model.json';cfg.write_text(json.dumps({'model':str(root)}))
    # A separate interpreter makes this assertion independent of pytest imports.
    code='''import builtins,runpy,sys
original=builtins.__import__
def guarded(name,*args,**kwargs):
    assert name.split('.')[0] not in {'torch','vllm','safetensors'}, 'structural preflight imported a runtime'
    return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=[sys.argv[1],sys.argv[2],'--json']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'scripts/validate_deepseek_checkpoint.py'),str(cfg)],
        capture_output=True,text=True,timeout=5)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['serialized_dtypes']==['F32','F8_E8M0']
    assert report['runtime_validation']=='not_checked'


def test_nested_quantization_profile_is_reported_without_being_promoted_to_validation(snapshot):
    config=json.loads((snapshot/'config.json').read_text())
    config['text_config']={'quantization_config':{'quant_method':'fp8','scale_fmt':'ue8m0'}}
    (snapshot/'config.json').write_text(json.dumps(config))
    report=validate(snapshot)
    assert report['quantization_profile']=={'quant_method':'fp8','scale_fmt':'ue8m0'}
    assert report['serialized_dtypes']==['BF16','F32']
    assert report['runtime_format_checks_required']==[]


def test_real_cpu_ue8m0_serialization_if_runtime_is_installed():
    pytest.importorskip('torch')
    pytest.importorskip('safetensors.torch')
    # Real dependency qualification, with no model weights or GPU initialization.
    helper()._check_ue8m0_serialization()
