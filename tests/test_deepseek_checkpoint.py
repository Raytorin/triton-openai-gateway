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

def validate(root):return helper().validate_checkpoint({'model':str(root)})

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
