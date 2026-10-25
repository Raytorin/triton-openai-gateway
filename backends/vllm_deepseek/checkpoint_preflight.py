# SPDX-License-Identifier: Apache-2.0
"""Bounded, stdlib-only local checkpoint admission. Never reads tensor payloads.

This checks serialization structure, not hashes, GPU capacity or Triton READY.
Snapshot paths may be HF cache symlinks; publish immutable snapshots because the
engine reopens files after this check. Limits below are local admission policy.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import stat
import struct

INDEX_LIMIT = 32 * 1024**2
CONFIG_LIMIT = 2 * 1024**2
HEADER_LIMIT = 16 * 1024**2
SHARD_LIMIT = 4096
TENSOR_LIMIT = 1_000_000
DEPTH_LIMIT = 64
LFS = b'version https://git-lfs.github.com/spec/v1'
DTYPE_BITS = {'BOOL':8, 'U8':8, 'I8':8, 'F8_E4M3':8, 'F8_E5M2':8,
              'I16':16, 'U16':16, 'F16':16, 'BF16':16,
              'I32':32, 'U32':32, 'F32':32, 'I64':64, 'U64':64, 'F64':64}
ARCHITECTURE = 'DeepseekV41ForCausalLM'


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@contextmanager
def _regular(path):
    path = Path(path)
    descriptor = None
    try:
        resolved = path.resolve(strict=True)
        before = resolved.stat()
        descriptor = os.open(resolved, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOFOLLOW)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f'{path}: expected regular file')
        if _identity(before) != _identity(opened):
            raise ValueError(f'{path}: changed before validation')
        prefix = os.pread(descriptor, 8, 0)
        if prefix == LFS[:8] and os.pread(descriptor, len(LFS), 0) == LFS:
            raise ValueError(f'{path}: Git LFS pointer; materialize the actual file')
        yield descriptor, opened
        if (_identity(opened) != _identity(os.fstat(descriptor))
                or path.resolve(strict=True) != resolved
                or _identity(opened) != _identity(resolved.stat())):
            raise ValueError(f'{path}: changed during validation; use an immutable snapshot')
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{path}: cannot validate regular local file: {error}') from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def _parse(raw, path):
    try:
        text = raw.decode('utf-8')
        depth, quoted, escaped = 0, False, False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == '\\':
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in '[{':
                depth += 1
                if depth > DEPTH_LIMIT:
                    raise ValueError(f'JSON nesting exceeds local limit {DEPTH_LIMIT}')
            elif char in ']}':
                depth -= 1
        def invalid_constant(value):
            raise ValueError(f'non-finite JSON value {value}')
        def finite_float(value):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f'non-finite JSON value {value}')
            return number
        result = json.loads(text, object_pairs_hook=_object_pairs,
                            parse_constant=invalid_constant, parse_float=finite_float)
        if not isinstance(result, dict):
            raise ValueError('JSON root must be an object')
        return result
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError(f'{path}: invalid JSON: {error}') from error


def read_json(path, limit=CONFIG_LIMIT):
    with _regular(path) as (descriptor, info):
        if info.st_size > limit:
            raise ValueError(f'{path}: JSON exceeds local byte limit {limit}')
        raw = os.pread(descriptor, info.st_size, 0)
        if len(raw) != info.st_size:
            raise ValueError(f'{path}: truncated JSON')
        return _parse(raw, path)


def _local_root(value, name):
    if not isinstance(value, str) or not value or not Path(value).is_absolute():
        raise ValueError(f'{name}: require an absolute local snapshot directory (materialize Hub/remote models first)')
    try:
        root = Path(value).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f'{name}: not a directory: {root}')
        return root
    except (OSError, RuntimeError) as error:
        raise ValueError(f'{name}: invalid snapshot directory {value}: {error}') from error


def _shard_name(value):
    if (not isinstance(value, str) or not value.endswith('.safetensors')
            or '\\' in value or '\x00' in value or ':' in value
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError(f'unsafe shard path {value!r}; require normalized relative POSIX .safetensors path')
    return value


def _validate_header(path, descriptor, info):
    prefix = os.pread(descriptor, 8, 0)
    if len(prefix) != 8:
        raise ValueError(f'{path}: truncated safetensors prefix')
    length, = struct.unpack('<Q', prefix)
    if length > HEADER_LIMIT:
        raise ValueError(f'{path}: safetensors header exceeds local limit {HEADER_LIMIT}')
    if length > info.st_size - 8:
        raise ValueError(f'{path}: truncated safetensors header')
    raw = os.pread(descriptor, length, 8)
    if len(raw) != length:
        raise ValueError(f'{path}: truncated safetensors header')
    header = _parse(raw, path)
    payload = info.st_size - 8 - length
    spans, names = [], set()
    for name, tensor in header.items():
        if name == '__metadata__':
            if not isinstance(tensor, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in tensor.items()):
                raise ValueError(f'{path}: invalid safetensors metadata')
            continue
        if not name or not isinstance(tensor, dict):
            raise ValueError(f'{path}: invalid tensor {name!r}')
        dtype = tensor.get('dtype')
        if not isinstance(dtype,str) or dtype not in DTYPE_BITS:
            raise ValueError(f'{path}: unsupported serialized dtype {dtype!r} for {name}')
        shape, offsets = tensor.get('shape'), tensor.get('data_offsets')
        if (not isinstance(shape,list) or len(shape)>64
                or any(type(dim) is not int or dim<0 for dim in shape)):
            raise ValueError(f'{path}: invalid shape for {name}')
        if (not isinstance(offsets,list) or len(offsets)!=2
                or any(type(x) is not int or x<0 for x in offsets)):
            raise ValueError(f'{path}: invalid offsets for {name}')
        start, end = offsets
        if end < start or end > payload:
            raise ValueError(f'{path}: tensor {name} extends outside payload')
        elements = 0 if 0 in shape else 1
        for dim in shape:
            elements *= dim
            if elements > payload * 8 + 1:
                raise ValueError(f'{path}: tensor size exceeds payload for {name}')
        bits = elements * DTYPE_BITS[dtype]
        if bits % 8 or end-start != bits//8:
            raise ValueError(f'{path}: serialized tensor size mismatch for {name}')
        names.add(name); spans.append((start,end))
        if len(names)>TENSOR_LIMIT:
            raise ValueError(f'{path}: tensor count exceeds local limit {TENSOR_LIMIT}')
    cursor = 0
    for start,end in sorted(spans):
        if start != cursor:
            issue = 'gap' if start>cursor else 'overlap'
            raise ValueError(f'{path}: tensor span {issue} at byte {start}')
        cursor = end
    if cursor != payload:
        raise ValueError(f'{path}: incomplete tensor coverage of payload')
    return names, payload


def validate_runtime_registries(architectures, tokenizers, version):
    """Membership only: do not import native model classes or launch inspection."""
    if ARCHITECTURE not in architectures:
        raise ValueError(f'vLLM {version}: missing native architecture registration {ARCHITECTURE}')
    if 'deepseek_v41' not in tokenizers:
        raise ValueError(f'vLLM {version}: missing tokenizer registration deepseek_v41')


def check_runtime():
    import vllm
    from vllm.model_executor.models import ModelRegistry
    from vllm.tokenizers.registry import TokenizerRegistry
    validate_runtime_registries(ModelRegistry.get_supported_archs(), TokenizerRegistry.tokenizers, vllm.__version__)


def validate_checkpoint(engine_config):
    """Validate the actual model root read by vLLM; model_weights is bookkeeping."""
    if not isinstance(engine_config,dict):
        raise ValueError('engine configuration must be an object')
    if any(key in engine_config for key in ('hf_overrides','hf_config_path')):
        raise ValueError('HF config overrides are unsupported by local checkpoint admission')
    root = _local_root(engine_config.get('model'), 'model')
    config = read_json(root/'config.json')
    if config.get('model_type') != 'deepseek_v41':
        raise ValueError(f'{root}/config.json: model_type must be deepseek_v41')
    if config.get('architectures') != [ARCHITECTURE]:
        raise ValueError(f'{root}/config.json: architectures must contain only {ARCHITECTURE}')
    tokenizer_root = _local_root(engine_config.get('tokenizer',str(root)), 'tokenizer')
    read_json(tokenizer_root/'tokenizer_config.json')
    for name in ('tokenizer.json','chat_template.jinja'):
        with _regular(tokenizer_root/name) as (_,info):
            if info.st_size == 0:
                raise ValueError(f'{tokenizer_root/name}: empty tokenizer artifact')
    index = read_json(root/'model.safetensors.index.json', INDEX_LIMIT)
    weight_map = index.get('weight_map')
    if not isinstance(weight_map,dict) or not weight_map:
        raise ValueError('index weight_map must be a nonempty object')
    if len(weight_map)>TENSOR_LIMIT:
        raise ValueError(f'index tensor count exceeds local limit {TENSOR_LIMIT}')
    expected = {}
    for name,filename in weight_map.items():
        if not name or name=='__metadata__':
            raise ValueError('index contains invalid tensor name')
        expected.setdefault(_shard_name(filename),set()).add(name)
    if len(expected)>SHARD_LIMIT:
        raise ValueError(f'index shard count exceeds local limit {SHARD_LIMIT}')
    seen, all_names = set(), set()
    serialized, allocated, payload_bytes, largest = 0, 0, 0, 0
    for filename in sorted(expected):
        path = root/filename
        with _regular(path) as (descriptor,info):
            identity = (info.st_dev,info.st_ino)
            if identity in seen:
                raise ValueError(f'{path}: duplicate shard alias')
            seen.add(identity)
            names,payload = _validate_header(path,descriptor,info)
            if names != expected[filename]:
                raise ValueError(f'{path}: indexed tensor closure mismatch (missing/extra/misassigned tensor)')
            if all_names.intersection(names):
                raise ValueError(f'{path}: duplicate tensor across shards')
            all_names.update(names)
            serialized += info.st_size; allocated += info.st_blocks*512
            payload_bytes += payload; largest = max(largest,info.st_size)
    metadata = index.get('metadata',{})
    if not isinstance(metadata,dict):
        raise ValueError('index metadata must be an object')
    total = metadata.get('total_size')
    if total is not None and (type(total) is not int or total != payload_bytes):
        raise ValueError('index metadata.total_size differs from serialized tensor payload bytes')
    return {'model_root':str(root), 'tokenizer_root':str(tokenizer_root),
            'shard_count':len(expected), 'tensor_count':len(all_names),
            'tensor_payload_bytes':payload_bytes, 'serialized_bytes':serialized,
            'allocated_bytes':allocated, 'largest_shard_bytes':largest,
            'validation_level':'structural_headers_only',
            'supplied_revision':engine_config.get('revision'),
            'limits':{'index_bytes':INDEX_LIMIT,'header_bytes':HEADER_LIMIT,
                      'shards':SHARD_LIMIT,'tensors':TENSOR_LIMIT}}
