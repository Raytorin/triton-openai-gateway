#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local no-weight Triton DeepSeek preflight-failure/exit smoke (inside image).

This never constructs an engine. It does not reproduce a native vLLM constructor
fault or prove GPU qualification. Run separately from a serving Triton process.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def log_snapshot(log):
    log.seek(0, os.SEEK_END)
    length = log.tell()
    stat = os.fstat(log.fileno())
    if length != stat.st_size:
        raise OSError('log changed while its size was inspected')
    return length, stat.st_mtime_ns, stat.st_ctime_ns


def read_log_chunk(log, length):
    chunk = log.read(length)
    if len(chunk) != length:
        raise OSError(f'short log read: expected {length} bytes, received {len(chunk)}')
    return chunk


def read_log_excerpt(log, length):
    """Keep the first cause and final messages without reading an entire log."""
    limit = 32768
    log.seek(0)
    prefix = read_log_chunk(log, min(length, limit))
    tail_length = min(max(0, length - limit), limit)
    omitted = max(0, length - len(prefix) - tail_length)
    separator = f'\n... {omitted} log bytes omitted ...\n'.encode() if omitted else b''
    log.seek(length - tail_length)
    return (prefix + separator + read_log_chunk(log, tail_length)).decode(errors='replace')


def inspect_complete_log(log, snapshot):
    """Decide from a stable complete log, independently of its display excerpt."""
    length = snapshot[0]
    if length > 64 * 1024 * 1024:
        raise OSError('log exceeds the 64 MiB complete-inspection limit')
    markers = (b'deepseek-preflight-probe', b'config.json', b'sigsegv', b'segmentation fault')
    found = set()
    overlap = max(map(len, markers)) - 1
    previous = b''
    remaining = length
    deadline = time.monotonic() + 5
    log.seek(0)
    while remaining:
        if time.monotonic() > deadline:
            raise OSError('complete log inspection exceeded its 5 second limit')
        chunk = read_log_chunk(log, min(remaining, 32768))
        remaining -= len(chunk)
        window = previous + chunk.lower()
        found.update(marker for marker in markers if marker in window)
        previous = window[-overlap:]
    if log_snapshot(log) != snapshot:
        raise OSError('log changed during complete log inspection')
    if time.monotonic() > deadline:
        raise OSError('complete log inspection exceeded its 5 second limit')
    return all(marker in found for marker in markers[:2]), any(marker in found for marker in markers[2:])


def cleanup_probe(process):
    """Bound cleanup to the child/session created solely for this local probe."""
    def send(sig):
        try:
            if os.name == 'posix':
                os.killpg(process.pid, sig)
            else:
                process.send_signal(sig)
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The parent may exit while a normal descendant ignores SIGTERM.
        # Its isolated process group still belongs to this probe.
        send(signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        raise RuntimeError('probe process did not exit after cleanup signals') from None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tritonserver',default='/opt/tritonserver/bin/tritonserver')
    parser.add_argument('--environment',default='/opt/envs/vllm032-cu129')
    parser.add_argument('--timeout',type=float,default=45)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='deepseek-stub-') as directory:
        root=Path(directory);model=root/'DeepSeek-preflight-probe';version=model/'1';version.mkdir(parents=True)
        # A deliberately incomplete local snapshot must fail before GPU creation.
        (version/'model.json').write_text(json.dumps({'model':str(version),'language_model_only':True,'enable_mm_embeds':False}))
        (model/'config.pbtxt').write_text('''name: "DeepSeek-preflight-probe"
backend: "vllm_deepseek"
max_batch_size: 0
model_transaction_policy { decoupled: true }
instance_group [{ kind: KIND_MODEL count: 1 }]
input [{ name: "text_input" data_type: TYPE_STRING dims: [1] }]
output [{ name: "text_output" data_type: TYPE_STRING dims: [-1] }]
parameters { key: "EXECUTION_ENV_PATH" value { string_value: "'''+args.environment+'''" } }
''')
        with tempfile.TemporaryFile() as log:
            process = None
            failure = None
            cleanup_failure = None
            try:
                process = subprocess.Popen([args.tritonserver,'--model-repository='+directory,
                    '--model-control-mode=explicit','--load-model=DeepSeek-preflight-probe',
                    '--exit-on-error=true','--allow-http=false','--allow-grpc=false','--allow-metrics=false'],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=(os.name == 'posix'),
                    env={**os.environ,'DEEPSEEK_CACHE_ARCH':'cpu'})
                returncode = process.wait(timeout=args.timeout)
            except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                failure = 'interrupted' if isinstance(error, KeyboardInterrupt) else str(error)
            finally:
                if failure is not None and process is not None:
                    try:
                        cleanup_probe(process)
                    except (OSError, RuntimeError) as error:
                        cleanup_failure = str(error)
                output = ''
                try:
                    snapshot = log_snapshot(log)
                    output = read_log_excerpt(log, snapshot[0])
                    intended, reported_crash = inspect_complete_log(log, snapshot)
                except (OSError, ValueError) as error:
                    inspection_failure = f'complete Triton log inspection failed: {error}'
                    if failure is None:
                        failure = inspection_failure
                    else:
                        output += f'\n{inspection_failure}'
            if failure is not None:
                print(f'native preflight smoke failed: {failure}\nTriton log:\n{output}', file=sys.stderr)
                if cleanup_failure is not None:
                    print(f'probe cleanup failed: {cleanup_failure}', file=sys.stderr)
                return 1
        bad_exit=returncode!=1 or reported_crash
        if bad_exit or not intended:
            reason = '; reported SIGSEGV/segmentation fault found in complete Triton log' if reported_crash else ''
            print(f'native preflight smoke failed (exit={returncode}{reason}):\n{output}',file=sys.stderr);return 1
        print('Native no-weight preflight failure surfaced; Triton exited 1 without a reported SIGSEGV. GPU constructor/load/unload remain separate gates.')
        return 0

if __name__=='__main__':raise SystemExit(main())
