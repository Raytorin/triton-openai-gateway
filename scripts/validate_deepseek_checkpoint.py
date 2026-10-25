#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read-only structural checkpoint admission; no tensor reads or downloads."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config',type=Path)
    parser.add_argument('--json',action='store_true')
    parser.add_argument('--check-runtime',action='store_true',help='also check installed native registry membership')
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]/'backends/vllm_deepseek/checkpoint_preflight.py'
    if not source.is_file():
        source=Path('/opt/tritonserver/backends/vllm_deepseek/checkpoint_preflight.py')
    spec=importlib.util.spec_from_file_location('checkpoint_preflight',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        config=module.read_json(args.model_config)
        if args.check_runtime:
            module.check_runtime()
        report=module.validate_checkpoint(config)
    except (ValueError,ImportError,RuntimeError) as error:
        print(f'checkpoint preflight failed: {error}',file=sys.stderr)
        return 1
    print(json.dumps(report,indent=2) if args.json else
          f"structural_headers_only: {report['shard_count']} shards, {report['serialized_bytes']} serialized bytes; not a hash, memory-fit or READY check")
    return 0

if __name__=='__main__':
    raise SystemExit(main())
