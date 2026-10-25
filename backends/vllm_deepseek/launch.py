# SPDX-License-Identifier: Apache-2.0
"""Exec a DeepSeek process with native/tool/cache/pip isolation already selected."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def execute(command, root=None, cuda=None, stub=False):
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location('deepseek_preexec_runtime', here / 'runtime.py')
    runtime = importlib.util.module_from_spec(spec); spec.loader.exec_module(runtime)
    environment = runtime.prepare_environment(root or runtime.ENV_ROOT, cuda or runtime.CUDA_ROOT, stub=stub)
    os.execve(command[0], list(command), environment)


def stock_stub():
    execute(['/opt/tritonserver/backends/python/triton_python_backend_stub', *sys.argv[1:]], stub=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root')
    parser.add_argument('--cuda')
    parser.add_argument('--exec', dest='command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command or not Path(args.command[0]).is_absolute():
        parser.error('--exec requires an absolute executable path')
    execute(args.command, args.root, args.cuda)


if __name__ == '__main__':
    main()
