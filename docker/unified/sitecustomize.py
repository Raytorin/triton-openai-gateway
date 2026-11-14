# SPDX-License-Identifier: Apache-2.0
"""Give embedded Python a real interpreter for generic multiprocessing workers.

The stub launcher adds this trusted directory before CPython initialization.
Leave backend/model import paths and task policies to their owners.
"""
import multiprocessing
import os
import sys


python = os.environ.get('TRITON_RUNTIME_PYTHON')
if python:
    sys.executable = python
    multiprocessing.set_executable(python)
