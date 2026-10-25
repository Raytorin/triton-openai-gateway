# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
"""Default Qwen/BGE frontend remains usable without the optional recipe."""
import os
from pathlib import Path
import subprocess
import sys


def test_default_qwen_and_bge_http_work_when_recipe_import_is_unavailable():
    root = Path(__file__).resolve().parents[1]
    script = '''
import importlib.abc, sys
class BlockRecipe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'deepseek_recipe' or fullname.startswith('deepseek_recipe.'):
            raise ImportError('optional DeepSeek parser is deliberately unavailable')
sys.meta_path.insert(0, BlockRecipe())
import gateway.app
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider',
    'tests/test_request_thinking.py', 'tests/test_hybrid_embeddings.py']))
'''
    result = subprocess.run([sys.executable, '-c', script], cwd=root,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
