"""Verifier dependency closures must survive their actual image COPY layout."""
import re
import runpy
import shlex
import shutil
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def materialize_verifier_inputs(dockerfile,image_root,verifier_path):
    """Stage context-file COPY inputs up to verification; execute no build RUNs."""
    text=re.sub(r'\\\s*\n',' ',dockerfile.read_text())
    instructions=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        if line.startswith('FROM '):instructions=[]
        else:instructions.append(line)
    for instruction in instructions:
        words=shlex.split(instruction)
        if words[0]=='COPY' and not any(word.startswith('--from=') for word in words[1:]):
            sources,destination=words[1:-1],words[-1]
            for source in sources:
                source=ROOT/source
                if not source.is_file():continue
                target=image_root/destination.lstrip('/')
                if destination.endswith('/') or len(sources)>1:target/=source.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
        if words[0]=='RUN' and f'python3 {verifier_path}' in instruction:
            return image_root/verifier_path.lstrip('/')
    raise AssertionError('Image recipe never runs its runtime verifier')


@pytest.mark.parametrize('recipe,installed_verifier',[
    ('Dockerfile.triton-gateway','/tmp/verify-runtime.py'),
    ('Dockerfile.triton-gateway-deepseek','/tmp/verify-qwen-runtime.py'),
])
def test_installed_runtime_verifier_can_read_its_requirement_contract(tmp_path,recipe,installed_verifier):
    installed=materialize_verifier_inputs(ROOT/recipe,tmp_path,installed_verifier)
    # The real module reads adjacent pins while constructing EXPECTED_VERSIONS.
    # Default run_name avoids main's checks against this test runner's packages.
    packaged=runpy.run_path(str(installed))
    source=runpy.run_path(str(ROOT/'docker/verify-runtime.py'))
    assert packaged['EXPECTED_VERSIONS']==source['EXPECTED_VERSIONS']
    assert packaged['REQUIREMENTS_FILE'].parent==installed.parent
