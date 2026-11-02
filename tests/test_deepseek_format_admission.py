"""A decoder rejection must stop real checkpoint admission before engine startup."""
import json
import sys

import pytest

from test_deepseek_checkpoint import snapshot, ue8m0_snapshot
from test_deepseek_runtime import load_policy


def test_ue8m0_decoder_failure_blocks_shared_constructor(monkeypatch, tmp_path, snapshot):
    policy = load_policy(monkeypatch)
    root = ue8m0_snapshot(snapshot)
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'model.json').write_text(json.dumps({
        'model': str(root),
        'language_model_only': True,
        'enable_mm_embeds': False,
        'tokenizer_mode': 'deepseek_v41',
        'trust_remote_code': False,
        'safetensors_load_strategy': 'lazy',
    }))
    monkeypatch.setattr(sys.modules['triton_python_backend_utils'], 'get_model_dir',
                        lambda: str(model_dir), raising=False)

    checkpoint = policy._preflight_model.__globals__['_checkpoint']
    reports = []
    cause = ValueError('installed CPU decoder does not support UE8M0')
    failure = RuntimeError('runtime cannot deserialize F8_E8M0 (UE8M0) scales')

    def reject_runtime(report):
        reports.append(report)
        raise failure from cause

    monkeypatch.setattr(checkpoint, 'check_runtime', reject_runtime)
    constructed = []

    class SharedEngine:
        def initialize(self, args):
            constructed.append(args)
            self.supported_tasks = ('generate',)

    class Model(policy, SharedEngine):
        pass

    model = Model()
    with pytest.raises(RuntimeError, match='F8_E8M0') as raised:
        model.initialize({})

    assert raised.value is failure
    assert raised.value.__cause__ is cause
    assert constructed == []
    assert len(reports) == 1
    assert reports[0] is model.checkpoint_report
    assert reports[0]['serialized_dtypes'] == ['F32', 'F8_E8M0']
    assert reports[0]['safetensors_load_strategy'] == 'lazy'
    assert reports[0]['runtime_format_checks_required'] == ['F8_E8M0']
    assert reports[0]['runtime_validation'] == 'not_checked'
