"""Independent, literal reference IDs exercise the offline comparison contract."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/deepseek'
REVISION = '314113f8a7dadb388215396ba5fbee2c610a04d6'


class SyntheticTokenizer:
    """A tiny independent lexicon, never loaded from the reference ID lists."""
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    pieces = {
        '<｜begin▁of▁sentence｜>': 0, '<｜User｜>': 3, '<｜Assistant｜>': 4,
        '<｜System｜>': 5, '<think>': 6, '</think>': 7,
        'hello': 10, 'Reasoning Effort: ': 11, '50': 12,
        ' (range 1-100, the higher the value, the more thorough the reasoning)\n\n': 13,
        'Привет 👋': 14, '62': 16, '75': 17, '100': 18, 'a': 19, 'b': 20,
    }

    def __init__(self, path):
        self.chat_template = (path / 'chat_template.jinja').read_text()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=['jinja2.ext.loopcontrols'])
        env.filters['tojson'] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
        env.globals['raise_exception'] = lambda message: (_ for _ in ()).throw(ValueError(message))
        self.template = env.from_string(self.chat_template)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.pop('tokenize') is False
        return self.template.render(messages=messages, **kwargs)

    def encode(self, text, *, add_special_tokens=True):
        ids = [0] if add_special_tokens else []
        while text:
            piece = next((p for p in sorted(self.pieces, key=len, reverse=True) if text.startswith(p)), None)
            assert piece is not None, repr(text)
            ids.append(self.pieces[piece])
            text = text[len(piece):]
        return ids


@pytest.fixture
def qualification(tmp_path):
    from gateway.deepseek_tokenizer_validation import validate_tokenizer
    checkpoint = tmp_path / 'snapshot'
    checkpoint.mkdir()
    (checkpoint / 'config.json').write_text('{"model_type":"deepseek_v41"}')
    (checkpoint / 'tokenizer_config.json').write_text('{"bos_token":"<｜begin▁of▁sentence｜>"}')
    (checkpoint / 'tokenizer.json').write_text('{"synthetic":true}')
    (checkpoint / 'chat_template.jinja').write_bytes((FIXTURE / 'v41-flash/chat_template.jinja').read_bytes())
    config = {'model': str(checkpoint), 'revision': REVISION}
    reference = {
        'version': 1,
        'checkpoint': {'repo_id': 'deepseek-ai/DeepSeek-V4.1-Flash', 'revision': REVISION},
        'source': {'kind': 'synthetic-test-fixture', 'description': 'Literal, manually derived tiny-vocabulary IDs.'},
        'artifacts': {p.name: sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()},
        'special_token_ids': {'bos_token_id': 0, 'eos_token_id': 1, 'pad_token_id': 2},
        'cases': json.loads((FIXTURE / 'token-cases.json').read_text())['cases'],
    }
    return validate_tokenizer, config, reference, SyntheticTokenizer(checkpoint)


def test_literals_match_without_a_duplicate_bos(qualification):
    validate, config, reference, tokenizer = qualification
    report = validate(config, reference, tokenizer=tokenizer)
    assert report['matched'] is True
    assert report['reference_kind'] == 'synthetic-test-fixture'
    assert report['case_count'] == 6
    assert all(item['matched'] for item in report['cases'])


def test_wrong_token_id_reports_first_difference_without_prompt_payload(qualification):
    validate, config, reference, tokenizer = qualification
    reference['cases'][0]['expected_token_ids'][2] = 99
    report = validate(config, reference, tokenizer=tokenizer)
    assert report['matched'] is False
    case = report['cases'][0]
    assert case['token_ids_match'] is False
    assert case['first_token_difference'] == {'index': 2, 'expected': 99, 'actual': 10}
    assert 'expected_prompt' not in json.dumps(report)


def test_extra_bos_from_runtime_is_detected(qualification):
    validate, config, reference, tokenizer = qualification
    encode = tokenizer.encode
    tokenizer.encode = lambda text, **kwargs: [0] + encode(text, **kwargs)
    report = validate(config, reference, tokenizer=tokenizer)
    assert not report['matched']
    assert report['cases'][0]['first_token_difference'] == {'index': 1, 'expected': 3, 'actual': 0}


def test_mismatched_rendered_prompt_is_detected(qualification):
    validate, config, reference, tokenizer = qualification
    reference['cases'][0]['expected_prompt'] = 'different, independent reference'
    report = validate(config, reference, tokenizer=tokenizer)
    assert not report['matched']
    assert not report['cases'][0]['prompt_match']


@pytest.mark.parametrize('field', ['bos_token_id', 'eos_token_id', 'pad_token_id'])
def test_special_token_policy_is_compared_not_silently_repaired(qualification, field):
    validate, config, reference, tokenizer = qualification
    reference['special_token_ids'][field] = 42
    report = validate(config, reference, tokenizer=tokenizer)
    assert not report['matched']
    assert field in report['special_token_mismatches']
    assert getattr(tokenizer, field) != 42


def test_tokenizer_artifact_change_fails_before_encoding(qualification):
    validate, config, reference, tokenizer = qualification
    (Path(config['model']) / 'tokenizer.json').write_text('{"changed":true}')
    tokenizer.encode = lambda *a, **kw: pytest.fail('must validate artifact identity first')
    with pytest.raises(ValueError, match='tokenizer.json.*hash'):
        validate(config, reference, tokenizer=tokenizer)


def test_checkpoint_revision_must_match_reference(qualification):
    validate, config, reference, tokenizer = qualification
    config['revision'] = '0' * 40
    with pytest.raises(ValueError, match='revision'):
        validate(config, reference, tokenizer=tokenizer)


@pytest.mark.parametrize('bad', [True, -1, '10', 1.5])
def test_reference_ids_must_be_nonnegative_integers(qualification, bad):
    validate, config, reference, tokenizer = qualification
    reference['cases'][0]['expected_token_ids'][0] = bad
    with pytest.raises(ValueError, match='token IDs'):
        validate(config, reference, tokenizer=tokenizer)


def test_missing_artifact_digest_is_not_an_identity_check(qualification):
    validate, config, reference, tokenizer = qualification
    del reference['artifacts']['tokenizer.json']
    with pytest.raises(ValueError, match='artifacts'):
        validate(config, reference, tokenizer=tokenizer)


def test_reference_cannot_request_automatic_remote_code_execution(qualification):
    validate, config, reference, tokenizer = qualification
    reference['source'] = {'kind': 'execute-checkpoint-code', 'path': 'encoding/encoding.py'}
    with pytest.raises(ValueError, match='reference source'):
        validate(config, reference, tokenizer=tokenizer)


@pytest.mark.parametrize('kind', [[], {}, None, True, 1])
def test_malformed_reference_source_kind_is_a_validation_error(qualification, kind):
    validate, config, reference, tokenizer = qualification
    reference['source']['kind'] = kind
    with pytest.raises(ValueError, match='reference source'):
        validate(config, reference, tokenizer=tokenizer)


def test_invalid_reference_request_does_not_echo_private_values(qualification):
    validate, config, reference, tokenizer = qualification
    reference['cases'][0]['request']['max_tokens'] = 'private-invalid-value'
    with pytest.raises(ValueError) as error:
        validate(config, reference, tokenizer=tokenizer)
    assert 'private-invalid-value' not in str(error.value)


def test_reviewed_encoder_reference_requires_a_digest(qualification):
    validate, config, reference, tokenizer = qualification
    reference['source'] = {'kind': 'reviewed-checkpoint-encoder'}
    with pytest.raises(ValueError, match='encoder SHA256'):
        validate(config, reference, tokenizer=tokenizer)


def test_reference_json_rejects_duplicate_identity_fields(tmp_path):
    from gateway.deepseek_tokenizer_validation import read_reference
    path = tmp_path / 'reference.json'
    path.write_text('{"version":1,"version":1}')
    with pytest.raises(ValueError, match='duplicate reference key'):
        read_reference(path)


@pytest.mark.parametrize('operation', ['reference', 'artifact'])
def test_nonregular_input_is_rejected_without_reading(tmp_path, monkeypatch, operation):
    import gateway.deepseek_tokenizer_validation as validation
    fifo = tmp_path / 'input.fifo'
    os.mkfifo(fifo)
    original_open = Path.open
    def never_open_fifo(path, *args, **kwargs):
        if path == fifo:
            pytest.fail('nonregular file must not be opened through a blocking reader')
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', never_open_fifo)
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('nonregular file must not be read'))
    with pytest.raises(ValueError, match='regular'):
        (validation.read_reference if operation == 'reference' else validation._artifact_hash)(fifo)


def test_hf_style_symlink_to_regular_blob_is_supported(tmp_path):
    from gateway.deepseek_tokenizer_validation import read_reference, _artifact_hash
    target = tmp_path / 'blob'
    target.write_bytes(b'{"version":1}')
    alias = tmp_path / 'reference.json'
    alias.symlink_to(target)
    assert read_reference(alias) == {'version': 1}
    assert _artifact_hash(alias) == sha256(b'{"version":1}').hexdigest()


@pytest.mark.parametrize('operation,limit', [('reference', 32 * 1024 * 1024), ('artifact', 64 * 1024 * 1024)])
def test_opened_size_limit_is_enforced_before_reads(tmp_path, monkeypatch, operation, limit):
    import gateway.deepseek_tokenizer_validation as validation
    path = tmp_path / 'oversized'
    with path.open('wb') as stream:
        stream.truncate(limit + 1)
    monkeypatch.setattr(os, 'pread', lambda *args: pytest.fail('oversized file must not be read'))
    with pytest.raises(ValueError, match='byte limit'):
        (validation.read_reference if operation == 'reference' else validation._artifact_hash)(path)


@pytest.mark.parametrize('operation', ['reference', 'artifact'])
def test_opened_file_growth_is_bounded_and_rejected(tmp_path, monkeypatch, operation):
    import gateway.deepseek_tokenizer_validation as validation
    path = tmp_path / 'input.json'
    initial = b'{"version":1}'
    path.write_bytes(initial)
    original = os.pread
    reads = []
    def growing(fd, count, offset):
        reads.append((count, offset))
        result = original(fd, count, offset)
        with path.open('r+b') as stream:
            stream.truncate(validation.MAX_ARTIFACT_BYTES + 1)
        return result
    monkeypatch.setattr(os, 'pread', growing)
    with pytest.raises(ValueError, match='changed'):
        (validation.read_reference if operation == 'reference' else validation._artifact_hash)(path)
    assert reads and sum(count for count, _ in reads) <= len(initial)


@pytest.mark.parametrize('operation', ['reference', 'artifact'])
def test_path_replacement_during_read_is_rejected(tmp_path, monkeypatch, operation):
    import gateway.deepseek_tokenizer_validation as validation
    original_path = tmp_path / 'original'
    replacement = tmp_path / 'replacement'
    original_path.write_text('{"version":1}')
    replacement.write_text('{"version":2}')
    alias = tmp_path / 'input.json'
    alias.symlink_to(original_path)
    original = os.pread
    def retarget(fd, count, offset):
        result = original(fd, count, offset)
        alias.unlink()
        alias.symlink_to(replacement)
        return result
    monkeypatch.setattr(os, 'pread', retarget)
    with pytest.raises(ValueError, match='changed'):
        (validation.read_reference if operation == 'reference' else validation._artifact_hash)(alias)


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '-Infinity', '1e999'])
def test_reference_json_rejects_nonfinite_numbers(tmp_path, number):
    from gateway.deepseek_tokenizer_validation import read_reference
    path = tmp_path / 'reference.json'
    path.write_text('{"metadata":' + number + '}')
    with pytest.raises(ValueError, match='non-finite'):
        read_reference(path)


def test_cli_rejects_changed_identity_before_loading_runtime(qualification, tmp_path):
    _, config, reference, _ = qualification
    reference['artifacts']['tokenizer.json'] = '0' * 64
    config_path, reference_path = tmp_path / 'model.json', tmp_path / 'reference.json'
    config_path.write_text(json.dumps(config))
    reference_path.write_text(json.dumps(reference))
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_deepseek_tokenizer.py'),
        str(config_path), '--reference', str(reference_path), '--json'], text=True, capture_output=True)
    assert result.returncode == 1
    assert 'tokenizer.json: artifact hash' in result.stderr
    assert 'Traceback' not in result.stderr


def test_cli_help_does_not_require_transformers():
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_deepseek_tokenizer.py'), '--help'],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert '--reference' in result.stdout


def test_pinned_checkpoint_token_ids_match_independent_reference():
    """Opt in with reviewed local tokenizer artifacts; never download weights."""
    value = os.environ.get('DEEPSEEK_TOKENIZER_SNAPSHOT')
    if not value:
        pytest.skip('set DEEPSEEK_TOKENIZER_SNAPSHOT to the pinned local tokenizer snapshot for real token-ID parity')
    snapshot = Path(value)
    assert snapshot.is_absolute() and snapshot.is_dir(), 'DEEPSEEK_TOKENIZER_SNAPSHOT must be an existing absolute local directory'
    from gateway.deepseek_tokenizer_validation import read_reference, validate_tokenizer
    reference = read_reference(FIXTURE / 'v41-flash/token-reference.json')
    assert reference['source']['kind'] == 'reviewed-checkpoint-encoder'
    report = validate_tokenizer({'model': str(snapshot), 'revision': reference['checkpoint']['revision']}, reference)
    assert report['case_count'] == 16
    assert report['matched'], report
    assert not report['special_token_mismatches']
