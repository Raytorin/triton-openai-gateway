"""Real protobuf config admission and HTTP routing across the watcher boundary."""
import json
from unittest.mock import AsyncMock

import pytest

from test_deepseek_checkpoint import snapshot
from test_deepseek_watcher import checkout, run, config_text
from test_deepseek_protocol import endpoint, post, body, DSML, TOOLS, ARGUMENTS, anyio_backend
import gateway.app as app_module
import gateway.generation as generation
import gateway.registry as registry_module
from gateway.registry import ModelRegistry

@pytest.mark.anyio
async def test_watcher_identity_selects_real_registry_deepseek_http(tmp_path, snapshot, config_text, endpoint, monkeypatch):
    target = checkout(tmp_path, snapshot)
    (target.parent/'config.pbtxt').write_text(config_text)
    (target/'gateway.json').write_text('{"reasoning":{"mode":"separate"},"generation":{"context_window":32768}}')
    result = run(tmp_path, 'scan_once')
    assert result.returncode == 0, result.stderr
    active = tmp_path/'models-active'
    assert sorted(p.name for p in active.iterdir()) == ['selected']
    monkeypatch.setattr(registry_module, 'MODELS_ACTIVE_DIR', active)
    registry = ModelRegistry()
    monkeypatch.setattr(generation, 'registry', registry)
    monkeypatch.setattr(app_module, 'registry', registry)
    tokenizer = AsyncMock(return_value=(endpoint.tokenizer, target))
    monkeypatch.setattr(registry, 'get_tokenizer_async', tokenizer)
    assert registry.resolve('selected') == target
    assert registry.get_backend('selected') == 'vllm_deepseek'
    assert registry.get_tool_parser('selected') == 'deepseek_v41'
    endpoint.output = 'Inspect.</think>Final answer.'
    response = await post(body(model='selected', chat_template_kwargs={'enable_thinking':True}))
    assert response.status_code == 200, response.text
    message = response.json()['choices'][0]['message']
    assert message['content'] == 'Final answer.'
    assert message['reasoning_content'] == 'Inspect.'
    endpoint.output = DSML
    response = await post(body(model='selected', tools=TOOLS, chat_template_kwargs={'enable_thinking':False}))
    assert response.status_code == 200, response.text
    choice = response.json()['choices'][0]
    assert choice['finish_reason'] == 'tool_calls'
    call = choice['message']['tool_calls'][0]['function']
    assert call['name'] == 'typed'
    assert json.loads(call['arguments']) == ARGUMENTS
    assert endpoint.models == ['selected', 'selected']
    tokenizer.reset_mock()
    before = len(endpoint.records)
    response = await post(body(model='selected', messages=[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,eA=='}}]}]))
    assert response.status_code == 400, response.text
    assert 'text generation only' in response.text
    response = await post({'model':'selected','input':'hello'}, '/v1/embeddings')
    assert response.status_code == 400, response.text
    assert 'text generation' in response.text
    tokenizer.assert_not_awaited()
    assert len(endpoint.records) == before

@pytest.mark.parametrize('invalid', ['backend: "vllm_deepseek" name: 12', 'backend: "vllm_deepseek" name: ""', 'backend: "vllm_deepseek" name: "selected" name: "other"', 'backend: "vllm_deepseek'])
def test_real_registry_fails_closed_for_malformed_config(tmp_path, snapshot, invalid, monkeypatch):
    from fastapi import HTTPException
    target = checkout(tmp_path, snapshot)
    (target.parent/'config.pbtxt').write_text(invalid)
    active = tmp_path/'models-active'; active.mkdir()
    (active/'selected').symlink_to(target)
    monkeypatch.setattr(registry_module, 'MODELS_ACTIVE_DIR', active)
    with pytest.raises(HTTPException) as failure:
        ModelRegistry().get_backend('selected')
    assert failure.value.status_code == 500


@pytest.mark.parametrize('text', [
    r"backend:'vllm_deepseek' name:'caf\303' '\251'",
    "backend:'vllm_deepseek' name:'café'",
    r"backend:'vllm_deepseek' name:'caf\u00e9'",
    r"backend:'vllm_deepseek' name:'\x73elected'",
    r"backend:'vllm_deepseek' name:'\x7'",
])
def test_shared_reader_decodes_protobuf_strings_like_triton(tmp_path, text):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2
    from gateway.model_config import read_model_config
    parsed = text_format.Parse(text, model_config_pb2.ModelConfig())
    config = tmp_path/'config.pbtxt'; config.write_text(text)
    assert read_model_config(config) == {'name':parsed.name, 'backend':parsed.backend}


@pytest.mark.parametrize('literal', [
    '0.5f', '1F', '1.e-2f', '.5F',
    '+inf', '-inf', '+infinity', '-Infinity', '+nan', '-NaN',
    '-inff', '-nanF',
])
def test_histogram_numeric_literals_preserve_reader_and_registry_identity(tmp_path, monkeypatch, literal):
    from google.protobuf import text_format
    from tritonclient.grpc import model_config_pb2
    from gateway.model_config import read_model_config
    text = '''name: "qwen" backend: "vllm_multimodal"
model_metrics { metric_control {
    metric_identifier { family: "nv_inference_request_duration_us" }
    histogram_options { buckets: [%s] }
} }
''' % literal
    # Qualify non-identity scalar syntax with the parser used by Triton clients.
    parsed = text_format.Parse(text, model_config_pb2.ModelConfig())
    assert len(parsed.model_metrics.metric_control[0].histogram_options.buckets) == 1
    version = tmp_path / 'models' / 'qwen' / '1'
    version.mkdir(parents=True)
    config = version.parent / 'config.pbtxt'
    config.write_text(text)
    active = tmp_path / 'models-active'
    active.mkdir()
    (active / 'qwen').symlink_to(version)
    monkeypatch.setattr(registry_module, 'MODELS_ACTIVE_DIR', active)
    assert read_model_config(config) == {'name': 'qwen', 'backend': 'vllm_multimodal'}
    assert ModelRegistry().get_backend('qwen') == 'vllm_multimodal'


@pytest.mark.parametrize('identity', [
    'name: 0.5f backend: "vllm_deepseek"',
    'name: "selected" backend: +inf',
    'name: -nan backend: "vllm_deepseek"',
])
def test_numeric_literals_cannot_be_model_identity(tmp_path, identity):
    from gateway.model_config import read_model_config
    config = tmp_path / 'config.pbtxt'
    config.write_text(identity)
    with pytest.raises(ValueError):
        read_model_config(config)


@pytest.mark.parametrize('kind', ['directory', 'broken_link', 'loop'])
def test_registry_config_read_errors_do_not_disable_capability_checks(tmp_path, snapshot, kind, monkeypatch):
    from fastapi import HTTPException
    target = checkout(tmp_path, snapshot)
    config = target.parent/'config.pbtxt'; config.unlink()
    if kind == 'directory': config.mkdir()
    elif kind == 'broken_link': config.symlink_to(target.parent/'missing')
    else: config.symlink_to(config)
    active = tmp_path/'models-active'; active.mkdir()
    (active/'selected').symlink_to(target)
    monkeypatch.setattr(registry_module, 'MODELS_ACTIVE_DIR', active)
    with pytest.raises(HTTPException) as failure:
        ModelRegistry().validate_route('selected', 'embeddings')
    assert failure.value.status_code == 500
