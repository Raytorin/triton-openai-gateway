"""Regression coverage for Triton message-list syntax in the shared reader."""
import pytest
from google.protobuf import text_format
from tritonclient.grpc import model_config_pb2
from gateway.model_config import read_model_config


@pytest.mark.parametrize("colon", ["", ":"])
@pytest.mark.parametrize("brackets", [("{", "}"), ("<", ">")])
def test_message_lists_match_triton(tmp_path, colon, brackets):
    start, end = brackets
    text = f'''name: "Qwen3.8-27B-FP8"
backend: "vllm_multimodal"
model_transaction_policy {{ decoupled: true }}
instance_group{colon} [ {start} kind: KIND_MODEL count: 1 {end} ]
parameters{colon} [ {start} key: "GPU_DEVICE_IDS" value {{ string_value: "0" }} {end} ]
input{colon} [ {start} name: "text_input" data_type: TYPE_STRING dims: [1] {end} ]
'''
    parsed = text_format.Parse(text, model_config_pb2.ModelConfig())
    config = tmp_path / "config.pbtxt"
    config.write_text(text)
    assert parsed.name == "Qwen3.8-27B-FP8"
    assert parsed.parameters["GPU_DEVICE_IDS"].string_value == "0"
    assert read_model_config(config) == {"name": "Qwen3.8-27B-FP8", "backend": "vllm_multimodal"}


@pytest.mark.parametrize("field", ["instance_group", "parameters", "input"])
def test_empty_message_list_without_colon(tmp_path, field):
    text = f'name: "qwen" backend: "vllm_multimodal" {field} []'
    text_format.Parse(text, model_config_pb2.ModelConfig())
    config = tmp_path / "config.pbtxt"
    config.write_text(text)
    assert read_model_config(config) == {"name": "qwen", "backend": "vllm_multimodal"}


@pytest.mark.parametrize("invalid", [
    "instance_group [1]",
    "instance_group [{ kind: KIND_MODEL }, 1]",
    'parameters ["not a message"]',
    'input { name: "text" data_type: TYPE_STRING dims [1] }',
    "instance_group [{ kind: KIND_MODEL }",
])
def test_colonless_lists_do_not_admit_scalar_or_malformed_values(tmp_path, invalid):
    config = tmp_path / "config.pbtxt"
    config.write_text('name: "qwen" backend: "vllm_multimodal" ' + invalid)
    with pytest.raises(ValueError):
        read_model_config(config)
