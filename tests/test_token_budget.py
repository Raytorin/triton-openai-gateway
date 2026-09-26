# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import patch, AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from gateway.schemas import ChatCompletionRequest
from gateway.token_budget import load_generation_limits, choose_budget, requested_output_limit
from gateway import generation
from gateway.admission import AdmissionLease


def request(**fields):
    return ChatCompletionRequest(model="test", messages=[{"role": "user", "content": "hello"}], **fields)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, False, "512"])
@pytest.mark.parametrize("field", ["max_tokens", "max_completion_tokens"])
def test_output_limit_is_strict_positive_integer(field, value):
    with pytest.raises(ValidationError):
        request(**{field: value})


def test_aliases_and_null():
    assert requested_output_limit(request()) is None
    assert requested_output_limit(request(max_tokens=None)) is None
    assert requested_output_limit(request(max_tokens=10, max_completion_tokens=10)) == 10
    with pytest.raises(HTTPException) as error:
        requested_output_limit(request(max_tokens=10, max_completion_tokens=11))
    assert error.value.status_code == 400


def limits(tmp_path, *, thinking=False, config=None):
    (tmp_path / "model.json").write_text('{"max_model_len":32768}')
    if config is not None:
        (tmp_path / "gateway.json").write_text(json.dumps({"generation": config}))
    return load_generation_limits(tmp_path, thinking=thinking)


def test_automatic_limit_preserves_input(tmp_path):
    normal = limits(tmp_path)
    budget = choose_budget(normal, None, prompt_tokens=30000, media_tokens=0, safety_margin=64)
    assert budget.effective == 2704
    assert budget.source == "global"
    with pytest.raises(HTTPException):
        choose_budget(normal, 4096, prompt_tokens=30000, media_tokens=0, safety_margin=64)
    assert choose_budget(normal, 64, prompt_tokens=30000, media_tokens=0, safety_margin=64).effective == 64


def test_reasoning_and_model_profile(tmp_path):
    assert limits(tmp_path, thinking=True).default == 8192
    configured = limits(tmp_path, thinking=True, config={"reasoning_default_output_tokens": 2048})
    assert configured.default == 2048
    assert configured.source == "model"


def test_runtime_context_required_and_override_cannot_expand(tmp_path):
    with pytest.raises(HTTPException, match="Unknown runtime context"):
        load_generation_limits(tmp_path, thinking=False)
    (tmp_path / "gateway.json").write_text('{"generation":{"context_window":4096}}')
    assert load_generation_limits(tmp_path, thinking=False).context_window == 4096
    (tmp_path / "model.json").write_text('{"max_model_len":2048}')
    assert load_generation_limits(tmp_path, thinking=False).context_window == 2048


def test_cap_and_media_reserve(tmp_path):
    policy = limits(tmp_path)
    with pytest.raises(HTTPException) as e:
        choose_budget(policy, 32769, prompt_tokens=1, media_tokens=0, safety_margin=64)
    assert e.value.status_code == 400
    budget = choose_budget(policy, None, prompt_tokens=30000, media_tokens=1000, safety_margin=64)
    assert budget.effective == 1704
    with pytest.raises(HTTPException):
        choose_budget(policy, None, prompt_tokens=32704, media_tokens=0, safety_margin=64)


def test_invalid_config(tmp_path):
    with pytest.raises(HTTPException):
        limits(tmp_path, config={"max_output_tokens":65536})
    with pytest.raises(HTTPException):
        limits(tmp_path, config={"default_output_tokens":0})


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return " ".join(str(m.get("content", "")) for m in messages)
    def encode(self, text, **kwargs):
        return text.split()
    def __call__(self, text, **kwargs):
        return type("Encoded", (), {"input_ids": self.encode(text)})()


def test_generation_keeps_history_and_reports_effective_budget(tmp_path):
    import asyncio
    (tmp_path / "model.json").write_text('{"max_model_len":100}')
    (tmp_path / "gateway.json").write_text('{"context_compression":{"mode":"summarize"}}')
    req = request()
    req.messages = [type(req.messages[0])(role="user", content="old question"),
                    type(req.messages[0])(role="assistant", content="old answer"),
                    type(req.messages[0])(role="user", content="new question")]
    infer = AsyncMock(return_value="ok")
    async def run():
        with (patch.object(generation.registry, "resolve", return_value=tmp_path),
              patch.object(generation.registry, "validate_route"),
              patch.object(generation.registry, "get_backend", return_value="vllm"),
              patch.object(generation.registry, "get_tokenizer_async", AsyncMock(return_value=(Tokenizer(), tmp_path))),
              patch.object(generation.admission, "acquire", AsyncMock(return_value=AdmissionLease([]))),
              patch.object(generation, "call_triton_multimodal", infer)):
            result = await generation.generate(req)
        assert result.headers["X-Output-Token-Limit"] == "30"
        assert result.extensions["context_status"]["action"] == "none"
        assert "old question old answer new question" in infer.call_args.args[1]
        assert infer.call_args.args[2]["max_tokens"] == 30
    asyncio.run(run())


def test_cap_below_fallback_defaults_is_valid(tmp_path):
    policy = limits(tmp_path, config={"max_output_tokens": 1024})
    assert choose_budget(policy, None, prompt_tokens=10, media_tokens=0, safety_margin=64).effective == 1024
