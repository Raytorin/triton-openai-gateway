# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from gateway.admission import AdmissionLease
import gateway.app as gateway_app
from gateway.openai_contract import normalize_system_messages
from gateway.prompt import build_sampling_parameters, render_chat_prompt
from gateway.schemas import ChatCompletionRequest


def request_with(**overrides):
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Return JSON"}],
    }
    payload.update(overrides)
    return ChatCompletionRequest.model_validate(payload)


class OpenAIParameterTests(unittest.TestCase):
    def test_seed_is_forwarded_to_sampling_parameters(self):
        sampling = build_sampling_parameters(request_with(seed=777))

        self.assertEqual(777, sampling["seed"])

    def test_json_object_maps_to_serialized_vllm_structured_outputs(self):
        sampling = build_sampling_parameters(
            request_with(response_format={"type": "json_object"})
        )

        self.assertIsInstance(sampling["structured_outputs"], str)
        self.assertEqual(
            {"json_object": True},
            json.loads(sampling["structured_outputs"]),
        )

    def test_json_schema_maps_to_serialized_vllm_structured_outputs(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        sampling = build_sampling_parameters(
            request_with(
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": True,
                        "schema": schema,
                    },
                }
            )
        )

        self.assertIsInstance(sampling["structured_outputs"], str)
        self.assertEqual(
            {"json": schema},
            json.loads(sampling["structured_outputs"]),
        )

    def test_unknown_response_format_is_rejected(self):
        with self.assertRaises(ValidationError):
            request_with(response_format={"type": "definitely_not_a_type"})


class SystemMessageContractTests(unittest.TestCase):
    def test_system_messages_are_merged_and_moved_to_front(self):
        conversation = [
            {"role": "system", "content": "Base instruction."},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Draft"},
            {"role": "system", "content": "Answer briefly."},
        ]

        normalized, count, moved = normalize_system_messages(conversation)

        self.assertEqual(2, count)
        self.assertEqual(1, moved)
        self.assertEqual("system", normalized[0]["role"])
        self.assertEqual(
            "Base instruction.\n\nAnswer briefly.",
            normalized[0]["content"],
        )
        self.assertEqual(
            ["system", "user", "assistant"],
            [message["role"] for message in normalized],
        )

    def test_non_text_system_content_is_rejected_with_400(self):
        with self.assertRaises(HTTPException) as context:
            normalize_system_messages(
                [
                    {"role": "user", "content": "Question"},
                    {
                        "role": "system",
                        "content": [{"type": "image_url", "image_url": {}}],
                    },
                ]
            )

        self.assertEqual(400, context.exception.status_code)

    def test_template_contract_error_is_returned_as_400(self):
        class RejectingTokenizer:
            def apply_chat_template(self, *_args, **_kwargs):
                raise ValueError("system role is only allowed at the beginning")

        with self.assertRaises(HTTPException) as context:
            render_chat_prompt(
                RejectingTokenizer(),
                [{"role": "user", "content": "Question"}],
            )

        self.assertEqual(400, context.exception.status_code)
        self.assertIn("chat template", str(context.exception.detail))


class FinishReasonRegressionTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_max_tokens_returns_length(self):
        class Tokenizer:
            def apply_chat_template(self, conversation, **_kwargs):
                return " ".join(
                    str(message.get("content", "")) for message in conversation
                )

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return str(text).split()

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return type("Tokenized", (), {"input_ids": self.encode(text)})()

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                '{"max_model_len":1024}',
                encoding="utf-8",
            )
            request = request_with(max_tokens=3)
            with (
                patch.object(gateway_app.registry, "resolve", return_value=model_path),
                patch.object(gateway_app.registry, "validate_route"),
                patch.object(gateway_app.registry, "get_backend", return_value="vllm"),
                patch.object(
                    gateway_app.registry,
                    "get_tokenizer_async",
                    new=AsyncMock(return_value=(Tokenizer(), model_path)),
                ),
                patch.object(
                    gateway_app.admission,
                    "acquire",
                    new=AsyncMock(return_value=AdmissionLease([])),
                ),
                patch.object(
                    gateway_app,
                    "call_triton_multimodal",
                    new=AsyncMock(return_value="one two three"),
                ),
            ):
                response = await gateway_app.create_chat_completion(request)

        self.assertEqual("length", response["choices"][0]["finish_reason"])


if __name__ == "__main__":
    unittest.main()
