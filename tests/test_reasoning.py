# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
import numpy as np

from gateway.admission import AdmissionLease
import gateway.app as gateway_app
import gateway.generation as generation
from gateway.prompt import (
    build_usage,
    completion_reached_token_limit,
    render_chat_prompt,
)
from gateway.reasoning import (
    ReasoningSettings,
    load_reasoning_settings,
    split_reasoning_output,
)
from gateway.schemas import ChatCompletionRequest
from gateway.triton_client import (
    stream_tool_aware_response,
    stream_triton_multimodal_to_openai,
)


class FakeTokenizer:
    def __init__(self):
        self.enable_thinking = None

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        del tokenize
        self.enable_thinking = kwargs.get("enable_thinking")
        content = " ".join(str(message.get("content", "")) for message in conversation)
        return f"{content} assistant" if add_generation_prompt else content

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return type("Tokenized", (), {"input_ids": self.encode(text)})()


def reasoning_settings(
    mode: str = "separate",
    response_field: str = "reasoning_content",
) -> ReasoningSettings:
    return ReasoningSettings(
        configured_mode=mode,
        mode=mode,
        parser="qwen3",
        response_field=response_field,
        supported=True,
    )


def parse_sse(events):
    payloads = []
    for event in events:
        if event == "data: [DONE]\n\n":
            continue
        payloads.append(json.loads(event.removeprefix("data: ").strip()))
    return payloads


class ReasoningParserTests(unittest.TestCase):
    def test_qwen_close_only_output_is_split(self):
        result = split_reasoning_output(
            "Check the facts first.</think>The final answer.",
            reasoning_settings(),
        )

        self.assertEqual("Check the facts first.", result.reasoning)
        self.assertEqual("The final answer.", result.content)
        self.assertTrue(result.detected)

    def test_legacy_think_block_is_split_without_service_tags(self):
        result = split_reasoning_output(
            "<think>Private reasoning.</think>Visible answer.",
            reasoning_settings(),
        )

        self.assertEqual("Private reasoning.", result.reasoning)
        self.assertEqual("Visible answer.", result.content)
        self.assertNotIn("<think>", result.content)

    def test_tool_call_implicitly_ends_qwen_reasoning(self):
        result = split_reasoning_output(
            'Need current data.<tool_call>{"name":"weather","arguments":{}}</tool_call>',
            reasoning_settings(),
        )

        self.assertEqual("Need current data.", result.reasoning)
        self.assertTrue(result.content.startswith("<tool_call>"))

    def test_disabled_mode_defensively_removes_accidental_reasoning(self):
        disabled = reasoning_settings(mode="disabled")
        result = split_reasoning_output(
            "<think>Do not expose.</think>Safe answer.",
            disabled,
        )

        self.assertEqual("Do not expose.", result.reasoning)
        self.assertEqual("Safe answer.", result.content)

    def test_incomplete_reasoning_never_falls_through_to_content(self):
        result = split_reasoning_output(
            "<think>Still reasoning",
            reasoning_settings(mode="hidden"),
        )

        self.assertEqual("Still reasoning", result.reasoning)
        self.assertEqual("", result.content)
        self.assertTrue(result.incomplete)


class ReasoningSettingsTests(unittest.TestCase):
    def test_auto_detects_qwen_and_request_can_only_hide(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3"}',
                encoding="utf-8",
            )
            (model_path / "tokenizer_config.json").write_text(
                '{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}',
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate"}}',
                encoding="utf-8",
            )
            visible = load_reasoning_settings(model_path)
            hidden = load_reasoning_settings(
                model_path,
                include_reasoning=False,
            )

        self.assertEqual("qwen3", visible.parser)
        self.assertEqual("separate", visible.mode)
        self.assertEqual("hidden", hidden.mode)

    def test_client_cannot_reveal_server_hidden_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"hidden","parser":"qwen3"}}',
                encoding="utf-8",
            )
            loaded = load_reasoning_settings(
                model_path,
                include_reasoning=True,
            )

        self.assertEqual("hidden", loaded.mode)
        self.assertFalse(loaded.expose_reasoning)

    def test_structured_output_can_force_thinking_off(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3"}',
                encoding="utf-8",
            )
            (model_path / "tokenizer_config.json").write_text(
                '{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}',
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate"}}',
                encoding="utf-8",
            )
            loaded = load_reasoning_settings(
                model_path,
                force_disable=True,
            )

        self.assertEqual("separate", loaded.configured_mode)
        self.assertEqual("disabled", loaded.mode)
        self.assertFalse(loaded.enable_thinking)

    def test_enabled_mode_without_supported_parser_is_operator_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate","parser":"auto"}}',
                encoding="utf-8",
            )
            with self.assertRaises(HTTPException) as context:
                load_reasoning_settings(model_path)

        self.assertEqual(500, context.exception.status_code)

    def test_qwen3_instruct_without_thinking_template_is_not_auto_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3"}',
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate","parser":"auto"}}',
                encoding="utf-8",
            )
            with self.assertRaises(HTTPException):
                load_reasoning_settings(model_path)


class ReasoningPromptAndUsageTests(unittest.TestCase):
    def test_chat_template_receives_enable_thinking(self):
        tokenizer = FakeTokenizer()

        render_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "Question"}],
            enable_thinking=True,
        )

        self.assertTrue(tokenizer.enable_thinking)

    def test_usage_reports_reasoning_tokens_as_completion_details(self):
        usage = build_usage(
            FakeTokenizer(),
            "one two",
            "reason one answer two",
            reasoning_text="reason one",
        )

        self.assertEqual(4, usage["completion_tokens"])
        self.assertEqual(
            2,
            usage["completion_tokens_details"]["reasoning_tokens"],
        )

    def test_completion_limit_is_detected_from_usage(self):
        self.assertTrue(
            completion_reached_token_limit(
                {"completion_tokens": 512},
                {"max_tokens": 512},
            )
        )
        self.assertFalse(
            completion_reached_token_limit(
                {"completion_tokens": 511},
                {"max_tokens": 512},
            )
        )


class ReasoningStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reasoning_precedes_tool_call_in_sse(self):
        request = ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "Weather?"}],
            stream=True,
        )
        generated = (
            "<think>Need weather data.</think>"
            '<tool_call>{"name":"weather","arguments":{"city":"London"}}</tool_call>'
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object"},
                },
            }
        ]
        with patch(
            "gateway.triton_client.call_triton",
            new=AsyncMock(return_value=generated),
        ):
            events = [
                event
                async for event in stream_tool_aware_response(
                    request,
                    FakeTokenizer(),
                    "prompt",
                    {},
                    tools,
                    reasoning_settings=reasoning_settings(),
                )
            ]

        payloads = parse_sse(events)
        deltas = [payload["choices"][0]["delta"] for payload in payloads]
        reasoning_index = next(
            index
            for index, delta in enumerate(deltas)
            if "reasoning_content" in delta
        )
        tool_index = next(
            index for index, delta in enumerate(deltas) if "tool_calls" in delta
        )
        self.assertLess(reasoning_index, tool_index)
        self.assertEqual(
            "Need weather data.",
            deltas[reasoning_index]["reasoning_content"],
        )
        self.assertEqual(
            "tool_calls",
            payloads[-1]["choices"][0]["finish_reason"],
        )

    async def test_incremental_stream_separates_reasoning_and_content(self):
        request = ChatCompletionRequest(
            model="qwen",
            messages=[{"role": "user", "content": "Question"}],
            stream=True,
        )

        class Result:
            def __init__(self, text):
                self.text = text

            def as_numpy(self, name):
                if name != "text_output":
                    return None
                return np.asarray([self.text.encode("utf-8")], dtype=np.object_)

        async def results(*args, **kwargs):
            del args, kwargs
            yield Result("<think>Inspecting evidence carefully")
            yield Result(
                "<think>Inspecting evidence carefully</think>"
                "This is the final answer."
            )

        with patch(
            "gateway.triton_client._stream_grpc_results",
            new=results,
        ):
            events = [
                event
                async for event in stream_triton_multimodal_to_openai(
                    request,
                    FakeTokenizer(),
                    "prompt",
                    {},
                    [],
                    reasoning_settings=reasoning_settings(),
                )
            ]

        payloads = parse_sse(events)
        deltas = [payload["choices"][0]["delta"] for payload in payloads]
        reasoning = "".join(
            delta.get("reasoning_content", "")
            for delta in deltas
        )
        content = "".join(delta.get("content", "") for delta in deltas)
        self.assertEqual("Inspecting evidence carefully", reasoning)
        self.assertEqual("This is the final answer.", content)
        self.assertNotIn("<think>", reasoning + content)
        self.assertGreater(
            payloads[-1]["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0,
        )


class ReasoningEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_stream_response_exposes_reasoning_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                '{"max_model_len":1024}',
                encoding="utf-8",
            )
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3"}',
                encoding="utf-8",
            )
            (model_path / "tokenizer_config.json").write_text(
                '{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}',
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate"}}',
                encoding="utf-8",
            )
            tokenizer = FakeTokenizer()
            request = ChatCompletionRequest(
                model="qwen",
                messages=[{"role": "user", "content": "Question"}],
            )
            with (
                patch.object(
                    gateway_app.registry,
                    "resolve",
                    return_value=model_path,
                ),
                patch.object(gateway_app.registry, "validate_route"),
                patch.object(
                    gateway_app.registry,
                    "get_backend",
                    return_value="vllm",
                ),
                patch.object(
                    gateway_app.registry,
                    "get_tokenizer_async",
                    new=AsyncMock(return_value=(tokenizer, model_path)),
                ),
                patch.object(
                    gateway_app.admission,
                    "acquire",
                    new=AsyncMock(return_value=AdmissionLease([])),
                ),
                patch.object(
                    generation,
                    "call_triton_multimodal",
                    new=AsyncMock(
                        return_value="Inspect first.</think>Final response."
                    ),
                ),
            ):
                response = await gateway_app.create_chat_completion(request)

        message = response["choices"][0]["message"]
        self.assertEqual("Inspect first.", message["reasoning_content"])
        self.assertEqual("Final response.", message["content"])
        self.assertEqual("separate", response["reasoning_status"]["mode"])
        self.assertTrue(tokenizer.enable_thinking)

    async def test_structured_output_is_returned_as_visible_content(self):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "retries": {"type": "integer"},
            },
            "required": ["status", "retries"],
            "additionalProperties": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                '{"max_model_len":1024}',
                encoding="utf-8",
            )
            (model_path / "config.json").write_text(
                '{"model_type":"qwen3"}',
                encoding="utf-8",
            )
            (model_path / "tokenizer_config.json").write_text(
                '{"chat_template":"{% if enable_thinking %}<think>{% endif %}"}',
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                '{"reasoning":{"mode":"separate"}}',
                encoding="utf-8",
            )
            tokenizer = FakeTokenizer()
            request = ChatCompletionRequest.model_validate(
                {
                    "model": "qwen",
                    "messages": [
                        {"role": "user", "content": "Return structured JSON"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "pipeline_result",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                }
            )
            with (
                patch.object(
                    gateway_app.registry,
                    "resolve",
                    return_value=model_path,
                ),
                patch.object(gateway_app.registry, "validate_route"),
                patch.object(
                    gateway_app.registry,
                    "get_backend",
                    return_value="vllm",
                ),
                patch.object(
                    gateway_app.registry,
                    "get_tokenizer_async",
                    new=AsyncMock(return_value=(tokenizer, model_path)),
                ),
                patch.object(
                    gateway_app.admission,
                    "acquire",
                    new=AsyncMock(return_value=AdmissionLease([])),
                ),
                patch.object(
                    generation,
                    "call_triton_multimodal",
                    new=AsyncMock(return_value='{"status":"ok","retries":2}'),
                ),
            ):
                response = await gateway_app.create_chat_completion(request)

        message = response["choices"][0]["message"]
        self.assertEqual(
            {"status": "ok", "retries": 2},
            json.loads(message["content"]),
        )
        self.assertNotIn("reasoning_content", message)
        self.assertEqual("disabled", response["reasoning_status"]["mode"])
        self.assertFalse(tokenizer.enable_thinking)


if __name__ == "__main__":
    unittest.main()
