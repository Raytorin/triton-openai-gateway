import json
import tempfile
import unittest
from pathlib import Path

from gateway.tool_parsers import (
    QWEN3_CODER_PARSER,
    detect_model_tool_parser,
    extract_tool_calls,
    normalize_tool_calls_for_template,
)
from gateway.prompt import build_conversation, has_tool_result, tool_choice_instruction
from gateway.schemas import ChatCompletionRequest


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                    "days": {"type": "integer"},
                    "include_wind": {"type": "boolean"},
                    "filters": {"type": "array"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string"}},
            },
        },
    },
]


class ToolParserTests(unittest.TestCase):
    def test_assistant_tool_call_message_may_omit_content(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Какая погода?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location":"London"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": '{"temperature":18}',
                    },
                ],
                "debug": True,
            }
        )

        conversation = build_conversation(request.messages)

        self.assertEqual("", conversation[1]["content"])
        self.assertTrue(has_tool_result(conversation))
        self.assertTrue(request.debug)

    def test_auto_tool_followup_instruction_requires_final_answer(self):
        instruction = tool_choice_instruction("auto", has_tool_result=True)

        self.assertIn("answer the user directly", instruction)
        self.assertIn("do not return an empty response", instruction)

    def test_existing_json_xml_format_is_preserved(self):
        text = (
            "Calling the tool.\n"
            '<tool_call>{"name":"get_current_weather","arguments":'
            '{"location":"London","unit":"celsius"}}</tool_call>'
        )

        calls, content = extract_tool_calls(text, TOOLS)

        self.assertEqual("Calling the tool.", content)
        self.assertEqual("get_current_weather", calls[0]["function"]["name"])
        self.assertEqual(
            {"location": "London", "unit": "celsius"},
            json.loads(calls[0]["function"]["arguments"]),
        )

    def test_qwen3_coder_xml_is_converted_to_openai_format(self):
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>
London, UK
</parameter>
<parameter=unit>
celsius
</parameter>
<parameter=days>
3
</parameter>
<parameter=include_wind>
true
</parameter>
<parameter=filters>
["rain", "wind"]
</parameter>
</function>
</tool_call>"""

        calls, content = extract_tool_calls(text, TOOLS, QWEN3_CODER_PARSER)

        self.assertEqual("", content)
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertEqual("function", calls[0]["type"])
        self.assertEqual("get_current_weather", calls[0]["function"]["name"])
        self.assertEqual(
            {
                "location": "London, UK",
                "unit": "celsius",
                "days": 3,
                "include_wind": True,
                "filters": ["rain", "wind"],
            },
            json.loads(calls[0]["function"]["arguments"]),
        )

    def test_universal_fallback_parses_qwen_xml_without_parser_file(self):
        text = """<tool_call>
<function=get_time>
<parameter=timezone>
Europe/Moscow
</parameter>
</function>
</tool_call>"""

        calls, content = extract_tool_calls(text, TOOLS)

        self.assertEqual("", content)
        self.assertEqual("get_time", calls[0]["function"]["name"])
        self.assertEqual(
            {"timezone": "Europe/Moscow"},
            json.loads(calls[0]["function"]["arguments"]),
        )

    def test_multiple_qwen_tool_calls_are_preserved(self):
        text = """<tool_call>
<function=get_current_weather>
<parameter=location>London</parameter>
</function>
</tool_call>
<tool_call>
<function=get_time>
<parameter=timezone>Europe/London</parameter>
</function>
</tool_call>"""

        calls, content = extract_tool_calls(text, TOOLS, QWEN3_CODER_PARSER)

        self.assertEqual("", content)
        self.assertEqual(
            ["get_current_weather", "get_time"],
            [call["function"]["name"] for call in calls],
        )

    def test_raw_function_is_used_as_truncated_output_fallback(self):
        text = """<tool_call>
<function=get_time>
<parameter=timezone>UTC</parameter>
</function>"""

        calls, content = extract_tool_calls(text, TOOLS)

        self.assertEqual("", content)
        self.assertEqual("get_time", calls[0]["function"]["name"])

    def test_malformed_tool_call_remains_normal_content(self):
        text = "<tool_call><parameter=location>London</parameter></tool_call>"

        calls, content = extract_tool_calls(text, TOOLS, QWEN3_CODER_PARSER)

        self.assertEqual([], calls)
        self.assertEqual(text, content)

    def test_model_parser_is_detected_without_importing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            parser_path = model_path / "qwen3coder_tool_parser.py"
            parser_path.write_text("raise RuntimeError('must not be imported')\n")

            self.assertEqual(
                QWEN3_CODER_PARSER,
                detect_model_tool_parser(model_path),
            )

    def test_missing_model_parser_uses_universal_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(detect_model_tool_parser(Path(directory)))

    def test_openai_argument_string_is_normalized_for_qwen_template(self):
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "get_time",
                    "arguments": '{"timezone":"Europe/Moscow"}',
                },
            }
        ]

        normalized = normalize_tool_calls_for_template(tool_calls)

        self.assertEqual(
            {"timezone": "Europe/Moscow"},
            normalized[0]["function"]["arguments"],
        )


if __name__ == "__main__":
    unittest.main()
