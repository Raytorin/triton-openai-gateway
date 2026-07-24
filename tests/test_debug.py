import unittest
from unittest.mock import patch

from gateway import debug
from gateway.schemas import ChatCompletionRequest


class DebugLoggingTests(unittest.TestCase):
    def test_request_debug_logs_tool_metadata_without_payload(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "secret question"}],
                "debug": True,
            }
        )
        conversation = [
            {"role": "user", "content": "secret question"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "lookup", "arguments": {}},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "secret result",
            },
        ]

        with patch.object(debug, "DEBUG_LOG_PAYLOADS", False), patch.object(
            debug, "log_event"
        ) as log_event:
            debug.log_chat_request_debug(request, conversation, [], "fallback")
            debug.log_chat_prompt_debug(
                request,
                "secret rendered prompt",
                prompt_tokens=12,
                reserved_media_tokens=0,
            )
            debug.log_chat_response_debug(
                request,
                "secret generated response",
                "final answer",
                [],
                "stop",
            )

        events = [call.args[1] for call in log_event.call_args_list]
        self.assertEqual(
            ["chat.debug.request", "chat.debug.prompt", "chat.debug.response"],
            events,
        )
        for call in log_event.call_args_list:
            self.assertNotIn("prompt_preview", call.kwargs)
            self.assertNotIn("generated_preview", call.kwargs)
        request_fields = log_event.call_args_list[0].kwargs
        self.assertEqual("call_1", request_fields["messages"][-1]["tool_call_id"])
        self.assertEqual(["lookup"], request_fields["messages"][1]["tool_call_names"])

    def test_global_debug_logs_request_without_request_flag(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "question"}],
            }
        )

        with patch.object(debug, "GATEWAY_DEBUG", True), patch.object(
            debug, "log_event"
        ) as log_event:
            debug.log_chat_request_debug(
                request,
                [{"role": "user", "content": "question"}],
                [],
                None,
            )

        log_event.assert_called_once()
        self.assertEqual("chat.debug.request", log_event.call_args.args[1])

    def test_debug_is_disabled_by_default(self):
        request = ChatCompletionRequest.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "question"}],
            }
        )

        with patch.object(debug, "GATEWAY_DEBUG", False), patch.object(
            debug, "log_event"
        ) as log_event:
            debug.log_chat_request_debug(
                request,
                [{"role": "user", "content": "question"}],
                [],
                None,
            )

        log_event.assert_not_called()


if __name__ == "__main__":
    unittest.main()
