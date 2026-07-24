import unittest

from fastapi import HTTPException

from gateway.prompt import fit_conversation_to_context


class FakeTokenizer:
    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    ):
        del tokenize, kwargs
        messages = [
            f"{message['role']} {message.get('content', '')}"
            for message in conversation
        ]
        if add_generation_prompt:
            messages.append("assistant")
        return " ".join(messages)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()


class ContextWindowTests(unittest.TestCase):
    def test_oldest_turns_are_removed_until_prompt_fits(self):
        conversation = [
            {"role": "system", "content": "Отвечай кратко"},
            {"role": "user", "content": "старый вопрос " * 10},
            {"role": "assistant", "content": "старый ответ " * 10},
            {"role": "user", "content": "новый вопрос " * 4},
            {"role": "assistant", "content": "новый ответ " * 4},
            {"role": "user", "content": "Кто ты?"},
        ]

        fitted, prompt, prompt_tokens, dropped = fit_conversation_to_context(
            FakeTokenizer(),
            conversation,
            tools=None,
            max_model_len=40,
            max_completion_tokens=8,
            safety_margin_tokens=2,
        )

        self.assertEqual(2, dropped)
        self.assertLessEqual(prompt_tokens, 30)
        self.assertEqual("system", fitted[0]["role"])
        self.assertEqual("Кто ты?", fitted[-1]["content"])
        self.assertNotIn("старый вопрос", prompt)
        self.assertIn("новый вопрос", prompt)

    def test_media_reserve_reduces_available_history(self):
        conversation = [
            {"role": "user", "content": "история " * 12},
            {"role": "assistant", "content": "ответ " * 8},
            {"role": "user", "content": "Что на видео?"},
        ]

        fitted, _, _, dropped = fit_conversation_to_context(
            FakeTokenizer(),
            conversation,
            tools=None,
            max_model_len=40,
            max_completion_tokens=8,
            reserved_media_tokens=16,
            safety_margin_tokens=2,
        )

        self.assertEqual(2, dropped)
        self.assertEqual(1, len(fitted))

    def test_latest_message_that_cannot_fit_returns_clear_error(self):
        conversation = [
            {"role": "system", "content": "system " * 10},
            {"role": "user", "content": "current " * 20},
        ]

        with self.assertRaisesRegex(HTTPException, "latest user message"):
            fit_conversation_to_context(
                FakeTokenizer(),
                conversation,
                tools=None,
                max_model_len=24,
                max_completion_tokens=8,
                safety_margin_tokens=2,
            )


if __name__ == "__main__":
    unittest.main()
