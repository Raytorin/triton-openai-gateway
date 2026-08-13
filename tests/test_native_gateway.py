import base64
import json
import unittest
from unittest.mock import patch

try:
    from gateway.multimodal import (
        MediaPayload,
        MediaPayloads,
        extract_media_payloads,
        focus_current_media_context,
        isolate_latest_media_turn,
        reclassify_media_content,
        scope_media_history,
        strip_gateway_metadata,
    )
    from gateway.triton_client import _build_grpc_generate_inputs
except ImportError as exc:
    raise unittest.SkipTest(f"gateway runtime dependencies are unavailable: {exc}")


class FakeInferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, value):
        self.data = value


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return str(text).split()


class NativeGatewayTests(unittest.TestCase):
    def test_current_media_focus_preserves_bounded_text_history(self):
        conversation = [
            {"role": "system", "content": "Отвечай точно."},
            {"role": "user", "content": "Мы обсуждали инфраструктуру."},
            {"role": "assistant", "content": "Да, Triton и gateway."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что тут?"},
                    {"type": "video", "video": "dmlkZW8="},
                ],
            },
        ]
        media = extract_media_payloads(conversation)

        focused, kept, dropped = focus_current_media_context(
            conversation,
            FakeTokenizer(),
            media,
            history_max_tokens=64,
        )

        self.assertEqual(2, kept)
        self.assertEqual(0, dropped)
        self.assertIn("primary source", focused[0]["content"])
        self.assertEqual("Мы обсуждали инфраструктуру.", focused[1]["content"])
        self.assertEqual("Что тут?", focused[-1]["content"][0]["text"])

    def test_current_media_focus_drops_history_outside_budget(self):
        conversation = [
            {"role": "user", "content": "Старый вопрос " * 20},
            {"role": "assistant", "content": "Старый длинный ответ " * 20},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на фото?"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
        ]
        media = extract_media_payloads(conversation)

        focused, kept, dropped = focus_current_media_context(
            conversation,
            FakeTokenizer(),
            media,
            history_max_tokens=16,
        )

        self.assertEqual(0, kept)
        self.assertEqual(2, dropped)
        self.assertEqual(2, len(focused))
        self.assertEqual("system", focused[0]["role"])

    def test_current_media_focus_can_defer_history_limit_to_compressor(self):
        conversation = [
            {"role": "user", "content": "Старый вопрос " * 20},
            {"role": "assistant", "content": "Старый ответ " * 20},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на фото?"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
        ]
        media = extract_media_payloads(conversation)

        focused, kept, dropped = focus_current_media_context(
            conversation,
            FakeTokenizer(),
            media,
            history_max_tokens=None,
        )

        self.assertEqual(2, kept)
        self.assertEqual(0, dropped)
        self.assertIn("Старый вопрос", focused[1]["content"])
        self.assertEqual("Что на фото?", focused[-1]["content"][0]["text"])

    def test_current_media_focus_preserves_tool_result_after_latest_user(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на изображении и какая погода?"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Москва"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "name": "get_weather",
                "content": '{"temperature": 18}',
            },
        ]
        media = extract_media_payloads(conversation)

        focused, kept, dropped = focus_current_media_context(
            conversation,
            FakeTokenizer(),
            media,
            history_max_tokens=64,
        )

        self.assertEqual(0, kept)
        self.assertEqual(0, dropped)
        self.assertEqual(["system", "user", "assistant", "tool"], [
            message["role"] for message in focused
        ])
        self.assertEqual("call_weather", focused[-1]["tool_call_id"])

        isolated, removed = isolate_latest_media_turn(conversation)
        self.assertEqual(0, removed)
        self.assertEqual(["user", "assistant", "tool"], [
            message["role"] for message in isolated
        ])

    def test_current_media_focus_neutralizes_historical_media_turns(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на старой картинке?"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
            {"role": "assistant", "content": "На изображении цифра 2."},
            {"role": "user", "content": "Запомни название проекта."},
            {"role": "assistant", "content": "Проект называется Triton Gateway."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что происходит на видео?"},
                    {"type": "video", "video": "dmlkZW8="},
                ],
            },
        ]

        scoped, removed_media = scope_media_history(conversation)
        media = extract_media_payloads(scoped)
        focused, kept, dropped = focus_current_media_context(
            scoped,
            FakeTokenizer(),
            media,
            history_max_tokens=256,
        )

        self.assertEqual(1, removed_media)
        self.assertEqual(4, kept)
        self.assertEqual(0, dropped)
        self.assertNotIn("цифра 2", json.dumps(focused, ensure_ascii=False))
        self.assertIn("Historical attachment omitted", json.dumps(focused))
        self.assertIn("earlier attachment omitted", json.dumps(focused))
        self.assertIn("Triton Gateway", json.dumps(focused, ensure_ascii=False))
        self.assertNotIn("_gateway_historical_media", json.dumps(focused))

    def test_gateway_media_markers_are_removed_without_focus_mode(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Старое изображение"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
            {"role": "user", "content": "Текстовый вопрос"},
        ]

        scoped, _ = scope_media_history(conversation)
        cleaned = strip_gateway_metadata(scoped)

        self.assertNotIn("_gateway_historical_media", cleaned[0])
        self.assertEqual("Старое изображение", cleaned[0]["content"])

    def test_only_latest_user_turn_keeps_active_media(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что на картинке?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                    },
                ],
            },
            {"role": "assistant", "content": "На картинке цифра 2."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "А что это?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:application/pdf;base64,JVBERi0xLjcK"
                        },
                    },
                ],
            },
        ]

        scoped, removed = scope_media_history(conversation)
        media = extract_media_payloads(scoped)

        self.assertEqual(1, removed)
        self.assertEqual([], media.images)
        self.assertEqual(1, len(media.pdfs))
        self.assertEqual("Что на картинке?", scoped[0]["content"])

    def test_text_follow_up_does_not_reprocess_historical_media(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Проанализируй PDF"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:application/pdf;base64,JVBERi0xLjcK"
                        },
                    },
                ],
            },
            {"role": "assistant", "content": "Документ описывает систему."},
            {"role": "user", "content": "Уточни основную мысль."},
        ]

        scoped, removed = scope_media_history(conversation)

        self.assertEqual(1, removed)
        self.assertFalse(extract_media_payloads(scoped).has_any)
        self.assertEqual("Проанализируй PDF", scoped[0]["content"])

    def test_all_media_history_mode_preserves_legacy_behavior(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Старое изображение"},
                    {"type": "image", "image": "aW1hZ2U="},
                ],
            },
            {"role": "user", "content": "Новый вопрос"},
        ]

        scoped, removed = scope_media_history(conversation, mode="all")

        self.assertIs(scoped, conversation)
        self.assertEqual(0, removed)

    def test_new_media_turn_isolated_from_previous_text_answers(self):
        conversation = [
            {"role": "system", "content": "Отвечай по текущему вложению."},
            {"role": "user", "content": "Что на старой картинке?"},
            {"role": "assistant", "content": "На изображении цифра 2."},
            {"role": "user", "content": "О чем старый PDF?"},
            {"role": "assistant", "content": "Документ описывает Dognauts."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Что тут?"},
                    {"type": "video", "video": "dmlkZW8="},
                ],
            },
        ]

        isolated, removed = isolate_latest_media_turn(conversation)

        self.assertEqual(4, removed)
        self.assertEqual(2, len(isolated))
        self.assertEqual("system", isolated[0]["role"])
        self.assertEqual("Что тут?", isolated[1]["content"][0]["text"])

    def test_pdf_signature_overrides_incorrect_image_mime_type(self):
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,JVBERi0xLjcK"
                        },
                    }
                ],
            }
        ]

        media = extract_media_payloads(conversation)

        self.assertEqual([], media.images)
        self.assertEqual(1, len(media.pdfs))
        self.assertEqual(["pdf"], media.order)

    def test_mp4_signature_overrides_incorrect_image_content_type(self):
        mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
        encoded = base64.b64encode(mp4).decode("ascii")
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    }
                ],
            }
        ]

        media = extract_media_payloads(conversation)

        self.assertEqual([], media.images)
        self.assertEqual(1, len(media.videos))
        self.assertEqual("video/unknown", media.videos[0].mime_type)
        self.assertEqual("mp4", media.videos[0].format)
        self.assertEqual(["video"], media.order)

        rewritten = reclassify_media_content(conversation)
        self.assertEqual("video", rewritten[0]["content"][0]["type"])
        self.assertEqual(
            f"data:image/png;base64,{encoded}",
            rewritten[0]["content"][0]["video"],
        )

    def test_wav_signature_overrides_incorrect_video_content_type(self):
        wav = b"RIFF" + b"\x00" * 4 + b"WAVEfmt " + b"\x00" * 24
        encoded = base64.b64encode(wav).decode("ascii")
        conversation = [
            {
                "role": "user",
                "content": [{"type": "video", "video": encoded}],
            }
        ]

        media = extract_media_payloads(conversation)

        self.assertEqual([], media.videos)
        self.assertEqual(1, len(media.audios))
        self.assertEqual(["audio"], media.order)

    def test_native_media_is_mapped_to_backend_tensors(self):
        media = MediaPayloads(
            images=[MediaPayload("image-data", "image/png", "png")],
            videos=[MediaPayload("video-data", "video/mp4", "mp4")],
            audios=[MediaPayload("audio-data", "audio/wav", "wav")],
            pdfs=[MediaPayload("pdf-data", "application/pdf", "pdf")],
            order=["pdf", "image", "video", "audio"],
            parameters={"video_fps": 1.5, "pdf_dpi": 96},
        )

        with patch("gateway.triton_client.grpcclient.InferInput", FakeInferInput):
            inputs = _build_grpc_generate_inputs(
                "prompt",
                {"max_tokens": 32},
                [],
                stream=True,
                media=media,
            )

        by_name = {item.name: item for item in inputs}
        self.assertEqual(
            {
                "text_input",
                "stream",
                "sampling_parameters",
                "exclude_input_in_output",
                "image",
                "video",
                "audio",
                "pdf",
                "media_parameters",
            },
            set(by_name),
        )

        parameters = json.loads(by_name["media_parameters"].data[0].decode("utf-8"))
        self.assertEqual(["pdf", "image", "video", "audio"], parameters["media_order"])
        self.assertEqual(1.5, parameters["video_fps"])
        self.assertEqual(96, parameters["pdf_dpi"])

        video = json.loads(by_name["video"].data[0].decode("utf-8"))
        self.assertEqual("video/mp4", video["mime_type"])
        self.assertEqual("video-data", video["data"])


if __name__ == "__main__":
    unittest.main()
