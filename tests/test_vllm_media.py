import json
import os
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.media_preprocessing import (
    VllmMediaProcessingError,
    _validate_remote_url,
)
from gateway.vllm_media import (
    MediaSummary,
    PdfEmbeddingContext,
    VisualItem,
    _build_pdf_text_chunks,
    _conversation_with_media_evidence,
    _is_document_summary_request,
    _retrieve_pdf_text_chunks,
    _split_text,
    has_extended_vllm_media,
    load_vllm_media_settings,
    materialize_native_remote_media,
    prepare_vllm_media_conversation,
    requires_gateway_media_preprocessing,
    select_video_frame_indices,
)


@dataclass(frozen=True)
class MediaPayloadStub:
    data: str
    mime_type: str | None = None
    format: str | None = None


@dataclass(frozen=True)
class MediaPayloadsStub:
    images: list[MediaPayloadStub]
    videos: list[MediaPayloadStub]
    audios: list[MediaPayloadStub]
    pdfs: list[MediaPayloadStub]
    order: list[str]
    parameters: dict = field(default_factory=dict)


class VllmMediaTests(unittest.TestCase):
    def test_remote_media_rejects_private_address(self):
        address = (2, 1, 6, "", ("127.0.0.1", 80))
        with patch("gateway.media_preprocessing.socket.getaddrinfo", return_value=[address]):
            with self.assertRaisesRegex(VllmMediaProcessingError, "non-public"):
                _validate_remote_url("http://example.test/video.mp4", False)

    def test_remote_media_allows_private_address_only_when_enabled(self):
        with patch("gateway.media_preprocessing.socket.getaddrinfo") as resolver:
            _validate_remote_url("http://internal.test/video.mp4", True)
        resolver.assert_not_called()

    def test_video_sampling_covers_full_duration_and_honors_limit(self):
        indices = select_video_frame_indices(
            total_frames=300,
            source_fps=30.0,
            sample_fps=2.0,
            max_frames=8,
        )

        self.assertEqual(8, len(indices))
        self.assertEqual(0, indices[0])
        self.assertEqual(299, indices[-1])
        self.assertEqual(sorted(set(indices)), indices)

    def test_model_image_limit_caps_pdf_and_video_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                json.dumps({"limit_mm_per_prompt": {"image": 2}}),
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "vllm_multimodal": {
                            "pdf_chunk_pages": 5,
                            "video_chunk_frames": 6,
                            "video_max_frames": 12,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_vllm_media_settings(model_path)

        self.assertEqual(2, settings.image_limit)
        self.assertEqual(2, settings.pdf_chunk_pages)
        self.assertEqual(2, settings.video_chunk_frames)
        self.assertEqual(12, settings.video_max_frames)

    def test_environment_overrides_gateway_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                json.dumps({"limit_mm_per_prompt": {"image": 4}}),
                encoding="utf-8",
            )
            (model_path / "gateway.json").write_text(
                json.dumps({"vllm_multimodal": {"pdf_chunk_pages": 3}}),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"VLLM_MEDIA_PDF_CHUNK_PAGES": "1"},
                clear=True,
            ):
                settings = load_vllm_media_settings(model_path)

        self.assertEqual(1, settings.pdf_chunk_pages)

    def test_pdf_rag_settings_are_loaded_from_gateway_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text("{}", encoding="utf-8")
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "vllm_multimodal": {
                            "pdf_rag": {
                                "embedding_model": "Qwen3-Embedding-4B",
                                "dimensions": 1024,
                                "chunk_chars": 1200,
                                "chunk_overlap": 100,
                                "top_k": 4,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_vllm_media_settings(model_path)

        self.assertEqual("Qwen3-Embedding-4B", settings.pdf_embedding_model)
        self.assertEqual(1024, settings.pdf_embedding_dimensions)
        self.assertEqual(1200, settings.pdf_rag_chunk_chars)
        self.assertEqual(100, settings.pdf_rag_chunk_overlap)
        self.assertEqual(4, settings.pdf_rag_top_k)
        self.assertEqual("latest", settings.media_history_mode)
        self.assertFalse(settings.reset_history_on_new_media)
        self.assertTrue(settings.focus_current_media)
        self.assertEqual(512, settings.media_history_max_tokens)

    def test_media_history_budget_is_loaded_from_gateway_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text("{}", encoding="utf-8")
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "vllm_multimodal": {
                            "media_history_max_tokens": 1536,
                        }
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                settings = load_vllm_media_settings(model_path)

        self.assertEqual(1536, settings.media_history_max_tokens)

    def test_final_conversation_removes_media_and_adds_all_evidence(self):
        conversation = [
            {"role": "system", "content": "Be precise."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize the document."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:application/pdf;base64,AAAA"},
                    },
                ],
            },
        ]

        result = _conversation_with_media_evidence(
            conversation,
            "Summarize the document.",
            [
                MediaSummary(label="PDF 1, page 1", text="First fact."),
                MediaSummary(label="PDF 1, page 2", text="Second fact."),
            ],
        )

        self.assertEqual("Be precise.", result[0]["content"])
        self.assertNotIn("image_url", result[1]["content"])
        self.assertIn("First fact.", result[1]["content"])
        self.assertIn("Second fact.", result[1]["content"])

    def test_text_chunking_preserves_all_content(self):
        text = " ".join(f"word-{index}" for index in range(100))
        chunks = _split_text(text, 256)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(text.replace(" ", ""), "".join(chunks).replace(" ", ""))

    def test_document_summary_request_does_not_use_retrieval(self):
        self.assertTrue(_is_document_summary_request("О чем данный документ?"))
        self.assertTrue(_is_document_summary_request("Summarize this document"))
        self.assertFalse(_is_document_summary_request("Какой срок указан в договоре?"))

    def test_pdf_text_chunks_keep_page_labels_and_overlap(self):
        text = " ".join(f"word-{index}" for index in range(100))
        chunks = _build_pdf_text_chunks(
            [("PDF 1, page 1/1", text)],
            chunk_chars=256,
            overlap=32,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all("PDF 1, page 1/1" in chunk.label for chunk in chunks))

    def test_extended_media_detection(self):
        self.assertTrue(
            has_extended_vllm_media(
                SimpleNamespace(images=[], videos=[object()], audios=[], pdfs=[])
            )
        )
        self.assertFalse(
            has_extended_vllm_media(
                SimpleNamespace(images=[object()], videos=[], audios=[], pdfs=[])
            )
        )

    def test_native_backend_preprocesses_pdf_and_configured_audio_asr(self):
        pdf_media = SimpleNamespace(
            images=[], videos=[], audios=[], pdfs=[object()]
        )
        video_media = SimpleNamespace(
            images=[], videos=[object()], audios=[], pdfs=[]
        )
        audio_media = SimpleNamespace(
            images=[], videos=[], audios=[object()], pdfs=[]
        )

        self.assertTrue(
            requires_gateway_media_preprocessing("vllm_multimodal", pdf_media)
        )
        self.assertFalse(
            requires_gateway_media_preprocessing("vllm_multimodal", video_media)
        )
        self.assertFalse(
            requires_gateway_media_preprocessing("vllm_multimodal", audio_media)
        )
        self.assertTrue(
            requires_gateway_media_preprocessing(
                "vllm_multimodal",
                audio_media,
                SimpleNamespace(audio_asr_model="/models/whisper"),
            )
        )
        self.assertTrue(requires_gateway_media_preprocessing("vllm", video_media))


class VllmMediaOrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_pdf_retrieval_selects_highest_cosine_score(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text("{}", encoding="utf-8")
            (model_path / "gateway.json").write_text(
                json.dumps({"vllm_multimodal": {"pdf_rag": {"top_k": 1}}}),
                encoding="utf-8",
            )
            settings = load_vllm_media_settings(model_path)

        with patch(
            "gateway.triton_client.call_triton_embeddings",
            new_callable=AsyncMock,
            side_effect=[
                ([1.0, 0.0], 2),
                ([0.0, 1.0], 3),
                ([0.9, 0.1], 3),
            ],
        ) as embed:
            selected = await _retrieve_pdf_text_chunks(
                [
                    ("PDF 1, page 1/2", "Нерелевантный раздел."),
                    ("PDF 1, page 2/2", "Срок договора 12 месяцев."),
                ],
                "Какой срок договора?",
                PdfEmbeddingContext("embedding-model"),
                settings,
            )

        self.assertEqual(3, embed.await_count)
        self.assertEqual(1, len(selected))
        self.assertIn("12 месяцев", selected[0].text)

    async def test_native_remote_media_is_materialized_before_backend(self):
        media = MediaPayloadsStub(
            images=[MediaPayloadStub("https://example.test/image.png", "image/remote-url", "png")],
            videos=[],
            audios=[],
            pdfs=[],
            order=["image"],
        )

        def materialize(payload, media_type, work_dir, index, settings):
            path = work_dir / "image.png"
            path.write_bytes(b"image-bytes")
            return path

        with (
            tempfile.TemporaryDirectory() as model_directory,
            patch("gateway.vllm_media._materialize_payload", side_effect=materialize),
        ):
            result = await materialize_native_remote_media(
                Path(model_directory),
                media,
            )

        self.assertEqual("aW1hZ2UtYnl0ZXM=", result.images[0].data)
        self.assertEqual("png", result.images[0].format)

    async def test_pdf_chunks_are_synthesized_before_final_text_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text(
                json.dumps({"limit_mm_per_prompt": {"image": 2}}),
                encoding="utf-8",
            )
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "О чем данный документ?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:application/pdf;base64,AAAA"
                            },
                        },
                    ],
                }
            ]
            media = SimpleNamespace(
                images=[],
                videos=[],
                audios=[],
                pdfs=[SimpleNamespace(data="AAAA", format="pdf")],
            )

            with (
                patch(
                    "gateway.vllm_media._render_pdf_pages",
                    return_value=[
                        VisualItem(label="PDF 1, page 1/4", data="a"),
                        VisualItem(label="PDF 1, page 2/4", data="b"),
                        VisualItem(label="PDF 1, page 3/4", data="c"),
                        VisualItem(label="PDF 1, page 4/4", data="d"),
                    ],
                ),
                patch(
                    "gateway.vllm_media._generate_visual_summary",
                    new_callable=AsyncMock,
                    return_value="Chunk facts.",
                ) as generate_summary,
                patch(
                    "gateway.vllm_media._generate_text_summary",
                    new_callable=AsyncMock,
                    return_value="Coherent document synthesis.",
                ) as generate_synthesis,
            ):
                result = await prepare_vllm_media_conversation(
                    "test-model",
                    model_path,
                    object(),
                    conversation,
                    media,
                    {"max_tokens": 128, "temperature": 0.2},
                )

        self.assertEqual(2, generate_summary.await_count)
        generate_synthesis.assert_awaited_once()
        self.assertIn("Coherent document synthesis.", result[-1]["content"])
        self.assertNotIn("PDF 1, page", result[-1]["content"])
        self.assertNotIn("image_url", result[-1]["content"])

    async def test_targeted_pdf_question_uses_configured_embedding_retrieval(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "model.json").write_text("{}", encoding="utf-8")
            settings = load_vllm_media_settings(model_path)
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Какой срок указан в договоре?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:application/pdf;base64,AAAA"},
                        },
                    ],
                }
            ]
            media = SimpleNamespace(
                images=[], videos=[], audios=[],
                pdfs=[SimpleNamespace(data="AAAA", format="pdf")],
            )

            with (
                patch(
                    "gateway.vllm_media._extract_pdf_text_pages",
                    return_value=[("PDF 1, page 2/3", "Срок договора 12 месяцев. " * 20)],
                ),
                patch(
                    "gateway.vllm_media._retrieve_pdf_text_chunks",
                    new_callable=AsyncMock,
                    return_value=[
                        MediaSummary(
                            label="PDF 1, page 2/3",
                            text="Срок договора составляет 12 месяцев.",
                        )
                    ],
                ) as retrieve,
                patch("gateway.vllm_media._render_pdf_pages") as render_pages,
            ):
                result = await prepare_vllm_media_conversation(
                    "chat-model",
                    model_path,
                    object(),
                    conversation,
                    media,
                    {"max_tokens": 128},
                    settings=settings,
                    pdf_embedding=PdfEmbeddingContext("embedding-model"),
                )

        retrieve.assert_awaited_once()
        render_pages.assert_not_called()
        self.assertIn("12 месяцев", result[-1]["content"])


if __name__ == "__main__":
    unittest.main()
