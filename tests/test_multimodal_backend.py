import base64
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

try:
    import av
    import numpy as np
    from PIL import Image
    import pymupdf
except ImportError as exc:  # These packages are installed in the release image.
    raise unittest.SkipTest(f"multimodal runtime dependencies are unavailable: {exc}")


BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backends" / "vllm_multimodal"
sys.path.insert(0, str(BACKEND_ROOT))

from utils.media import (  # noqa: E402
    build_multimodal_prompt,
    decode_audio_values,
    decode_video_values,
    expand_pdf_placeholders,
    select_video_frame_indices,
)
from utils.observability import log_event as backend_log_event  # noqa: E402
from utils.device_config import local_parallel_world_size  # noqa: E402


def _encoded_envelope(data: bytes, mime_type: str, media_format: str) -> bytes:
    return json.dumps(
        {
            "data": base64.b64encode(data).decode("ascii"),
            "mime_type": mime_type,
            "format": media_format,
        }
    ).encode("utf-8")


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def _pdf_bytes() -> bytes:
    document = pymupdf.open()
    for page_number in range(2):
        page = document.new_page(width=200, height=100)
        page.insert_text((20, 40), f"Page {page_number + 1}")
    data = document.tobytes()
    document.close()
    return data


def _wav_bytes() -> bytes:
    output = BytesIO()
    samples = (np.sin(np.linspace(0, np.pi * 8, 8000)) * 16000).astype(np.int16)
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(samples.tobytes())
    return output.getvalue()


def _video_bytes() -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4") as output:
        container = av.open(output.name, mode="w")
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(20):
            pixels = np.zeros((48, 64, 3), dtype=np.uint8)
            pixels[:, :, index % 3] = index * 10
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        output.seek(0)
        return output.read()


class MultimodalBackendTests(unittest.TestCase):
    def test_device_validation_counts_local_data_parallel_ranks(self):
        topology = local_parallel_world_size(
            {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 2,
            }
        )

        self.assertEqual((2, 1, 2, 4), topology)

    def test_device_validation_uses_local_data_parallel_size(self):
        topology = local_parallel_world_size(
            {
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 1,
                "data_parallel_size": 8,
                "data_parallel_size_local": 2,
            }
        )

        self.assertEqual((2, 1, 2, 4), topology)

    def test_backend_error_log_uses_cef_and_includes_model(self):
        class Logger:
            message = ""
            method = ""

            def log_error(self, message):
                self.message = message
                self.method = "error"

        logger = Logger()
        with patch.dict("os.environ", {"LOG_FORMAT": "cef"}, clear=False):
            backend_log_event(
                logger,
                "request.failed",
                level="error",
                model="test-model",
                request_id="request-1",
                error="invalid=value",
            )

        self.assertEqual("error", logger.method)
        self.assertIn("|request.failed|request.failed|8|", logger.message)
        self.assertIn("model=test-model", logger.message)
        self.assertIn("request_id=request-1", logger.message)
        self.assertIn("error=invalid\\=value", logger.message)

    def test_request_rejects_too_many_media_items(self):
        with patch.dict(
            "os.environ", {"VLLM_MULTIMODAL_MAX_MEDIA_ITEMS": "1"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "media items"):
                build_multimodal_prompt(
                    "prompt",
                    image_values=[b"first", b"second"],
                )

    def test_total_pdf_page_limit_applies_across_documents(self):
        pdf = _encoded_envelope(_pdf_bytes(), "application/pdf", "pdf")
        with patch.dict(
            "os.environ", {"VLLM_MULTIMODAL_MAX_PDF_PAGES": "3"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "PDF has 2 pages"):
                build_multimodal_prompt(
                    "<|image_pad|><|image_pad|>",
                    pdf_values=[pdf, pdf],
                    parameters={"pdf_dpi": 72, "pdf_max_pixels": 65536},
                )

    def test_total_video_frame_limit_applies_across_videos(self):
        video = _encoded_envelope(_video_bytes(), "video/mp4", "mp4")
        with patch.dict(
            "os.environ",
            {"VLLM_MULTIMODAL_MAX_TOTAL_VIDEO_FRAMES": "3"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "video frames"):
                decode_video_values(
                    [video, video],
                    {
                        "video_fps": 2,
                        "video_max_frames": 2,
                        "video_max_pixels": 4096,
                    },
                )

    def test_total_audio_duration_limit_applies_across_files(self):
        audio = _encoded_envelope(_wav_bytes(), "audio/wav", "wav")
        with patch.dict(
            "os.environ", {"VLLM_MULTIMODAL_MAX_AUDIO_SECONDS": "1"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "total audio duration"):
                decode_audio_values(
                    [audio, audio],
                    {"audio_sample_rate": 16000},
                )

    def test_video_sampling_covers_entire_duration(self):
        indices = select_video_frame_indices(300, 30.0, 2.0, 8)
        self.assertEqual(8, len(indices))
        self.assertEqual(0, indices[0])
        self.assertEqual(299, indices[-1])

    def test_pdf_placeholder_expansion_preserves_media_order(self):
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        prompt = f"before{marker}middle{marker}after"
        expanded = expand_pdf_placeholders(
            prompt,
            [2],
            media_order=["pdf", "image"],
            regular_image_count=1,
        )
        self.assertEqual(3, expanded.count(marker))
        self.assertIn(marker * 2, expanded)

    def test_pdf_is_rendered_into_ordered_image_items(self):
        marker = "<|vision_start|><|image_pad|><|vision_end|>"
        prompt = f"document:{marker} image:{marker}"
        result = build_multimodal_prompt(
            prompt,
            image_values=[_encoded_envelope(_image_bytes((255, 0, 0)), "image/png", "png")],
            pdf_values=[_encoded_envelope(_pdf_bytes(), "application/pdf", "pdf")],
            parameters={
                "media_order": ["pdf", "image"],
                "pdf_dpi": 72,
                "pdf_max_pixels": 65536,
            },
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(3, len(result["multi_modal_data"]["image"]))
        self.assertEqual(3, result["prompt"].count(marker))

    def test_video_contains_qwen_metadata(self):
        videos = decode_video_values(
            [_encoded_envelope(_video_bytes(), "video/mp4", "mp4")],
            {"video_fps": 2, "video_max_frames": 4, "video_max_pixels": 4096},
        )
        frames, metadata = videos[0]
        self.assertGreaterEqual(frames.shape[0], 2)
        self.assertLessEqual(frames.shape[0], 4)
        self.assertEqual(frames.shape[0], len(metadata["frames_indices"]))
        self.assertFalse(metadata["do_sample_frames"])

    def test_audio_is_resampled_for_vllm(self):
        audios = decode_audio_values(
            [_encoded_envelope(_wav_bytes(), "audio/wav", "wav")],
            {"audio_sample_rate": 16000},
        )
        samples, sample_rate = audios[0]
        self.assertEqual(16000, sample_rate)
        self.assertGreater(samples.size, 8000)
        self.assertEqual(np.float32, samples.dtype)


if __name__ == "__main__":
    unittest.main()
