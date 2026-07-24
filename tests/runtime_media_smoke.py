"""CPU-only smoke test intended to run inside the project Triton image."""

import base64
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gateway.vllm_media import (
    _extract_video_frames,
    _render_pdf_pages,
    load_vllm_media_settings,
)


def create_pdf(path: Path) -> None:
    document = pymupdf.open()
    for page_number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"Runtime test page {page_number}")
    document.save(path)
    document.close()


def create_video(path: Path) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mpeg4", rate=10)
    stream.width = 64
    stream.height = 64
    stream.pix_fmt = "yuv420p"
    for index in range(20):
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = min(index * 12, 255)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        model_path = root / "model"
        work_dir = root / "work"
        model_path.mkdir()
        work_dir.mkdir()
        (model_path / "model.json").write_text(
            json.dumps({"limit_mm_per_prompt": {"image": 2}}),
            encoding="utf-8",
        )
        (model_path / "gateway.json").write_text(
            json.dumps(
                {
                    "vllm_multimodal": {
                        "pdf_dpi": 72,
                        "video_fps": 2,
                        "video_max_frames": 4,
                        "pdf_max_pixels": 65536,
                        "video_max_pixels": 65536,
                    }
                }
            ),
            encoding="utf-8",
        )
        settings = load_vllm_media_settings(model_path)

        pdf_path = root / "source.pdf"
        create_pdf(pdf_path)
        pdf_payload = SimpleNamespace(
            data=base64.b64encode(pdf_path.read_bytes()).decode("ascii"),
            format="pdf",
        )
        pages = _render_pdf_pages(pdf_payload, work_dir, 1, settings)
        assert len(pages) == 3
        assert pages[0].mime_type == "image/png"
        base64.b64decode(pages[0].data, validate=True)

        video_path = root / "source.mp4"
        create_video(video_path)
        video_payload = SimpleNamespace(
            data=base64.b64encode(video_path.read_bytes()).decode("ascii"),
            format="mp4",
        )
        frames = _extract_video_frames(video_payload, work_dir, 1, settings)
        assert 1 < len(frames) <= 4
        assert "timestamp" in frames[-1].label
        base64.b64decode(frames[-1].data, validate=True)

    print("Triton OpenAI Gateway PDF/video preprocessing: OK")


if __name__ == "__main__":
    main()
