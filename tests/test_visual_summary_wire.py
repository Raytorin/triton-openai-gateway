"""Visual preprocessing negotiates the runtime's image input contract."""
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tritonclient.utils import deserialize_bytes_tensor

from gateway import triton_client as tc, vllm_media
from gateway.generation_telemetry import (
    get_generation_telemetry, reset_generation_telemetry, start_generation_telemetry,
)
from test_generation_metadata_transport import Result, envelope, run_async
from test_reasoning import FakeTokenizer


class VisualRuntime:
    """Runtime double: reject undeclared tensors and inspect serialized images."""
    def __init__(self, *, custom_media=False, metadata=False, discovery_error=False):
        self.custom_media = custom_media
        self.metadata = metadata
        self.discovery_error = discovery_error
        self.requests = []

    async def get_model_metadata(self, *args, **kwargs):
        if self.discovery_error:
            raise RuntimeError("metadata discovery unavailable")
        inputs = ["text_input", "stream", "sampling_parameters", "exclude_input_in_output", "image"]
        if self.custom_media:
            inputs.append("media_parameters")
        if self.metadata:
            inputs.append("return_generation_metadata")
        return {"inputs": [{"name": name} for name in inputs],
                "outputs": [{"name": name} for name in (
                    ["text_output", "generation_metadata"] if self.metadata else ["text_output"])]}

    async def generate(self, operation, model, inputs, outputs, **kwargs):
        tensors = {tensor.name(): tensor for tensor in inputs}
        self.requests.append(tensors)
        assert set(tensors) <= {
            "text_input", "stream", "sampling_parameters", "exclude_input_in_output", "image",
            *(["media_parameters"] if self.custom_media else []),
            *(["return_generation_metadata"] if self.metadata else []),
        }, "runtime rejects undeclared input tensors"
        images = [value.decode() for value in deserialize_bytes_tensor(tensors["image"]._get_content())]
        if self.custom_media:
            assert [json.loads(value)["data"] for value in images] == ["AA==", "AQ=="]
            parameters = json.loads(deserialize_bytes_tensor(tensors["media_parameters"]._get_content())[0])
            assert parameters["image_origins"] == ["pdf", "video"]
            assert parameters["media_order"] == ["image", "image"]
        else:
            assert images == ["AA==", "AQ=="]
        facts = {**envelope(), "vision_usage": {
            "status": "available", "source": "engine", "unit": "embedding_positions",
            "by_source": {"pdf": 7, "video": 5},
        }} if self.metadata else None
        yield Result(b"Media facts.", facts)


@run_async
@pytest.mark.parametrize("custom_media,metadata,discovery_error", [
    (False, False, False),
    (False, True, False),
    (False, False, True),
    (True, False, False),
    (True, True, False),
])
async def test_visual_summary_uses_only_advertised_runtime_tensors(custom_media, metadata, discovery_error):
    runtime = VisualRuntime(custom_media=custom_media, metadata=metadata, discovery_error=discovery_error)
    telemetry_handle = start_generation_telemetry("visual-wire", "/v1/chat/completions", time.monotonic())
    try:
        with (patch.object(tc, "get_grpc_client", AsyncMock(return_value=runtime)),
              patch.object(tc, "_stream_grpc_results", runtime.generate)):
            result = await vllm_media._generate_visual_summary(
                "visual-wire", FakeTokenizer(), [
                    vllm_media.VisualItem("Page 1", "AA==", source_kind="pdf"),
                    vllm_media.VisualItem("Frame 1", "AQ==", source_kind="video"),
                ], "Describe the media", [], {}, SimpleNamespace(summary_max_tokens=128),
            )
        assert result == "Media facts."
        assert len(runtime.requests) == 1
        assert get_generation_telemetry().visual_usage.snapshot()["preprocessing"] == {
            "status": "available" if metadata else "unavailable", "source": "engine",
            "unit": "embedding_positions", "by_source": {"pdf": 7, "video": 5} if metadata else {},
            "calls": 1,
        }
    finally:
        reset_generation_telemetry(telemetry_handle)
