import sys
import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from gateway.registry import ModelRegistry


class _AutoTokenizer:
    calls = 0

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        cls.calls += 1
        return object()


class RegistryTests(unittest.TestCase):
    def _create_model(
        self,
        root: Path,
        model_name: str,
        model_json: str,
        backend: str = "vllm",
    ) -> tuple[Path, Path]:
        active = root / "active"
        active.mkdir(exist_ok=True)
        model_root = root / model_name
        model_path = model_root / "1"
        model_path.mkdir(parents=True)
        (model_path / "model.json").write_text(model_json, encoding="utf-8")
        (model_root / "config.pbtxt").write_text(
            f'backend: "{backend}"\n',
            encoding="utf-8",
        )
        (active / model_name).symlink_to(model_path)
        return active, model_path

    def test_tokenizer_load_observes_duration_without_timer_private_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            active = root / "active"
            model_path = root / "model" / "1"
            active.mkdir()
            model_path.mkdir(parents=True)
            (active / "model-a").symlink_to(model_path)

            transformers = ModuleType("transformers")
            transformers.AutoTokenizer = _AutoTokenizer
            duration = MagicMock()
            loads = MagicMock()
            _AutoTokenizer.calls = 0

            with (
                patch("gateway.registry.MODELS_ACTIVE_DIR", active),
                patch("gateway.registry.TOKENIZER_LOAD_DURATION") as histogram,
                patch("gateway.registry.TOKENIZER_LOADS", loads),
                patch.dict(sys.modules, {"transformers": transformers}),
            ):
                histogram.labels.return_value = duration
                registry = ModelRegistry()
                first, _ = registry.get_tokenizer("model-a")
                second, _ = registry.get_tokenizer("model-a")

            self.assertIs(first, second)
            self.assertEqual(_AutoTokenizer.calls, 1)
            duration.observe.assert_called_once()
            duration.time.assert_not_called()
            loads.labels.assert_called_once_with("model-a", "success")

    def test_embedding_model_rejects_chat_route_with_rag_guidance(self):
        with tempfile.TemporaryDirectory() as directory:
            active, model_path = self._create_model(
                Path(directory),
                "embedding-model",
                '{"runner":"pooling","convert":"embed"}',
            )
            with patch("gateway.registry.MODELS_ACTIVE_DIR", active):
                registry = ModelRegistry()
                self.assertEqual(
                    registry.get_capabilities("embedding-model", model_path),
                    frozenset({"embeddings"}),
                )
                registry.validate_route("embedding-model", "embeddings", model_path)
                with self.assertRaises(HTTPException) as context:
                    registry.validate_route("embedding-model", "chat", model_path)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("/v1/embeddings", str(context.exception.detail))
            self.assertIn("generative model", str(context.exception.detail))

    def test_vllm_generation_model_rejects_embeddings_route(self):
        with tempfile.TemporaryDirectory() as directory:
            active, model_path = self._create_model(
                Path(directory),
                "chat-model",
                '{"runner":"generate","max_model_len":32768}',
            )
            with patch("gateway.registry.MODELS_ACTIVE_DIR", active):
                registry = ModelRegistry()
                registry.validate_route("chat-model", "chat", model_path)
                with self.assertRaises(HTTPException) as context:
                    registry.validate_route("chat-model", "embeddings", model_path)

            self.assertEqual(context.exception.status_code, 400)
            self.assertIn("text generation", str(context.exception.detail))

    def test_auto_task_detection_is_delegated_to_backend(self):
        for backend in ("vllm", "vllm_multimodal"):
            for config in ('{"max_model_len":32768}', '{"runner":"auto"}'):
                with self.subTest(backend=backend, config=config):
                    with tempfile.TemporaryDirectory() as directory:
                        active, model_path = self._create_model(
                            Path(directory), "auto-model", config, backend
                        )
                        with patch("gateway.registry.MODELS_ACTIVE_DIR", active):
                            registry = ModelRegistry()
                            self.assertIsNone(registry.get_capabilities("auto-model"))
                            registry.validate_route("auto-model", "embeddings", model_path)
                            registry.validate_route("auto-model", "chat", model_path)


if __name__ == "__main__":
    unittest.main()
