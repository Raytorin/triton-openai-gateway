import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import tempfile
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from gateway.registry import ModelRegistry


class RegistryConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_switch_during_load_uses_new_tokenizer_and_model_path(self):
        registry = ModelRegistry()
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        first, second = object(), object()

        def load(path, **_kwargs):
            if path == "/v1":
                loop.call_soon_threadsafe(started.set)
                release.wait(2)
                return first
            self.assertEqual(path, "/v2")
            return second

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = type("Loader", (), {"from_pretrained": staticmethod(load)})
        with (
            patch.object(registry, "resolve", return_value=Path("/v1")) as path,
            patch.object(registry, "resolve_tokenizer_path", side_effect=lambda value: value),
            patch.dict(sys.modules, {"transformers": transformers}),
        ):
            old = asyncio.create_task(registry.get_tokenizer_async("model"))
            await asyncio.wait_for(started.wait(), 1)
            path.return_value = Path("/v2")
            new = asyncio.create_task(registry.get_tokenizer_async("model"))
            try:
                await asyncio.sleep(0.02)
            finally:
                release.set()
                results = await asyncio.gather(old, new)
            self.assertEqual(results[0], (first, Path("/v1")))
            self.assertEqual(results[1], (second, Path("/v2")))

    async def test_cold_waiters_do_not_block_cached_tokenizer(self):
        registry = ModelRegistry()
        cached = object()
        registry._cache["warm"] = ("/warm", cached)
        started = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
        calls = []

        def load(path, **_kwargs):
            calls.append(path)
            loop.call_soon_threadsafe(started.set)
            release.wait(2)
            return object()

        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = type("Loader", (), {"from_pretrained": staticmethod(load)})
        with (
            patch.object(registry, "resolve", side_effect=lambda name: Path("/") / name),
            patch.object(registry, "resolve_tokenizer_path", side_effect=lambda path: path),
            patch.dict(sys.modules, {"transformers": transformers}),
        ):
            cold = [asyncio.create_task(registry.get_tokenizer_async("cold")) for _ in range(5)]
            await asyncio.wait_for(started.wait(), 1)
            try:
                warm, _ = await asyncio.wait_for(registry.get_tokenizer_async("warm"), 0.2)
                self.assertIs(warm, cached)
                cold[0].cancel()
                await asyncio.gather(cold[0], return_exceptions=True)
            finally:
                release.set()
                results = await asyncio.gather(*cold, return_exceptions=True)
            self.assertEqual(calls, ["/cold"])
            self.assertTrue(all(result[0] is results[1][0] for result in results[1:]))

    async def test_failed_tokenizer_load_can_be_retried_and_path_change_reloads(self):
        registry = ModelRegistry()
        first, second = object(), object()
        loader = MagicMock(side_effect=[ValueError("broken tokenizer"), first, second])
        transformers = ModuleType("transformers")
        transformers.AutoTokenizer = type("Loader", (), {"from_pretrained": loader})
        with (
            patch.object(registry, "resolve", return_value=Path("/model")),
            patch.object(registry, "resolve_tokenizer_path", return_value=Path("/v1")) as path,
            patch.dict(sys.modules, {"transformers": transformers}),
        ):
            with self.assertRaisesRegex(ValueError, "broken tokenizer"):
                await registry.get_tokenizer_async("model")
            self.assertIs((await registry.get_tokenizer_async("model"))[0], first)
            path.return_value = Path("/v2")
            self.assertIs((await registry.get_tokenizer_async("model"))[0], second)


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
