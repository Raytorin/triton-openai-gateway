# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from pydantic import ValidationError

from gateway import app as gateway_app
from gateway.embeddings import (
    HybridEmbeddingSettings,
    load_hybrid_embedding_settings,
    reject_hybrid_options_on_dense_endpoint,
    validate_hybrid_embedding_request,
)
from gateway.schemas import EmbeddingsRequest, HybridEmbeddingsRequest
from gateway.triton_client import (
    _build_grpc_embedding_inputs,
    _validate_hybrid_embedding_payload,
)


class _InferInput:
    def __init__(self, name, shape, datatype):
        self.name = name
        self.shape = shape
        self.datatype = datatype
        self.data = None

    def set_data_from_numpy(self, data):
        self.data = data


class HybridEmbeddingSchemaTests(unittest.TestCase):
    def test_defaults_request_dense_and_sparse(self):
        request = HybridEmbeddingsRequest(model="bge-m3", input="query")

        self.assertEqual(request.output_types, ["dense", "sparse"])

    def test_compatibility_aliases_are_normalized(self):
        sparse = HybridEmbeddingsRequest(
            model="bge-m3",
            input="query",
            output_type="sparse",
        )
        dense = HybridEmbeddingsRequest(
            model="bge-m3",
            input="query",
            return_sparse=False,
        )

        self.assertEqual(sparse.output_types, ["sparse"])
        self.assertEqual(dense.output_types, ["dense"])

    def test_alias_and_canonical_output_types_cannot_be_combined(self):
        with self.assertRaises(ValidationError):
            HybridEmbeddingsRequest(
                model="bge-m3",
                input="query",
                output_types=["dense"],
                return_sparse=True,
            )

    def test_dense_endpoint_rejects_sparse_extension_instead_of_ignoring_it(self):
        request = EmbeddingsRequest(
            model="bge-m3",
            input="query",
            return_sparse=True,
        )

        with self.assertRaises(HTTPException) as context:
            reject_hybrid_options_on_dense_endpoint(request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("/v1/hybrid_embeddings", context.exception.detail)

    def test_dense_endpoint_accepts_standard_openai_fields(self):
        request = EmbeddingsRequest(
            model="bge-m3",
            input="query",
            user="request-owner",
        )

        reject_hybrid_options_on_dense_endpoint(request)


class HybridEmbeddingConfigurationTests(unittest.TestCase):
    def test_model_configuration_controls_outputs_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "embeddings": {
                            "hybrid": {
                                "enabled": True,
                                "output_types": ["dense", "sparse"],
                                "max_batch_size": 4,
                                "default_sparse_top_k": 128,
                                "max_sparse_top_k": 512,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = load_hybrid_embedding_settings(model_path)

        self.assertTrue(settings.enabled)
        self.assertEqual(settings.output_types, frozenset({"dense", "sparse"}))
        self.assertEqual(settings.max_batch_size, 4)
        self.assertEqual(settings.default_sparse_top_k, 128)
        self.assertEqual(settings.max_sparse_top_k, 512)

    def test_request_validation_applies_default_top_k(self):
        request = HybridEmbeddingsRequest(model="bge-m3", input="query")
        settings = HybridEmbeddingSettings(
            enabled=True,
            output_types=frozenset({"dense", "sparse"}),
            default_sparse_top_k=128,
            max_sparse_top_k=512,
        )

        sparse_top_k = validate_hybrid_embedding_request(request, settings, 1)

        self.assertEqual(sparse_top_k, 128)

    def test_request_validation_rejects_disabled_model(self):
        request = HybridEmbeddingsRequest(model="dense-only", input="query")

        with self.assertRaises(HTTPException) as context:
            validate_hybrid_embedding_request(
                request,
                HybridEmbeddingSettings(),
                1,
            )

        self.assertEqual(context.exception.status_code, 400)


class HybridEmbeddingTritonContractTests(unittest.TestCase):
    def test_grpc_request_contains_hybrid_parameters(self):
        with patch("gateway.triton_client.grpcclient.InferInput", _InferInput):
            inputs = _build_grpc_embedding_inputs(
                "query",
                None,
                output_types=["dense", "sparse"],
                sparse_top_k=64,
            )

        payload = json.loads(inputs[0].data[0].decode("utf-8"))
        self.assertEqual(payload["input"], "query")
        self.assertEqual(payload["output_types"], ["dense", "sparse"])
        self.assertEqual(payload["sparse_format"], "indices_values")
        self.assertEqual(payload["sparse_top_k"], 64)

    def test_backend_payload_is_normalized(self):
        result = _validate_hybrid_embedding_payload(
            {
                "dense": [0.1, 0.2],
                "sparse": {"indices": [42, 108], "values": [0.9, 0.4]},
            },
            ["dense", "sparse"],
        )

        self.assertEqual(result["dense"], [0.1, 0.2])
        self.assertEqual(result["sparse"]["indices"], [42, 108])

    def test_backend_payload_rejects_non_finite_weights(self):
        with self.assertRaises(HTTPException) as context:
            _validate_hybrid_embedding_payload(
                {
                    "sparse": {
                        "indices": [42],
                        "values": [float("nan")],
                    }
                },
                ["sparse"],
            )

        self.assertEqual(context.exception.status_code, 502)


class HybridEmbeddingEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_returns_dense_and_sparse_results(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "embeddings": {
                            "hybrid": {
                                "enabled": True,
                                "output_types": ["dense", "sparse"],
                                "max_batch_size": 4,
                                "max_sparse_top_k": 512,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            request = HybridEmbeddingsRequest(
                model="bge-m3",
                input=["first", "second"],
                sparse_top_k=64,
            )

            with (
                patch.object(gateway_app.registry, "get_backend", return_value="python"),
                patch.object(gateway_app.registry, "resolve", return_value=model_path),
                patch.object(
                    gateway_app,
                    "call_triton_hybrid_embeddings",
                    new=AsyncMock(
                        side_effect=[
                            (
                                {
                                    "dense": [0.1, 0.2],
                                    "sparse": {
                                        "indices": [10],
                                        "values": [0.8],
                                    },
                                },
                                2,
                            ),
                            (
                                {
                                    "dense": [0.3, 0.4],
                                    "sparse": {
                                        "indices": [20],
                                        "values": [0.7],
                                    },
                                },
                                3,
                            ),
                        ]
                    ),
                ),
            ):
                response = await gateway_app.create_hybrid_embeddings.__wrapped__(
                    request
                )

        self.assertEqual(response["object"], "hybrid_embedding.list")
        self.assertEqual(response["usage"]["prompt_tokens"], 5)
        self.assertEqual(response["data"][0]["embedding"], [0.1, 0.2])
        self.assertEqual(
            response["data"][1]["sparse_embedding"]["indices"],
            [20],
        )

    async def test_stock_vllm_backend_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "embeddings": {
                            "hybrid": {
                                "enabled": True,
                                "output_types": ["dense", "sparse"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            request = HybridEmbeddingsRequest(model="bge-m3", input="query")

            with (
                patch.object(gateway_app.registry, "get_backend", return_value="vllm"),
                patch.object(gateway_app.registry, "resolve", return_value=model_path),
            ):
                with self.assertRaises(HTTPException) as context:
                    await gateway_app.create_hybrid_embeddings.__wrapped__(request)

        self.assertEqual(context.exception.status_code, 400)


class HybridEmbeddingExampleTests(unittest.TestCase):
    def test_python_backend_example_is_valid_and_offline_only(self):
        root = Path(__file__).resolve().parents[1]
        model_path = root / "examples" / "bge-m3-hybrid" / "1" / "model.py"
        source = model_path.read_text(encoding="utf-8")

        compile(source, str(model_path), "exec")
        self.assertIn("local_files_only=True", source)
        self.assertIn("trust_remote_code=False", source)
        self.assertIn("sparse_linear.pt", source)

    def test_litellm_passthrough_uses_non_cohere_route(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "examples" / "litellm.config.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("path: /v1/hybrid_embeddings", source)
        self.assertIn(
            "target: http://triton-inference-server:8080/v1/hybrid_embeddings",
            source,
        )
        self.assertNotIn("/v1/embeddings/hybrid", source)


if __name__ == "__main__":
    unittest.main()
