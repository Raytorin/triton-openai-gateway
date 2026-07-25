# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from gateway.admission import AdmissionLease
import gateway.app as gateway_app
from gateway.rerank import build_rerank_documents, build_rerank_response
from gateway.rerank_strategies import resolve_rerank_strategy
from gateway.schemas import RerankRequest


class RerankStrategyTests(unittest.TestCase):
    def _model_path(
        self,
        directory: str,
        rerank_config: dict | None = None,
    ) -> Path:
        model_path = Path(directory)
        if rerank_config is not None:
            (model_path / "gateway.json").write_text(
                json.dumps({"rerank": rerank_config}),
                encoding="utf-8",
            )
        return model_path

    def _response(
        self,
        request: RerankRequest,
        model_path: Path,
        scores: list[float],
    ) -> dict:
        documents = build_rerank_documents(request)
        strategy = resolve_rerank_strategy(request, model_path)
        return build_rerank_response(request, documents, scores, strategy)

    def test_default_strategy_preserves_top_n_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second", "third"],
                top_n=2,
            )
            response = self._response(
                request,
                self._model_path(directory),
                [0.1, 0.9, 0.5],
            )

        self.assertEqual([item["index"] for item in response["results"]], [1, 2])
        self.assertEqual(response["meta"]["selection"]["strategy"], "top_n")
        self.assertEqual(response["meta"]["selection"]["source"], "builtin")

    def test_configured_threshold_strategy_accepts_allowed_top_n_override(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._model_path(
                directory,
                {
                    "default_strategy": "strict",
                    "strategies": {
                        "strict": {
                            "method": "top_n_and_threshold",
                            "parameters": {
                                "score_threshold": 0.5,
                                "top_n": 3,
                            },
                            "allow_request_parameters": ["top_n"],
                            "version": "2",
                        }
                    },
                },
            )
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second", "third"],
                selection={
                    "strategy": "strict",
                    "parameters": {"top_n": 1},
                },
            )
            response = self._response(request, model_path, [0.7, 0.9, 0.4])

        self.assertEqual([item["index"] for item in response["results"]], [1])
        self.assertEqual(response["meta"]["selection"]["method"], "top_n_and_threshold")
        self.assertEqual(response["meta"]["selection"]["version"], "2")

    def test_configured_strategy_rejects_disallowed_request_override(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._model_path(
                directory,
                {
                    "strategies": {
                        "strict": {
                            "method": "score_threshold",
                            "parameters": {"score_threshold": 0.5},
                        }
                    }
                },
            )
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first"],
                selection={
                    "strategy": "strict",
                    "parameters": {"score_threshold": 0.8},
                },
            )
            with self.assertRaises(HTTPException) as context:
                resolve_rerank_strategy(request, model_path)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("does not allow", str(context.exception.detail))

    def test_configured_strategy_can_require_allowed_request_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._model_path(
                directory,
                {
                    "strategies": {
                        "request-threshold": {
                            "method": "score_threshold",
                            "allow_request_parameters": ["score_threshold"],
                        }
                    }
                },
            )
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second"],
                selection={
                    "strategy": "request-threshold",
                    "parameters": {"score_threshold": 0.5},
                },
            )
            response = self._response(request, model_path, [0.2, 0.9])

        self.assertEqual([item["index"] for item in response["results"]], [1])

    def test_invalid_configured_strategy_is_reported_as_server_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = self._model_path(
                directory,
                {
                    "strategies": {
                        "missing-threshold": {
                            "method": "score_threshold",
                        }
                    }
                },
            )
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first"],
                selection="missing-threshold",
            )
            with self.assertRaises(HTTPException) as context:
                resolve_rerank_strategy(request, model_path)

        self.assertEqual(context.exception.status_code, 500)
        self.assertIn("must configure or allow", str(context.exception.detail))

    def test_builtin_score_threshold_allows_request_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second", "third"],
                selection={
                    "strategy": "score_threshold",
                    "parameters": {
                        "score_threshold": 0.6,
                        "top_n": 2,
                    },
                },
            )
            response = self._response(
                request,
                self._model_path(directory),
                [0.5, 0.9, 0.7],
            )

        self.assertEqual([item["index"] for item in response["results"]], [1, 2])

    def test_metadata_filter_uses_document_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=[
                    {"text": "Docker", "metadata": {"language": "en"}},
                    {"text": "Контейнеры", "metadata": {"language": "ru"}},
                    {"text": "Kubernetes", "metadata": {"language": "en"}},
                ],
                selection={
                    "strategy": "metadata_filter",
                    "parameters": {
                        "filters": {"language": "ru"},
                        "top_n": 2,
                    },
                },
            )
            response = self._response(
                request,
                self._model_path(directory),
                [0.99, 0.8, 0.7],
            )

        self.assertEqual([item["index"] for item in response["results"]], [1])

    def test_diversity_removes_near_duplicate_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=[
                    "Docker container platform",
                    "Docker container platform",
                    "PostgreSQL relational database",
                ],
                selection={
                    "strategy": "diversity",
                    "parameters": {
                        "max_similarity": 0.8,
                        "top_n": 3,
                    },
                },
            )
            response = self._response(
                request,
                self._model_path(directory),
                [0.9, 0.8, 0.7],
            )

        self.assertEqual([item["index"] for item in response["results"]], [0, 2])

    def test_selection_and_custom_top_cannot_be_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first"],
                selection="top_n",
                custom_top="score_threshold",
            )
            with self.assertRaises(HTTPException) as context:
                resolve_rerank_strategy(request, self._model_path(directory))

        self.assertEqual(context.exception.status_code, 400)

    def test_sqlite_strategy_source_is_loaded_without_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            database_path = model_path / "strategies.db"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE rerank_strategies (
                        model TEXT NOT NULL,
                        name TEXT NOT NULL,
                        method TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        allowed_request_parameters_json TEXT NOT NULL,
                        version TEXT NOT NULL,
                        enabled INTEGER NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO rerank_strategies VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "reranker",
                        "database-strict",
                        "score_threshold",
                        '{"score_threshold":0.75}',
                        '["top_n"]',
                        "db-1",
                        1,
                    ),
                )
            (model_path / "gateway.json").write_text(
                json.dumps(
                    {
                        "rerank": {
                            "database": {
                                "driver": "sqlite",
                                "path": "strategies.db",
                                "refresh_seconds": 0.01,
                                "required": True,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second"],
                selection="database-strict",
                top_n=1,
            )
            response = self._response(request, model_path, [0.8, 0.9])
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    UPDATE rerank_strategies
                    SET parameters_json = ?, version = ?
                    WHERE model = ? AND name = ?
                    """,
                    (
                        '{"score_threshold":0.95}',
                        "db-2",
                        "reranker",
                        "database-strict",
                    ),
                )
            time.sleep(0.02)
            updated_response = self._response(request, model_path, [0.8, 0.9])

        self.assertEqual([item["index"] for item in response["results"]], [1])
        self.assertEqual(response["meta"]["selection"]["source"], "database")
        self.assertEqual(response["meta"]["selection"]["version"], "db-1")
        self.assertEqual(
            [item["index"] for item in updated_response["results"]],
            [],
        )
        self.assertEqual(
            updated_response["meta"]["selection"]["version"],
            "db-2",
        )

    def test_unknown_strategy_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first"],
                selection="missing",
            )
            with self.assertRaises(HTTPException) as context:
                resolve_rerank_strategy(request, self._model_path(directory))

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("Available strategies", str(context.exception.detail))


class RerankEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_applies_strategy_after_backend_scoring(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            request = RerankRequest(
                model="reranker",
                query="containers",
                documents=["first", "second"],
                selection={
                    "strategy": "score_threshold",
                    "parameters": {"score_threshold": 0.5},
                },
            )
            with (
                patch.object(gateway_app.registry, "resolve", return_value=model_path),
                patch.object(gateway_app.registry, "validate_route"),
                patch.object(
                    gateway_app.registry,
                    "get_backend",
                    return_value="python",
                ),
                patch.object(
                    gateway_app.admission,
                    "acquire",
                    new=AsyncMock(return_value=AdmissionLease([])),
                ),
                patch.object(
                    gateway_app,
                    "call_triton_rerank",
                    new=AsyncMock(return_value=[0.2, 0.9]),
                ) as triton_call,
            ):
                response = await gateway_app.rerank(request)

        self.assertEqual([item["index"] for item in response["results"]], [1])
        triton_call.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
