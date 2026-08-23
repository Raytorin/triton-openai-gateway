# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import gateway.app as gateway_app
from gateway.rerank import (
    _read_rerank_execution_settings,
    iter_rerank_batches,
    load_rerank_execution_settings,
    plan_rerank_execution,
)
from gateway.schemas import RerankRequest


def rerank_request(document_count: int, **overrides) -> RerankRequest:
    payload = {
        "model": "reranker",
        "query": "query",
        "documents": [f"document-{index}" for index in range(document_count)],
        "return_documents": True,
    }
    payload.update(overrides)
    return RerankRequest.model_validate(payload)


class RerankExecutionPlanningTests(unittest.TestCase):
    def tearDown(self):
        _read_rerank_execution_settings.cache_clear()

    def test_default_plan_splits_fifteen_documents_into_safe_batches(self):
        request = rerank_request(15)
        documents = list(request.documents)

        plan = plan_rerank_execution(request, documents, Path("/missing"))
        batches = list(iter_rerank_batches(documents, plan))

        self.assertEqual(4, plan.batch_size)
        self.assertEqual(4, plan.batch_count)
        self.assertEqual([4, 4, 4, 3], [len(batch) for batch in batches])

    def test_longer_sequences_reduce_effective_batch_size(self):
        request = rerank_request(5, batch_size=8, max_length=4096)

        plan = plan_rerank_execution(request, list(request.documents), Path("/missing"))

        self.assertEqual(2, plan.batch_size)
        self.assertEqual(3, plan.batch_count)

    def test_oversized_client_batch_is_rejected(self):
        request = rerank_request(10, batch_size=16)

        with self.assertRaises(HTTPException) as context:
            plan_rerank_execution(request, list(request.documents), Path("/missing"))

        self.assertEqual(400, context.exception.status_code)

    def test_document_limit_is_rejected_before_triton(self):
        request = rerank_request(257)

        with self.assertRaises(HTTPException) as context:
            plan_rerank_execution(request, list(request.documents), Path("/missing"))

        self.assertEqual(413, context.exception.status_code)

    def test_model_gateway_json_overrides_execution_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            (model_path / "gateway.json").write_text(
                '{"rerank":{"execution":{'
                '"default_batch_size":2,"max_batch_size":3,'
                '"max_batch_tokens":1024}}}',
                encoding="utf-8",
            )
            request = rerank_request(5)
            plan = plan_rerank_execution(request, list(request.documents), model_path)

        self.assertEqual(2, plan.batch_size)
        self.assertEqual(3, plan.batch_count)

    def test_gateway_json_changes_are_reloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory)
            config_path = model_path / "gateway.json"
            config_path.write_text(
                '{"rerank":{"execution":{"default_batch_size":2}}}',
                encoding="utf-8",
            )
            first = load_rerank_execution_settings(model_path)
            previous_mtime = config_path.stat().st_mtime_ns

            config_path.write_text(
                '{"rerank":{"execution":{"default_batch_size":3}}}',
                encoding="utf-8",
            )
            updated_mtime = previous_mtime + 1_000_000
            os.utime(config_path, ns=(updated_mtime, updated_mtime))
            second = load_rerank_execution_settings(model_path)

        self.assertEqual(2, first.default_batch_size)
        self.assertEqual(3, second.default_batch_size)


class RerankEndpointBatchingTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        _read_rerank_execution_settings.cache_clear()

    async def test_endpoint_merges_scores_from_all_micro_batches(self):
        request = rerank_request(10)

        async def score_batch(_model, _query, documents, *_args):
            return [float(document.rsplit("-", 1)[1]) for document in documents]

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(gateway_app.registry, "resolve", return_value=Path(directory)),
                patch.object(gateway_app.registry, "get_backend", return_value="python"),
                patch.object(
                    gateway_app,
                    "call_triton_rerank",
                    new=AsyncMock(side_effect=score_batch),
                ) as infer,
            ):
                response = await gateway_app.rerank.__wrapped__(request)

        self.assertEqual(3, infer.await_count)
        self.assertEqual(10, len(response["results"]))
        self.assertEqual(9, response["results"][0]["index"])
        self.assertEqual(4, response["meta"]["execution"]["batch_size"])
        self.assertEqual(3, response["meta"]["execution"]["batch_count"])


if __name__ == "__main__":
    unittest.main()
