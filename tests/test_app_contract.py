# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

import unittest

import httpx

from gateway.app import app


EXPECTED_OPERATIONS = {
    "/health": {"get"},
    "/ready": {"get"},
    "/metrics": {"get"},
    "/v1/models": {"get"},
    "/v1/embeddings": {"post"},
    "/v1/hybrid_embeddings": {"post"},
    "/rerank": {"post"},
    "/v1/rerank": {"post"},
    "/v2/rerank": {"post"},
    "/v1/chat/completions": {"post"},
}


class AppContractTests(unittest.IsolatedAsyncioTestCase):
    def test_openapi_contract_and_metadata(self):
        schema = app.openapi()

        self.assertEqual(schema["info"]["title"], "Triton OpenAI Gateway")
        self.assertEqual(schema["info"]["license"]["name"], "Apache-2.0")
        self.assertEqual(set(schema["paths"]), set(EXPECTED_OPERATIONS))
        for path, methods in EXPECTED_OPERATIONS.items():
            self.assertEqual(set(schema["paths"][path]), methods)

    async def test_health_metrics_and_validation_contract(self):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
        ) as client:
            health = await client.get("/health")
            metrics = await client.get("/metrics")
            invalid_chat = await client.post("/v1/chat/completions", json={})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(metrics.status_code, 200)
        self.assertIn(
            "triton_gateway_http_requests_total",
            metrics.text,
        )
        self.assertEqual(invalid_chat.status_code, 422)
        self.assertIsInstance(invalid_chat.json().get("detail"), list)
