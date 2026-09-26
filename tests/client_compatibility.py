# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP clients -> gateway ASGI -> deterministic generation fixture.

Run with the isolated requirements-client.txt environment:
python tests/client_compatibility.py --server-python /path/to/test-env/bin/python
No model, credentials, or remote inference required. Does not claim runtime proof.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serve():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from gateway.app import app
    from gateway.generation_types import GenerationResult, GenerationStream, GenerationEvent

    captured = []
    async def generate(request, **kwargs):
        if request.tools and not any(m.role == "tool" for m in request.messages):
            message = {"tool_calls": [{"id": "call_weather", "type": "function", "function": {
                "name": "weather", "arguments": '{"city":"Paris"}'}}]}
            finish = "tool_calls"
        else:
            message = {"content": '{"answer":"sunny"}' if request.response_format else "It is sunny"}
            finish = "stop"
        usage = {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
        headers = {"X-Output-Token-Limit": str(request.max_completion_tokens or 4096)}
        if not request.stream:
            return GenerationResult("chat-1", 123, request.model, message, finish, usage, headers=headers)
        async def events():
            delta = dict(message)
            if "tool_calls" in delta:
                delta["tool_calls"] = [dict(t, index=i) for i, t in enumerate(delta["tool_calls"])]
            yield GenerationEvent("chat-1", 123, request.model, delta)
            yield GenerationEvent("chat-1", 123, request.model, {}, finish, usage)
        return GenerationStream(events(), headers)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(captured).encode())
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            captured.append({"path": self.path, "body": body})
            with patch("gateway.responses.generation.generate", generate):
                response = TestClient(app).post(self.path, json=body)
            self.send_response(response.status_code)
            for k,v in response.headers.items():
                if k.lower() not in {"transfer-encoding", "connection"}:
                    self.send_header(k,v)
            self.end_headers()
            self.wfile.write(response.content)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    print(server.server_port, flush=True)
    server.serve_forever()


def check_clients(url):
    # Disable optional remote model-price/tokenizer lookups. Test traffic stays local.
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import httpx
    import openai
    import litellm
    from importlib.metadata import version
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    tool = {"type": "function", "name": "weather", "parameters": {
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}
    text = {"format": {"type": "json_schema", "name": "answer", "strict": True,
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                   "required": ["answer"], "additionalProperties": False}}}
    client = openai.OpenAI(base_url=url+"/v1", api_key="local-fixture", max_retries=0)
    def lite(**kwargs):
        return litellm.responses(model="openai/test", api_base=url+"/v1", api_key="local-fixture", **kwargs)
    for name, create in [("openai", lambda **kw: client.responses.create(model="test", **kw)), ("litellm", lite)]:
        response = create(input="Hello", store=False)
        assert response.output_text == "It is sunny", (name, response)
        events = list(create(input="Hello", stream=True, max_output_tokens=77, store=False))
        assert events[-1].type == "response.completed", (name, events)
        assert events[-1].response.output[0].content[0].text == "It is sunny"
        assert json.loads(create(input="JSON", text=text, store=False).output_text) == {"answer": "sunny"}
        history = [{"role": "user", "content": "Weather?"}]
        first = create(input=history, tools=[tool], store=False)
        call = first.output[0]
        assert call.type == "function_call" and call.name == "weather"
        history += [call.model_dump(exclude_none=True), {"type": "function_call_output", "call_id": call.call_id, "output": "sunny"}]
        second = create(input=history, tools=[tool], store=False)
        assert second.output_text == "It is sunny"
        tools_stream = list(create(input="Weather?", tools=[tool], stream=True, store=False))
        assert tools_stream[-1].response.output[0].arguments == '{"city":"Paris"}'
        schema_stream = list(create(input="JSON", text=text, stream=True, store=False))
        assert json.loads(schema_stream[-1].response.output[0].content[0].text) == {"answer": "sunny"}
        print(f"{name}: text, streaming, function roundtrip, tool stream, JSON Schema and schema stream passed")
    # Exercise the SDK event accumulator, not just event iteration.
    with client.responses.stream(model="test", input="Hello", store=False) as stream:
        response = stream.get_final_response()
        assert response.output_text == "It is sunny"
    captured = httpx.get(url+"/captured").json()
    assert all(item["path"] == "/v1/responses" for item in captured)
    assert "max_output_tokens" not in captured[0]["body"]
    assert "max_output_tokens" not in captured[7]["body"]
    assert captured[1]["body"]["max_output_tokens"] == 77
    assert captured[8]["body"]["max_output_tokens"] == 77
    print(json.dumps({"openai": version("openai"), "litellm": version("litellm"),
        "requests": len(captured), "route": "/v1/responses", "injected_default_limit": False}))
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--server-python", default=sys.executable)
    args = parser.parse_args()
    if args.serve:
        serve()
    else:
        process = subprocess.Popen([args.server_python, __file__, "--serve"], stdout=subprocess.PIPE, text=True, cwd=ROOT)
        try:
            port = int(process.stdout.readline().strip())
            check_clients(f"http://127.0.0.1:{port}")
        finally:
            process.terminate()
            process.wait(timeout=10)
