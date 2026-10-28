"""Keep successful probes quiet while retaining failures and HTTP metrics."""
import asyncio
import logging

import pytest
from prometheus_client import REGISTRY

from gateway.observability import RequestContextMiddleware


@pytest.fixture
def gateway_logs(caplog):
    logger = logging.getLogger("triton-chat-gateway")
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


async def request(path, status=200, method="GET", error=None):
    async def app(scope, receive, send):
        if error is not None:
            raise error
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    sent = []

    async def send(message):
        sent.append(message)

    await RequestContextMiddleware(app)(
        {"type": "http", "method": method, "path": path,
         "headers": [(b"x-request-id", b"probe-request")]},
        receive, send,
    )
    return sent


@pytest.mark.parametrize("path", ["/health", "/ready"])
@pytest.mark.parametrize("method,status", [("GET", 200), ("GET", 204), ("HEAD", 200)])
def test_successful_probe_is_quiet_at_info_and_keeps_metrics_and_request_id(
    gateway_logs, path, method, status,
):
    labels = {"method": method, "path": path, "status": str(status)}
    before = REGISTRY.get_sample_value("triton_gateway_http_requests_total", labels) or 0
    sent = asyncio.run(request(path, status, method))
    assert gateway_logs.records == []
    assert sent[0]["status"] == status
    assert dict(sent[0]["headers"])[b"x-request-id"] == b"probe-request"
    after = REGISTRY.get_sample_value("triton_gateway_http_requests_total", labels)
    assert after == before + 1
    assert REGISTRY.get_sample_value(
        "triton_gateway_http_requests_inflight", {"method": method, "path": path},
    ) == 0


@pytest.mark.parametrize("path", ["/health", "/ready"])
@pytest.mark.parametrize("status,level", [(307, logging.INFO), (405, logging.WARNING), (503, logging.ERROR)])
def test_probe_redirects_and_errors_remain_visible(gateway_logs, path, status, level):
    asyncio.run(request(path, status))
    completed = [record for record in gateway_logs.records
                 if record.event == "http.request.completed"]
    assert len(completed) == 1
    assert completed[0].levelno == level
    assert completed[0].status_code == status
    assert completed[0].request_id == "probe-request"


@pytest.mark.parametrize("method,path", [("GET", "/v1/models"), ("GET", "/health/custom"), ("POST", "/health")])
def test_other_successful_requests_keep_info_logs(gateway_logs, method, path):
    asyncio.run(request(path, method=method))
    completed = [record for record in gateway_logs.records
                 if record.event == "http.request.completed"]
    assert len(completed) == 1
    assert completed[0].levelno == logging.INFO


def test_probe_exception_keeps_error_details(gateway_logs):
    with pytest.raises(RuntimeError, match="probe failed"):
        asyncio.run(request("/ready", error=RuntimeError("probe failed")))
    errors = [record for record in gateway_logs.records if record.levelno == logging.ERROR]
    failed = next(record for record in errors if record.event == "http.request.failed")
    assert failed.error_type == "RuntimeError"
    assert failed.exc_info[0] is RuntimeError
    assert any(record.event == "http.request.completed" and record.status_code == 500
               for record in errors)


def test_successful_probes_are_available_at_debug(gateway_logs):
    logging.getLogger("triton-chat-gateway").setLevel(logging.DEBUG)
    asyncio.run(request("/health"))
    completed = [record for record in gateway_logs.records
                 if record.event == "http.request.completed"]
    assert len(completed) == 1
    assert completed[0].levelno == logging.DEBUG
