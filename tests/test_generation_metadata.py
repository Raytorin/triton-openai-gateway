"""Engine completion facts take precedence over decoded-text heuristics."""
import json

import pytest
from fastapi import HTTPException

from gateway.generation_metadata import (
    GenerationMetadata, GenerationOutput, GenerationMetadataState,
    decode_generation_metadata, metadata_of, resolve_finish_reason, apply_engine_usage,
)


def envelope(**output):
    return {"version": 1, "finished": True, "prompt_tokens": 42,
            "outputs": [{"index": 0, "finish_reason": "stop", "stop_reason": 248046,
                         "completion_tokens": 1, "first_token_id": 248046,
                         "last_token_id": 248046, **output}]}


def test_empty_eos_retains_engine_count_and_stop():
    metadata = decode_generation_metadata(json.dumps(envelope()))
    result = GenerationOutput("", metadata)
    assert result == "" and isinstance(result, str)
    assert metadata_of(result).completion_tokens == 1
    assert resolve_finish_reason(metadata, token_limit_reached=True) == "stop"
    assert apply_engine_usage({"prompt_tokens": 40, "completion_tokens": 0, "total_tokens": 40}, metadata) == {
        "prompt_tokens": 42, "completion_tokens": 1, "total_tokens": 43}


@pytest.mark.parametrize("reason,expected", [("stop", "tool_calls"), ("length", "length")])
def test_engine_reason_controls_tool_completion(reason, expected):
    metadata = decode_generation_metadata(envelope(finish_reason=reason))
    assert resolve_finish_reason(metadata, tool_calls=True) == expected


@pytest.mark.parametrize("reason", ["abort", "error", "unsupported", None])
def test_unsuccessful_engine_completion_is_not_success(reason):
    with pytest.raises(HTTPException) as error:
        resolve_finish_reason(decode_generation_metadata(envelope(finish_reason=reason)))
    assert error.value.status_code == 502


def test_legacy_empty_result_does_not_infer_length():
    assert resolve_finish_reason(None, token_limit_reached=False) == "stop"
    assert resolve_finish_reason(None, token_limit_reached=True) == "length"
    assert metadata_of("legacy") is None


def test_stream_counts_are_cumulative_and_empty_final_is_retained():
    state = GenerationMetadataState()
    first = envelope(completion_tokens=2, finish_reason=None, last_token_id=8)
    first["finished"] = False
    state.update_json(first)
    state.update_json(envelope(completion_tokens=3, last_token_id=9))
    assert state.metadata.completion_tokens == 3
    assert state.metadata.finished
    assert resolve_finish_reason(state.metadata) == "stop"


def test_unfinished_metadata_requires_terminal_event_or_explicit_local_stop():
    metadata = GenerationMetadata(finished=False, completion_tokens=1)
    with pytest.raises(HTTPException):
        resolve_finish_reason(metadata)
    assert resolve_finish_reason(metadata, stopped_locally=True) == "stop"


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"finished": "false"},
    {"prompt_tokens": -1}, {"prompt_tokens": True}, {"outputs": []},
    {"outputs": [{"index": 0, "completion_tokens": -1}]},
    {"outputs": [{"index": 0, "completion_tokens": 1, "stop_reason": True}]},
])
def test_invalid_metadata_is_an_upstream_error(change):
    with pytest.raises(HTTPException) as error:
        decode_generation_metadata({**envelope(), **change})
    assert error.value.status_code == 502


def test_decreasing_cumulative_count_is_rejected():
    state = GenerationMetadataState()
    first = envelope(completion_tokens=3, finish_reason=None)
    first["finished"] = False
    state.update_json(first)
    with pytest.raises(HTTPException):
        state.update_json(envelope(completion_tokens=2))


@pytest.mark.parametrize("reason", ["length", "abort"])
def test_terminal_engine_reason_wins_over_a_local_stop(reason):
    metadata = decode_generation_metadata(envelope(finish_reason=reason))
    if reason == "abort":
        with pytest.raises(HTTPException):
            resolve_finish_reason(metadata, stopped_locally=True)
    else:
        assert resolve_finish_reason(metadata, stopped_locally=True) == "length"


def test_empty_diagnostics_do_not_log_stop_string_or_content():
    import logging
    from gateway.generation_metadata import log_completion
    records = []
    class Capture(logging.Handler):
        def emit(self, record):records.append(record)
    logger = logging.getLogger("triton-chat-gateway")
    handler = Capture()
    logger.addHandler(handler)
    try:
        metadata = GenerationMetadata(finished=True, finish_reason="stop", stop_reason="private-stop-value", prompt_tokens=42, completion_tokens=1, first_token_id=8, last_token_id=8)
        log_completion("model", "", metadata)
        log_completion("model", "private-answer", metadata)
    finally:
        logger.removeHandler(handler)
    assert len(records) == 1
    assert records[0].event == "generation.empty_output"
    assert records[0].completion_tokens == 1
    assert records[0].first_token_id == 8
    assert "private-stop-value" not in str(records[0].__dict__)
    assert "private-answer" not in str(records[0].__dict__)


@pytest.mark.parametrize("missing", [True, False])
def test_terminal_packet_must_supply_its_own_cumulative_count(missing):
    state = GenerationMetadataState()
    first = envelope(completion_tokens=2, finish_reason=None)
    first["finished"] = False
    state.update_json(first)
    terminal = envelope(completion_tokens=None)
    if missing:
        del terminal["outputs"][0]["completion_tokens"]
    with pytest.raises(HTTPException) as error:
        state.update_json(terminal)
    assert error.value.status_code == 502
    assert not state.metadata.finished
    assert state.metadata.completion_tokens == 2
