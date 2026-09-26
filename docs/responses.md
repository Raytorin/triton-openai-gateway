# Responses API

**Language:** English | [Русский](responses.ru.md)

`POST /v1/responses` and `POST /responses` provide stateless generation with
JSON or Responses SSE. This is a supported subset, not full OpenAI parity.
Chat, Responses, LoRA selection and admission use one generation service.

## Supported contract

| Parameter / feature | Behavior |
| --- | --- |
| `model`, string `input` | Required; uses the configured model chat template |
| Message list `input`, `instructions` | Roles: user, assistant, system, developer; developer becomes system |
| `input_text`, `output_text` | Text parts, including replay of returned assistant messages |
| `input_image.image_url` | HTTP(S) or image data URL; user messages only |
| `tools` | Client-executed `function` definitions: name, description, parameters, strict=false/null |
| `tool_choice` | auto, none, required, or `{type: function, name: ...}` |
| `function_call`, `function_call_output` | Replay calls and string outputs with matching `call_id`; duplicate/unmatched calls rejected |
| `text.format` | text, json_object, or json_schema with name/schema/strict; requires backend support for constrained output |
| `temperature`, `top_p` | Passed to the existing sampling layer |
| `max_output_tokens` | Positive integer or null; shared budget with Chat |
| `lora_name` | Gateway extension; same routing as Chat |
| `metadata` | Echoed, never persisted |
| `truncation` | disabled (default) or auto; no hidden summarization |
| `stream` | Native Responses events, not Chat chunks |
| `store`, `background` | Omitted/null/false only; no persistence or background execution |
| `reasoning` | Empty/null only; model configuration controls thinking |
| `parallel_tool_calls` | Omitted/null/true; false is rejected because the backend cannot enforce serial calls |

Unsupported fields return 400 before inference: built-in tools, strict function
argument enforcement, file_id, input audio/video/PDF, text verbosity, nonempty
reasoning settings, previous_response_id, conversation, store=true, background=true.
Use Chat for audio/video/PDF. Retrieval/deletion/cancellation by response ID,
Conversations, compact, generated audio/images and legacy `/v1/completions`
are outside this release. Functions execute in the client, never in the gateway.

**Deliberate default difference:** omitted `store` means **false**. Every turn
must contain its own history; IDs do not identify retrievable server objects.
Include each returned function call and its output in the next input. Calls and
outputs must precede the next message. Instructions are supplied by the caller.

## Token and context policy

Chat uses `max_completion_tokens` or its legacy alias `max_tokens`; Responses
uses `max_output_tokens`. Equal Chat aliases are allowed; different values return
400. Booleans, strings, fractional/zero/negative limits are rejected. Missing/null
selects an automatic budget: min(profile default, server/model cap, free context).
Defaults are 4096 normally and 8192 with thinking; the configurable global cap
starts at 32768. This cap is an operational policy, not a model specification.

Free context uses the configured runtime window minus the rendered instructions,
tools, history, additional image reserve and safety margin (64 by default).
Media reserve subtracts placeholders already counted in the prompt. Image reserve
is a conservative estimate from the configured pixel bound, not exact processor
usage. Remaining media is recalculated after truncation. Images use the existing
bounded downloader and decoder, are resized within `image_max_pixels`, and
normalized to JPEG. `detail` is accepted; the pixel bound governs preprocessing.
Vision is detected from local `config.json` vision settings; for other verified
vision runtimes set `generation.supports_vision=true` in `gateway.json`.
`supports_vision=false` rejects images before inference.

Fitting history is preserved, even when less than the default output budget
remains. Explicit limits are never silently changed. With `truncation=auto`, only
oldest complete turns are removed; instructions, latest user turn and associated
function calls/results remain. With disabled truncation, overflow is an error.
Thinking, text and function arguments consume the same budget. A length finish
produces `status=incomplete` and `incomplete_details.reason=max_output_tokens`,
even if thinking exhausted the budget before any visible text. No automatic retry.

The current Triton text transports expose text without authoritative token usage
or finish metadata; usage and limit detection are estimated by retokenization.
`X-Token-Usage-Source: estimated` makes this explicit. `X-Output-Token-Limit` and
`X-Output-Token-Limit-Source` report the selected budget. Raw thinking is never
presented as an OpenAI reasoning summary. See [configuration](configuration.md)
for defaults, caps and mandatory runtime context configuration. A complete
[model configuration example](../examples/responses/gateway.json) is provided;
adjust its context window to the real runtime.

## Python example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="configured-by-proxy")
response = client.responses.create(
    model="your-model", input="Say hello", store=False, max_output_tokens=128,
)
print(response.output_text)

with client.responses.stream(model="your-model", input="Say hello", store=False) as stream:
    for event in stream:
        if event.type == "response.output_text.delta":
            print(event.delta, end="")
    final = stream.get_final_response()
```

For tools, use flat Responses definitions (`{"type":"function","name":"..."}`),
not the nested Chat shape. For multiple turns append `response.output` as dicts
and the matching `function_call_output` items to your original input history.

## Streaming and errors

Events include response.created/in_progress; output_item.added/done;
content_part.added/done; output_text.delta/done; function_call_arguments.delta/done;
and exactly one terminal response.completed/incomplete/failed. All carry increasing
`sequence_number`; item IDs, call IDs and output indices stay consistent.
The stream ends after the terminal event, without Chat's `[DONE]` sentinel.
Some existing Python and tool-aware Triton paths buffer generation before
emitting deltas. Responses does not claim incremental tool decoding on those paths.

Errors before headers use an HTTP error envelope. Errors after headers emit
`error`, then `response.failed`. Disconnect closes the backend iterator and
releases admission, including during request preparation. gRPC cancellation is
propagated; actual GPU cancellation on a particular backend still needs a runtime
smoke check. HTTP transport closure cannot guarantee immediate GPU abort.

Request IDs use existing middleware. Prometheus labels include the two fixed
routes, never individual response or item IDs.

## Compatibility and release validation

`requirements-client.txt` pins OpenAI SDK 2.54.0 and LiteLLM 1.102.1. The local
HTTP fixture verifies text/output_text, SDK stream accumulation, tool round trips,
streamed tools and JSON Schema for both clients. It captures actual forwarded
paths and limits. With LiteLLM `model="openai/test"` and `api_base=.../v1`, requests
reach `/v1/responses`; this tested SDK configuration adds no default output limit.
A deployed LiteLLM proxy may have different versions, mappings or defaults.

```bash
python -m venv .local/client-venv
.local/client-venv/bin/python -m pip install -r requirements-client.txt
.local/client-venv/bin/python tests/client_compatibility.py --server-python .venv/bin/python
```

The fixture does not validate a live Triton GPU or deployed LiteLLM proxy.
Before release, run an approved runtime smoke for text, images, functions,
JSON Schema, long/thinking output and client disconnect. Check forwarded route,
max_output_tokens, measured backend cancellation and released admission slots.
No DevZone inference or deployment is part of the local CI suite.
