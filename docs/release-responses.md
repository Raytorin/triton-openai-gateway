# Responses first release — unreleased

**Language:** English | [Русский](release-responses.ru.md)

Adds stateless POST /v1/responses and /responses: text, bounded image inputs,
client functions, JSON/JSON Schema and Responses SSE. Generation, LoRA,
reasoning policy and admission are shared with Chat.

## Migration

Automatic Chat output defaults change from 256 to 4096, or 8192 with thinking.
Set GATEWAY_DEFAULT_OUTPUT_TOKENS and GATEWAY_REASONING_DEFAULT_OUTPUT_TOKENS to
256 to retain the old fallback. The default server cap is 32768; model caps may
reduce it. Explicit output limits are never silently clamped. Chat aliases must
agree and must be positive integers. Missing/null budgets preserve fitting
history and shrink to remaining context. Runtime context must be supplied in
model.json max_model_len or gateway.json generation.context_window.

Responses defaults to store=false, including when omitted. Persistence,
background jobs, server-side conversation continuation, generated audio/images,
compact and legacy completions are not included. See the
[compatibility table](responses.md) before migrating clients.

## Validation and rollout

Local CI includes regression tests and isolated, pinned OpenAI SDK/LiteLLM HTTP
fixtures. These verify the actual /v1/responses route and forwarded output limit,
but do not prove compatibility with a deployed proxy or real GPU runtime.
Runtime smoke is a separate release gate requiring explicit authorization for
DevZone targets. Test disconnect and backend cancellation on every supported
transport used in production. Stateful features remain a subsequent release;
no SQLite or identity proxy contract is enabled by this change.

## Conditions for user rollout

The first release operates without a database; persistence and related APIs are
deferred. Before exposing the route to users:

- Configure the actual context window for every published model and verify its
  tools/vision/JSON Schema on the chosen runtime. The example gateway.json does
  not establish capabilities for an arbitrary model.
- Check both the direct route and the deployed LiteLLM proxy, including its
  defaults and parameter mapping. Both Responses routes must pass through the
  existing proxy authentication and quotas.
- Check proxy timeouts and SSE buffering, `incomplete` termination, errors after
  streaming starts, and admission release on disconnect. HTTP 200 for SSE does
  not by itself mean generation completed successfully.
- Measure queues, latency and GPU use with defaults 4096/8192 under an agreed
  workload. `max_output_tokens` is an upper bound, not a promised response length.
  Set lower per-model defaults/caps where needed. Admission is process-local;
  adding replicas does not establish a shared load limit for one Triton server.
- Build and check the release container, record its digest and the previous
  working image for rollback. Local Python CI does not exercise the full NVIDIA
  base runtime. When restoring defaults to 256, account for gateway.json
  overrides; global environment variables do not override per-model settings.

The temporary Accelerate audit exception expires at 2026-10-10 00:00 UTC and does
not fix the vulnerability; see [SECURITY.md](../SECURITY.md) for its scope.
Live runtime checks and any DevZone changes require separate authorization for
the specific target and scope.
