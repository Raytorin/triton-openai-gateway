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
