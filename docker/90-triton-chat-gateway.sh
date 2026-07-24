#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export TRITON_BASE_URL="${TRITON_BASE_URL:-http://127.0.0.1:8000}"
export MODELS_ACTIVE_DIR="${MODELS_ACTIVE_DIR:-/tmp/models-active}"
export GATEWAY_PORT="${GATEWAY_PORT:-8080}"
log_level="${LOG_LEVEL:-INFO}"
log_level="${log_level,,}"

uvicorn_args=(
    --app-dir /opt/tritonserver
    gateway.app:app
    --host 0.0.0.0
    --port "${GATEWAY_PORT}"
    --log-level "${log_level}"
)
if [[ "${UVICORN_ACCESS_LOG:-false}" != "true" ]]; then
    uvicorn_args+=(--no-access-log)
fi

uvicorn "${uvicorn_args[@]}" &
gateway_pid=$!
echo "$(date '+%F %T') started Triton chat gateway on port ${GATEWAY_PORT} (pid=${gateway_pid})"

# NVIDIA's entrypoint later execs Triton as PID 1, while Uvicorn remains its
# child. Stop Uvicorn when Triton begins shutdown so its keep-alive connection
# cannot hold Triton's HTTP service open until exit_timeout.
(
    while kill -0 "${gateway_pid}" 2>/dev/null; do
        if curl -fsS --max-time 1 "${TRITON_BASE_URL}/v2/health/live" >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    failures=0
    while kill -0 "${gateway_pid}" 2>/dev/null; do
        if curl -fsS --max-time 1 "${TRITON_BASE_URL}/v2/health/live" >/dev/null 2>&1; then
            failures=0
        else
            failures=$((failures + 1))
            if (( failures >= 3 )); then
                echo "$(date '+%F %T') Triton stopped responding; terminating gateway pid=${gateway_pid}"
                kill -TERM "${gateway_pid}" 2>/dev/null || true
                break
            fi
        fi
        sleep 1
    done
) &

return 0 2>/dev/null || exit 0
