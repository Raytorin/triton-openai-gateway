#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export TRITON_BASE_URL="${TRITON_BASE_URL:-http://127.0.0.1:8000}"
gateway_model_root="${WATCHER_MODEL_DIR:-${TMP_ROOT:-${TMPDIR:-/tmp}}}"
export MODELS_ACTIVE_DIR="${MODELS_ACTIVE_DIR:-${gateway_model_root%/}/models-active}"
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

# This hook is sourced before the NVIDIA entrypoint execs Triton. Keep the
# supervisor in a child shell so it does not block startup or replace PID 1.
gateway_supervise() (
    gateway_pid=""
    interval="${GATEWAY_HEALTH_INTERVAL_SECONDS:-1}"
    timeout="${GATEWAY_HEALTH_TIMEOUT_SECONDS:-1}"
    threshold="${GATEWAY_HEALTH_FAILURE_THRESHOLD:-3}"
    startup_grace="${GATEWAY_STARTUP_GRACE_SECONDS:-60}"
    stop_grace="${GATEWAY_STOP_GRACE_SECONDS:-10}"
    restart_delay="${GATEWAY_RESTART_DELAY_SECONDS:-2}"
    for value in "${interval}" "${timeout}" "${threshold}" "${stop_grace}" "${restart_delay}"; do
        if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
            echo "gateway watchdog intervals and thresholds must be positive integers" >&2
            exit 1
        fi
    done
    if [[ ! "${startup_grace}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "GATEWAY_STARTUP_GRACE_SECONDS must be a nonnegative integer" >&2
        exit 1
    fi

    stop_gateway() {
        local deadline
        [[ -n "${gateway_pid}" ]] || return 0
        if kill -0 "${gateway_pid}" 2>/dev/null; then
            kill -TERM "${gateway_pid}" 2>/dev/null || true
            deadline=$((SECONDS + stop_grace))
            while kill -0 "${gateway_pid}" 2>/dev/null && (( SECONDS < deadline )); do
                sleep 1
            done
            if kill -0 "${gateway_pid}" 2>/dev/null; then
                echo "$(date '+%F %T') gateway did not stop; killing pid=${gateway_pid}"
                kill -KILL "${gateway_pid}" 2>/dev/null || true
            fi
        fi
        wait "${gateway_pid}" 2>/dev/null || true
        gateway_pid=""
    }
    trap stop_gateway EXIT
    trap 'exit 0' TERM INT

    start_gateway() {
        uvicorn "${uvicorn_args[@]}" &
        gateway_pid=$!
        started_at=${SECONDS}
        gateway_seen=false
        gateway_failures=0
        echo "$(date '+%F %T') started Triton chat gateway on port ${GATEWAY_PORT} (pid=${gateway_pid})"
    }

    triton_seen=false
    triton_failures=0
    start_gateway
    while true; do
        if curl -fsS --max-time "${timeout}" "${TRITON_BASE_URL}/v2/health/live" >/dev/null 2>&1; then
            triton_seen=true
            triton_failures=0
        elif [[ "${triton_seen}" == true ]]; then
            triton_failures=$((triton_failures + 1))
            if (( triton_failures >= threshold )); then
                echo "$(date '+%F %T') Triton stopped responding; stopping gateway supervisor"
                exit 0
            fi
            # Do not restart Uvicorn while Triton might be shutting down.
            sleep "${interval}"
            continue
        fi

        restart_reason=""
        if ! kill -0 "${gateway_pid}" 2>/dev/null; then
            restart_reason="process exited"
        elif curl -fsS --max-time "${timeout}" "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
            gateway_seen=true
            gateway_failures=0
        elif [[ "${gateway_seen}" == true ]] || (( SECONDS - started_at >= startup_grace )); then
            gateway_failures=$((gateway_failures + 1))
            if (( gateway_failures >= threshold )); then
                restart_reason="health check failed ${gateway_failures} times"
            fi
        fi

        if [[ -n "${restart_reason}" ]]; then
            echo "$(date '+%F %T') restarting gateway: ${restart_reason} (pid=${gateway_pid})"
            stop_gateway
            sleep "${restart_delay}"
            # The next loop checks Triton before starting another process.
            if [[ "${triton_seen}" == true ]] && ! curl -fsS --max-time "${timeout}" "${TRITON_BASE_URL}/v2/health/live" >/dev/null 2>&1; then
                echo "$(date '+%F %T') Triton stopped during gateway restart; exiting supervisor"
                exit 0
            fi
            start_gateway
        fi
        sleep "${interval}"
    done
)

gateway_supervise &
gateway_supervisor_pid=$!
echo "$(date '+%F %T') started gateway supervisor (pid=${gateway_supervisor_pid})"

return 0 2>/dev/null || exit 0
