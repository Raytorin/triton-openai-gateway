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
    gateway_start=""
    supervisor_pid=${BASHPID}
    run_dir="${GATEWAY_RUN_DIR:-/tmp/triton-gateway-${UID}}"
    umask 077
    mkdir -m 700 -- "${run_dir}" 2>/dev/null || true
    if [[ ! -d "${run_dir}" || -L "${run_dir}" || ! -O "${run_dir}" ]] || [[ "$(stat -c %a -- "${run_dir}")" != 700 ]]; then
        echo "unsafe gateway supervisor runtime directory: expected an owned directory with mode 0700" >&2
        exit 1
    fi
    # Never unlink the lock file: another hook must lock the same inode.
    if ! command -v flock >/dev/null; then
        echo "gateway supervisor requires flock" >&2
        exit 1
    fi
    exec 9>"${run_dir}/supervisor.lock"
    if ! flock -n 9; then
        echo "gateway supervisor is already running"
        exit 0
    fi
    exec 8>"${run_dir}/control.lock"
    generation=0
    phase=starting
    completed_request=""
    request_id=""
    request_base=0
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

    process_start() {
        local pid=$1 record
        local -a fields
        [[ -r "/proc/${pid}/stat" ]] || return 1
        record=$(<"/proc/${pid}/stat")
        read -r -a fields <<< "${record##*) }"
        [[ "${fields[0]:-Z}" != Z && "${fields[0]:-X}" != X ]] || return 1
        printf '%s' "${fields[19]:-}"
    }
    supervisor_start=$(process_start "${supervisor_pid}")

    publish_state() {
        {
            printf 'supervisor_pid=%s\nsupervisor_start=%s\n' "${supervisor_pid}" "${supervisor_start}"
            printf 'gateway_pid=%s\ngateway_start=%s\ngeneration=%s\n' "${gateway_pid:-0}" "${gateway_start}" "${generation}"
            printf 'phase=%s\ncompleted_request=%s\n' "${phase}" "${completed_request}"
            printf 'gateway_port=%s\ntriton_url=%s\n' "${GATEWAY_PORT}" "${TRITON_BASE_URL%/}"
        } >"${run_dir}/.state.${supervisor_pid}"
        mv -f -- "${run_dir}/.state.${supervisor_pid}" "${run_dir}/state"
    }

    gateway_alive() {
        [[ -n "${gateway_pid}" && -n "${gateway_start}" ]] &&
            [[ "$(process_start "${gateway_pid}" || true)" == "${gateway_start}" ]]
    }

    check_request() {
        local started base token extra
        flock 8
        if [[ -f "${run_dir}/restart-request" ]]; then
            read -r started base token extra <"${run_dir}/restart-request" || true
            if [[ "${started}" == "${supervisor_start}" && "${base}" =~ ^[1-9][0-9]*$ && "${token}" =~ ^[a-f0-9]{32}$ && -z "${extra}" ]] && (( base <= generation )); then
                request_id=${token}
                request_base=${base}
            else
                rm -f -- "${run_dir}/restart-request"
            fi
        fi
        if [[ -n "${request_id}" && "${phase}" == running ]] && (( generation > request_base )); then
            completed_request=${request_id}
            publish_state
            rm -f -- "${run_dir}/restart-request"
            request_id=""
        fi
        flock -u 8
    }

    stop_gateway() {
        local deadline
        [[ -n "${gateway_pid}" ]] || return 0
        if gateway_alive; then
            kill -TERM "${gateway_pid}" 2>/dev/null || true
            deadline=$((SECONDS + stop_grace))
            while gateway_alive && (( SECONDS < deadline )); do
                sleep 1
            done
            if gateway_alive; then
                echo "$(date '+%F %T') gateway did not stop; killing pid=${gateway_pid}"
                kill -KILL "${gateway_pid}" 2>/dev/null || true
            fi
        fi
        wait "${gateway_pid}" 2>/dev/null || true
        gateway_pid=""
        gateway_start=""
    }
    cleanup() {
        trap '' TERM INT
        phase=stopping
        publish_state || true
        stop_gateway
        phase=stopped
        publish_state || true
    }
    trap cleanup EXIT
    trap 'exit 0' TERM INT

    start_gateway() {
        # The child must not retain either supervisor/control lock after our exit.
        uvicorn "${uvicorn_args[@]}" 8>&- 9>&- &
        gateway_pid=$!
        gateway_start=$(process_start "${gateway_pid}" || true)
        generation=$((generation + 1))
        phase=starting
        started_at=${SECONDS}
        gateway_seen=false
        gateway_failures=0
        publish_state
        echo "$(date '+%F %T') started Triton chat gateway on port ${GATEWAY_PORT} (pid=${gateway_pid})"
    }

    triton_seen=false
    triton_failures=0
    flock 8
    rm -f -- "${run_dir}/restart-request"
    flock -u 8
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

        check_request
        restart_reason=""
        if [[ -n "${request_id}" ]] && (( generation <= request_base )); then
            restart_reason="operator request"
        elif ! gateway_alive; then
            restart_reason="process exited"
        elif curl -fsS --max-time "${timeout}" "http://127.0.0.1:${GATEWAY_PORT}/health" >/dev/null 2>&1; then
            gateway_seen=true
            gateway_failures=0
            phase=running
            publish_state
            check_request
        elif [[ "${gateway_seen}" == true ]] || (( SECONDS - started_at >= startup_grace )); then
            gateway_failures=$((gateway_failures + 1))
            if (( gateway_failures >= threshold )); then
                restart_reason="health check failed ${gateway_failures} times"
            fi
        fi

        if [[ -n "${restart_reason}" ]]; then
            phase=restarting
            publish_state
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
