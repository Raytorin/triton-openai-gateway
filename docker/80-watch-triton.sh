#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WATCH_TRITON_BIN="${WATCH_TRITON_BIN:-/usr/local/bin/watch_triton.sh}"

fail_startup() {
    echo "$(date '+%F %T') watcher startup failed: $*" >&2
    return 1
}

is_remote_path() {
    [[ "$1" == *"://"* ]]
}

prepare_directory() {
    local path="$1"
    local label="$2"

    if is_remote_path "${path}"; then
        fail_startup "${label} must be a local filesystem path"
        return 1
    fi
    if [[ -e "${path}" && ! -d "${path}" ]]; then
        fail_startup "${label} is not a directory: ${path}"
        return 1
    fi
    if [[ ! -d "${path}" ]]; then
        if ! mkdir -p -- "${path}"; then
            fail_startup "cannot create ${label}: ${path}"
            return 1
        fi
        echo "$(date '+%F %T') created ${label}: ${path}"
    else
        echo "$(date '+%F %T') using existing ${label}: ${path}"
    fi
    if [[ ! -r "${path}" || ! -w "${path}" || ! -x "${path}" ]]; then
        fail_startup "${label} must be readable, writable and searchable: ${path}"
        return 1
    fi
}

resolve_watcher_directories() {
    local model_dir=""
    local model_dir_source=""
    local active_dir=""
    local active_dir_source=""

    if [[ -v WATCHER_MODEL_DIR ]]; then
        if [[ -z "${WATCHER_MODEL_DIR}" ]]; then
            fail_startup "WATCHER_MODEL_DIR is set but empty"
            return 1
        fi
        model_dir="${WATCHER_MODEL_DIR}"
        model_dir_source="WATCHER_MODEL_DIR"
    elif [[ -v TMP_ROOT ]]; then
        if [[ -z "${TMP_ROOT}" ]]; then
            fail_startup "TMP_ROOT is set but empty"
            return 1
        fi
        model_dir="${TMP_ROOT}"
        model_dir_source="TMP_ROOT"
    elif [[ -n "${TMPDIR:-}" ]]; then
        model_dir="${TMPDIR}"
        model_dir_source="TMPDIR"
    else
        model_dir="/tmp"
        model_dir_source="default"
    fi

    if [[ -v MODELS_ACTIVE_DIR ]]; then
        if [[ -z "${MODELS_ACTIVE_DIR}" ]]; then
            fail_startup "MODELS_ACTIVE_DIR is set but empty"
            return 1
        fi
        active_dir="${MODELS_ACTIVE_DIR}"
        active_dir_source="MODELS_ACTIVE_DIR"
    else
        active_dir="${model_dir%/}/models-active"
        active_dir_source="derived"
    fi

    if [[ "${active_dir}" == "${model_dir}" ]]; then
        fail_startup "MODELS_ACTIVE_DIR must differ from watcher model directory"
        return 1
    fi

    export WATCHER_MODEL_DIR="${model_dir}"
    export WATCHER_MODEL_DIR_SOURCE="${model_dir_source}"
    export MODELS_ACTIVE_DIR="${active_dir}"
    export MODELS_ACTIVE_DIR_SOURCE="${active_dir_source}"

    # Compatibility with release 5 watcher scripts.
    export TMP_ROOT="${model_dir}"

    # Keep NVIDIA frontend and the custom gateway on the same active registry.
    export TRITON_FRONTEND_LOCAL_MODEL_REPOSITORY="${active_dir}"
}

if ! resolve_watcher_directories; then
    return 1 2>/dev/null || exit 1
fi

if ! prepare_directory "${WATCHER_MODEL_DIR}" "watcher model directory"; then
    return 1 2>/dev/null || exit 1
fi
if ! prepare_directory "${MODELS_ACTIVE_DIR}" "models-active directory"; then
    return 1 2>/dev/null || exit 1
fi

echo "$(date '+%F %T') selected watcher model directory ${WATCHER_MODEL_DIR} (source=${WATCHER_MODEL_DIR_SOURCE})"
echo "$(date '+%F %T') selected models-active directory ${MODELS_ACTIVE_DIR} (source=${MODELS_ACTIVE_DIR_SOURCE})"

if [[ ! -x "${WATCH_TRITON_BIN}" ]]; then
    fail_startup "watcher is not executable: ${WATCH_TRITON_BIN}"
    return 1 2>/dev/null || exit 1
fi

"${WATCH_TRITON_BIN}" &
echo "$(date '+%F %T') started ${WATCH_TRITON_BIN} in background (pid=$!)"

return 0 2>/dev/null || exit 0
