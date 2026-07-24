#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export TMP_ROOT="${TMP_ROOT:-/tmp}"
WATCH_TRITON_BIN="${WATCH_TRITON_BIN:-/usr/local/bin/watch_triton.sh}"

fail_startup() {
    echo "$(date '+%F %T') watcher startup failed: $*" >&2
    return 1
}

prepare_tmp_root() {
    local path="$1"

    if [[ -e "${path}" && ! -d "${path}" ]]; then
        fail_startup "TMP_ROOT is not a directory: ${path}"
        return 1
    fi

    if [[ ! -d "${path}" ]]; then
        if ! mkdir -p -- "${path}"; then
            fail_startup "cannot create TMP_ROOT: ${path}"
            return 1
        fi
        echo "$(date '+%F %T') created TMP_ROOT directory: ${path}"
    else
        echo "$(date '+%F %T') using existing TMP_ROOT directory: ${path}"
    fi

    if [[ ! -r "${path}" || ! -w "${path}" || ! -x "${path}" ]]; then
        fail_startup "TMP_ROOT must be readable, writable and searchable: ${path}"
        return 1
    fi
}

if ! prepare_tmp_root "${TMP_ROOT}"; then
    return 1 2>/dev/null || exit 1
fi

if [[ ! -x "${WATCH_TRITON_BIN}" ]]; then
    fail_startup "watcher is not executable: ${WATCH_TRITON_BIN}"
    return 1 2>/dev/null || exit 1
fi

"${WATCH_TRITON_BIN}" &
echo "$(date '+%F %T') started ${WATCH_TRITON_BIN} in background (pid=$!)"

return 0 2>/dev/null || exit 0
