#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WATCHER_MODEL_DIR="${WATCHER_MODEL_DIR:-${TMP_ROOT:-${TMPDIR:-/tmp}}}"
WATCHER_MODEL_DIR_SOURCE="${WATCHER_MODEL_DIR_SOURCE:-direct}"
VERSION_DIR_REGEX="${VERSION_DIR_REGEX:-^[0-9]+$}"
MODELS_ACTIVE_DIR="${MODELS_ACTIVE_DIR:-${WATCHER_MODEL_DIR%/}/models-active}"

# Remember published links so an external unlink is an unload, not a signal
# to publish the same temporary checkout again on the next scan.
declare -A published_targets=()
declare -A config_signatures=() config_backends=() config_failed=() cleanup_attempted=()
cleanup_recovery_error=""
cleanup_recovery_pending=false
WATCHER_CLEANUP_GRACE_SECONDS="${WATCHER_CLEANUP_GRACE_SECONDS:-300}"
WATCHER_CLEANUP_INTERVAL_SECONDS="${WATCHER_CLEANUP_INTERVAL_SECONDS:-30}"
WATCHER_PROC_ROOT="${WATCHER_PROC_ROOT:-/proc}"
if [[ ! "${WATCHER_CLEANUP_GRACE_SECONDS}" =~ ^[0-9]+$ || ! "${WATCHER_CLEANUP_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "watcher cleanup grace and interval must be non-negative integer seconds" >&2
  exit 2
fi
WATCHER_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

read_model_field() {
  local helper="${WATCHER_SOURCE_ROOT}/gateway/model_config.py"
  [[ -f "${helper}" ]] || helper="${WATCHER_SOURCE_ROOT}/../libexec/triton-openai-gateway/model_config.py"
  python3 "${helper}" "$(dirname -- "$1")/config.pbtxt" "$2"
}

read_model_backend() {
  local config_path
  config_path="$(dirname -- "$1")/config.pbtxt"
  if [[ ! -e "${config_path}" && ! -L "${config_path}" ]]; then
    return 0
  fi
  read_model_field "$1" backend
}

# Called directly (not through command substitution), so failure caching survives
# each scan. Both link and target stat identities invalidate an old result.
prepare_model_backend() {
  local target="$1" config signature result
  config="$(dirname -- "${target}")/config.pbtxt"
  signature="$(stat -c '%d:%i:%s:%y:%z' -- "${config}" 2>/dev/null || true)
$(stat -Lc '%d:%i:%s:%y:%z' -- "${config}" 2>/dev/null || true)"
  if [[ "${config_signatures[${config}]+present}" && "${config_signatures[${config}]}" == "${signature}" ]]; then
    MODEL_BACKEND="${config_backends[${config}]-}"
    return "${config_failed[${config}]}"
  fi
  config_signatures["${config}"]="${signature}"
  if result="$(read_model_backend "${target}" 2>&1)"; then
    config_backends["${config}"]="${result}"
    config_failed["${config}"]=0
    MODEL_BACKEND="${result}"
    return 0
  fi
  config_backends["${config}"]=""
  config_failed["${config}"]=1
  MODEL_BACKEND=""
  printf '%s\n' "${result}" >&2
  return 1
}

model_cleanup_helper() {
  local helper="${WATCHER_SOURCE_ROOT}/gateway/model_cleanup.py"
  [[ -f "${helper}" ]] || helper="${WATCHER_SOURCE_ROOT}/../libexec/triton-openai-gateway/model_cleanup.py"
  printf '%s\n' "${helper}"
}

recover_pending_cleanup() {
  local pending result
  for pending in "${WATCHER_MODEL_DIR}"/.watcher-cleanup-*; do
    [[ -d "${pending}" && ! -L "${pending}" ]] || continue
    if result="$(python3 "$(model_cleanup_helper)" --recover "${WATCHER_MODEL_DIR}" "${MODELS_ACTIVE_DIR}" 2>&1)"; then
      [[ -z "${result}" ]] || printf '%s\n' "${result}"
      cleanup_recovery_error=""
      cleanup_recovery_pending=false
      return 0
    fi
    if [[ "${result}" != "${cleanup_recovery_error}" ]]; then
      printf '%s\n' "${result}" >&2
      cleanup_recovery_error="${result}"
    fi
    cleanup_recovery_pending=true
    return 1
  done
  cleanup_recovery_error=""
  cleanup_recovery_pending=false
}

cleanup_unlinked_target() {
  local target="$1" mode="${2:-version}" helper
  [[ "${cleanup_recovery_pending}" == false ]] || return 1
  prepare_model_backend "${target}" || return 1
  if [[ "${cleanup_attempted[${target}]+present}" ]] && (( SECONDS - cleanup_attempted[${target}] < 10#${WATCHER_CLEANUP_INTERVAL_SECONDS} )); then
    return 1
  fi
  cleanup_attempted["${target}"]="${SECONDS}"
  helper="$(model_cleanup_helper)"
  if python3 "${helper}" "${WATCHER_MODEL_DIR}" "${MODELS_ACTIVE_DIR}" "${target}" \
    "${MODEL_BACKEND}" "${WATCHER_CLEANUP_GRACE_SECONDS}" "${WATCHER_PROC_ROOT}" "${mode}"; then
    unset 'cleanup_attempted[$target]'
    return 0
  else
    return "$?"
  fi
}

remember_existing_links() {
  local link_path
  [[ -d "${MODELS_ACTIVE_DIR}" ]] || return 0
  while IFS= read -r -d '' link_path; do
    published_targets["${link_path}"]="$(readlink -m -- "${link_path}")"
  done < <(find "${MODELS_ACTIVE_DIR}" -mindepth 1 -maxdepth 1 -type l -print0)
}

cleanup_removed_links() {
  local link_path previous_target current_target
  for link_path in "${!published_targets[@]}"; do
    previous_target="${published_targets[${link_path}]}"
    current_target=""
    if [[ -L "${link_path}" ]]; then
      current_target="$(readlink -m -- "${link_path}")"
    fi
    if [[ "${current_target}" != "${previous_target}" ]]; then
      cleanup_unlinked_target "${previous_target}" || continue
      unset 'published_targets[$link_path]'
    fi
  done
}

sync_model_json() {
  local target_dir="$1"
  local model_file="${target_dir}/model.json"
  local status=0
  local backend
  prepare_model_backend "${target_dir}" || return 1
  backend="${MODEL_BACKEND}"

  if python3 - "${model_file}" "${target_dir}" "${WATCHER_SOURCE_ROOT}" "${backend}" <<'PY'
import importlib.util
import json
import os
import pathlib
import re
import stat
import sys
import tempfile

file_path, target_dir, source_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
backend = sys.argv[4]
model_path = str(target_dir)
changed = False

try:
    before = file_path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("model.json must be a regular file, not a final symlink")
    if backend == "vllm_deepseek":
        helper = source_root / "backends/vllm_deepseek/checkpoint_preflight.py"
        if not helper.is_file():
            helper = pathlib.Path('/opt/tritonserver/backends/vllm_deepseek/checkpoint_preflight.py')
        spec = importlib.util.spec_from_file_location('checkpoint_preflight', helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = module.read_json(file_path)
        module.validate_checkpoint(data)
        # Validate exactly the root passed to vLLM. No writes to weight snapshots.
        model_path = data['model']
    else:
        fd = os.open(file_path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("model.json must contain an object")
except (OSError, ValueError, ImportError, AttributeError) as error:
    print(f"checkpoint/config admission failed for {file_path}: {error}", file=sys.stderr)
    raise SystemExit(4)

if str(data.get("load_format", "")).lower() == "gguf":
    gguf_files = sorted(target_dir.glob("*.gguf"))
    if not gguf_files:
        raise SystemExit(5)
    model_path = str(gguf_files[0])

if "swap_space" in data:
    del data["swap_space"]
    changed = True
if data.get("model") != model_path:
    data["model"] = model_path
    changed = True
if not changed:
    raise SystemExit(0)

# Same-filesystem replace: readers see complete old/new JSON. Never follow the
# final path, and preserve existing permissions/ownership rather than tempfile's.
temporary = None
try:
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=file_path.parent, prefix='.model-json-', delete=False) as fh:
        temporary = fh.name
        os.fchown(fh.fileno(), before.st_uid, before.st_gid)
        os.fchmod(fh.fileno(), stat.S_IMODE(before.st_mode))
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        # Path normalization must not change the source revision's priority,
        # including on the first scan or after a watcher restart.
        os.utime(fh.fileno(), ns=(before.st_atime_ns, before.st_mtime_ns))
        os.fsync(fh.fileno())
    current = file_path.lstat()
    if (current.st_dev,current.st_ino,current.st_mtime_ns,current.st_size) != (before.st_dev,before.st_ino,before.st_mtime_ns,before.st_size):
        raise ValueError('model.json changed during atomic replacement')
    os.replace(temporary, file_path)
    temporary = None
except (OSError, ValueError) as error:
    print(f"atomic model config update failed for {file_path}: {error}", file=sys.stderr)
    raise SystemExit(4)
finally:
    if temporary is not None:
        os.unlink(temporary)

raise SystemExit(3)
PY
  then
    return 0
  else
    status=$?
  fi

  case "${status}" in
    3)
      echo "$(date '+%F %T') updated model config in ${model_file}"
      ;;
    4)
      echo "$(date '+%F %T') skipped invalid ${model_file}"
      return 1
      ;;
    5)
      echo "$(date '+%F %T') skipped gguf model without .gguf file in ${target_dir}"
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_model_name() {
  local target_dir="$1"
  local model_name
  if [[ -e "${target_dir}/../config.pbtxt" || -L "${target_dir}/../config.pbtxt" ]]; then
    model_name="$(read_model_field "${target_dir}" name)" || return 1
    if [[ -n "${model_name}" ]]; then
      printf '%s\n' "${model_name}"
      return 0
    fi
  fi

  python3 - "${target_dir}" <<'PY'
import json
import pathlib
import sys

target_dir = pathlib.Path(sys.argv[1]).resolve()
model_root = target_dir.parent

model_json = target_dir / "model.json"
if model_json.is_file():
    try:
        data = json.loads(model_json.read_text(encoding="utf-8"))
    except Exception:
        data = None
    if isinstance(data, dict):
        for key in ("served_model_name", "model_name", "name"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                print(value.strip())
                raise SystemExit(0)

hf_config = target_dir / "config.json"
if hf_config.is_file():
    try:
        data = json.loads(hf_config.read_text(encoding="utf-8"))
    except Exception:
        data = None
    if isinstance(data, dict):
        for key in ("_name_or_path", "name_or_path"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                print(value.strip().split("/")[-1])
                raise SystemExit(0)

print(model_root.name)
PY
}

prefer_target() {
  local candidate="$1"
  local current="$2"
  local candidate_version current_version candidate_mtime current_mtime

  [[ -e "${current}" ]] || return 0

  candidate_version="$(basename "${candidate}")"
  current_version="$(basename "${current}")"

  if [[ "${candidate_version}" =~ ^[0-9]+$ && "${current_version}" =~ ^[0-9]+$ ]]; then
    if (( candidate_version > current_version )); then
      return 0
    fi
    if (( candidate_version < current_version )); then
      return 1
    fi
  fi

  candidate_mtime="$(stat -c %Y "${candidate}/model.json" 2>/dev/null || stat -c %Y "${candidate}" 2>/dev/null || echo 0)"
  current_mtime="$(stat -c %Y "${current}/model.json" 2>/dev/null || stat -c %Y "${current}" 2>/dev/null || echo 0)"

  (( candidate_mtime >= current_mtime ))
}

sync_model_symlink() {
  local target_dir="$1"
  local model_name link_path existing_target

  model_name="$(resolve_model_name "${target_dir}")" || return 0
  [[ -n "${model_name}" && "${model_name}" != "." && "${model_name}" != ".." && "${model_name}" != */* ]] || return 0

  mkdir -p "${MODELS_ACTIVE_DIR}"
  link_path="${MODELS_ACTIVE_DIR}/${model_name}"

  if [[ -L "${link_path}" ]]; then
    existing_target="$(readlink -m "${link_path}" 2>/dev/null || true)"
  else
    existing_target=""
  fi

  if [[ -n "${existing_target}" && "${existing_target}" == "$(readlink -f "${target_dir}")" ]]; then
    published_targets["${link_path}"]="${existing_target}"
    return 0
  fi

  if [[ -n "${existing_target}" ]] && ! prefer_target "${target_dir}" "${existing_target}"; then
    return 0
  fi

  ln -sfnT "${target_dir}" "${link_path}"
  published_targets["${link_path}"]="$(readlink -m -- "${target_dir}")"
  if [[ -n "${existing_target}" ]]; then
    cleanup_unlinked_target "${existing_target}" || true
  fi
  echo "$(date '+%F %T') updated symlink ${link_path} -> ${target_dir}"
}

cleanup_stale_symlinks() {
  local link_path
  local target_path

  [[ -d "${MODELS_ACTIVE_DIR}" ]] || return 0

  while IFS= read -r link_path; do
    target_path="$(readlink -m "${link_path}" 2>/dev/null || true)"
    if [[ -z "${target_path}" || ! -e "${target_path}" ]]; then
      rm -f "${link_path}"
      echo "$(date '+%F %T') removed stale symlink ${link_path}"
      if [[ -n "${target_path}" ]]; then
        cleanup_unlinked_target "${target_path}" || true
      fi
    fi
  done < <(find "${MODELS_ACTIVE_DIR}" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | sort)
}

scan_target() {
  local target="$1"
  local version_dir
  version_dir="$(basename "${target}")"

  [[ "${version_dir}" =~ ${VERSION_DIR_REGEX} ]] || return 0
  local backend
  prepare_model_backend "${target}" || return 0
  backend="${MODEL_BACKEND}"
  if [[ -e "${target}/.deepseek-retired" || -L "${target}/.deepseek-retired" ]]; then
    return 0
  fi
  if [[ -e "${target}/.watcher-retired.json" || -L "${target}/.watcher-retired.json" ]]; then
    cleanup_unlinked_target "${target}" || true
    return 0
  fi
  [[ -f "${target}/model.json" || -f "${target}/model.py" ]] || return 0

  # Select before path repair: rewriting an old model.json must not make an
  # unpublished duplicate appear newer than the currently selected copy.
  local model_name existing_target
  model_name="$(resolve_model_name "${target}")" || return 0
  [[ -n "${model_name}" && "${model_name}" != "." && "${model_name}" != ".." && "${model_name}" != */* ]] || return 0
  if [[ -L "${MODELS_ACTIVE_DIR}/${model_name}" ]]; then
    existing_target="$(readlink -m -- "${MODELS_ACTIVE_DIR}/${model_name}")"
    if [[ "${existing_target}" != "$(readlink -f -- "${target}")" ]] && ! prefer_target "${target}" "${existing_target}"; then
      # Preserve distinct versions and backend changes; only an older duplicate
      # of the same model/version can be collected without publication.
      if [[ "$(basename -- "${existing_target}")" == "${version_dir}" ]] && prepare_model_backend "${existing_target}" && [[ "${MODEL_BACKEND}" == "${backend}" ]]; then
        cleanup_unlinked_target "${target}" || true
      fi
      return 0
    fi
  fi

  if [[ -f "${target}/model.json" ]]; then
    sync_model_json "${target}" || return 0
  fi
  sync_model_symlink "${target}"
}

scan_once() {
  local folder
  local target

  recover_pending_cleanup || true
  cleanup_stale_symlinks
  cleanup_removed_links

  while IFS= read -r folder; do
    # A deferred metadata recovery must not freeze unrelated valid models.
    # Do not use the legacy missing-config fallback while files are staged.
    if [[ "${cleanup_recovery_pending}" == true && ! -f "${folder}/config.pbtxt" ]]; then
      continue
    fi
    # An unloaded version may already be gone while its shared metadata is
    # waiting for the grace period. Keep collecting it after watcher restarts.
    if [[ -e "${folder}/.watcher-retired.json" || -L "${folder}/.watcher-retired.json" ]]; then
      cleanup_unlinked_target "${folder}/0" empty || true
    fi
    while IFS= read -r target; do
      scan_target "${target}"
    done < <(find "${folder}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  done < <(find "${WATCHER_MODEL_DIR}" -maxdepth 1 -type d -name 'folder*' 2>/dev/null | sort)
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "$(date '+%F %T') watching ${WATCHER_MODEL_DIR} (source=${WATCHER_MODEL_DIR_SOURCE}, models_active_dir=${MODELS_ACTIVE_DIR}) ..."
  remember_existing_links
  while true; do
    scan_once
    sleep 1
  done
fi
