#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

WATCHER_MODEL_DIR="${WATCHER_MODEL_DIR:-${TMP_ROOT:-${TMPDIR:-/tmp}}}"
WATCHER_MODEL_DIR_SOURCE="${WATCHER_MODEL_DIR_SOURCE:-direct}"
VERSION_DIR_REGEX="${VERSION_DIR_REGEX:-^[0-9]+$}"
MODELS_ACTIVE_DIR="${MODELS_ACTIVE_DIR:-${WATCHER_MODEL_DIR%/}/models-active}"

sync_model_json() {
  local target_dir="$1"
  local model_file="${target_dir}/model.json"
  local status=0

  if python3 - "${model_file}" "${target_dir}" <<'PY'
import json
import pathlib
import sys

file_path, target_dir = sys.argv[1], pathlib.Path(sys.argv[2])
model_path = str(target_dir)
changed = False

try:
    with open(file_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
except (OSError, json.JSONDecodeError):
    raise SystemExit(4)

if not isinstance(data, dict):
    raise SystemExit(4)

if str(data.get("load_format", "")).lower() == "gguf":
    gguf_files = sorted(target_dir.glob("*.gguf"))
    if not gguf_files:
        raise SystemExit(5)
    model_path = str(gguf_files[0])

# vLLM 0.22 removed this AsyncEngineArgs parameter. Rewrite only the temporary
# S3 checkout; the source model repository remains untouched.
if "swap_space" in data:
    del data["swap_space"]
    changed = True

if data.get("model") != model_path:
    data["model"] = model_path
    changed = True

if not changed:
    raise SystemExit(0)

with open(file_path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
    fh.write("\n")

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

  python3 - "${target_dir}" <<'PY'
import json
import pathlib
import re
import sys

target_dir = pathlib.Path(sys.argv[1]).resolve()
model_root = target_dir.parent

config_pbtxt = model_root / "config.pbtxt"
if config_pbtxt.is_file():
    text = config_pbtxt.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'^\s*name\s*:\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        match = re.search(r'^\s*name\s*:\s*([^\s"]+)', text, re.MULTILINE)
    if match:
        print(match.group(1))
        raise SystemExit(0)

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

  model_name="$(resolve_model_name "${target_dir}")"
  [[ -n "${model_name}" ]] || return 0

  mkdir -p "${MODELS_ACTIVE_DIR}"
  link_path="${MODELS_ACTIVE_DIR}/${model_name}"

  if [[ -L "${link_path}" ]]; then
    existing_target="$(readlink -f "${link_path}" 2>/dev/null || true)"
  else
    existing_target=""
  fi

  if [[ -n "${existing_target}" && "${existing_target}" == "$(readlink -f "${target_dir}")" ]]; then
    return 0
  fi

  if [[ -n "${existing_target}" ]] && ! prefer_target "${target_dir}" "${existing_target}"; then
    return 0
  fi

  ln -sfn "${target_dir}" "${link_path}"
  echo "$(date '+%F %T') updated symlink ${link_path} -> ${target_dir}"
}

cleanup_stale_symlinks() {
  local link_path
  local target_path

  [[ -d "${MODELS_ACTIVE_DIR}" ]] || return 0

  while IFS= read -r link_path; do
    target_path="$(readlink -f "${link_path}" 2>/dev/null || true)"
    if [[ -z "${target_path}" || ! -e "${target_path}" ]]; then
      rm -f "${link_path}"
      echo "$(date '+%F %T') removed stale symlink ${link_path}"
    fi
  done < <(find "${MODELS_ACTIVE_DIR}" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | sort)
}

scan_target() {
  local target="$1"
  local version_dir
  version_dir="$(basename "${target}")"

  [[ "${version_dir}" =~ ${VERSION_DIR_REGEX} ]] || return 0
  [[ -f "${target}/model.json" || -f "${target}/model.py" ]] || return 0

  if [[ -f "${target}/model.json" ]]; then
    sync_model_json "${target}" || return 0
  fi
  sync_model_symlink "${target}"
}

scan_once() {
  local folder
  local target

  cleanup_stale_symlinks

  while IFS= read -r folder; do
    while IFS= read -r target; do
      scan_target "${target}"
    done < <(find "${folder}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  done < <(find "${WATCHER_MODEL_DIR}" -maxdepth 1 -type d -name 'folder*' 2>/dev/null | sort)
}

echo "$(date '+%F %T') watching ${WATCHER_MODEL_DIR} (source=${WATCHER_MODEL_DIR_SOURCE}, models_active_dir=${MODELS_ACTIVE_DIR}) ..."

while true; do
  scan_once
  sleep 1
done
