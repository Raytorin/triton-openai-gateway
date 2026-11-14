#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
# Choose the shared Python/native/compiler paths before hooks start any process.
exec /opt/envs/vllm032-cu129/bin/python3.12 -I \
    /opt/tritonserver/unified/launch.py --exec \
    /bin/bash /opt/nvidia/nvidia_entrypoint.sh "$@"
