# Copyright 2026 Raytorin and Triton OpenAI Gateway contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any


def local_parallel_world_size(engine_config: dict[str, Any]) -> tuple[int, int, int, int]:
    tp_size = int(engine_config.get("tensor_parallel_size", 1))
    pp_size = int(engine_config.get("pipeline_parallel_size", 1))
    dp_size = int(engine_config.get("data_parallel_size", 1))
    local_dp_size = int(engine_config.get("data_parallel_size_local", dp_size))
    return tp_size, pp_size, local_dp_size, tp_size * pp_size * local_dp_size
