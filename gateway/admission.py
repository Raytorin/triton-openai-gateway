# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .metrics import (
    ADMISSION_INFLIGHT,
    ADMISSION_QUEUED,
    ADMISSION_REJECTED,
    ADMISSION_WAIT,
)


@dataclass(frozen=True)
class GatePolicy:
    max_inflight: int
    max_queue: int
    queue_timeout_seconds: float


@dataclass(frozen=True)
class AdmissionPolicy:
    global_gate: GatePolicy
    model_gate: GatePolicy


class _BoundedGate:
    def __init__(self, scope: str, route: str, model: str, policy: GatePolicy):
        self.scope = scope
        self.route = route
        self.model = model
        self.policy = policy
        self._condition = asyncio.Condition()
        self._inflight = 0
        self._queued = 0

    async def acquire(self) -> float:
        started_at = time.monotonic()
        async with self._condition:
            if self._inflight >= self.policy.max_inflight:
                if self._queued >= self.policy.max_queue:
                    self._reject("queue_full")
                self._queued += 1
                self._set_metrics()
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._inflight < self.policy.max_inflight
                        ),
                        timeout=self.policy.queue_timeout_seconds,
                    )
                except TimeoutError as exc:
                    self._reject("queue_timeout", exc)
                finally:
                    self._queued -= 1
                    self._set_metrics()

            self._inflight += 1
            self._set_metrics()
        wait_seconds = time.monotonic() - started_at
        ADMISSION_WAIT.labels(self.scope, self.route, self.model).observe(wait_seconds)
        return wait_seconds

    async def release(self) -> None:
        async with self._condition:
            if self._inflight > 0:
                self._inflight -= 1
            self._set_metrics()
            self._condition.notify(1)

    def _set_metrics(self) -> None:
        labels = (self.scope, self.route, self.model)
        ADMISSION_INFLIGHT.labels(*labels).set(self._inflight)
        ADMISSION_QUEUED.labels(*labels).set(self._queued)

    def _reject(self, reason: str, cause: BaseException | None = None) -> None:
        ADMISSION_REJECTED.labels(
            self.scope,
            self.route,
            self.model,
            reason,
        ).inc()
        error = HTTPException(
            status_code=429,
            detail=(
                f"Gateway is overloaded for model '{self.model}' and route "
                f"'{self.route}' ({reason})"
            ),
            headers={"Retry-After": "1"},
        )
        if cause is not None:
            raise error from cause
        raise error


class AdmissionLease:
    def __init__(self, gates: list[_BoundedGate], wait_seconds: float = 0.0):
        self._gates = gates
        self.wait_seconds = max(float(wait_seconds), 0.0)
        self._released = False
        self._lock = asyncio.Lock()

    async def release(self) -> None:
        async with self._lock:
            if self._released:
                return
            self._released = True
            for gate in reversed(self._gates):
                await gate.release()


class AdmissionController:
    def __init__(self) -> None:
        self._gates: dict[tuple[Any, ...], _BoundedGate] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, route: str, model: str, model_path: Path) -> AdmissionLease:
        policy = load_admission_policy(model_path, route)
        global_gate = await self._gate(
            ("global", policy.global_gate),
            "global",
            "all",
            "all",
            policy.global_gate,
        )
        model_gate = await self._gate(
            ("model", route, model, policy.model_gate),
            "model",
            route,
            model,
            policy.model_gate,
        )
        acquired: list[_BoundedGate] = []
        wait_seconds = 0.0
        try:
            wait_seconds += await global_gate.acquire()
            acquired.append(global_gate)
            wait_seconds += await model_gate.acquire()
            acquired.append(model_gate)
        except BaseException:
            for gate in reversed(acquired):
                await gate.release()
            raise
        return AdmissionLease(acquired, wait_seconds)

    async def _gate(
        self,
        key: tuple[Any, ...],
        scope: str,
        route: str,
        model: str,
        policy: GatePolicy,
    ) -> _BoundedGate:
        async with self._lock:
            gate = self._gates.get(key)
            if gate is None:
                gate = _BoundedGate(scope, route, model, policy)
                self._gates[key] = gate
            return gate


def load_admission_policy(model_path: Path, route: str) -> AdmissionPolicy:
    config_path = model_path / "gateway.json"
    modified_ns = config_path.stat().st_mtime_ns if config_path.is_file() else 0
    configured = _read_admission_config(str(config_path), modified_ns)
    route_config = configured.get(route, {})
    if not isinstance(route_config, dict):
        route_config = {}

    global_policy = GatePolicy(
        max_inflight=_env_int("GATEWAY_MAX_INFLIGHT_REQUESTS", 256),
        max_queue=_env_int("GATEWAY_MAX_QUEUE_SIZE", 512, allow_zero=True),
        queue_timeout_seconds=_env_float("GATEWAY_QUEUE_TIMEOUT_SECONDS", 30.0),
    )
    prefix = "MEDIA" if route == "media" else route.upper()
    if route == "media":
        default_inflight, default_queue = 4, 16
    elif route == "rerank":
        # A Python reranker normally owns one GPU model instance.
        default_inflight, default_queue = 1, 64
    else:
        default_inflight, default_queue = 64, 256
    model_policy = GatePolicy(
        max_inflight=_configured_int(
            route_config,
            "max_inflight",
            f"GATEWAY_{prefix}_MAX_INFLIGHT_REQUESTS",
            default_inflight,
        ),
        max_queue=_configured_int(
            route_config,
            "max_queue",
            f"GATEWAY_{prefix}_MAX_QUEUE_SIZE",
            default_queue,
            allow_zero=True,
        ),
        queue_timeout_seconds=_configured_float(
            route_config,
            "queue_timeout_seconds",
            f"GATEWAY_{prefix}_QUEUE_TIMEOUT_SECONDS",
            global_policy.queue_timeout_seconds,
        ),
    )
    return AdmissionPolicy(global_gate=global_policy, model_gate=model_policy)


@lru_cache(maxsize=256)
def _read_admission_config(config_path: str, modified_ns: int) -> dict[str, Any]:
    del modified_ns
    path = Path(config_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    configured = payload.get("admission", {}) if isinstance(payload, dict) else {}
    return configured if isinstance(configured, dict) else {}


def _configured_int(
    configured: dict[str, Any],
    key: str,
    env_name: str,
    default: int,
    *,
    allow_zero: bool = False,
) -> int:
    if env_name in os.environ:
        return _env_int(env_name, default, allow_zero=allow_zero)
    try:
        value = int(configured.get(key, default))
    except (TypeError, ValueError):
        value = default
    minimum = 0 if allow_zero else 1
    return max(value, minimum)


def _configured_float(
    configured: dict[str, Any],
    key: str,
    env_name: str,
    default: float,
) -> float:
    if env_name in os.environ:
        return _env_float(env_name, default)
    try:
        value = float(configured.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(value, 0.001)


def _env_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    minimum = 0 if allow_zero else 1
    return max(value, minimum)


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    return max(value, 0.001)
