# SPDX-FileCopyrightText: 2026 Raytorin
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU work with cooperative cancellation and request-context propagation."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from functools import partial
import os
import math
from threading import Event
import time
from typing import Callable, TypeVar

import anyio
from anyio.lowlevel import RunVar
from fastapi import HTTPException

from .metrics import CPU_ACTIVE, CPU_QUEUED, CPU_REJECTED, CPU_WAIT, CPU_DURATION

T = TypeVar("T")
_cancel_event: ContextVar[Event | None] = ContextVar("cpu_cancel_event", default=None)
_pool: RunVar[CpuWorkPool] = RunVar("gateway_cpu_pool")


def checkpoint() -> None:
    """Stop between tokenizer calls; an individual native encode cannot be interrupted."""
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise asyncio.CancelledError()


class CpuWorkPool:
    def __init__(self, workers: int = 4, max_queue: int = 64, timeout: float = 30.0):
        if workers < 1 or max_queue < 0 or timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("Invalid CPU work pool limits")
        self._slots = asyncio.Semaphore(workers)
        self._threads = anyio.CapacityLimiter(workers)
        self.max_queue = max_queue
        self.timeout = timeout
        self.queued = 0
        self.active = 0

    def _reject(self, reason: str) -> None:
        CPU_REJECTED.labels(reason).inc()
        raise HTTPException(429, f"Gateway CPU work queue {reason}", headers={"Retry-After": "1"})

    @staticmethod
    def _invoke(stop: Event, function: Callable[..., T], args, kwargs) -> T:
        token = _cancel_event.set(stop)
        try:
            checkpoint()
            result = function(*args, **kwargs)
            checkpoint()
            return result
        finally:
            _cancel_event.reset(token)

    async def run(self, function: Callable[..., T], *args, **kwargs) -> T:
        waiting = time.monotonic()
        if self._slots.locked():
            if self.queued >= self.max_queue:
                self._reject("full")
            self.queued += 1
            CPU_QUEUED.inc()
            try:
                try:
                    await asyncio.wait_for(self._slots.acquire(), self.timeout)
                except TimeoutError:
                    self._reject("timeout")
            finally:
                self.queued -= 1
                CPU_QUEUED.dec()
        else:
            await self._slots.acquire()
        CPU_WAIT.observe(time.monotonic() - waiting)
        started = time.monotonic()
        self.active += 1
        CPU_ACTIVE.inc()
        stop = Event()
        task = None
        try:
            task = asyncio.create_task(anyio.to_thread.run_sync(
                partial(self._invoke, stop, function, args, kwargs), limiter=self._threads,
            ))
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            stop.set()
            # Do not release CPU/admission capacity while a native call still runs.
            # Drain without blocking the event loop, including repeated cancellation.
            with anyio.CancelScope(shield=True):
                while task is not None and not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                if task is not None and task.done() and not task.cancelled():
                    task.exception()
            raise
        finally:
            self.active -= 1
            CPU_ACTIVE.dec()
            CPU_DURATION.observe(time.monotonic() - started)
            self._slots.release()


async def run_cpu(function: Callable[..., T], *args, **kwargs) -> T:
    # AnyIO's run-local storage isolates pools across event loops and test clients.
    try:
        pool = _pool.get()
    except LookupError:
        pool = CpuWorkPool(
            workers=int(os.environ.get("GATEWAY_CPU_WORKERS", "4")),
            max_queue=int(os.environ.get("GATEWAY_CPU_MAX_QUEUE", "64")),
            timeout=float(os.environ.get("GATEWAY_CPU_QUEUE_TIMEOUT_SECONDS", "30")),
        )
        _pool.set(pool)
    return await pool.run(function, *args, **kwargs)
