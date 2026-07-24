import asyncio

try:
    import uvloop
except ImportError:  # pragma: no cover - uvloop is unavailable on some platforms.
    uvloop = None


if uvloop is not None:
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
