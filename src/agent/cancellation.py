"""Cooperative cancellation for async I/O that may never yield another byte."""
import asyncio
from contextlib import suppress
from threading import Event


class RequestCancelled(Exception):
    """The caller cancelled an in-flight request; this is not a provider error."""


async def cancellable(awaitable, cancel_event: Event | None):
    """Cancel AND join I/O on the owner loop; never strand an executor thread."""
    task = asyncio.ensure_future(awaitable)
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled("Research cancelled")
            done, _ = await asyncio.wait({task}, timeout=.05)
            if done:
                if cancel_event is not None and cancel_event.is_set():
                    raise RequestCancelled("Research cancelled")
                return task.result()
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task
