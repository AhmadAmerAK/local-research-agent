"""Strands Ollama adapter with bounded HTTP waits and cooperative stream cancellation.

Request construction and chunk conversion remain the installed SDK's. Its current
Ollama adapter ignores cancel_signal and does not explicitly close its AsyncClient.
Only transport lifetime differs here; inference options are untouched.
"""
import asyncio

import httpx
import ollama
from strands.models.ollama import OllamaModel
from strands.types.exceptions import ContextWindowOverflowException

from .cancellation import RequestCancelled, cancellable

OLLAMA_HTTP_TIMEOUT = httpx.Timeout(connect=5, read=60, write=15, pool=5)
MODEL_REQUEST_DEADLINE = 180.0


class CancellableOllamaModel(OllamaModel):
    def __init__(self, host, *, cancel_event=None, **kwargs):
        super().__init__(host, **kwargs)
        self.cancel_event = cancel_event

    async def stream(self, messages, tool_specs=None, system_prompt=None, *,
                     tool_choice=None, cancel_signal=None, **kwargs):
        if cancel_signal is None:
            cancel_signal = self.cancel_event
        request = self.format_request(messages, tool_specs, system_prompt)
        client = ollama.AsyncClient(self.host, **self.client_args)
        response = None
        event = None
        tool_requested = False
        try:
            async with asyncio.timeout(MODEL_REQUEST_DEADLINE):
                response = await cancellable(client.chat(**request), cancel_signal)
                yield self.format_chunk({"chunk_type": "message_start"})
                yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})
                while True:
                    try:
                        event = await cancellable(anext(response), cancel_signal)
                    except StopAsyncIteration:
                        break
                    for tool_call in event.message.tool_calls or []:
                        for kind in ("content_start", "content_delta", "content_stop"):
                            yield self.format_chunk({"chunk_type": kind, "data_type": "tool", "data": tool_call})
                        tool_requested = True
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text",
                                             "data": event.message.content})
        except RequestCancelled:
            # No fabricated messageStop: native Strands records stop_reason=cancelled.
            return
        except ollama.ResponseError as error:
            if any(message in str(error).lower() for message in self.OVERFLOW_MESSAGES):
                raise ContextWindowOverflowException(str(error)) from error
            raise
        finally:
            try:
                if response is not None:
                    await response.aclose()
            finally:
                await client.close()
        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "message_stop",
                                 "data": "tool_use" if tool_requested else (event.done_reason if event else None)})
        if event is not None:
            yield self.format_chunk({"chunk_type": "metadata", "data": event})
