"""Pending-confirmation registry for HITL tool gating.

When the agent loop pauses to ask the user about a tool call, it
needs an awaitable that resolves once the user clicks confirm or
decline.  The streaming worker that holds the SSE connection
isn't necessarily the worker that receives the ``POST /chat/confirm``
HTTP callback — under uvicorn ``--workers N`` the two land on
different processes — so we route decisions through a pubsub
channel any worker can publish to.

Two implementations:

* :class:`InMemoryStore` (default).  An in-process dict of
  ``asyncio.Future`` objects keyed by ``request_id``.  Single-
  worker development; doesn't survive a process restart.
* :class:`RedisStore` (opt-in).  Each worker SUBSCRIBES to a
  per-request channel; the decision POST PUBLISHes from any
  worker.  Requires the ``redis>=5.0`` extra.

Pick by setting the ``REDIS_URL`` env var; absent → in-memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


log = logging.getLogger(__name__)


ConfirmationDecision = Literal["confirm", "decline"]


@dataclass(frozen=True)
class ConfirmationOutcome:
    """The user's decision on a pending confirmation."""

    decision: ConfirmationDecision
    note: str | None = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class PendingConfirmationStore(ABC):
    """Bidirectional channel: the agent loop awaits, the chat-confirm
    HTTP route resolves.

    Lifecycle for one confirmation:

    1. Streaming worker calls :meth:`register(request_id)` and
       receives a future.  It then ``await``s the future.
    2. Some worker (the same or another) receives the user's
       decision via HTTP and calls :meth:`resolve(request_id, ...)`.
    3. The future fires; the agent loop continues.

    On client disconnect or stream cancel, the streaming worker
    calls :meth:`cancel(request_id)` to drop the registration.
    """

    @abstractmethod
    async def register(self, request_id: str) -> "asyncio.Future[ConfirmationOutcome]":
        """Register a pending confirmation; return the awaitable."""

    @abstractmethod
    async def resolve(
        self, request_id: str, outcome: ConfirmationOutcome
    ) -> bool:
        """Push a decision.  Returns ``True`` if a future was waiting,
        ``False`` if the id is unknown / already resolved / cancelled."""

    @abstractmethod
    async def cancel(self, request_id: str) -> None:
        """Drop a pending registration without resolving it.  Idempotent."""

    async def aclose(self) -> None:  # pragma: no cover - default is a no-op
        """Release any held resources (subscriptions, connections)."""


# ---------------------------------------------------------------------------
# In-memory (default)
# ---------------------------------------------------------------------------


class InMemoryStore(PendingConfirmationStore):
    """Local-process pending map.  Sufficient for single-worker dev
    deployments and the test suite."""

    def __init__(self) -> None:
        self._futures: dict[str, asyncio.Future[ConfirmationOutcome]] = {}

    async def register(self, request_id: str) -> asyncio.Future[ConfirmationOutcome]:
        if request_id in self._futures:
            raise ValueError(f"request_id {request_id!r} already registered")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ConfirmationOutcome] = loop.create_future()
        self._futures[request_id] = fut
        return fut

    async def resolve(
        self, request_id: str, outcome: ConfirmationOutcome
    ) -> bool:
        fut = self._futures.pop(request_id, None)
        if fut is None or fut.done():
            return False
        fut.set_result(outcome)
        return True

    async def cancel(self, request_id: str) -> None:
        fut = self._futures.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.cancel()


# ---------------------------------------------------------------------------
# Redis pubsub (opt-in)
# ---------------------------------------------------------------------------


class RedisStore(PendingConfirmationStore):
    """Cross-worker pending map.  Each ``register`` SUBSCRIBES to a
    per-request channel; ``resolve`` PUBLISHES the decision so it
    reaches whichever worker holds the awaiting future.

    The Redis client is created from the ``url`` (e.g.
    ``redis://localhost:6379``).  Channel keys are namespaced under
    ``aai-confirmation:<request_id>``.
    """

    _CHANNEL_PREFIX = "aai-confirmation:"

    def __init__(self, url: str) -> None:
        try:
            import redis.asyncio as aioredis  # noqa: F401
        except ImportError as err:  # pragma: no cover
            raise RuntimeError(
                "RedisStore requires the 'redis' extra: "
                "pip install ai-assistant-client[redis]"
            ) from err
        self._url = url
        self._client = self._make_client()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._futures: dict[str, asyncio.Future[ConfirmationOutcome]] = {}

    def _make_client(self):
        import redis.asyncio as aioredis

        return aioredis.from_url(self._url, decode_responses=True)

    def _channel(self, request_id: str) -> str:
        return f"{self._CHANNEL_PREFIX}{request_id}"

    async def register(self, request_id: str) -> asyncio.Future[ConfirmationOutcome]:
        if request_id in self._futures:
            raise ValueError(f"request_id {request_id!r} already registered")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[ConfirmationOutcome] = loop.create_future()
        self._futures[request_id] = fut
        # Spawn a subscriber task; it sets the future when a
        # message arrives on this request's channel.
        self._tasks[request_id] = asyncio.create_task(
            self._subscribe_and_resolve(request_id, fut)
        )
        return fut

    async def _subscribe_and_resolve(
        self,
        request_id: str,
        fut: asyncio.Future[ConfirmationOutcome],
    ) -> None:
        pubsub = self._client.pubsub()
        await pubsub.subscribe(self._channel(request_id))
        try:
            async for msg in pubsub.listen():
                if msg.get("type") != "message":
                    continue
                try:
                    data = json.loads(msg["data"])
                    outcome = ConfirmationOutcome(
                        decision=data["decision"], note=data.get("note")
                    )
                except (KeyError, ValueError, TypeError):
                    log.warning(
                        "RedisStore: malformed confirmation payload for %s",
                        request_id,
                    )
                    continue
                if not fut.done():
                    fut.set_result(outcome)
                return
        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(self._channel(request_id))
                await pubsub.aclose()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass

    async def resolve(
        self, request_id: str, outcome: ConfirmationOutcome
    ) -> bool:
        payload = json.dumps(
            {"decision": outcome.decision, "note": outcome.note}
        )
        # PUBLISH returns the number of subscribers that received it.
        # Zero means nobody was waiting (unknown request, expired,
        # or the streaming worker died).
        delivered = await self._client.publish(
            self._channel(request_id), payload
        )
        return bool(delivered)

    async def cancel(self, request_id: str) -> None:
        fut = self._futures.pop(request_id, None)
        if fut is not None and not fut.done():
            fut.cancel()
        task = self._tasks.pop(request_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def aclose(self) -> None:  # pragma: no cover - shutdown path
        for task in list(self._tasks.values()):
            task.cancel()
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_confirmation_store() -> PendingConfirmationStore:
    """Pick a store based on env: ``REDIS_URL`` → :class:`RedisStore`,
    else :class:`InMemoryStore`."""
    url = os.environ.get("REDIS_URL", "").strip()
    if url:
        return RedisStore(url)
    return InMemoryStore()
