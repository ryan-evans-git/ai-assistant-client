"""Tests for the pending-confirmation store.

Both the in-memory (default) and Redis-backed implementations are
exercised — Redis via fakeredis-asyncio so the test suite stays
fully offline.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_assistant_client.confirmation_store import (
    ConfirmationOutcome,
    InMemoryStore,
    RedisStore,
    make_confirmation_store,
)


# ---------------------------------------------------------------------------
# InMemoryStore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inmemory_register_and_resolve_round_trips() -> None:
    store = InMemoryStore()
    fut = await store.register("rid-1")
    delivered = await store.resolve(
        "rid-1", ConfirmationOutcome(decision="confirm", note="LGTM")
    )
    assert delivered is True
    outcome = await fut
    assert outcome.decision == "confirm"
    assert outcome.note == "LGTM"


@pytest.mark.asyncio
async def test_inmemory_resolve_unknown_id_returns_false() -> None:
    store = InMemoryStore()
    delivered = await store.resolve(
        "ghost", ConfirmationOutcome(decision="confirm")
    )
    assert delivered is False


@pytest.mark.asyncio
async def test_inmemory_resolve_after_resolve_returns_false() -> None:
    """Second resolve on the same id is a no-op (id was popped)."""
    store = InMemoryStore()
    fut = await store.register("rid-2")
    await store.resolve("rid-2", ConfirmationOutcome(decision="confirm"))
    await fut
    delivered = await store.resolve("rid-2", ConfirmationOutcome(decision="decline"))
    assert delivered is False


@pytest.mark.asyncio
async def test_inmemory_cancel_drops_future() -> None:
    store = InMemoryStore()
    fut = await store.register("rid-3")
    await store.cancel("rid-3")
    assert fut.cancelled()
    # Subsequent resolve no-ops.
    delivered = await store.resolve("rid-3", ConfirmationOutcome(decision="confirm"))
    assert delivered is False


@pytest.mark.asyncio
async def test_inmemory_cancel_unknown_id_is_noop() -> None:
    store = InMemoryStore()
    await store.cancel("ghost")  # must not raise


@pytest.mark.asyncio
async def test_inmemory_double_register_raises() -> None:
    store = InMemoryStore()
    await store.register("rid-4")
    with pytest.raises(ValueError, match="already registered"):
        await store.register("rid-4")


@pytest.mark.asyncio
async def test_inmemory_wait_for_timeout() -> None:
    """The store doesn't enforce timeout itself — callers wrap in
    asyncio.wait_for.  Verify that pattern works."""
    store = InMemoryStore()
    fut = await store.register("rid-5")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(fut, timeout=0.05)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_confirmation_store_defaults_to_inmemory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REDIS_URL", raising=False)
    store = make_confirmation_store()
    assert isinstance(store, InMemoryStore)


def test_make_confirmation_store_picks_redis_when_url_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We don't actually connect — just verify the factory picks the
    right class.  Constructing RedisStore needs the redis pkg, which
    is in the dev install for this repo's coverage check."""
    pytest.importorskip("redis")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    store = make_confirmation_store()
    assert isinstance(store, RedisStore)


# ---------------------------------------------------------------------------
# RedisStore (via fakeredis)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_redis_store(monkeypatch: pytest.MonkeyPatch) -> RedisStore:
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("redis")

    monkeypatch.setattr(
        RedisStore,
        "_make_client",
        lambda self: fakeredis.aioredis.FakeRedis(decode_responses=True),
    )
    return RedisStore("redis://fake")


@pytest.mark.asyncio
async def test_redis_store_register_resolve_round_trips(
    fake_redis_store: RedisStore,
) -> None:
    store = fake_redis_store
    fut = await store.register("rid-r1")
    # Give the subscriber task a tick to actually subscribe.
    await asyncio.sleep(0.05)
    delivered = await store.resolve(
        "rid-r1", ConfirmationOutcome(decision="confirm", note="ok")
    )
    assert delivered is True
    outcome = await asyncio.wait_for(fut, timeout=1.0)
    assert outcome.decision == "confirm"
    assert outcome.note == "ok"
    await store.aclose()


@pytest.mark.asyncio
async def test_redis_store_resolve_unknown_returns_false(
    fake_redis_store: RedisStore,
) -> None:
    store = fake_redis_store
    delivered = await store.resolve("ghost", ConfirmationOutcome(decision="confirm"))
    assert delivered is False
    await store.aclose()


@pytest.mark.asyncio
async def test_redis_store_cancel_drops_subscriber(
    fake_redis_store: RedisStore,
) -> None:
    store = fake_redis_store
    fut = await store.register("rid-r2")
    await asyncio.sleep(0.05)
    await store.cancel("rid-r2")
    assert fut.cancelled()
    await store.aclose()


@pytest.mark.asyncio
async def test_redis_store_double_register_raises(
    fake_redis_store: RedisStore,
) -> None:
    store = fake_redis_store
    await store.register("rid-r3")
    with pytest.raises(ValueError, match="already registered"):
        await store.register("rid-r3")
    await store.aclose()
