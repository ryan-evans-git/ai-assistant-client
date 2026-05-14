"""Tests for the SQL / native-async :class:`MemoryStore` backends.

Same coverage philosophy as the transcript / conversation SQL
tests:

1. **Functional / round-trip** against a real ``sqlite3``
   connection (no external DB needed in CI).  Exercises every
   protocol method end-to-end, including the
   per-user-isolation contract.
2. **Dialect correctness** for asyncpg / aiomysql via a
   captured-cursor mock — asserts the SQL we emit uses the
   right placeholder style without standing up a real DB.
"""

from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from ai_assistant_client.persistence import (
    AiomysqlMemoryStore,
    AsyncpgMemoryStore,
    Dialect,
    SqlMemoryStore,
    make_memory_store,
)
from ai_assistant_client.persistence.factory import (
    MEMORY_BACKEND_ENV,
    MEMORY_SQLITE_PATH_ENV,
)


# ---------------------------------------------------------------------------
# Real sqlite — functional round trip
# ---------------------------------------------------------------------------


def _sqlite_conn(tmp_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(tmp_path / "memory.sqlite3", check_same_thread=False)


@pytest.fixture
def sql_store(tmp_path: Path) -> SqlMemoryStore:
    return SqlMemoryStore(_sqlite_conn(tmp_path), dialect=Dialect.SQLITE)


async def test_sql_add_and_get_round_trip(sql_store: SqlMemoryStore) -> None:
    rec = await sql_store.add(
        user_id="alice", key="role", value="data scientist", tags=("work",)
    )
    again = await sql_store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "data scientist"
    assert again.tags == ("work",)


async def test_sql_list_returns_user_memories(
    sql_store: SqlMemoryStore,
) -> None:
    await sql_store.add(user_id="alice", key="a", value=1)
    await sql_store.add(user_id="alice", key="b", value=2)
    await sql_store.add(user_id="bob", key="c", value=3)

    alice_list = await sql_store.list(user_id="alice")
    assert {r.key for r in alice_list} == {"a", "b"}


async def test_sql_list_tag_filter(sql_store: SqlMemoryStore) -> None:
    await sql_store.add(user_id="alice", key="a", value=1, tags=("work",))
    await sql_store.add(user_id="alice", key="b", value=2, tags=("home",))
    work = await sql_store.list(user_id="alice", tags=("work",))
    assert {r.key for r in work} == {"a"}


async def test_sql_list_preserves_insertion_order(
    sql_store: SqlMemoryStore,
) -> None:
    """seq column drives the ORDER BY — 10 inserts must come back
    in insert order regardless of physical storage."""
    for i in range(10):
        await sql_store.add(user_id="alice", key=f"k{i}", value=i)
    rows = await sql_store.list(user_id="alice")
    assert [r.value for r in rows] == list(range(10))


async def test_sql_cross_user_get_raises_keyerror(
    sql_store: SqlMemoryStore,
) -> None:
    """Alice's memory_id is unreachable to Bob even though Bob
    knows the id — same KeyError as 'doesn't exist' so it can't
    be used as an enumeration oracle."""
    rec = await sql_store.add(user_id="alice", key="secret", value=42)
    with pytest.raises(KeyError):
        await sql_store.get(user_id="bob", memory_id=rec.memory_id)


async def test_sql_cross_user_update_raises_keyerror(
    sql_store: SqlMemoryStore,
) -> None:
    rec = await sql_store.add(user_id="alice", key="x", value=1)
    with pytest.raises(KeyError):
        await sql_store.update(
            user_id="bob", memory_id=rec.memory_id, value=99
        )
    again = await sql_store.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == 1


async def test_sql_cross_user_remove_raises_keyerror(
    sql_store: SqlMemoryStore,
) -> None:
    rec = await sql_store.add(user_id="alice", key="x", value=1)
    with pytest.raises(KeyError):
        await sql_store.remove(user_id="bob", memory_id=rec.memory_id)


async def test_sql_update_bumps_updated_at_preserves_created_at(
    sql_store: SqlMemoryStore,
) -> None:
    rec = await sql_store.add(user_id="alice", key="x", value=1)
    upd = await sql_store.update(
        user_id="alice", memory_id=rec.memory_id, value=2
    )
    assert upd.created_at == rec.created_at
    assert upd.updated_at >= rec.updated_at


async def test_sql_remove_then_get_raises(sql_store: SqlMemoryStore) -> None:
    rec = await sql_store.add(user_id="alice", key="x", value=1)
    await sql_store.remove(user_id="alice", memory_id=rec.memory_id)
    with pytest.raises(KeyError):
        await sql_store.get(user_id="alice", memory_id=rec.memory_id)


async def test_sql_forget_all_count_and_isolation(
    sql_store: SqlMemoryStore,
) -> None:
    await sql_store.add(user_id="alice", key="a", value=1)
    await sql_store.add(user_id="alice", key="b", value=2)
    bob_rec = await sql_store.add(user_id="bob", key="b", value=3)

    count = await sql_store.forget_all(user_id="alice")
    assert count == 2
    assert await sql_store.list(user_id="alice") == []

    # Bob's data untouched.
    bob = await sql_store.get(user_id="bob", memory_id=bob_rec.memory_id)
    assert bob.value == 3


async def test_sql_list_users(sql_store: SqlMemoryStore) -> None:
    await sql_store.add(user_id="alice", key="x", value=1)
    await sql_store.add(user_id="bob", key="y", value=2)
    users = await sql_store.list_users()
    assert set(users) == {"alice", "bob"}


async def test_sql_survives_reopen(tmp_path: Path) -> None:
    """Cross-connection durability — same DB file."""
    conn1 = sqlite3.connect(tmp_path / "m.sqlite3", check_same_thread=False)
    s1 = SqlMemoryStore(conn1, dialect=Dialect.SQLITE)
    rec = await s1.add(user_id="alice", key="x", value="hello")
    conn1.close()

    conn2 = sqlite3.connect(tmp_path / "m.sqlite3", check_same_thread=False)
    s2 = SqlMemoryStore(conn2, dialect=Dialect.SQLITE)
    again = await s2.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "hello"


# ---------------------------------------------------------------------------
# Dialect correctness — captured cursor mocks
# ---------------------------------------------------------------------------


class _CapturingCursor:
    def __init__(self, parent: "_CapturingConnection") -> None:
        self._parent = parent

    def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._parent.executes.append((sql, tuple(params or ())))

    def fetchone(self) -> Any:
        return self._parent.next_fetchone.pop(0) if self._parent.next_fetchone else (1,)

    def fetchall(self) -> list[Any]:
        return self._parent.next_fetchall.pop(0) if self._parent.next_fetchall else []

    def close(self) -> None:
        pass


class _CapturingConnection:
    def __init__(self) -> None:
        self.executes: list[tuple[str, tuple[Any, ...]]] = []
        self.next_fetchone: list[Any] = []
        self.next_fetchall: list[list[Any]] = []

    def cursor(self) -> _CapturingCursor:
        return _CapturingCursor(self)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


@pytest.mark.parametrize(
    "dialect,expected,forbidden",
    [
        (Dialect.SQLITE, "?", "%s"),
        (Dialect.POSTGRESQL, "%s", "?"),
        (Dialect.MYSQL, "%s", "?"),
    ],
)
async def test_sql_memory_uses_dialect_placeholders(
    dialect: Dialect, expected: str, forbidden: str
) -> None:
    conn = _CapturingConnection()
    store = SqlMemoryStore(conn, dialect=dialect)
    await store.add(user_id="alice", key="x", value=1)

    inserts = [
        sql for sql, _ in conn.executes if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert expected in inserts[0]
    assert forbidden not in inserts[0]


# ---------------------------------------------------------------------------
# asyncpg memory store — mock pool dialect check + isolation
# ---------------------------------------------------------------------------


class _AsyncpgMockConnection:
    def __init__(self, pool: "_AsyncpgMockPool") -> None:
        self._pool = pool

    async def execute(self, sql: str, *args: Any) -> None:
        self._pool.calls.append(("execute", sql, args))

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self._pool.calls.append(("fetchrow", sql, args))
        return (
            self._pool.next_fetchrow.pop(0)
            if self._pool.next_fetchrow
            else None
        )

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self._pool.calls.append(("fetch", sql, args))
        return self._pool.next_fetch.pop(0) if self._pool.next_fetch else []

    @asynccontextmanager
    async def transaction(self):  # type: ignore[no-untyped-def]
        yield


class _AsyncpgMockPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.next_fetchrow: list[Any] = []
        self.next_fetch: list[list[Any]] = []

    @asynccontextmanager
    async def acquire(self):  # type: ignore[no-untyped-def]
        yield _AsyncpgMockConnection(self)


async def test_asyncpg_memory_uses_dollar_placeholders() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetchrow.append([1])  # next-seq lookup
    store = AsyncpgMemoryStore(pool)
    await store.add(user_id="alice", key="x", value=1)

    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert "$1" in inserts[0]
    assert "?" not in inserts[0]


async def test_asyncpg_memory_cross_user_get_raises_keyerror() -> None:
    pool = _AsyncpgMockPool()
    # First call: get's SELECT returns None (wrong-owner check).
    pool.next_fetchrow.append(None)
    store = AsyncpgMemoryStore(pool)
    with pytest.raises(KeyError):
        await store.get(user_id="bob", memory_id="mem_alice_only")


# ---------------------------------------------------------------------------
# aiomysql memory store — mock pool dialect check
# ---------------------------------------------------------------------------


class _AiomysqlMockCursor:
    def __init__(self, pool: "_AiomysqlMockPool") -> None:
        self._pool = pool
        self._current_result: Any = None
        self._current_result_all: list[Any] = []

    async def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> None:
        self._pool.calls.append(("execute", sql, tuple(params or ())))
        # Only pop from the queues when the statement is a SELECT.
        # DDL (CREATE TABLE / CREATE INDEX) and INSERT / UPDATE /
        # DELETE don't call fetch* afterwards, so consuming the
        # queue on those would starve the next real SELECT.
        sql_head = sql.lstrip().lower().split(None, 1)
        is_select = bool(sql_head) and sql_head[0] == "select"
        if is_select:
            self._current_result = (
                self._pool.next_fetchone.pop(0)
                if self._pool.next_fetchone
                else None
            )
            self._current_result_all = (
                self._pool.next_fetchall.pop(0)
                if self._pool.next_fetchall
                else []
            )

    async def fetchone(self) -> Any:
        return self._current_result

    async def fetchall(self) -> list[Any]:
        return self._current_result_all

    async def __aenter__(self) -> "_AiomysqlMockCursor":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None


class _AiomysqlMockConnection:
    def __init__(self, pool: "_AiomysqlMockPool") -> None:
        self._pool = pool

    def cursor(self) -> _AiomysqlMockCursor:
        return _AiomysqlMockCursor(self._pool)

    async def commit(self) -> None:
        self._pool.commits += 1


class _AiomysqlMockPool:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[Any, ...]]] = []
        self.next_fetchone: list[Any] = []
        self.next_fetchall: list[list[Any]] = []
        self.commits = 0

    @asynccontextmanager
    async def acquire(self):  # type: ignore[no-untyped-def]
        yield _AiomysqlMockConnection(self)


async def test_aiomysql_memory_uses_percent_s_placeholders() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append([1])
    store = AiomysqlMemoryStore(pool)
    await store.add(user_id="alice", key="x", value=1)

    inserts = [
        sql for _, sql, _ in pool.calls if sql.lstrip().startswith("INSERT")
    ]
    assert inserts
    assert "%s" in inserts[0]
    assert "?" not in inserts[0]


async def test_aiomysql_memory_cross_user_remove_raises_keyerror() -> None:
    pool = _AiomysqlMockPool()
    # SELECT-1 check returns None → wrong owner.
    pool.next_fetchone.append(None)
    store = AiomysqlMemoryStore(pool)
    with pytest.raises(KeyError):
        await store.remove(user_id="bob", memory_id="mem_alice_only")


# ---------------------------------------------------------------------------
# asyncpg memory — coverage for read / update / remove / forget paths
# ---------------------------------------------------------------------------


async def test_asyncpg_memory_get_round_trip() -> None:
    pool = _AsyncpgMockPool()
    # get's SELECT returns one row: (mkey, value_json, tags_json, created, updated)
    pool.next_fetchrow.append(
        ("role", '"data scientist"', '["work"]', "t0", "t0")
    )
    store = AsyncpgMemoryStore(pool)
    record = await store.get(user_id="alice", memory_id="mem_x")
    assert record.value == "data scientist"
    assert record.tags == ("work",)


async def test_asyncpg_memory_list_with_tag_filter_unions() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetch.append(
        [
            ("m1", "a", '1', '["work"]', "t0", "t0"),
            ("m2", "b", '2', '["home"]', "t0", "t0"),
            ("m3", "c", '3', '["work", "urgent"]', "t0", "t0"),
        ]
    )
    store = AsyncpgMemoryStore(pool)
    rows = await store.list(user_id="alice", tags=("work",))
    keys = {r.key for r in rows}
    assert keys == {"a", "c"}


async def test_asyncpg_memory_list_no_tag_filter_returns_all() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetch.append(
        [
            ("m1", "a", '1', '[]', "t0", "t0"),
            ("m2", "b", '2', '[]', "t0", "t0"),
        ]
    )
    store = AsyncpgMemoryStore(pool)
    rows = await store.list(user_id="alice")
    assert {r.key for r in rows} == {"a", "b"}


async def test_asyncpg_memory_update_bumps_updated_at() -> None:
    pool = _AsyncpgMockPool()
    # update's SELECT returns (mkey, tags_json, created_at)
    pool.next_fetchrow.append(("x", '[]', "t0"))
    store = AsyncpgMemoryStore(pool)
    rec = await store.update(user_id="alice", memory_id="mem_x", value=99)
    assert rec.value == 99
    assert rec.created_at == "t0"
    assert rec.updated_at != "t0"


async def test_asyncpg_memory_update_unknown_raises_keyerror() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetchrow.append(None)
    store = AsyncpgMemoryStore(pool)
    with pytest.raises(KeyError):
        await store.update(user_id="alice", memory_id="missing", value=1)


async def test_asyncpg_memory_remove_round_trip() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetchrow.append((1,))  # SELECT 1 finds the row
    store = AsyncpgMemoryStore(pool)
    await store.remove(user_id="alice", memory_id="mem_x")
    # Last execute should be the DELETE.
    deletes = [c for c in pool.calls if c[0] == "execute" and "DELETE" in c[1]]
    assert deletes


async def test_asyncpg_memory_forget_all_returns_count() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetchrow.append((7,))  # COUNT(*) result
    store = AsyncpgMemoryStore(pool)
    count = await store.forget_all(user_id="alice")
    assert count == 7


async def test_asyncpg_memory_list_users() -> None:
    pool = _AsyncpgMockPool()
    pool.next_fetch.append([("alice",), ("bob",)])
    store = AsyncpgMemoryStore(pool)
    users = await store.list_users()
    assert set(users) == {"alice", "bob"}


# ---------------------------------------------------------------------------
# aiomysql memory — coverage for read / update / remove / forget paths
# ---------------------------------------------------------------------------


async def test_aiomysql_memory_get_round_trip() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append(
        ("role", '"data scientist"', '["work"]', "t0", "t0")
    )
    store = AiomysqlMemoryStore(pool)
    record = await store.get(user_id="alice", memory_id="mem_x")
    assert record.value == "data scientist"
    assert record.tags == ("work",)


async def test_aiomysql_memory_list_with_tag_filter_unions() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchall.append(
        [
            ("m1", "a", '1', '["work"]', "t0", "t0"),
            ("m2", "b", '2', '["home"]', "t0", "t0"),
            ("m3", "c", '3', '["work", "urgent"]', "t0", "t0"),
        ]
    )
    store = AiomysqlMemoryStore(pool)
    rows = await store.list(user_id="alice", tags=("work",))
    assert {r.key for r in rows} == {"a", "c"}


async def test_aiomysql_memory_list_no_tag_filter() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchall.append(
        [
            ("m1", "a", '1', '[]', "t0", "t0"),
            ("m2", "b", '2', '[]', "t0", "t0"),
        ]
    )
    store = AiomysqlMemoryStore(pool)
    rows = await store.list(user_id="alice")
    assert len(rows) == 2


async def test_aiomysql_memory_update_bumps_updated_at() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append(("x", '[]', "t0"))
    store = AiomysqlMemoryStore(pool)
    rec = await store.update(user_id="alice", memory_id="mem_x", value=99)
    assert rec.value == 99


async def test_aiomysql_memory_update_unknown_raises_keyerror() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append(None)
    store = AiomysqlMemoryStore(pool)
    with pytest.raises(KeyError):
        await store.update(user_id="alice", memory_id="missing", value=1)


async def test_aiomysql_memory_remove_round_trip() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append((1,))
    store = AiomysqlMemoryStore(pool)
    await store.remove(user_id="alice", memory_id="mem_x")
    deletes = [c for c in pool.calls if "DELETE" in c[1]]
    assert deletes


async def test_aiomysql_memory_forget_all_returns_count() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchone.append((4,))
    store = AiomysqlMemoryStore(pool)
    count = await store.forget_all(user_id="alice")
    assert count == 4


async def test_aiomysql_memory_list_users() -> None:
    pool = _AiomysqlMockPool()
    pool.next_fetchall.append([("alice",), ("bob",)])
    store = AiomysqlMemoryStore(pool)
    users = await store.list_users()
    assert set(users) == {"alice", "bob"}


# ---------------------------------------------------------------------------
# Factory sqlite path
# ---------------------------------------------------------------------------


def test_factory_memory_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(MEMORY_BACKEND_ENV, "sqlite")
    monkeypatch.setenv(MEMORY_SQLITE_PATH_ENV, str(tmp_path / "m.sqlite3"))
    store = make_memory_store()
    assert isinstance(store, SqlMemoryStore)


async def test_factory_memory_sqlite_actually_writes(tmp_path: Path) -> None:
    """End-to-end through the factory: build, write, reopen, read."""
    path = tmp_path / "f.sqlite3"
    store = make_memory_store(kind="sqlite", sqlite_path=str(path))
    rec = await store.add(user_id="alice", key="role", value="dev")

    store2 = make_memory_store(kind="sqlite", sqlite_path=str(path))
    again = await store2.get(user_id="alice", memory_id=rec.memory_id)
    assert again.value == "dev"
