"""Shared pytest fixtures + cleanup helpers.

Currently just one helper: a sqlite connection factory that
tracks every connection it hands out and closes them on
teardown.  This eliminates the ``ResourceWarning: unclosed
database`` flood that the SQL store tests would otherwise
produce — those connections sit on a thread pool's reference
chain and only get GC'd at process exit, by which point pytest
has already reported each one as an unraisable warning.

Why a factory (rather than a single-connection fixture):
several tests deliberately open a *second* connection to the
same DB file to verify cross-connection durability of the
on-disk format.  A single-connection fixture would force those
tests to bypass it, so they'd still leak.  The factory yields
new connections on demand and tracks them all centrally.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Callable, Iterator

import pytest


@pytest.fixture
def make_sqlite_conn(
    tmp_path: Path,
) -> Iterator[Callable[..., sqlite3.Connection]]:
    """Yield a function that opens a sqlite connection and
    auto-closes on teardown.

    Usage::

        def test_x(make_sqlite_conn):
            conn = make_sqlite_conn()                # default name
            conn2 = make_sqlite_conn("alt.sqlite3")  # different DB

    Both connections are closed by the fixture finalizer.  Each
    call defaults to ``<tmp_path>/test.sqlite3`` so tests that
    don't need multiple files just call with no arguments.
    """
    opened: list[sqlite3.Connection] = []

    def _open(name: str = "test.sqlite3") -> sqlite3.Connection:
        path = tmp_path / name
        conn = sqlite3.connect(path, check_same_thread=False)
        opened.append(conn)
        return conn

    yield _open

    for conn in opened:
        try:
            conn.close()
        except Exception:
            # A test may have closed the connection itself
            # (e.g. the cross-connection durability tests); skip
            # so cleanup of remaining connections still runs.
            pass
