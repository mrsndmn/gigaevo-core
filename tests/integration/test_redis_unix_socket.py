"""Real-Redis integration smoke for unix-socket connections.

Runs against a Redis listening on the unix socket named by
``GIGAEVO_REDIS_SOCKET``; skipped automatically when that variable is unset
or nothing answers on it, so the test is safe to commit and harmless on
machines with a TCP-only server.

fakeredis cannot cover this: the point is precisely that redis-py accepts a
``unix://`` URL alongside the pool options ``RedisConnection`` always passes
(``max_connections``, ``health_check_interval``, ``socket_connect_timeout``,
``socket_timeout``, ``retry_on_timeout``). A ``unix://`` URL that parses fine
but blows up on those keyword arguments would leave socket mode broken while
every unit test still passed.

Start a suitable server with::

    redis-server --port 0 --unixsocket /tmp/gigaevo-test.sock
    export GIGAEVO_REDIS_SOCKET=/tmp/gigaevo-test.sock
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest
import redis.asyncio as redis_async

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import Program
from gigaevo.utils.redis_conn import SOCKET_ENV, build_redis_url


# Captured at import time on purpose. The autouse conftest fixture deletes
# SOCKET_ENV from os.environ so unit tests see TCP defaults, and it runs before
# these tests do — reading the variable inside a test would always come back
# empty and silently skip the whole module.
_SOCKET_PATH = (os.environ.get(SOCKET_ENV) or "").strip() or None


def _socket_path() -> str | None:
    return _SOCKET_PATH


async def _reachable(url: str) -> bool:
    try:
        r = redis_async.from_url(url, socket_connect_timeout=1)
        try:
            await r.ping()
        finally:
            await r.aclose()
        return True
    except Exception:
        return False


@pytest.fixture
async def socket_storage():
    """RedisProgramStorage bound to the unix socket, under a unique prefix."""
    path = _socket_path()
    if path is None:
        pytest.skip(f"{SOCKET_ENV} not set")

    url = build_redis_url(socket=path)
    assert url.startswith("unix://"), url
    if not await _reachable(url):
        pytest.skip(f"no Redis answering on {path}")

    prefix = f"socket-test-{uuid.uuid4().hex[:8]}"
    storage = RedisProgramStorage(
        RedisProgramStorageConfig(redis_url=url, key_prefix=prefix)
    )
    try:
        yield storage, prefix
    finally:
        r = redis_async.from_url(url)
        try:
            keys = [k async for k in r.scan_iter(match=f"{prefix}:*")]
            if keys:
                await r.delete(*keys)
        finally:
            await r.aclose()
        with contextlib.suppress(Exception):
            await storage.close()


class TestUnixSocketStorage:
    async def test_round_trip_over_socket(self, socket_storage) -> None:
        storage, _ = socket_storage
        program = Program(code="def entrypoint():\n    return 1\n")

        await storage.add(program)
        loaded = await storage.get(program.id)

        assert loaded is not None
        assert loaded.id == program.id
        assert loaded.code == program.code

    async def test_db_index_from_url_query_is_honoured(self) -> None:
        # The db travels as a query parameter in unix:// URLs rather than as a
        # path segment, so a value written to db 3 must be invisible on db 0.
        path = _socket_path()
        if path is None:
            pytest.skip(f"{SOCKET_ENV} not set")
        url_db3 = build_redis_url(db=3, socket=path)
        if not await _reachable(url_db3):
            pytest.skip(f"no Redis answering on {path}")

        key = f"socket-db-probe-{uuid.uuid4().hex[:8]}"
        r3 = redis_async.from_url(url_db3)
        r0 = redis_async.from_url(build_redis_url(db=0, socket=path))
        try:
            await r3.set(key, "written-to-db-3")
            assert await r3.get(key) == b"written-to-db-3"
            assert await r0.get(key) is None
        finally:
            with contextlib.suppress(Exception):
                await r3.delete(key)
            await r3.aclose()
            await r0.aclose()
