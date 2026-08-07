"""Load Program objects from a live Redis DB produced by a GigaEvo run."""

from __future__ import annotations

import asyncio

from gigaevo.database.redis import RedisProgramStorageConfig
from gigaevo.database.redis_program_storage import RedisProgramStorage
from gigaevo.programs.program import EXCLUDE_STAGE_RESULTS, Program
from gigaevo.utils.redis_conn import build_redis_url


def load_programs_from_redis(
    *,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    prefix: str = "",
    socket: str | None = None,
) -> list[Program]:
    """Load all programs from a live Redis DB (read-only, no lock acquisition).

    Args:
        host: Redis server hostname.
        port: Redis server port.
        db: Redis database index (0-15).
        socket: Unix socket path; overrides host/port when set. Defaults to
                $GIGAEVO_REDIS_SOCKET.
        prefix: Key prefix matching the run's ``problem.name``
                (e.g. ``"chains/hotpotqa/static"``).

    Returns:
        All programs stored in the DB, with stage_results excluded for speed.
    """
    redis_url = build_redis_url(host, port, db, socket)
    config = RedisProgramStorageConfig(
        redis_url=redis_url,
        key_prefix=prefix,
        read_only=True,
    )

    async def _load() -> list[Program]:
        async with RedisProgramStorage(config) as storage:
            return await storage.get_all(exclude=EXCLUDE_STAGE_RESULTS)

    return asyncio.run(_load())
