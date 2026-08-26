"""Redis connection targets: TCP or unix domain socket.

GigaEvo opens Redis connections in two shapes: URLs handed to
``redis.asyncio.from_url`` and keyword arguments handed to ``redis.Redis``.
Both are produced here, so a single switch routes every connection in the
process over a unix socket instead of TCP:

* the ``redis.socket`` key in the Hydra config (``run.py`` and everything
  instantiated from ``config/``), or
* the ``GIGAEVO_REDIS_SOCKET`` environment variable, which also covers the
  call sites that hardcode a default URL (endpoint pool, balanced chat,
  ``tools/``) and therefore have no config to read.

The environment variable exists because a half-converted process is worse
than an unconverted one: with a socket-only Redis (``port 0`` plus
``unixsocket`` in ``redis.conf``) any connection still built from host/port
fails outright instead of silently reaching some other server.

Access control differs between the two modes. A TCP listener on loopback is
reachable by every local user, so restricting it needs ``requirepass``. A
unix socket is a filesystem object: placing it in a directory with mode
0700 limits it to the owning user, with no password anywhere in the config.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

__all__ = [
    "SOCKET_ENV",
    "build_redis_url",
    "redis_connect_kwargs",
    "resolve_socket_path",
]

SOCKET_ENV = "GIGAEVO_REDIS_SOCKET"

DEFAULT_HOST = "localhost"
DEFAULT_PORT = 6379
DEFAULT_DB = 0


def resolve_socket_path(socket: str | None = None) -> str | None:
    """Return the unix socket path to use, or ``None`` for TCP.

    An explicit ``socket`` argument wins; otherwise ``$GIGAEVO_REDIS_SOCKET``
    is consulted. Blank values in either place mean "not set" so that
    ``socket: null`` in YAML and ``GIGAEVO_REDIS_SOCKET=`` both fall back to
    TCP rather than producing a ``unix://`` URL with an empty path.
    """
    candidate = socket if socket is not None else os.environ.get(SOCKET_ENV)
    if candidate is None:
        return None
    candidate = str(candidate).strip()
    if not candidate:
        return None
    return str(Path(candidate).expanduser())


def build_redis_url(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db: int = DEFAULT_DB,
    socket: str | None = None,
) -> str:
    """Build a redis-py connection URL for ``redis.asyncio.from_url``.

    Returns ``unix://<path>?db=<db>`` when a socket is configured and
    ``redis://<host>:<port>/<db>`` otherwise. The socket path is percent
    encoded so that paths containing characters with meaning in a URL
    (``?``, ``#``, spaces) survive redis-py's ``urlparse`` round trip.
    """
    sock = resolve_socket_path(socket)
    if sock:
        return f"unix://{quote(sock)}?db={int(db)}"
    return f"redis://{host}:{int(port)}/{int(db)}"


def redis_connect_kwargs(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    db: int = DEFAULT_DB,
    socket: str | None = None,
) -> dict[str, Any]:
    """Build keyword arguments for a direct ``redis.Redis(...)`` constructor.

    ``unix_socket_path`` and ``host``/``port`` are mutually exclusive in
    redis-py: passing the former selects ``UnixDomainSocketConnection`` and
    the latter are ignored. Only the applicable pair is returned so the
    caller cannot accidentally imply a TCP target that is never used.
    """
    sock = resolve_socket_path(socket)
    if sock:
        return {"unix_socket_path": sock, "db": int(db)}
    return {"host": host, "port": int(port), "db": int(db)}
