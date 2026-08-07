"""Unit tests for TCP/unix-socket connection target selection."""

from __future__ import annotations

import pytest

from gigaevo.utils.redis_conn import (
    SOCKET_ENV,
    build_redis_url,
    redis_connect_kwargs,
    resolve_socket_path,
)


@pytest.fixture(autouse=True)
def _clear_socket_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep an ambient GIGAEVO_REDIS_SOCKET out of the assertions."""
    monkeypatch.delenv(SOCKET_ENV, raising=False)


class TestResolveSocketPath:
    def test_none_without_env_means_tcp(self) -> None:
        assert resolve_socket_path() is None

    def test_explicit_path_wins_over_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SOCKET_ENV, "/from/env.sock")
        assert resolve_socket_path("/explicit.sock") == "/explicit.sock"

    def test_env_used_when_no_explicit_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SOCKET_ENV, "/from/env.sock")
        assert resolve_socket_path() == "/from/env.sock"

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_means_tcp_not_empty_socket(self, blank: str) -> None:
        # `socket: null` in YAML arrives as None, but an empty string from a
        # shell (`GIGAEVO_REDIS_SOCKET=`) must not yield "unix://?db=0".
        assert resolve_socket_path(blank) is None

    def test_blank_env_falls_back_to_tcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(SOCKET_ENV, "")
        assert resolve_socket_path() is None

    def test_tilde_is_expanded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOME", "/home/someone")
        assert resolve_socket_path("~/redis.sock") == "/home/someone/redis.sock"


class TestBuildRedisUrl:
    def test_tcp_default(self) -> None:
        assert build_redis_url() == "redis://localhost:6379/0"

    def test_tcp_explicit(self) -> None:
        assert build_redis_url("example", 6380, 3) == "redis://example:6380/3"

    def test_socket_url_carries_db_as_query(self) -> None:
        assert (
            build_redis_url(socket="/run/redis.sock") == "unix:///run/redis.sock?db=0"
        )

    def test_socket_overrides_host_and_port(self) -> None:
        url = build_redis_url("example", 6380, 7, "/run/redis.sock")
        assert url == "unix:///run/redis.sock?db=7"

    def test_socket_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SOCKET_ENV, "/run/redis.sock")
        assert build_redis_url(db=2) == "unix:///run/redis.sock?db=2"

    def test_socket_path_is_percent_encoded(self) -> None:
        # A '?' in the path would otherwise swallow the db query parameter.
        url = build_redis_url(socket="/run/odd?name.sock")
        assert url == "unix:///run/odd%3Fname.sock?db=0"

    def test_string_db_is_coerced(self) -> None:
        # OmegaConf can hand the resolver stringified numbers.
        assert build_redis_url("h", "6379", "4") == "redis://h:6379/4"  # type: ignore[arg-type]


class TestRedisConnectKwargs:
    def test_tcp_shape(self) -> None:
        assert redis_connect_kwargs("h", 6380, 2) == {
            "host": "h",
            "port": 6380,
            "db": 2,
        }

    def test_socket_shape_omits_host_and_port(self) -> None:
        # redis-py ignores host/port when unix_socket_path is given; emitting
        # them anyway would suggest a TCP target that is never contacted.
        assert redis_connect_kwargs("h", 6380, 2, "/run/redis.sock") == {
            "unix_socket_path": "/run/redis.sock",
            "db": 2,
        }

    def test_socket_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(SOCKET_ENV, "/run/redis.sock")
        assert redis_connect_kwargs(db=5) == {
            "unix_socket_path": "/run/redis.sock",
            "db": 5,
        }
