"""Resolve credentials for the internal OpenAI-compatible inference endpoint.

Order of resolution (first hit wins), so that nothing secret has to live in
this repo:

1. environment: ``AIRI_API_KEY`` / ``AIRI_BASE_URL`` (or ``OPENAI_API_KEY`` /
   ``OPENAI_BASE_URL`` as the standard aliases);
2. env files, in order: ``$AIRI_ENV_FILE``, ``<repo>/.env``,
   ``~/.config/airi/env``, ``~/.../graphify/.env`` (where the JWT was first
   provisioned).

The token is a JWT with an expiry; `resolve()` decodes the `exp` claim (without
verifying the signature — we are not the issuer) and raises early with a clear
message instead of letting the endpoint answer 401 mid-run.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

from .configs import DEFAULT_BASE_URL, REPO_ROOT

ENV_FILE_CANDIDATES: tuple[Path, ...] = (
    REPO_ROOT / ".env",
    Path.home() / ".config" / "airi" / "env",
    REPO_ROOT.parent / "graphify" / ".env",
)


@dataclass(frozen=True)
class Credentials:
    base_url: str
    api_key: str
    source: str
    expires_at: float | None = None

    @property
    def expires_in_days(self) -> float | None:
        if self.expires_at is None:
            return None
        return (self.expires_at - time.time()) / 86400.0


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = path.read_text()
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip("'\"")
    return out


def _jwt_exp(token: str) -> float | None:
    """Best-effort `exp` claim of a JWT. Returns None for opaque keys."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        pad = "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(parts[1] + pad))
    except Exception:
        return None
    exp = payload.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def resolve(require_valid: bool = True) -> Credentials:
    """Find the endpoint URL + token, or raise with actionable instructions."""
    key = os.environ.get("AIRI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    url = os.environ.get("AIRI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    source = "environment"

    if not key:
        candidates = list(ENV_FILE_CANDIDATES)
        if os.environ.get("AIRI_ENV_FILE"):
            candidates.insert(0, Path(os.environ["AIRI_ENV_FILE"]))
        for path in candidates:
            env = _parse_env_file(path)
            key = env.get("AIRI_API_KEY") or env.get("OPENAI_API_KEY")
            if key:
                url = url or env.get("AIRI_BASE_URL") or env.get("OPENAI_BASE_URL")
                source = str(path)
                break

    if not key:
        tried = "\n  ".join(str(p) for p in ENV_FILE_CANDIDATES)
        raise RuntimeError(
            "No inference token found. Set AIRI_API_KEY (or OPENAI_API_KEY), "
            f"or put it in one of:\n  {tried}"
        )

    exp = _jwt_exp(key)
    if require_valid and exp is not None and exp < time.time():
        raise RuntimeError(
            f"The token from {source} expired at "
            f"{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime(exp))}. "
            "Reissue it before running."
        )

    return Credentials(
        base_url=url or DEFAULT_BASE_URL,
        api_key=key,
        source=source,
        expires_at=exp,
    )
