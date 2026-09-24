"""Endpoint client for the probe.

A copy — not an import — of the request/retry loop in
`experiments.noregeo_synth.recovery_eval.request`, so that `geostate` depends
on `noregeo_synth` only through the small allowlist in `tests/geostate/
test_boundaries.py` (ADR 0004). Credentials are still resolved by the shared
`credentials` module: the token lives outside the repo and expires.

Semantics kept from the original: every request is cached by directory; a
directory with a manifest but no result is an interrupted request and is
refused rather than silently re-rolled; only transient transport errors are
retried, with exponential backoff capped by the config.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from experiments.noregeo_synth import credentials as _credentials

from .configs import INTERMEDIATE_CERT, OUT_DIR, TLS_BUNDLE_NAME, E0Config


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # unique per process so two writers of one file never share a temp name
    temporary = path.with_suffix(f"{path.suffix}.tmp{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def is_retryable(exc: Exception) -> bool:
    """Transient transport shapes only; auth/config/bad-request errors are not."""
    import openai

    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            openai.AuthenticationError,
            openai.PermissionDeniedError,
            openai.BadRequestError,
            openai.NotFoundError,
        ),
    ):
        return False
    return isinstance(
        exc,
        (
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APIResponseValidationError,
            json.JSONDecodeError,
        ),
    )


def ca_bundle(directory: str | Path | None = None) -> Path:
    """certifi's roots plus the endpoint's missing intermediate, written once."""
    import certifi

    target = Path(directory or OUT_DIR) / TLS_BUNDLE_NAME
    content = Path(certifi.where()).read_text() + "\n" + INTERMEDIATE_CERT.read_text()
    if not target.exists() or target.read_text() != content:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(content)
        tmp.replace(target)
    return target


def make_client(cfg: E0Config, ca_dir: str | Path | None = None):
    """OpenAI client for the endpoint; verifies TLS against the bundle.

    `SSL_CERT_FILE`, when set (the MLS launchers set it), wins over the bundle
    written next to the run.
    """
    import ssl

    import httpx
    from openai import OpenAI

    creds = _credentials.resolve(expected_base_url=cfg.base_url)
    cafile = os.environ.get("SSL_CERT_FILE") or str(ca_bundle(ca_dir))
    context = ssl.create_default_context(cafile=cafile)
    client = OpenAI(
        base_url=creds.base_url,
        api_key=creds.api_key,
        timeout=cfg.request_timeout,
        max_retries=0,  # the loop below owns every retry
        http_client=httpx.Client(verify=context, timeout=cfg.request_timeout),
    )
    return client, creds


def seed_for(cfg: E0Config, salt: str) -> int:
    digest = hashlib.sha256(f"{cfg.seed}:{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def request(
    client,
    cfg: E0Config,
    model: str,
    messages: list[dict[str, Any]],
    directory: str | Path,
    salt: str,
) -> dict[str, Any]:
    """One chat completion, cached at `directory/result.json`."""
    directory = Path(directory)
    cached = directory / "result.json"
    if cached.exists():
        return read_json(cached)
    directory.mkdir(parents=True, exist_ok=True)
    if (
        (directory / "request.json").exists()
        or list(directory.glob("raw_*.json"))
        or list(directory.glob("error_*.json"))
    ):
        raise RuntimeError(f"incomplete request at {directory}; refusing silent reroll")
    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        seed=seed_for(cfg, salt),
    )
    if cfg.reasoning_effort is not None:
        kwargs["extra_body"] = {"reasoning_effort": cfg.reasoning_effort}
    write_json(directory / "request.json", kwargs)
    started = time.monotonic()
    for attempt in range(cfg.max_transport_retries + 1):
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - every failure is recorded
            write_json(
                directory / f"error_{attempt:02d}.json",
                {"type": type(exc).__name__, "error": str(exc)},
            )
            if is_retryable(exc) and attempt < cfg.max_transport_retries:
                time.sleep(min(2 ** (attempt + 1), cfg.retry_backoff_cap))
                continue
            result = {
                "error": f"{type(exc).__name__}: {exc}",
                "content": None,
                "wall_time_s": time.monotonic() - started,
            }
            write_json(cached, result)
            return result
        write_json(directory / f"raw_{attempt:02d}.json", response.model_dump())
        choice = response.choices[0]
        result = {
            "content": choice.message.content or "",
            "error": None,
            "finish_reason": choice.finish_reason,
            "usage": response.usage.model_dump() if response.usage else {},
            "wall_time_s": time.monotonic() - started,
        }
        write_json(cached, result)
        return result
    raise AssertionError("unreachable")
