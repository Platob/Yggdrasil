"""Local Ollama -backed :class:`TokenEngine`.

Talks to a local `Ollama <https://ollama.com>`_ server (default
``http://localhost:11434``, override with ``OLLAMA_HOST``) over its native
chat API — so any open model you've ``ollama pull``-ed runs on this machine,
free and private. Available only when the server answers.

Every call rides the project's :class:`~yggdrasil.http_.HTTPSession` (its
connection pooling, retry budget, and response parsing) — no bespoke HTTP — so
probes get a quick single-shot waiting profile and the real calls reuse the
shared pool.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, ClassVar, Iterator, Optional

from yggdrasil.dataclasses.waiting import WaitingConfig

from ..engine import DEFAULT_MAX_TOKENS, Completion
from .local import LocalEngine

__all__ = ["OllamaEngine"]

#: Quick, single-shot waiting profile for liveness probes — short timeout.
_PROBE = WaitingConfig(timeout=2.0, retries=0, max_attempts=1)

#: Seconds an ``available()`` probe result is trusted before re-checking. A
#: single ``ygg loki`` command asks ``available()`` several times (engine()
#: + engines() + per-turn select()); the probe is a network round-trip to the
#: Ollama server, so caching it collapses that burst into one call — the bulk
#: of ``ygg loki`` startup latency when a local model server is configured.
#: Short enough that a server started later in a long REPL is still detected.
_PROBE_TTL = 30.0

#: A no-retry HTTPSession for liveness probes (cached). The shared session
#: retries a refused connection ~8× with backoff — right for real calls, but it
#: would stall ``available()`` for tens of seconds when Ollama isn't running, so
#: a probe gets its own zero-retry policy. Still an HTTPSession — all HTTP stays
#: centralized there.
_PROBE_SESSION: "Any" = None


def _probe_session() -> "Any":
    global _PROBE_SESSION
    if _PROBE_SESSION is None:
        from yggdrasil.http_ import HTTPSession

        class _NoRetrySession(HTTPSession):
            def _build_retry(self):
                return super()._build_retry().new(
                    total=0, connect=0, read=0, status=0, other=0,
                )

        _PROBE_SESSION = _NoRetrySession()
    return _PROBE_SESSION


class OllamaEngine(LocalEngine):
    """Reason with a local model served by Ollama, sized to the workstation."""

    name = "ollama"
    #: Fallback when the resource tier isn't in the ladder.
    default_model: ClassVar[str] = "qwen2.5:3b"
    #: Resource tier → Qwen2.5 instruct model (Apache-2.0, strong, ``ollama
    #: pull``-able). A modest CPU box gets 3B; more RAM/GPU climbs the ladder.
    RESOURCE_MODELS: ClassVar[dict[str, str]] = {
        "small": "qwen2.5:3b",     # ≥ 8 GB CPU — the lightweight bootstrap
        "medium": "qwen2.5:7b",    # ≥ 16 GB
        "large": "qwen2.5:14b",    # ≥ 32 GB
        "xlarge": "qwen2.5:32b",   # CUDA GPU
    }

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None,
                 host: Optional[str] = None) -> None:
        super().__init__(model=model or os.getenv("YGG_LOKI_OLLAMA_MODEL"), tier=tier)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        #: Memoized liveness probe — (monotonic timestamp, result), TTL-bounded.
        self._probe: "Optional[tuple[float, bool]]" = None

    def _session(self, *, probe: bool = False):
        """The HTTPSession to use — pooling, retry, and parsing in one place.

        ``probe=True`` returns the zero-retry probe session for liveness checks
        so they fail fast; the default is the shared, retrying session.
        """
        if probe:
            return _probe_session()
        from yggdrasil.http_ import HTTPSession

        return HTTPSession()

    def available(self) -> bool:
        now = time.monotonic()
        if self._probe is not None and now - self._probe[0] < _PROBE_TTL:
            return self._probe[1]
        try:
            resp = self._session(probe=True).get(f"{self.host}/api/tags",
                                                 raise_error=False, wait=_PROBE)
            ok = resp.status_code == 200
        except Exception:
            ok = False
        self._probe = (now, ok)
        return ok

    # -- lazy model install ------------------------------------------------

    def installed_models(self) -> list[str]:
        """Models already pulled onto this Ollama server (empty if unreachable)."""
        try:
            tags = self._session(probe=True).get(f"{self.host}/api/tags",
                                                 raise_error=False, wait=_PROBE).json()
        except Exception:
            return []
        return [m.get("name", "") for m in tags.get("models", []) if m.get("name")]

    def has_model(self, model: str) -> bool:
        """True when *model* (with or without a ``:tag``) is already pulled."""
        names = self.installed_models()
        return model in names or f"{model}:latest" in names or any(
            n.split(":")[0] == model for n in names
        )

    def pull(self, model: Optional[str] = None, *, timeout: float = 1800.0,
             on_progress: "Optional[Callable[[dict[str, Any]], None]]" = None) -> str:
        """Download *model* onto the Ollama server (lazy — the heavy bit).

        Defaults to :attr:`bootstrap_model`, the lightweight brain. Without
        *on_progress* this does the non-streaming pull (one final status object).
        With *on_progress* it **streams** Ollama's NDJSON progress events —
        ``{"status", "completed", "total"}`` per chunk — so a caller can render a
        live download bar; the final status string is returned either way.
        """
        model = model or self.bootstrap_model
        wait = WaitingConfig(timeout=timeout, retries=0, max_attempts=1)
        if on_progress is None:
            resp = self._session().post(
                f"{self.host}/api/pull", json={"name": model, "stream": False}, wait=wait,
            )
            return resp.json().get("status", "unknown")
        # Streamed pull: leave the body un-preloaded (``send_config`` stream) and
        # parse the NDJSON event log as it arrives, reporting each progress tick.
        resp = self._session().post(
            f"{self.host}/api/pull", json={"name": model, "stream": True},
            send_config={"stream": True}, wait=wait,
        )
        status = "unknown"
        for event in _iter_ndjson(resp):
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            status = event.get("status", status)
            on_progress(event)
        return status

    def ensure(self, model: Optional[str] = None,
               on_progress: "Optional[Callable[[dict[str, Any]], None]]" = None) -> dict[str, Any]:
        """Make sure *model* is available, pulling it only if missing.

        Returns ``{"model", "was_present", "status"}`` — the lazy-install
        receipt so a caller (the ``setup`` skill) can report what it did. An
        *on_progress* callback streams the pull's download progress.
        """
        model = model or self.bootstrap_model
        if self.has_model(model):
            return {"model": model, "was_present": True, "status": "already installed"}
        return {"model": model, "was_present": False,
                "status": self.pull(model, on_progress=on_progress)}

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ) -> Completion:
        model = self.resolve_model(messages=messages, system=system, tier=tier)
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        resp = self._session().post(
            f"{self.host}/api/chat",
            json={"model": model, "messages": msgs, "stream": False,
                  "options": {"num_predict": min(max_tokens, 1024)}},
            wait=WaitingConfig(timeout=180.0, retries=0, max_attempts=1),
        )
        data = resp.json()
        text = data.get("message", {}).get("content", "")
        self._record(model,
                     input_tokens=data.get("prompt_eval_count"),
                     output_tokens=data.get("eval_count"),
                     messages=messages, system=system, text=text)
        return Completion(text=text, model=model, raw=data)


def _iter_ndjson(resp: Any) -> "Iterator[dict[str, Any]]":
    """Yield JSON objects from a newline-delimited streaming response body.

    Ollama's ``/api/pull`` (and ``/api/generate``) emit one JSON object per
    line as the download proceeds. Reads the body incrementally through the
    response's ``.stream()`` chunks, buffering across chunk boundaries, and
    skips any blank or partial-trailing fragment that doesn't parse.
    """
    buffer = b""
    for chunk in resp.stream():
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    tail = buffer.strip()
    if tail:
        try:
            yield json.loads(tail)
        except json.JSONDecodeError:
            pass
