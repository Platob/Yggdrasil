"""Local Ollama -backed :class:`TokenEngine`.

Talks to a local `Ollama <https://ollama.com>`_ server (default
``http://localhost:11434``, override with ``OLLAMA_HOST``) over its native
chat API — so any open model you've ``ollama pull``-ed runs on this machine,
free and private. Available only when the server answers. Uses stdlib HTTP so
it needs no extra dependency.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, TokenEngine

__all__ = ["OllamaEngine"]


class OllamaEngine(TokenEngine):
    """Reason with a local model served by Ollama."""

    name = "ollama"
    local = True
    default_model: ClassVar[str] = "llama3.2"
    MODELS: ClassVar[dict[str, str]] = {"fast": "llama3.2:1b", "deep": "llama3.2"}
    #: Lightweight, free, broadly-capable entry model: Qwen2.5 3B Instruct —
    #: ~2GB, Apache-2.0, strong instruction-following, runs on a modest CPU.
    #: Smart enough for basic install/config and for routing harder work up.
    BOOTSTRAP_MODEL: ClassVar[str] = "qwen2.5:3b"
    bootstrap_model: ClassVar[str] = "qwen2.5:3b"

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None,
                 host: Optional[str] = None) -> None:
        super().__init__(model=model or os.getenv("YGG_LOKI_OLLAMA_MODEL"), tier=tier)
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=0.5) as r:
                return r.status == 200
        except Exception:
            return False

    # -- lazy model install ------------------------------------------------

    def installed_models(self) -> list[str]:
        """Models already pulled onto this Ollama server (empty if unreachable)."""
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2.0) as r:
                tags = json.loads(r.read())
        except Exception:
            return []
        return [m.get("name", "") for m in tags.get("models", []) if m.get("name")]

    def has_model(self, model: str) -> bool:
        """True when *model* (with or without a ``:tag``) is already pulled."""
        names = self.installed_models()
        return model in names or f"{model}:latest" in names or any(
            n.split(":")[0] == model for n in names
        )

    def pull(self, model: Optional[str] = None, *, timeout: float = 1800.0) -> str:
        """Download *model* onto the Ollama server (lazy — the heavy bit).

        Streams ``POST /api/pull`` to completion and returns the final status
        line. Defaults to :attr:`bootstrap_model`, the lightweight brain.
        """
        model = model or self.bootstrap_model
        body = json.dumps({"name": model, "stream": True}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/pull", data=body,
            headers={"Content-Type": "application/json"},
        )
        status = "unknown"
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:  # newline-delimited JSON progress
                line = raw.strip()
                if not line:
                    continue
                try:
                    status = json.loads(line).get("status", status)
                except Exception:
                    pass
        return status

    def ensure(self, model: Optional[str] = None) -> dict[str, Any]:
        """Make sure *model* is available, pulling it only if missing.

        Returns ``{"model", "was_present", "status"}`` — the lazy-install
        receipt so a caller (the ``setup`` skill) can report what it did.
        """
        model = model or self.bootstrap_model
        if self.has_model(model):
            return {"model": model, "was_present": True, "status": "already installed"}
        return {"model": model, "was_present": False, "status": self.pull(model)}

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
        body = json.dumps({
            "model": model, "messages": msgs, "stream": False,
            "options": {"num_predict": min(max_tokens, 1024)},
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        text = data.get("message", {}).get("content", "")
        self._record(model,
                     input_tokens=data.get("prompt_eval_count"),
                     output_tokens=data.get("eval_count"),
                     messages=messages, system=system, text=text)
        return Completion(text=text, model=model, raw=data)
