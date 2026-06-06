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
