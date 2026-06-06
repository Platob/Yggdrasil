"""Local HuggingFace -backed :class:`TokenEngine`.

Runs an **open-source model on this workstation** via the ``transformers``
text-generation pipeline — free, private, offline. Defaults to a small Qwen
instruct model so it loads fast on CPU; override with ``YGG_LOKI_HF_MODEL``
(any chat/instruct model id) and ``YGG_LOKI_HF_DEVICE`` (e.g. ``"cuda"``).
Available only when ``transformers`` + ``torch`` are installed.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion, TokenEngine

__all__ = ["TransformersEngine"]


class TransformersEngine(TokenEngine):
    """Reason with a local HuggingFace model (``transformers`` pipeline)."""

    name = "transformers"
    local = True
    #: Small open instruct models — fast tier loads on CPU; deep is larger.
    default_model: ClassVar[str] = "Qwen/Qwen2.5-1.5B-Instruct"
    MODELS: ClassVar[dict[str, str]] = {
        "fast": "Qwen/Qwen2.5-0.5B-Instruct",
        "deep": "Qwen/Qwen2.5-1.5B-Instruct",
    }
    #: One pipeline per model, shared across instances (weights load once).
    _PIPES: ClassVar[dict[str, Any]] = {}

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        super().__init__(model=model or os.getenv("YGG_LOKI_HF_MODEL"), tier=tier)
        self.device = device or os.getenv("YGG_LOKI_HF_DEVICE")

    def available(self) -> bool:
        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)

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
        pipe = self._PIPES.get(model)
        if pipe is None:
            from transformers import pipeline

            pipe = self._PIPES[model] = pipeline(
                "text-generation", model=model,
                device=self.device, trust_remote_code=False,
            )
        chat = ([{"role": "system", "content": system}] if system else []) + list(messages)
        # Cap new tokens for a local model so CPU runs stay responsive.
        out = pipe(chat, max_new_tokens=min(max_tokens, 512),
                   do_sample=False, return_full_text=False)
        gen = out[0]["generated_text"]
        if isinstance(gen, list):  # chat pipeline returns the message list
            last = gen[-1]
            text = last.get("content", "") if isinstance(last, dict) else str(last)
        else:
            text = str(gen)
        self._record(model, messages=messages, system=system, text=text)
        return Completion(text=text, model=model)
