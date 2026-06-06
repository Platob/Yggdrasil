"""Local HuggingFace -backed :class:`TokenEngine`.

Runs an **open-source model on this workstation** via the ``transformers``
text-generation pipeline — free, private, offline. The default model is **sized
to the machine** (:class:`LocalEngine` + :mod:`yggdrasil.loki.resources`): a
small Qwen instruct model on a modest CPU box, a larger one as RAM/GPU allow.
Override with ``YGG_LOKI_HF_MODEL`` (any chat/instruct id) and
``YGG_LOKI_HF_DEVICE`` (e.g. ``"cuda"``). Available only when ``transformers``
+ ``torch`` are installed.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, ClassVar, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion
from .local import LocalEngine

__all__ = ["TransformersEngine"]


class TransformersEngine(LocalEngine):
    """Reason with a local HuggingFace model (``transformers`` pipeline)."""

    name = "transformers"
    #: Fallback when the resource tier isn't in the ladder.
    default_model: ClassVar[str] = "Qwen/Qwen2.5-1.5B-Instruct"
    #: Resource tier → open Qwen2.5 instruct model. Bigger box → bigger model.
    RESOURCE_MODELS: ClassVar[dict[str, str]] = {
        "small": "Qwen/Qwen2.5-1.5B-Instruct",   # ≥ 8 GB CPU
        "medium": "Qwen/Qwen2.5-3B-Instruct",     # ≥ 16 GB
        "large": "Qwen/Qwen2.5-7B-Instruct",      # ≥ 32 GB
        "xlarge": "Qwen/Qwen2.5-14B-Instruct",    # CUDA GPU
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
            from ..runtime import load

            load("torch")  # the pipeline backend — auto-installed if missing
            pipe = self._PIPES[model] = load("transformers").pipeline(
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
