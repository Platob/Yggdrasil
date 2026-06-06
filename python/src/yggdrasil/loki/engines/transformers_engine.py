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
import logging
import os
import threading
from typing import Any, ClassVar, Iterator, Optional

from ..engine import DEFAULT_MAX_TOKENS, Completion
from .local import LocalEngine

__all__ = ["TransformersEngine"]

#: Local-model progress logs ride this logger; ``ygg loki`` routes it to the
#: terminal (``style.install_logging``) so a long, otherwise-silent first load
#: on a CPU box (download weights → load → generate) reports what it's doing.
_log = logging.getLogger(__name__)


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

    def ready(self, model: Optional[str] = None) -> bool:
        """True when the pipeline for *model* (resolved if omitted) is loaded.

        Lets a caller (the CLI) warn that a turn is about to trigger the slow
        first load — download weights + build the pipeline — instead of going
        silent on a CPU box.
        """
        return (model or self.resolve_model()) in self._PIPES

    def _pipeline(self, model: str) -> Any:
        """The cached text-generation pipeline for *model*, built on first use.

        The build is the slow, silent part on a fresh box — weights download
        then load — so it's bracketed with progress logs (see :data:`_log`).
        """
        pipe = self._PIPES.get(model)
        if pipe is not None:
            return pipe
        from ..runtime import load

        _log.info("loading local model %s on %s — first run downloads weights, "
                  "this can take a while…", model, self.device or "cpu")
        load("torch")  # the pipeline backend — auto-installed if missing
        pipe = self._PIPES[model] = load("transformers").pipeline(
            "text-generation", model=model,
            device=self.device, trust_remote_code=False,
        )
        _log.info("local model %s ready", model)
        return pipe

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
        pipe = self._pipeline(model)
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

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tier: Optional[str] = None,
        **options: Any,
    ) -> Iterator[str]:
        """Generate live, token by token, via ``TextIteratorStreamer``.

        Without this the base :meth:`stream` runs the whole generation in one
        blocking :meth:`complete` and yields it at the end — so a slow CPU run
        prints nothing until it finishes. Here the pipeline runs on a worker
        thread and feeds a streamer the terminal drains as tokens arrive.
        """
        from ..runtime import load

        model = self.resolve_model(messages=messages, system=system, tier=tier)
        pipe = self._pipeline(model)
        chat = ([{"role": "system", "content": system}] if system else []) + list(messages)
        streamer = load("transformers").TextIteratorStreamer(
            pipe.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        worker = threading.Thread(
            target=pipe, args=(chat,), daemon=True,
            kwargs={"max_new_tokens": min(max_tokens, 512), "do_sample": False,
                    "return_full_text": False, "streamer": streamer},
        )
        worker.start()
        parts: list[str] = []
        for piece in streamer:
            parts.append(piece)
            yield piece
        worker.join()
        self._record(model, messages=messages, system=system, text="".join(parts))
