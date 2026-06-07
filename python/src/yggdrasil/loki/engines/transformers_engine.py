"""Local HuggingFace -backed :class:`TokenEngine`.

Runs an **open-source model on this workstation** via the ``transformers``
text-generation pipeline — free, private, offline. The default model is **sized
to the machine** (:class:`LocalEngine` + :mod:`yggdrasil.loki.resources`): a
small Qwen instruct model on a modest CPU box, a larger one as RAM/GPU allow.
Override with ``YGG_LOKI_HF_MODEL`` (any chat/instruct id) and
``YGG_LOKI_HF_DEVICE`` (e.g. ``"cuda"``). When the device is left unset the
engine **auto-detects an accelerator** (:func:`yggdrasil.loki.resources.accelerator`)
— NVIDIA ``cuda``, **Intel GPU** ``xpu``, or Apple ``mps`` — so a local model
lands on the GPU instead of the CPU. An **Intel NPU** (AI Boost) is detected and
flagged, but the HF pipeline can't target it directly (use OpenVINO /
``optimum-intel`` for that). Available only when ``transformers`` + ``torch``
are installed.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import threading
from typing import Any, ClassVar, Iterator, Optional

from .. import resources
from ..engine import DEFAULT_MAX_TOKENS, Completion
from .local import LocalEngine

__all__ = ["TransformersEngine"]

#: Local-model progress logs ride this logger; ``ygg loki`` routes it to the
#: terminal (``style.install_logging``) so a long, otherwise-silent first load
#: on a CPU box (download weights → load → generate) reports what it's doing.
_log = logging.getLogger(__name__)


def _brief(exc: object, limit: int = 200) -> str:
    """One-line, length-capped rendering of an exception (or its message).

    transformers reports a failed load by stuffing several nested tracebacks
    into a single error *string*; logged verbatim that's hundreds of lines per
    turn. Collapse the whitespace and cap it so the log stays one readable line.
    """
    msg = " ".join(str(exc).split())
    return msg if len(msg) <= limit else msg[: limit - 1] + "…"


#: Substrings (and the ``OSError`` type) that mark a load failure as a *corrupt
#: or partial download* rather than a runtime/device problem — the only case
#: that warrants a force re-fetch. A device-placement error (GPU not usable)
#: must NOT match, so it falls back to CPU instead of re-downloading the weights
#: on every run (the "it redownloads the model every time" pain).
_CORRUPT_SIGNALS = (
    "could not load", "safetensors", "corrupt", "incomplete", "truncat",
    "checkpoint", "no such file", "errno", "unexpectedly", "eof",
)


def _looks_corrupt(exc: BaseException) -> bool:
    """Whether *exc* looks like a corrupt/partial download (vs a device error)."""
    return isinstance(exc, OSError) or any(s in str(exc).lower() for s in _CORRUPT_SIGNALS)


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
    #: Models whose build already failed this process → the surfaced cause.
    #: A local load is slow (download weights → build the pipeline) and can
    #: fail late (corrupt cache, torch mismatch); without this the doomed load
    #: re-runs on *every* turn — the "it reloads every chat, it's too slow"
    #: pain. Remembering the failure makes the retry fail fast instead.
    _FAILED: ClassVar[dict[str, Exception]] = {}

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        super().__init__(model=model or os.getenv("YGG_LOKI_HF_MODEL"), tier=tier)
        self.device = device or os.getenv("YGG_LOKI_HF_DEVICE")

    def available(self) -> bool:
        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)

    def resolve_device(self) -> Optional[str]:
        """The device to load the pipeline on: an explicit pin (ctor arg /
        ``YGG_LOKI_HF_DEVICE``) wins; otherwise the best auto-detected
        accelerator — NVIDIA ``cuda``, **Intel GPU** ``xpu``, Apple ``mps`` —
        or ``None`` (CPU). Lets a local model use the GPU without configuration.
        """
        if self.device:
            return self.device
        return resources.accelerator()

    def ready(self, model: Optional[str] = None) -> bool:
        """True when the pipeline for *model* (resolved if omitted) is loaded.

        Lets a caller (the CLI) warn that a turn is about to trigger the slow
        first load — download weights + build the pipeline — instead of going
        silent on a CPU box.
        """
        return (model or self.resolve_model()) in self._PIPES

    def warm(self, model: Optional[str] = None) -> None:
        """Build the model's pipeline ahead of the first turn — best-effort.

        Loading a local model is slow and silent (download weights → build the
        pipeline); the ``ygg loki`` REPL calls this on a background thread so
        the wait overlaps the user picking a session and typing, instead of
        stalling the first submit. Swallows failures — they're cached in
        :attr:`_FAILED` and surfaced on the first real turn.
        """
        try:
            self._pipeline(model or self.resolve_model())
        except Exception:
            pass

    def _pipeline(self, model: str) -> Any:
        """The cached text-generation pipeline for *model*, built on first use.

        **The HuggingFace cache is reused** — a model downloaded once is *not*
        re-downloaded on a later run (the default load only fetches files that
        are missing). Two failure modes are handled distinctly so a transient
        problem never triggers a needless multi-GB re-download:

        - a **device** error (the GPU can't host the model) → fall back to CPU,
          keeping the cached weights;
        - a **corrupt / partial** download → repair with exactly one force
          re-fetch, then retry.

        A build that already failed this process is **not retried**: it raises
        the remembered cause straight away.
        """
        pipe = self._PIPES.get(model)
        if pipe is not None:
            return pipe
        failed = self._FAILED.get(model)
        if failed is not None:
            raise failed
        from ..runtime import load

        # Windows has no symlinks in the HF cache by default, so a download
        # falls back to *copies* — an interrupted weights fetch then leaves a
        # truncated file that loads as the generic "Could not load model … with
        # any of the following classes"; the corrupt-cache repair below re-fetches
        # exactly that. Quiet the (expected) symlink warning here.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        # Xet is HuggingFace's newer CAS transfer; behind a corporate proxy its
        # endpoint (cas-server.xethub.hf.co) is often blocked — a 403 there
        # aborts the weights fetch with a giant nested traceback. Fall back to
        # the classic LFS-over-HTTPS download, which rides the normal hub host.
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        load("torch")  # the pipeline backend — auto-installed if missing
        transformers = load("transformers")
        # transformers is loud — a failed load dumps several nested tracebacks
        # through its own logger. Pin it to errors so our one concise line is the
        # signal, not buried under framework noise.
        try:
            transformers.logging.set_verbosity_error()
        except Exception:
            pass
        # Auto-detect the accelerator (cuda / Intel xpu / mps) unless pinned, so
        # the model uses the GPU instead of the CPU. torch must be importable
        # first, hence after load("torch").
        device = self.resolve_device()
        _log.info("loading local model %s on %s — a first run downloads the weights "
                  "(cached for next time, with a progress bar), then runs locally…",
                  model, device or "cpu")
        if device is None and resources.has_npu():
            _log.info("an Intel NPU (AI Boost) was detected but the torch pipeline "
                      "runs on CPU — switch to the 'openvino' engine (ygg loki "
                      "/engine openvino) to run the model on the NPU.")
        pipe = self._build(load, transformers, model, device)
        self._PIPES[model] = pipe
        _log.info("local model %s ready", model)
        return pipe

    def _build(self, load: Any, transformers: Any, model: str, device: Optional[str]) -> Any:
        """Build the text-generation pipeline, reusing the cache.

        The plain build reuses any cached weights (HF fetches only what's
        missing). On a **device** failure it retries on CPU; on a **corrupt**
        cache it force-re-fetches once and retries. Any final failure is
        remembered (:attr:`_FAILED`) and re-raised with the unwrapped cause.
        """
        try:
            return transformers.pipeline("text-generation", model=model,
                                         device=device, trust_remote_code=False)
        except Exception as first:
            # A GPU placement failure (xpu/cuda/mps unusable for this model) →
            # run on CPU instead of re-downloading the weights. Skip when the
            # error already looks like a bad download (handled just below).
            if device is not None and not _looks_corrupt(first):
                _log.warning("local model %s couldn't load on %s (%s) — falling back "
                             "to CPU (weights kept, not re-downloaded)",
                             model, device, _brief(first))
                try:
                    return transformers.pipeline("text-generation", model=model,
                                                 device=None, trust_remote_code=False)
                except Exception as cpu_exc:
                    first = cpu_exc
            # A corrupt / partial download (chiefly on Windows) → repair the repo
            # once with a force re-fetch, then retry on CPU.
            if _looks_corrupt(first):
                _log.warning("local model %s failed to load (%s: %s) — re-fetching the "
                             "weights to repair a partial download, retrying once…",
                             model, type(first).__name__, _brief(first))
                try:
                    load("huggingface_hub").snapshot_download(model, force_download=True)
                    return transformers.pipeline("text-generation", model=model,
                                                 device=None, trust_remote_code=False)
                except Exception as exc:
                    first = exc
            # transformers masks the real reason behind a generic "Could not load
            # model …" — unwrap the cause so the log says *why*, and remember the
            # failure so the doomed load isn't re-attempted every turn.
            cause = first.__cause__ or first.__context__ or first
            err = RuntimeError(
                f"could not load local model {model!r}: "
                f"{type(cause).__name__}: {_brief(cause)}. Pin a smaller model "
                f"with YGG_LOKI_HF_MODEL, or clear the HuggingFace cache and retry."
            )
            self._FAILED[model] = err
            _log.warning("local model %s failed to load — %s", model, err)
            raise err from first

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
