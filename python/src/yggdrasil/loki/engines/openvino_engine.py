"""Local **OpenVINO** -backed :class:`TokenEngine` — runs a model on the Intel NPU.

Where :class:`TransformersEngine` runs a model through a torch pipeline (CPU, or
an Intel GPU via the XPU torch build), this engine loads the model with
`optimum-intel <https://github.com/huggingface/optimum-intel>`_ +
`OpenVINO <https://docs.openvino.ai>`_ and runs it on an **Intel NPU** (AI Boost)
— the inference accelerator a torch pipeline can't target — falling back to the
Intel GPU then CPU. It produces the same HuggingFace ``text-generation``
pipeline, so the ``complete`` / ``stream`` inference path is inherited from
:class:`TransformersEngine` unchanged; only the *model loader* differs.

Defaults to OpenVINO's pre-converted **int4** Qwen2.5 models (small, NPU-friendly,
no on-the-fly conversion); a plain HuggingFace id is converted to OpenVINO IR on
first load (``export=True``). Override with ``YGG_LOKI_OV_MODEL`` and pin the
device with ``YGG_LOKI_OV_DEVICE`` (``NPU`` / ``GPU`` / ``CPU``). Available only
when ``openvino`` + ``optimum`` are installed **and** an NPU or GPU is present
(CPU-only boxes are better served by the transformers / ollama engines).
"""
from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, ClassVar, Optional

from .transformers_engine import TransformersEngine, _brief

__all__ = ["OpenVINOEngine"]

#: Model load / device-fallback progress rides this logger; ``ygg loki`` routes
#: it to the terminal so a slow first load (download + IR conversion) isn't
#: silent — same treatment as the transformers engine.
_log = logging.getLogger(__name__)


def _is_ov_model(model: str) -> bool:
    """Whether *model* is already an OpenVINO-IR repo (loads without conversion).

    OpenVINO's own pre-converted models live under the ``OpenVINO/`` org and carry
    an ``-ov`` / ``-int4-ov`` suffix; anything else is a plain HF model that
    ``OVModelForCausalLM`` converts on first load (``export=True``).
    """
    low = model.lower()
    return low.startswith("openvino/") or "-ov" in low or "openvino" in low


class OpenVINOEngine(TransformersEngine):
    """Reason with a local model on the **Intel NPU** via OpenVINO / optimum-intel."""

    name = "openvino"
    #: Fallback when the resource tier isn't in the ladder.
    default_model: ClassVar[str] = "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov"
    #: Resource tier → OpenVINO pre-converted **int4** Qwen2.5 model. int4 keeps
    #: the model inside the NPU's memory budget; bigger boxes climb the ladder.
    RESOURCE_MODELS: ClassVar[dict[str, str]] = {
        "small": "OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov",   # ≥ 8 GB
        "medium": "OpenVINO/Qwen2.5-3B-Instruct-int4-ov",     # ≥ 16 GB
        "large": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",      # ≥ 32 GB
        "xlarge": "OpenVINO/Qwen2.5-7B-Instruct-int4-ov",     # NPU memory-bound
    }
    #: Own pipeline / failure caches — separate from the transformers engine's so
    #: an OpenVINO build and a torch build of the same id never collide.
    _PIPES: ClassVar[dict[str, Any]] = {}
    _FAILED: ClassVar[dict[str, Exception]] = {}

    def __init__(self, *, model: Optional[str] = None, tier: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        # Bypass TransformersEngine.__init__ (it reads the HF env vars); take the
        # OpenVINO ones instead. LocalEngine/TokenEngine init sets model + tier.
        super(TransformersEngine, self).__init__(
            model=model or os.getenv("YGG_LOKI_OV_MODEL"), tier=tier)
        self.device = device or os.getenv("YGG_LOKI_OV_DEVICE")
        #: Memoized OpenVINO device list — devices don't change within a process,
        #: and ``available()`` is probed several times per ``ygg loki`` command.
        self._device_list: Optional[list[str]] = None

    def available(self) -> bool:
        """True when OpenVINO + optimum are installed **and** an NPU/GPU is present.

        A CPU-only box is left to the transformers / ollama engines — this engine
        exists for the accelerators a torch pipeline can't reach (chiefly the NPU).
        Cheap: the package check is ``find_spec``; the device list is memoized.
        """
        if (importlib.util.find_spec("openvino") is None
                or importlib.util.find_spec("optimum") is None):
            return False
        return any(d == "NPU" or d.startswith("GPU") for d in self._devices())

    def _devices(self) -> list[str]:
        """OpenVINO's available devices (``["NPU", "GPU", "CPU"]`` …), memoized.

        Never raises — capability detection on an offline / partial install must
        degrade to an empty list, not blow up the parallel engine probe.
        """
        if self._device_list is None:
            try:
                import openvino

                self._device_list = list(openvino.Core().available_devices)
            except Exception:
                self._device_list = []
        return self._device_list

    def resolve_device(self) -> str:
        """The OpenVINO device to run on: an explicit pin wins, else the best
        present accelerator — **NPU** first (the whole point), then GPU."""
        if self.device:
            return self.device
        return self._device_chain()[0]

    def _device_chain(self) -> list[str]:
        """Ordered devices to try: a pin alone, else NPU → GPU → CPU (CPU always
        succeeds, so a model still runs if the NPU rejects it)."""
        if self.device:
            return [self.device]
        present = self._devices()
        chain = [d for d in ("NPU", "GPU") if any(p == d or p.startswith(d) for p in present)]
        chain.append("CPU")
        return chain

    @property
    def model_label(self) -> str:
        # Append the target device so `ygg loki engines` shows where it runs.
        return f"{super().model_label} @ {self.resolve_device()}"

    def _pipeline(self, model: str) -> Any:
        """The cached OpenVINO text-generation pipeline for *model*.

        Loads with ``OVModelForCausalLM`` (converting a plain HF model to
        OpenVINO IR on first use) and walks the device chain — **NPU → GPU →
        CPU** — so a model the NPU can't host still runs on the GPU/CPU instead
        of failing. A build that already failed this process is not retried.
        """
        pipe = self._PIPES.get(model)
        if pipe is not None:
            return pipe
        failed = self._FAILED.get(model)
        if failed is not None:
            raise failed
        from ..runtime import load

        load("openvino")  # the runtime — auto-installed (pulls in via optimum too)
        optimum = load("optimum.intel", "optimum[openvino]")
        transformers = load("transformers")
        try:
            transformers.logging.set_verbosity_error()
        except Exception:
            pass

        tokenizer = transformers.AutoTokenizer.from_pretrained(model)
        export = not _is_ov_model(model)   # convert a plain HF model to OV IR
        last: Optional[Exception] = None
        for device in self._device_chain():
            _log.info("loading OpenVINO model %s on %s%s — a first run downloads "
                      "(and converts) the weights, cached for next time…",
                      model, device, " · converting to OpenVINO IR" if export else "")
            try:
                ov_model = optimum.OVModelForCausalLM.from_pretrained(
                    model, device=device, export=export)
                pipe = transformers.pipeline(
                    "text-generation", model=ov_model, tokenizer=tokenizer)
                self._PIPES[model] = pipe
                _log.info("OpenVINO model %s ready on %s", model, device)
                return pipe
            except Exception as exc:
                last = exc
                _log.warning("OpenVINO model %s couldn't load on %s (%s) — trying the "
                             "next device", model, device, _brief(exc))

        cause = (last.__cause__ or last.__context__ or last) if last else None
        err = RuntimeError(
            f"could not load OpenVINO model {model!r} on any of "
            f"{self._device_chain()}: {type(cause).__name__ if cause else 'error'}: "
            f"{_brief(cause)}. Pin a smaller model with YGG_LOKI_OV_MODEL (the "
            f"OpenVINO/*-int4-ov models suit the NPU), or a device with "
            f"YGG_LOKI_OV_DEVICE."
        )
        self._FAILED[model] = err
        _log.warning("OpenVINO model %s failed to load — %s", model, err)
        raise err
