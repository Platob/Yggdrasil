"""Workstation resource probe — CPU / RAM / accelerator → a model size tier.

A local model is bounded by the machine it runs on, so Loki sizes it to the
box: the more muscle, the larger the *default* local model it reaches for, and
the more confidently it keeps light work local instead of paying a remote API.
Probed live (hardware is cheap to read; ``torch`` stays cached in
``sys.modules`` after the first import).

The accelerator probe spans the vendors a laptop actually ships today — NVIDIA
(``cuda``), **Intel GPU** (``xpu`` — Arc / integrated Xe), Apple Silicon
(``mps``), plus a separate flag for an **Intel NPU** (AI Boost) — so a local
model lands on the accelerator instead of crawling on the CPU.
"""
from __future__ import annotations

import os
from typing import Any, Optional

__all__ = ["snapshot", "size_tier", "can_run_local", "accelerator", "has_npu"]

#: RAM (GB) thresholds for the CPU size tiers; a CUDA GPU jumps to ``xlarge``.
_RAM_LARGE = 32.0
_RAM_MEDIUM = 16.0
#: Floor for running any local model at all (below this, prefer a remote API).
_RAM_MIN = 8.0
_CPU_MIN = 4


def accelerator() -> Optional[str]:
    """Best ``torch`` compute device for a local model on this box, or ``None``.

    Returns the device string the ``transformers`` pipeline accepts directly:
    ``"cuda"`` (NVIDIA), ``"xpu"`` (**Intel GPU** — Arc or integrated Xe, via
    the native XPU backend in recent torch / ``intel-extension-for-pytorch``),
    ``"mps"`` (Apple Silicon), or ``None`` for CPU-only. Drives the transformers
    engine's device when ``YGG_LOKI_HF_DEVICE`` is unset. Intel NPUs are
    reported separately by :func:`has_npu` — the HF pipeline can't target them.
    """
    try:
        import torch
    except Exception:
        return None
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        if getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
            return "xpu"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def has_npu() -> bool:
    """True when an **Intel NPU** (AI Boost) is reachable on this box.

    Probed via OpenVINO's device list — the HF text-generation pipeline can't
    run on the NPU directly (that path is OpenVINO / ``optimum-intel``), so Loki
    *reports* it (and hints at the toolchain) rather than routing a torch model
    onto it. Best-effort: ``False`` when OpenVINO isn't installed.
    """
    try:
        import openvino

        return "NPU" in openvino.Core().available_devices
    except Exception:
        return False


def snapshot() -> dict[str, Any]:
    """Current ``{cpu, ram_gb, gpu, accelerator, npu}`` for this workstation.

    ``accelerator`` is the best torch device (see :func:`accelerator`); ``gpu``
    stays the CUDA flag that drives the ``xlarge`` tier (a discrete NVIDIA GPU),
    and ``npu`` flags an Intel NPU (see :func:`has_npu`).
    """
    accel = accelerator()
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    return {
        "cpu": os.cpu_count() or 1,
        "ram_gb": round(ram_gb, 1),
        "gpu": accel == "cuda",
        "accelerator": accel,
        "npu": has_npu(),
    }


def size_tier(snap: Optional[dict[str, Any]] = None) -> str:
    """Model size tier for this box: ``small`` / ``medium`` / ``large`` / ``xlarge``.

    A CUDA GPU is ``xlarge``; otherwise RAM decides (≥ 32 GB large, ≥ 16 GB
    medium, else small). Local engines map this to a concrete model.
    """
    s = snap or snapshot()
    if s["gpu"]:
        return "xlarge"
    ram = s["ram_gb"]
    if ram >= _RAM_LARGE:
        return "large"
    if ram >= _RAM_MEDIUM:
        return "medium"
    return "small"


def can_run_local(snap: Optional[dict[str, Any]] = None) -> bool:
    """True when this box can comfortably host a local model (any GPU
    accelerator, or ≥ 4 cores + ≥ 8 GB RAM) — the gate for keeping light work
    local instead of remote. An Intel GPU (``xpu``) counts as much as CUDA."""
    s = snap or snapshot()
    return (s["gpu"] or s.get("accelerator") is not None
            or (s["cpu"] >= _CPU_MIN and s["ram_gb"] >= _RAM_MIN))
