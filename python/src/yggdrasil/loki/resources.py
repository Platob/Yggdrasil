"""Workstation resource probe — CPU / RAM / GPU → a model size tier.

A local model is bounded by the machine it runs on, so Loki sizes it to the
box: the more muscle, the larger the *default* local model it reaches for, and
the more confidently it keeps light work local instead of paying a remote API.
Probed live (hardware is cheap to read; ``torch`` stays cached in
``sys.modules`` after the first import).
"""
from __future__ import annotations

import os
from typing import Any, Optional

__all__ = ["snapshot", "size_tier", "can_run_local"]

#: RAM (GB) thresholds for the CPU size tiers; a CUDA GPU jumps to ``xlarge``.
_RAM_LARGE = 32.0
_RAM_MEDIUM = 16.0
#: Floor for running any local model at all (below this, prefer a remote API).
_RAM_MIN = 8.0
_CPU_MIN = 4


def snapshot() -> dict[str, Any]:
    """Current ``{cpu, ram_gb, gpu}`` for this workstation."""
    gpu = False
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:
        pass
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    return {"cpu": os.cpu_count() or 1, "ram_gb": round(ram_gb, 1), "gpu": gpu}


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
    """True when this box can comfortably host a local model (GPU, or ≥ 4 cores
    + ≥ 8 GB RAM) — the gate for keeping light work local instead of remote."""
    s = snap or snapshot()
    return s["gpu"] or (s["cpu"] >= _CPU_MIN and s["ram_gb"] >= _RAM_MIN)
