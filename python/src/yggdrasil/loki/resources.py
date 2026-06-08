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

import glob
import importlib.util
import os
import sys
from dataclasses import asdict, dataclass
from typing import Optional

__all__ = [
    "Resources", "snapshot", "size_tier", "can_run_local",
    "accelerator", "intel_gpu_present", "has_npu", "XPU_TORCH_INDEX",
]


@dataclass(frozen=True, slots=True)
class Resources:
    """A workstation's compute, as typed fields instead of a loose dict.

    ``gpu`` is the CUDA flag that drives the ``xlarge`` tier; ``accelerator`` is
    the best torch-usable device (cuda/xpu/mps, or ``None``); ``intel_gpu`` flags
    an Intel GPU physically present even when torch can't target it; ``npu`` an
    Intel NPU.
    """

    cpu: int = 1
    ram_gb: float = 0.0
    gpu: bool = False
    accelerator: Optional[str] = None
    intel_gpu: bool = False
    npu: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

#: The dedicated PyTorch wheel index that ships **Intel GPU (XPU)** support —
#: ``pip install --index-url <this> torch`` turns a *detected* Intel GPU into a
#: *torch-usable* one (``ygg loki setup`` offers this). The stock PyPI wheel is
#: CPU/CUDA-only, so a fresh box never drives the Intel GPU without it.
XPU_TORCH_INDEX = "https://download.pytorch.org/whl/xpu"

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
        # ``torch.xpu`` only exists on an XPU-enabled torch build; on a stock
        # CPU/CUDA wheel it's absent even when an Intel GPU is physically here.
        # ``intel-extension-for-pytorch`` (IPEX), if installed, registers the
        # backend on import — so try it before giving up on the GPU.
        xpu = getattr(torch, "xpu", None)
        if xpu is None and importlib.util.find_spec("intel_extension_for_pytorch"):
            import intel_extension_for_pytorch  # noqa: F401 — import registers torch.xpu
            xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            return "xpu"
    except Exception:
        pass
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return None


def intel_gpu_present() -> bool:
    """Whether an **Intel GPU** (Arc / integrated Xe) is *physically present* —
    independent of whether torch can target it.

    :func:`accelerator` only reports ``"xpu"`` when torch can actually drive the
    GPU (an XPU build / IPEX). But a laptop's Intel iGPU is worth *reporting*
    even on a stock CPU torch wheel — so this probes the OS directly: a DRM card
    with Intel's PCI vendor id on Linux, the video-controller list on Windows.
    Best-effort; ``False`` (e.g. on macOS, or when nothing matches).
    """
    # Linux: a DRM card whose PCI vendor id is Intel's (0x8086).
    try:
        for path in glob.glob("/sys/class/drm/card*/device/vendor"):
            try:
                with open(path) as fh:
                    if fh.read().strip().lower() == "0x8086":
                        return True
            except OSError:
                continue
    except Exception:
        pass
    if sys.platform == "win32" and "intel" in _win_devices("Win32_VideoController"):
        return True
    return False


def has_npu() -> bool:
    """True when an **Intel NPU** (AI Boost) is present on this box.

    OpenVINO's device list is the authoritative signal when installed (and the
    HF pipeline can't target the NPU directly — that path is OpenVINO /
    ``optimum-intel``). Without OpenVINO, fall back to **OS-level** signals so
    the NPU is still detected on a bare box: the Linux ``intel_vpu`` accel
    device, or the Windows "AI Boost" PnP entry. Best-effort; ``False`` when
    nothing matches.
    """
    try:
        import openvino

        if "NPU" in openvino.Core().available_devices:
            return True
    except Exception:
        pass
    return _os_has_npu()


def _os_has_npu() -> bool:
    """OS-level Intel NPU probe — no OpenVINO required.

    Linux: the ``intel_vpu`` kernel driver exposes ``/dev/accel/accelN`` (the
    driver is confirmed via sysfs so another vendor's accel card doesn't
    match). Windows: the NPU enumerates as an "AI Boost" PnP entity.
    """
    try:
        for dev in glob.glob("/sys/class/accel/accel*"):
            try:
                driver = os.path.basename(os.readlink(os.path.join(dev, "device", "driver")))
            except OSError:
                driver = ""
            if "vpu" in driver.lower():
                return True
        # A device node with no usable sysfs class still implies an accelerator.
        if not glob.glob("/sys/class/accel/accel*") and glob.glob("/dev/accel/accel*"):
            return True
    except Exception:
        pass
    if sys.platform == "win32":
        names = _win_devices("Win32_PnPEntity")
        if "ai boost" in names or "vpu" in names:
            return True
    return False


def _win_devices(cim_class: str) -> str:
    """Lowercased device names for a Windows CIM class (``""`` elsewhere / on
    failure). Tries PowerShell CIM (current) then legacy ``wmic`` — so it works
    across Windows versions that have dropped one or the other."""
    import subprocess

    commands = (
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance {cim_class} | Select-Object -ExpandProperty Name"],
        ["wmic", "path", cim_class, "get", "name"],
    )
    for cmd in commands:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.lower()
        except Exception:
            continue
    return ""


def snapshot() -> Resources:
    """Current :class:`Resources` for this box (typed, not a loose dict).

    ``accelerator`` is the best **torch-usable** device (see :func:`accelerator`);
    ``gpu`` stays the CUDA flag that drives the ``xlarge`` tier (a discrete
    NVIDIA GPU); ``intel_gpu`` flags an Intel GPU that's *physically present*
    even when torch can't target it (see :func:`intel_gpu_present`); and ``npu``
    flags an Intel NPU (see :func:`has_npu`).
    """
    accel = accelerator()
    ram_gb = 0.0
    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError, AttributeError):
        pass
    return Resources(
        cpu=os.cpu_count() or 1,
        ram_gb=round(ram_gb, 1),
        gpu=accel == "cuda",
        accelerator=accel,
        # Present even on a stock CPU torch wheel (where ``accelerator`` is None);
        # short-circuit when torch already reports the usable xpu device.
        intel_gpu=accel == "xpu" or intel_gpu_present(),
        npu=has_npu(),
    )


def size_tier(snap: Optional[Resources] = None) -> str:
    """Model size tier for this box: ``small`` / ``medium`` / ``large`` / ``xlarge``.

    A CUDA GPU is ``xlarge``; otherwise RAM decides (≥ 32 GB large, ≥ 16 GB
    medium, else small). Local engines map this to a concrete model.
    """
    s = snap or snapshot()
    if s.gpu:
        return "xlarge"
    if s.ram_gb >= _RAM_LARGE:
        return "large"
    if s.ram_gb >= _RAM_MEDIUM:
        return "medium"
    return "small"


def can_run_local(snap: Optional[Resources] = None) -> bool:
    """True when this box can comfortably host a local model (any GPU
    accelerator, or ≥ 4 cores + ≥ 8 GB RAM) — the gate for keeping light work
    local instead of remote. An Intel GPU (``xpu``) counts as much as CUDA."""
    s = snap or snapshot()
    return (s.gpu or s.accelerator is not None
            or (s.cpu >= _CPU_MIN and s.ram_gb >= _RAM_MIN))
