"""Back-compat shim — :class:`DeltaFolder` lives at :mod:`yggdrasil.io.nested.delta.delta_io`."""

from __future__ import annotations

from yggdrasil.io.nested.delta.delta_io import DeltaFolder, DeltaOptions

__all__ = ["DeltaFolder", "DeltaOptions"]
