"""Back-compat shim — :class:`DeltaFolder` lives at :mod:`yggdrasil.io.delta.delta_folder`."""

from __future__ import annotations

from yggdrasil.io.delta.delta_folder import DeltaFolder, DeltaOptions

__all__ = ["DeltaFolder", "DeltaOptions"]
