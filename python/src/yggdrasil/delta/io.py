"""Back-compat shim — :class:`DeltaIO` lives at :mod:`yggdrasil.io.nested.delta.delta_io`."""

from __future__ import annotations

from yggdrasil.io.nested.delta.delta_io import DeltaIO, DeltaOptions

__all__ = ["DeltaIO", "DeltaOptions"]
