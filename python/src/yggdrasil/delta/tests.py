"""Back-compat shim — :class:`DeltaTestCase` lives at :mod:`yggdrasil.io.delta.tests`."""

from __future__ import annotations

from yggdrasil.io.delta.tests import DeltaTestCase

__all__ = ["DeltaTestCase"]
