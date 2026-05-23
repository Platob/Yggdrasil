"""Shared fixtures for the yggdrasil HTTP test suite."""

from __future__ import annotations

import pytest

pytest.importorskip("yggdrasil")

from yggdrasil.io.bytes_io import BytesIO  # noqa: E402,F401
from yggdrasil.io.path import Path, LocalPath  # noqa: E402,F401
from yggdrasil.io.tabular import Tabular  # noqa: E402,F401
