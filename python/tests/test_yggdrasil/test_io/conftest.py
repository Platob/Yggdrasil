"""Shared fixtures for the rewritten yggdrasil.io test suite.

The previous test tree had grown a deep set of fixtures pinned to
implementation-internal behavior. The rewrite favors plain test
classes that build their own state. This conftest only declares the
things multiple files genuinely share, and skips the whole tree if
yggdrasil itself isn't importable.
"""

from __future__ import annotations

import pytest

pytest.importorskip("yggdrasil")

# Preheat the io.buffer / io.fs / io.tabular triangle before any test
# touches the package piecemeal. The trio sits in a circular-import
# triangle if a test starts at io.tabular (or any of its consumers)
# before yggdrasil.io.fs has finished loading. Importing the heavy
# leaves here once, in dependency order, breaks the cycle for every
# downstream test module.
from yggdrasil.io.buffer.bytes_io import BytesIO  # noqa: E402,F401
from yggdrasil.io.fs import Path, LocalPath  # noqa: E402,F401
from yggdrasil.io.buffer.base import TabularIO  # noqa: E402,F401
