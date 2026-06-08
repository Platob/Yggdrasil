"""Shared fixtures for the Loki test suite."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_lazy_import_cache():
    """Reset the process-wide optional-dependency cache around each test.

    ``yggdrasil.loki.runtime.load`` resolves through
    :func:`yggdrasil.lazy_imports._lazy_import`, which memoizes the imported
    module by name. Tests that inject *fake* modules (``anthropic`` / ``openai``
    / ``transformers`` / ``torch`` via ``patch.dict(sys.modules, …)``) would
    otherwise leak a fake into a later test through that cache. Clearing it
    before and after each test keeps engine-import tests isolated.
    """
    from yggdrasil import lazy_imports

    lazy_imports._LAZY_CACHE.clear()
    yield
    lazy_imports._LAZY_CACHE.clear()
