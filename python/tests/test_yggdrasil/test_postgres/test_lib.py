"""Probe-only tests for the optional-dependency guard.

The guard's job is to either return the imported module or raise
a clear ImportError. We don't try to install psycopg / adbc here —
``has_psycopg`` / ``has_adbc`` give a probe boolean either way.
"""

from __future__ import annotations

import importlib

from yggdrasil.postgres import lib


def test_has_psycopg_is_bool() -> None:
    assert isinstance(lib.has_psycopg(), bool)


def test_has_adbc_is_bool() -> None:
    assert isinstance(lib.has_adbc(), bool)


def test_lazy_module_object_present() -> None:
    # The lazy proxy is always returned even when the underlying
    # module isn't installed; only attribute access triggers the
    # import attempt.
    assert lib.psycopg is not None
    assert lib.adbc_dbapi is not None


def test_module_loader_callable() -> None:
    # Loaders should be callable; resolution is deferred.
    assert callable(lib.psycopg_module)
    assert callable(lib.adbc_dbapi_module)


def test_module_can_be_re_imported() -> None:
    # Re-import shouldn't reset the cached probe state in a way
    # that breaks subsequent calls.
    importlib.reload(lib)
    assert isinstance(lib.has_psycopg(), bool)
