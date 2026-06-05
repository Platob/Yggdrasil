"""Unit tests for yggdrasil.path.proxy.ProxyPathMixin.

The generic "be a path by delegating to an inner one" mixin: a thin
wrapper over an in-memory :class:`LocalPath` should expose the full
filesystem surface, route navigation back to the inner (inner-typed)
path, and let the wrapper's own attributes shadow the delegated ones.
"""
from __future__ import annotations

import pytest

from yggdrasil.path import LocalPath, Path, ProxyPathMixin


class Wrapper(ProxyPathMixin):
    """Minimal consumer: holds an inner path + one shadowing attribute."""

    def __init__(self, inner: Path, label: str = "wrapper") -> None:
        self._inner = inner
        self.label = label  # a wrapper-owned attribute (shadows delegation)

    def _internal_path(self) -> Path:
        return self._inner


@pytest.fixture
def inner(tmp_path):
    return LocalPath.from_(str(tmp_path / "f.txt"))


def test_requires_internal_path_hook():
    with pytest.raises(TypeError):
        ProxyPathMixin()  # abstract — can't instantiate


def test_inner_path_alias(inner):
    w = Wrapper(inner)
    assert w.inner_path is inner
    assert w._internal_path() is inner


def test_read_write_delegate_to_inner(inner):
    w = Wrapper(inner)
    w.write_bytes(b"hello")
    assert bytes(w.read_bytes()) == b"hello"
    assert w.exists() is True
    assert inner.exists() is True  # same backing file


def test_navigation_returns_inner_typed_paths(inner):
    w = Wrapper(inner)
    child = w / "sub" / "x.parquet"
    assert isinstance(child, LocalPath)        # left the wrapper
    assert not isinstance(child, Wrapper)
    assert isinstance(w.parent, LocalPath)


def test_fspath_and_str_mirror_inner(inner):
    w = Wrapper(inner)
    import os

    assert os.fspath(w) == os.fspath(inner)
    assert str(w) == str(inner)


def test_wrapper_attributes_shadow_delegation(inner):
    w = Wrapper(inner, label="raw_zone")
    # ``label`` is defined on the wrapper, so it is NOT delegated.
    assert w.label == "raw_zone"
    # ``name`` is not defined on the wrapper, so it delegates to the inner.
    assert w.name == inner.name == "f.txt"


def test_uses_inner_overrides(inner):
    class Loud(LocalPath):
        def full_path(self) -> str:  # an override on the inner type
            return "LOUD:" + super().full_path()

    loud = Loud.from_(inner.full_path())
    w = Wrapper(loud)
    # Delegation resolves the bound method on the inner instance, so the
    # inner's override wins.
    assert w.full_path().startswith("LOUD:")
