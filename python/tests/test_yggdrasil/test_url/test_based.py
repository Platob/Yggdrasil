"""Tests for :class:`URLBased` registration, dispatch, and abstract hooks."""
from __future__ import annotations

import pytest

from yggdrasil.enums import Scheme
from yggdrasil.url import URL
from yggdrasil.url.based import URLBased, _URL_BASED_REGISTRY

# Force the LocalPath import so the FILE handler is registered before
# any test inspects _URL_BASED_REGISTRY directly.
from yggdrasil.io.path.local_path import LocalPath  # noqa: F401


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestURLBasedRegistration:

    def test_concrete_scheme_is_registered(self):
        URLBased.for_scheme(Scheme.FILE)
        assert Scheme.FILE in _URL_BASED_REGISTRY
        cls = _URL_BASED_REGISTRY[Scheme.FILE]
        assert cls.scheme is Scheme.FILE

    def test_none_scheme_not_registered(self):
        class _AbstractNone(URLBased):
            scheme = None

            @classmethod
            def from_url(cls, url, **kw):
                ...

            def to_url(self):
                ...

        assert _AbstractNone not in _URL_BASED_REGISTRY.values()

    def test_empty_string_scheme_not_registered(self):
        class _AbstractEmpty(URLBased):
            scheme = ""

            @classmethod
            def from_url(cls, url, **kw):
                ...

            def to_url(self):
                ...

        assert _AbstractEmpty not in _URL_BASED_REGISTRY.values()

    def test_string_scheme_coerced_to_enum(self):
        original = _URL_BASED_REGISTRY[Scheme.FILE]
        try:
            class _FileCoerced(original):
                scheme = "file"

                @classmethod
                def from_url(cls, url, **kw):
                    ...

                def to_url(self):
                    ...

            assert _FileCoerced.scheme is Scheme.FILE
        finally:
            _URL_BASED_REGISTRY[Scheme.FILE] = original


# ---------------------------------------------------------------------------
# for_scheme
# ---------------------------------------------------------------------------


class TestForScheme:

    def test_for_scheme_with_enum(self):
        cls = URLBased.for_scheme(Scheme.FILE)
        assert cls.scheme is Scheme.FILE

    def test_for_scheme_with_string(self):
        cls = URLBased.for_scheme("file")
        assert cls.scheme is Scheme.FILE

    def test_for_scheme_returns_same_class(self):
        assert URLBased.for_scheme(Scheme.FILE) is URLBased.for_scheme("file")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestDispatch:

    def test_dispatch_file_url(self):
        obj = URLBased.dispatch("file:///tmp/x")
        assert isinstance(obj, LocalPath)

    def test_dispatch_parses_string_via_url(self):
        url = URL.from_("file:///tmp/y")
        obj = URLBased.dispatch(url)
        assert isinstance(obj, LocalPath)

    def test_dispatch_empty_scheme_falls_back_to_file(self):
        obj = URLBased.dispatch("/tmp/z")
        assert isinstance(obj, LocalPath)


# ---------------------------------------------------------------------------
# Abstract hooks
# ---------------------------------------------------------------------------


class TestDefaultHooks:

    def test_from_url_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            URLBased.from_url("file:///tmp/x")

    def test_to_url_raises_not_implemented(self):
        class _Bare(URLBased):
            scheme = None
        with pytest.raises(NotImplementedError):
            _Bare().to_url()
