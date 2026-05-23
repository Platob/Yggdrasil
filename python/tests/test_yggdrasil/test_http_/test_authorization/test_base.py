"""Unit tests for :class:`yggdrasil.io.authorization.Authorization`.

The base class is an ABC — these tests pin the contract every concrete
provider has to honor: ``authorization`` is required, ``__str__``
delegates to it, and the class can't be instantiated directly.
"""
from __future__ import annotations

import pytest

from yggdrasil.io.authorization import Authorization


class _StaticAuth(Authorization):
    """Minimal concrete provider for testing the base contract."""

    def __init__(self, header: str) -> None:
        self._header = header

    @property
    def authorization(self) -> str:
        return self._header


class TestAuthorizationContract:

    def test_abstract_authorization_property(self) -> None:
        # Authorization is an ABC; ``authorization`` is the one required
        # member — instantiating the base directly must fail.
        with pytest.raises(TypeError):
            Authorization()  # type: ignore[abstract]

    def test_missing_authorization_property_blocks_instantiation(self) -> None:
        class Incomplete(Authorization):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_returns_header(self) -> None:
        auth = _StaticAuth("Bearer abc")
        assert auth.authorization == "Bearer abc"

    def test_str_delegates_to_authorization(self) -> None:
        auth = _StaticAuth("Bearer xyz")
        assert str(auth) == "Bearer xyz"

    def test_str_reflects_property_changes(self) -> None:
        # ``__str__`` reads the property every call — concrete providers
        # rotate tokens behind the property and stale captured strings
        # would be a real bug.
        auth = _StaticAuth("Bearer one")
        assert str(auth) == "Bearer one"
        auth._header = "Bearer two"
        assert str(auth) == "Bearer two"

    def test_is_abstract_base_class(self) -> None:
        import abc
        assert isinstance(Authorization, abc.ABCMeta)
        assert issubclass(_StaticAuth, Authorization)
