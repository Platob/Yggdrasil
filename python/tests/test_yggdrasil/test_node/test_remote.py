"""Tests for ``yggdrasil.node.remote`` — the remote-function registry."""
from __future__ import annotations

import pytest

from yggdrasil.exceptions.api import NotFoundError
from yggdrasil.node import remote
from yggdrasil.node.remote import call_registered, get_registered, list_registered


class TestRegistry:
    def test_default_name_is_function_name(self):
        @remote()
        def _ygg_test_add(x, y):
            return x + y

        assert get_registered("_ygg_test_add") is _ygg_test_add
        assert "_ygg_test_add" in list_registered()

    def test_explicit_namespaced_name(self):
        @remote(name="ygg_test:mul")
        def _mul(x, y):
            return x * y

        assert call_registered("ygg_test:mul", (3, 4), {}) == 12

    def test_call_with_kwargs(self):
        @remote(name="ygg_test:greet")
        def _greet(name, *, greeting="hi"):
            return f"{greeting} {name}"

        assert call_registered("ygg_test:greet", ("ada",), {"greeting": "hello"}) == "hello ada"

    def test_missing_function_raises_with_suggestion(self):
        @remote(name="ygg_test:compute")
        def _compute():
            return 1

        with pytest.raises(NotFoundError) as exc:
            call_registered("ygg_test:computee", (), {})
        # Near-match hint should surface the real name.
        assert "ygg_test:compute" in str(exc.value)

    def test_list_is_sorted(self):
        names = list_registered()
        assert names == sorted(names)
