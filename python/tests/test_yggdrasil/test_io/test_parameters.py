"""Tests for yggdrasil.io.parameters.anonymize_parameters."""

from __future__ import annotations

import pytest

from yggdrasil.url.parameters import anonymize_parameters


class TestAnonymizeRemove:
    def test_drops_sensitive_keys(self):
        params = {"username": "alice", "password": "secret", "api_key": "abc"}
        result = anonymize_parameters(params, mode="remove")
        assert result == {"username": "alice"}

    def test_strips_keys_with_extra_whitespace(self):
        result = anonymize_parameters({"  Token  ": "x", "id": "1"}, mode="remove")
        assert "Token" in result or "id" in result
        # The sensitive Token key should be removed.
        assert all("token" not in k.lower() for k in result)

    def test_preserves_non_sensitive_pairs(self):
        params = {"region": "eu", "page": 3}
        assert anonymize_parameters(params, mode="remove") == {"region": "eu", "page": 3}


class TestAnonymizeRedact:
    def test_replaces_sensitive_values(self):
        params = {"password": "secret", "ok": True}
        result = anonymize_parameters(params, mode="redact")
        assert result == {"password": "<redacted>", "ok": True}


class TestAnonymizeNested:
    def test_dict_inside_dict(self):
        params = {"outer": {"password": "x", "keep": 1}}
        result = anonymize_parameters(params, mode="remove")
        assert result == {"outer": {"keep": 1}}

    def test_list_of_dicts(self):
        params = {"items": [{"token": "x"}, {"id": "y"}]}
        result = anonymize_parameters(params, mode="remove")
        assert result == {"items": [{}, {"id": "y"}]}


class TestAnonymizeContainerInputs:
    def test_list_input(self):
        result = anonymize_parameters([{"password": "x"}, {"id": "y"}], mode="remove")
        assert result == [{}, {"id": "y"}]

    def test_tuple_input_returns_tuple(self):
        result = anonymize_parameters(({"id": "1"}, {"password": "x"}), mode="remove")
        assert isinstance(result, tuple)
        assert result == ({"id": "1"}, {})

    def test_scalar_input_passthrough(self):
        assert anonymize_parameters("hello", mode="remove") == "hello"
        assert anonymize_parameters(123, mode="redact") == 123


class TestAnonymizeMode:
    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            anonymize_parameters({"a": 1}, mode="erase")  # type: ignore[arg-type]
