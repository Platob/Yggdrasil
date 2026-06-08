"""Tests for yggdrasil.url.parameters.anonymize_parameters."""
from __future__ import annotations

import pytest

from yggdrasil.url.parameters import anonymize_parameters


class TestAnonymizeRemove:

    def test_drops_password(self):
        assert anonymize_parameters({"password": "s3cret", "host": "localhost"}) == {"host": "localhost"}

    def test_drops_api_key(self):
        assert anonymize_parameters({"api_key": "abc123"}) == {}

    def test_drops_token(self):
        assert anonymize_parameters({"token": "tok"}) == {}

    def test_drops_secret(self):
        assert anonymize_parameters({"secret": "shh"}) == {}

    def test_drops_client_secret(self):
        assert anonymize_parameters({"client_secret": "cs"}) == {}

    def test_strips_whitespace_keys(self):
        assert anonymize_parameters({"  password  ": "val"}) == {}

    def test_preserves_non_sensitive(self):
        params = {"host": "localhost", "port": 5432, "database": "mydb"}
        assert anonymize_parameters(params) == params

    def test_case_insensitive_PASSWORD(self):
        assert anonymize_parameters({"PASSWORD": "x"}) == {}

    def test_case_insensitive_Token(self):
        assert anonymize_parameters({"Token": "x"}) == {}

    def test_case_insensitive_API_KEY(self):
        assert anonymize_parameters({"API_KEY": "x"}) == {}


class TestAnonymizeRedact:

    def test_redacts_sensitive_value(self):
        result = anonymize_parameters({"password": "s3cret", "host": "localhost"}, mode="redact")
        assert result == {"password": "<redacted>", "host": "localhost"}

    def test_keeps_non_sensitive_values(self):
        result = anonymize_parameters({"host": "localhost", "port": 5432}, mode="redact")
        assert result == {"host": "localhost", "port": 5432}

    def test_redacts_multiple_keys(self):
        result = anonymize_parameters({"token": "t", "secret": "s", "name": "ok"}, mode="redact")
        assert result == {"token": "<redacted>", "secret": "<redacted>", "name": "ok"}


class TestAnonymizeNested:

    def test_dict_inside_dict(self):
        params = {"config": {"password": "x", "host": "h"}}
        assert anonymize_parameters(params) == {"config": {"host": "h"}}

    def test_list_of_dicts(self):
        params = {"items": [{"token": "t", "name": "a"}, {"secret": "s", "name": "b"}]}
        result = anonymize_parameters(params)
        assert result == {"items": [{"name": "a"}, {"name": "b"}]}

    def test_tuple_of_dicts(self):
        params = {"items": ({"token": "t", "name": "a"},)}
        result = anonymize_parameters(params)
        assert result == {"items": ({"name": "a"},)}
        assert isinstance(result["items"], tuple)

    def test_deeply_nested(self):
        params = {"outer": [{"inner": {"api_key": "k", "url": "u"}}]}
        result = anonymize_parameters(params)
        assert result == {"outer": [{"inner": {"url": "u"}}]}


class TestAnonymizeContainerInputs:

    def test_list_input(self):
        params = [{"password": "p", "host": "h"}, {"token": "t", "name": "n"}]
        result = anonymize_parameters(params)
        assert result == [{"host": "h"}, {"name": "n"}]

    def test_tuple_input_returns_tuple(self):
        params = ({"password": "p", "host": "h"},)
        result = anonymize_parameters(params)
        assert result == ({"host": "h"},)
        assert isinstance(result, tuple)

    def test_scalar_str_passthrough(self):
        assert anonymize_parameters("hello") == "hello"

    def test_scalar_int_passthrough(self):
        assert anonymize_parameters(42) == 42

    def test_scalar_none_passthrough(self):
        assert anonymize_parameters(None) is None


class TestAnonymizeMode:

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            anonymize_parameters({"a": 1}, mode="mask")  # type: ignore[arg-type]

    def test_remove_mode(self):
        assert anonymize_parameters({"password": "x"}, mode="remove") == {}

    def test_redact_mode(self):
        assert anonymize_parameters({"password": "x"}, mode="redact") == {"password": "<redacted>"}


class TestSensitiveKeys:

    @pytest.mark.parametrize("key", [
        "password",
        "pass",
        "pwd",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "x_api_key",
        "secret",
        "client_secret",
        "authorization",
        "auth",
        "bearer",
    ])
    def test_each_sensitive_key_removed(self, key: str):
        assert anonymize_parameters({key: "value"}) == {}

    @pytest.mark.parametrize("key", [
        "password",
        "pass",
        "pwd",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "api_key",
        "apikey",
        "x_api_key",
        "secret",
        "client_secret",
        "authorization",
        "auth",
        "bearer",
    ])
    def test_each_sensitive_key_redacted(self, key: str):
        assert anonymize_parameters({key: "value"}, mode="redact") == {key: "<redacted>"}
