"""Tests for ``yggdrasil.environ.parameters.SystemParameters``."""
from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Mapping
from typing import Any
from unittest import mock

import pytest

from yggdrasil.environ import SystemParameters


class _FakeWidgets:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def get(self, name: str) -> str:
        return self._values[name]


class _FakeBindings:
    def __init__(self, values: dict[str, str]) -> None:
        self._values = values

    def keySet(self) -> list[str]:
        return list(self._values)

    def get(self, key: str) -> str:
        return self._values[key]


class _FakeEntryPoint:
    def __init__(self, bindings: Any) -> None:
        self._bindings = bindings

    def getCurrentBindings(self) -> Any:
        return self._bindings


class _FakeNotebook:
    def __init__(self, entry_point: _FakeEntryPoint) -> None:
        self.entry_point = entry_point


class _FakeDBUtils:
    def __init__(
        self,
        widgets: dict[str, str] | None = None,
        bindings: Any = None,
    ) -> None:
        self.widgets = _FakeWidgets(widgets or {})
        self.notebook = _FakeNotebook(_FakeEntryPoint(bindings))


# ============================================================================
# Mapping interface
# ============================================================================


class TestMappingInterface:
    def test_is_mapping_not_dict(self) -> None:
        params = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert isinstance(params, Mapping)
        assert not isinstance(params, dict)

    def test_getitem(self) -> None:
        params = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert params["a"] == "1"

    def test_getitem_missing_raises(self) -> None:
        params = SystemParameters(argv=None, dbutils=None)
        with pytest.raises(KeyError):
            params["nope"]

    def test_contains(self) -> None:
        params = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert "a" in params
        assert "nope" not in params
        assert 42 not in params  # non-string keys

    def test_get_with_default(self) -> None:
        params = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert params.get("nope", "fallback") == "fallback"

    def test_iter_and_len(self) -> None:
        params = SystemParameters({"a": "1", "b": "2"}, argv=None, dbutils=None)
        assert set(params) == {"a", "b"}
        assert len(params) == 2

    def test_keys_values_items(self) -> None:
        params = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert set(params.keys()) == {"a"}
        assert list(params.values()) == ["1"]
        assert list(params.items()) == [("a", "1")]


# ============================================================================
# Lazy dbutils — not probed until accessed
# ============================================================================


class TestLazyDBUtils:
    def test_dbutils_not_called_on_init(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils") as probe:
            SystemParameters(argv=None)
            probe.assert_not_called()

    def test_dbutils_probed_on_first_lookup(self) -> None:
        dbutils = _FakeDBUtils(bindings={"k": "v"})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils) as probe:
            params = SystemParameters(argv=None)
            probe.assert_not_called()
            assert params["k"] == "v"
            probe.assert_called_once()

    def test_dbutils_bindings_cached(self) -> None:
        dbutils = mock.MagicMock()
        bindings = _FakeBindings({"k": "v"})
        dbutils.notebook.entry_point.getCurrentBindings.return_value = bindings
        params = SystemParameters(argv=None, dbutils=dbutils)
        # Two reads — Java map only walked once.
        assert params["k"] == "v"
        assert params["k"] == "v"
        dbutils.notebook.entry_point.getCurrentBindings.assert_called_once()

    def test_dbutils_none_skips_probe(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils") as probe:
            params = SystemParameters({"k": "v"}, argv=None, dbutils=None)
            assert params["k"] == "v"
            probe.assert_not_called()


# ============================================================================
# Source precedence
# ============================================================================


class TestPrecedence:
    def test_explicit_over_argv(self) -> None:
        params = SystemParameters({"k": "explicit"}, argv=["--k=cli"], dbutils=None)
        assert params["k"] == "explicit"

    def test_argv_over_dbutils(self) -> None:
        dbutils = _FakeDBUtils(bindings={"k": "widget"})
        params = SystemParameters(argv=["--k=cli"], dbutils=dbutils)
        assert params["k"] == "cli"

    def test_dbutils_over_env(self) -> None:
        dbutils = _FakeDBUtils(bindings={"DATABRICKS_K": "widget"})
        env = {"DATABRICKS_K": "env"}
        with mock.patch.dict(os.environ, env, clear=True):
            params = SystemParameters(argv=None, dbutils=dbutils, env_prefix="DATABRICKS_")
            assert params["DATABRICKS_K"] == "widget"

    def test_env_only_when_no_other_source(self) -> None:
        env = {"DATABRICKS_K": "env"}
        with mock.patch.dict(os.environ, env, clear=True):
            params = SystemParameters(argv=None, dbutils=None, env_prefix="DATABRICKS_")
            assert params["DATABRICKS_K"] == "env"

    def test_env_prefix_filters(self) -> None:
        env = {"DATABRICKS_K": "1", "OTHER_K": "2"}
        with mock.patch.dict(os.environ, env, clear=True):
            params = SystemParameters(argv=None, dbutils=None, env_prefix="DATABRICKS_")
            assert "DATABRICKS_K" in params
            assert "OTHER_K" not in params


# ============================================================================
# Argv parsing
# ============================================================================


class TestFromArgv:
    def test_key_value_equals(self) -> None:
        params = SystemParameters.from_argv(["--name=alice", "--count=3"])
        assert params["name"] == "alice"
        assert params["count"] == "3"
        assert params.args == ()

    def test_key_value_space_separated(self) -> None:
        params = SystemParameters.from_argv(["--name", "alice"])
        assert params["name"] == "alice"

    def test_bare_flag_is_true(self) -> None:
        params = SystemParameters.from_argv(["--verbose"])
        assert params["verbose"] == "true"

    def test_positional_args_preserved(self) -> None:
        params = SystemParameters.from_argv(["one", "--k=v", "two"])
        assert params["k"] == "v"
        assert params.args == ("one", "two")

    def test_default_argv_reads_sys(self) -> None:
        with mock.patch.object(sys, "argv", ["prog", "--foo=bar"]):
            params = SystemParameters.from_argv()
        assert params["foo"] == "bar"

    def test_from_argv_skips_dbutils(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils") as probe:
            params = SystemParameters.from_argv(["--k=v"])
            assert params["k"] == "v"
            probe.assert_not_called()


# ============================================================================
# from_dbutils
# ============================================================================


class TestFromDBUtils:
    def test_raises_without_dbutils(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            with pytest.raises(RuntimeError, match="dbutils is not available"):
                SystemParameters.from_dbutils("foo")

    def test_named_widgets(self) -> None:
        dbutils = _FakeDBUtils(widgets={"name": "alice", "count": "3"})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils("name", "count")
        assert params["name"] == "alice"
        assert params["count"] == "3"

    def test_full_bindings_java_map(self) -> None:
        dbutils = _FakeDBUtils(bindings=_FakeBindings({"k": "v"}))
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils()
        assert params["k"] == "v"

    def test_full_bindings_plain_dict(self) -> None:
        dbutils = _FakeDBUtils(bindings={"k1": "v1", "k2": 2})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils()
        assert params["k1"] == "v1"
        assert params["k2"] == "2"


# ============================================================================
# from_environ
# ============================================================================


class TestFromEnviron:
    def test_no_prefix_returns_full_env(self) -> None:
        with mock.patch.dict(os.environ, {"A": "1", "B": "2"}, clear=True):
            params = SystemParameters.from_environ()
        assert params["A"] == "1"
        assert params["B"] == "2"

    def test_prefixes_filter(self) -> None:
        with mock.patch.dict(os.environ, {"X_A": "1", "Y_B": "2"}, clear=True):
            params = SystemParameters.from_environ("X_")
            assert "X_A" in params
            assert "Y_B" not in params


# ============================================================================
# from_ generic dispatch
# ============================================================================


class TestFromGeneric:
    def test_ellipsis_auto_fetches(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None), \
             mock.patch.object(sys, "argv", ["prog", "--k=v"]):
            params = SystemParameters.from_(...)
        assert params["k"] == "v"

    def test_none_auto_fetches(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None), \
             mock.patch.object(sys, "argv", ["prog"]):
            params = SystemParameters.from_(None)
        assert isinstance(params, SystemParameters)

    def test_identity_on_existing(self) -> None:
        original = SystemParameters({"a": "1"}, argv=None, dbutils=None)
        assert SystemParameters.from_(original) is original

    def test_mapping(self) -> None:
        params = SystemParameters.from_({"a": "1"})
        assert params["a"] == "1"

    def test_list_routes_to_argv(self) -> None:
        params = SystemParameters.from_(["--a=1", "pos"])
        assert params["a"] == "1"
        assert params.args == ("pos",)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot build SystemParameters"):
            SystemParameters.from_(42)


# ============================================================================
# Typed config via subclassing
# ============================================================================


class TestTypedSubclass:
    def test_basic_int_cast(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=42"], dbutils=None)
        assert cfg.count == 42
        assert isinstance(cfg.count, int)
        assert cfg["count"] == 42

    def test_default_when_missing(self) -> None:
        class Config(SystemParameters):
            count: int = 7

        cfg = Config(argv=None, dbutils=None)
        assert cfg.count == 7

    def test_string_field(self) -> None:
        class Config(SystemParameters):
            name: str = "default"

        cfg = Config(argv=["--name=alice"], dbutils=None)
        assert cfg.name == "alice"

    def test_bool_field_true(self) -> None:
        class Config(SystemParameters):
            verbose: bool = False

        cfg = Config(argv=["--verbose"], dbutils=None)
        assert cfg.verbose is True

    def test_bool_field_false(self) -> None:
        class Config(SystemParameters):
            verbose: bool = True

        cfg = Config({"verbose": "false"}, argv=None, dbutils=None)
        assert cfg.verbose is False

    def test_float_field(self) -> None:
        class Config(SystemParameters):
            ratio: float = 0.5

        cfg = Config(argv=["--ratio=3.14"], dbutils=None)
        assert cfg.ratio == 3.14

    def test_attribute_set_routes_to_explicit(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=5"], dbutils=None)
        assert cfg.count == 5
        cfg.count = 99
        assert cfg.count == 99
        # Set invalidates cache + writes to explicit overrides
        assert cfg["count"] == 99

    def test_dbutils_widget_value_casts(self) -> None:
        class Config(SystemParameters):
            count: int = 0

        dbutils = _FakeDBUtils(bindings={"count": "42"})
        cfg = Config(argv=None, dbutils=dbutils)
        assert cfg.count == 42

    def test_undeclared_key_stays_raw_string(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=5", "--extra=raw"], dbutils=None)
        assert cfg.count == 5
        assert cfg["extra"] == "raw"  # undeclared — no cast

    def test_subclass_inherits_parent_fields(self) -> None:
        class Base(SystemParameters):
            count: int = 1

        class Child(Base):
            name: str = "default"

        cfg = Child(argv=["--count=42", "--name=alice"], dbutils=None)
        assert cfg.count == 42
        assert cfg.name == "alice"

    def test_cast_error_is_loud(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=not_a_number"], dbutils=None)
        with pytest.raises(ValueError, match="cannot cast"):
            cfg.count

    def test_class_level_access_returns_descriptor(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        from yggdrasil.environ.parameters import _FieldDescriptor
        assert isinstance(Config.__dict__["count"], _FieldDescriptor)

    def test_cast_cache_avoids_redo(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=42"], dbutils=None)
        # Two reads should hit the cache on the second.
        with mock.patch("yggdrasil.environ.parameters.convert") as mock_convert:
            mock_convert.return_value = 999
            cfg._cast_cache.clear()
            assert cfg.count == 999
            assert cfg.count == 999
            mock_convert.assert_called_once()

    def test_datetime_field(self) -> None:
        class Config(SystemParameters):
            when: dt.datetime = dt.datetime(2020, 1, 1)

        cfg = Config(argv=["--when=2026-05-21T10:00:00"], dbutils=None)
        assert isinstance(cfg.when, dt.datetime)
        assert cfg.when.year == 2026
        assert cfg.when.month == 5
        assert cfg.when.day == 21

    def test_args_attribute_not_treated_as_field(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["pos1", "--count=5", "pos2"], dbutils=None)
        # ``args`` stays a tuple, not a descriptor.
        assert cfg.args == ("pos1", "pos2")
        assert cfg.count == 5


# ============================================================================
# as_dict / repr
# ============================================================================


class TestAccessors:
    def test_as_dict_includes_casts(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        cfg = Config(argv=["--count=5", "--name=alice"], dbutils=None)
        d = cfg.as_dict()
        assert d["count"] == 5
        assert d["name"] == "alice"
        assert type(d) is dict

    def test_repr_shows_sources(self) -> None:
        params = SystemParameters({"k": "v"}, argv=["--a=1", "pos"], dbutils=None)
        text = repr(params)
        assert "SystemParameters" in text
        assert "pos" in text
