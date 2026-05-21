"""Tests for ``yggdrasil.environ.parameters.SystemParameters``."""
from __future__ import annotations

import os
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


class TestFromArgv:
    def test_key_value_equals(self) -> None:
        params = SystemParameters.from_argv(["--name=alice", "--count=3"])
        assert params == {"name": "alice", "count": "3"}
        assert params.args == ()

    def test_key_value_space_separated(self) -> None:
        params = SystemParameters.from_argv(["--name", "alice", "--count", "3"])
        assert params == {"name": "alice", "count": "3"}

    def test_bare_flag_is_true(self) -> None:
        params = SystemParameters.from_argv(["--verbose", "--name=alice"])
        assert params == {"verbose": "true", "name": "alice"}

    def test_positional_args_preserved(self) -> None:
        params = SystemParameters.from_argv(
            ["one", "--name=alice", "two", "--count=3", "three"]
        )
        assert params == {"name": "alice", "count": "3"}
        assert params.args == ("one", "two", "three")

    def test_default_argv_reads_sys(self) -> None:
        with mock.patch.object(
            __import__("sys"), "argv", ["prog", "--foo=bar"]
        ):
            params = SystemParameters.from_argv()
        assert params["foo"] == "bar"

    def test_flag_followed_by_flag_stays_bool(self) -> None:
        params = SystemParameters.from_argv(["--a", "--b=2"])
        assert params == {"a": "true", "b": "2"}


class TestFromDBUtils:
    def test_raises_without_dbutils(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            with pytest.raises(RuntimeError, match="dbutils is not available"):
                SystemParameters.from_dbutils("foo")

    def test_named_widgets(self) -> None:
        dbutils = _FakeDBUtils(widgets={"name": "alice", "count": "3"})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils("name", "count")
        assert params == {"name": "alice", "count": "3"}

    def test_full_bindings_via_java_map(self) -> None:
        dbutils = _FakeDBUtils(bindings=_FakeBindings({"name": "alice", "count": "3"}))
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils()
        assert params == {"name": "alice", "count": "3"}

    def test_full_bindings_via_plain_dict(self) -> None:
        dbutils = _FakeDBUtils(bindings={"k1": "v1", "k2": 2})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.from_dbutils()
        assert params == {"k1": "v1", "k2": "2"}


class TestFromEnviron:
    def test_no_prefix_returns_full_env(self) -> None:
        with mock.patch.dict(os.environ, {"X_YGG_TEST_A": "1", "Y_OTHER": "2"}, clear=True):
            params = SystemParameters.from_environ()
        assert params["X_YGG_TEST_A"] == "1"
        assert params["Y_OTHER"] == "2"

    def test_with_prefix_filters(self) -> None:
        with mock.patch.dict(os.environ, {"X_YGG_TEST_A": "1", "Y_OTHER": "2"}, clear=True):
            params = SystemParameters.from_environ("X_YGG_TEST_")
        assert params == {"X_YGG_TEST_A": "1"}


class TestCurrent:
    def test_argv_only_when_no_dbutils(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            params = SystemParameters.current(argv=["pos1", "--name=alice"])
        assert params == {"name": "alice"}
        assert params.args == ("pos1",)

    def test_argv_overrides_dbutils_bindings(self) -> None:
        dbutils = _FakeDBUtils(bindings={"name": "from_widget", "extra": "v"})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.current(argv=["--name=from_cli"])
        assert params["name"] == "from_cli"
        assert params["extra"] == "v"

    def test_env_prefix_layer_under_bindings(self) -> None:
        dbutils = _FakeDBUtils(bindings={"DATABRICKS_JOB_ID": "from_widget"})
        env = {"DATABRICKS_JOB_ID": "from_env", "DATABRICKS_RUN_ID": "42", "OTHER": "x"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            params = SystemParameters.current(argv=[], env_prefix="DATABRICKS_")
        # Bindings override env on collision.
        assert params["DATABRICKS_JOB_ID"] == "from_widget"
        # Env-only keys still present.
        assert params["DATABRICKS_RUN_ID"] == "42"
        assert "OTHER" not in params

    def test_dbutils_bindings_failure_is_silent(self) -> None:
        class Broken:
            class notebook:
                class entry_point:
                    @staticmethod
                    def getCurrentBindings():
                        raise RuntimeError("no notebook entry point")
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=Broken()):
            params = SystemParameters.current(argv=["--k=v"])
        assert params == {"k": "v"}


class TestFromGeneric:
    def test_ellipsis_routes_to_current(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None), \
             mock.patch.object(__import__("sys"), "argv", ["prog", "--k=v"]):
            params = SystemParameters.from_(...)
        assert params["k"] == "v"

    def test_none_routes_to_current(self) -> None:
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None), \
             mock.patch.object(__import__("sys"), "argv", ["prog"]):
            params = SystemParameters.from_(None)
        assert isinstance(params, SystemParameters)

    def test_identity_on_existing_instance(self) -> None:
        original = SystemParameters({"a": "1"})
        assert SystemParameters.from_(original) is original

    def test_mapping_input(self) -> None:
        params = SystemParameters.from_({"a": "1"})
        assert params == {"a": "1"}

    def test_list_routes_to_argv(self) -> None:
        params = SystemParameters.from_(["--a=1", "pos"])
        assert params == {"a": "1"}
        assert params.args == ("pos",)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Cannot build SystemParameters"):
            SystemParameters.from_(42)


class TestDictBehavior:
    def test_is_dict_subclass(self) -> None:
        params = SystemParameters({"a": "1"})
        assert isinstance(params, dict)
        assert params["a"] == "1"

    def test_as_dict_returns_plain_dict(self) -> None:
        params = SystemParameters({"a": "1"}, args=("pos",))
        plain = params.as_dict()
        assert type(plain) is dict
        assert plain == {"a": "1"}

    def test_repr_includes_args(self) -> None:
        params = SystemParameters({"a": "1"}, args=("pos",))
        text = repr(params)
        assert "a" in text
        assert "pos" in text
