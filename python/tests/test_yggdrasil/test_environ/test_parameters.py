"""Tests for ``yggdrasil.environ.parameters.SystemParameters``."""
from __future__ import annotations

import datetime as dt
import os
import sys
from collections.abc import Mapping
from enum import Enum
from typing import Any
from unittest import mock

import pytest

from yggdrasil.environ import SystemParameters

# Force registration of the IO / Path converters.
from yggdrasil.io import Holder  # noqa: F401, E402
from yggdrasil.io.path.path import Path  # noqa: F401, E402


class _Color(Enum):
    """Module-scope enum so ``get_type_hints`` can resolve forward refs."""

    RED = "red"
    BLUE = "blue"


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


# ============================================================================
# Iterable target — CSV pre-split for multiselect-shaped sources
# ============================================================================


class TestIterableCasting:
    def test_csv_string_to_list_of_strings(self) -> None:
        class Config(SystemParameters):
            names: list[str] = []

        cfg = Config(argv=["--names=a,b,c"], dbutils=None)
        assert cfg.names == ["a", "b", "c"]

    def test_csv_string_to_list_of_ints(self) -> None:
        class Config(SystemParameters):
            ports: list[int] = []

        cfg = Config(argv=["--ports=80,443,8080"], dbutils=None)
        assert cfg.ports == [80, 443, 8080]

    def test_all_values_tag_filtered(self) -> None:
        from yggdrasil.environ.parameters import ALL_VALUES_TAG

        class Config(SystemParameters):
            names: list[str] = []

        cfg = Config(argv=[f"--names=a,{ALL_VALUES_TAG},b"], dbutils=None)
        assert cfg.names == ["a", "b"]

    def test_bare_list_annotation(self) -> None:
        class Config(SystemParameters):
            items: list = []

        cfg = Config(argv=["--items=x,y"], dbutils=None)
        assert cfg.items == ["x", "y"]

    def test_set_annotation(self) -> None:
        class Config(SystemParameters):
            tags: set[str] = set()

        cfg = Config(argv=["--tags=a,b,a"], dbutils=None)
        assert cfg.tags == {"a", "b"}

    def test_str_field_not_split(self) -> None:
        class Config(SystemParameters):
            csv: str = ""

        cfg = Config(argv=["--csv=a,b,c"], dbutils=None)
        # ``str`` is iterable in Python but explicitly excluded from the
        # CSV-split heuristic — the raw value passes through.
        assert cfg.csv == "a,b,c"


# ============================================================================
# Widget surface (init_widgets / init_job / from_environment)
# ============================================================================


class _RecordingWidgets:
    """Captures every ``dbutils.widgets.*`` call for inspection."""

    def __init__(self, existing: set[str] | None = None) -> None:
        self.existing = set(existing or ())
        self.calls: list[tuple[str, tuple, dict]] = []

    def get(self, name: str) -> str:
        if name not in self.existing:
            raise RuntimeError(f"no such widget: {name}")
        return ""

    def text(self, name: str, default: str, label: str) -> None:
        self.calls.append(("text", (name, default, label), {}))

    def dropdown(self, name: str, default: str, options: list, label: str) -> None:
        self.calls.append(("dropdown", (name, default, tuple(options), label), {}))

    def combobox(self, name: str, default: str, options: list, label: str) -> None:
        self.calls.append(("combobox", (name, default, tuple(options), label), {}))

    def multiselect(self, name: str, default: str, options: list, label: str) -> None:
        self.calls.append(("multiselect", (name, default, tuple(options), label), {}))


class _RecordingDBUtils:
    def __init__(self, existing: set[str] | None = None) -> None:
        self.widgets = _RecordingWidgets(existing=existing)
        self.notebook = mock.MagicMock()


class TestInitWidgets:
    def test_text_widget_for_string_field(self) -> None:
        class Config(SystemParameters):
            name: str = "alice"

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        assert any(call[0] == "text" and call[1][0] == "name" for call in dbutils.widgets.calls)
        text_call = next(c for c in dbutils.widgets.calls if c[1][0] == "name")
        assert text_call[1][1] == "alice"

    def test_dropdown_widget_for_bool(self) -> None:
        class Config(SystemParameters):
            verbose: bool = False

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        dd = next(c for c in dbutils.widgets.calls if c[0] == "dropdown")
        assert dd[1][0] == "verbose"
        assert dd[1][2] == ("true", "false")

    def test_multiselect_for_list(self) -> None:
        class Config(SystemParameters):
            names: list[str] = ["a", "b"]

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        ms = next(c for c in dbutils.widgets.calls if c[0] == "multiselect")
        assert ms[1][0] == "names"
        assert ms[1][2] == ("a", "b")

    def test_dropdown_for_enum(self) -> None:
        class Config(SystemParameters):
            color: _Color = _Color.RED

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        dd = next(c for c in dbutils.widgets.calls if c[0] == "dropdown")
        assert dd[1][0] == "color"
        assert dd[1][2] == ("red", "blue")

    def test_datetime_widget_isoformat(self) -> None:
        import datetime as dt

        class Config(SystemParameters):
            when: dt.datetime = dt.datetime(2026, 5, 21, 10, 0, 0)

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        text_call = next(c for c in dbutils.widgets.calls if c[1][0] == "when")
        assert text_call[1][1] == "2026-05-21T10:00:00"

    def test_skip_existing(self) -> None:
        class Config(SystemParameters):
            a: int = 1
            b: int = 2

        dbutils = _RecordingDBUtils(existing={"a"})
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        names = {c[1][0] for c in dbutils.widgets.calls}
        assert "b" in names
        assert "a" not in names

    def test_silent_no_op_without_dbutils(self) -> None:
        class Config(SystemParameters):
            a: int = 1

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            Config.init_widgets()  # no raise


class TestInitJobLogging:
    """``init_job(logging=...)`` activates the yggdrasil logger without
    double-installing a handler when one is already reachable upstream."""

    @pytest.fixture(autouse=True)
    def _restore_logger(self) -> Any:
        import logging

        ygg_logger = logging.getLogger("yggdrasil")
        saved_level = ygg_logger.level
        saved_handlers = list(ygg_logger.handlers)
        try:
            yield
        finally:
            ygg_logger.setLevel(saved_level)
            ygg_logger.handlers[:] = saved_handlers

    def test_default_sets_info_level(self) -> None:
        import logging

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job()

        ygg_logger = logging.getLogger("yggdrasil")
        assert ygg_logger.level == logging.INFO

    def test_explicit_level_sets_that_level(self) -> None:
        import logging

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job(logging=logging.DEBUG)

        ygg_logger = logging.getLogger("yggdrasil")
        assert ygg_logger.level == logging.DEBUG

    def test_true_is_alias_for_info(self) -> None:
        import logging

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job(logging=True)

        ygg_logger = logging.getLogger("yggdrasil")
        assert ygg_logger.level == logging.INFO

    def test_false_leaves_logger_untouched(self) -> None:
        import logging

        ygg_logger = logging.getLogger("yggdrasil")
        ygg_logger.setLevel(logging.WARNING)

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job(logging=False)

        assert ygg_logger.level == logging.WARNING

    def test_none_leaves_logger_untouched(self) -> None:
        import logging

        ygg_logger = logging.getLogger("yggdrasil")
        ygg_logger.setLevel(logging.WARNING)

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job(logging=None)

        assert ygg_logger.level == logging.WARNING

    def test_does_not_add_handler_when_one_reachable(self) -> None:
        # The conftest already wired a handler on the yggdrasil logger.
        # ``hasHandlers()`` returns True → init_job must skip adding one.
        import logging

        ygg_logger = logging.getLogger("yggdrasil")
        before = len(ygg_logger.handlers)
        assert before >= 1, "conftest should have installed a handler"

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
            SystemParameters.init_job()

        assert len(ygg_logger.handlers) == before

    def test_adds_handler_when_none_reachable(self) -> None:
        import logging

        ygg_logger = logging.getLogger("yggdrasil")
        # Strip every handler from the propagation chain so
        # ``hasHandlers()`` returns False.
        ygg_logger.handlers.clear()
        root = logging.getLogger()
        saved_root = list(root.handlers)
        root.handlers.clear()
        try:
            with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None):
                SystemParameters.init_job()
            assert len(ygg_logger.handlers) == 1
            assert isinstance(ygg_logger.handlers[0], logging.StreamHandler)
        finally:
            root.handlers[:] = saved_root


class TestNiceLabel:
    def test_snake_case_title_cased(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("start_date") == "Start Date"

    def test_acronyms_preserved(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("start_date_utc") == "Start Date UTC"
        assert nice_label("user_id") == "User ID"
        assert nice_label("api_url") == "API URL"
        assert nice_label("bidding_zone_eic") == "Bidding Zone EIC"

    def test_single_word(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("verbose") == "Verbose"

    def test_single_acronym(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("url") == "URL"

    def test_hyphen_treated_as_separator(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("data-dir") == "Data Dir"

    def test_empty_round_trips(self) -> None:
        from yggdrasil.environ import nice_label
        assert nice_label("") == ""
        assert nice_label("__") == "__"

    def test_widget_label_uses_nice_label(self) -> None:
        class Config(SystemParameters):
            start_date_utc: str = ""
            user_id: int = 0
            verbose: bool = False

        dbutils = _RecordingDBUtils()
        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=dbutils):
            Config.init_widgets()

        labels = {c[1][0]: c[1][-1] for c in dbutils.widgets.calls}
        assert labels["start_date_utc"] == "Start Date UTC"
        assert labels["user_id"] == "User ID"
        assert labels["verbose"] == "Verbose"


class TestFromEnvironment:
    def test_from_environment_returns_instance(self) -> None:
        class Config(SystemParameters):
            count: int = 1

        with mock.patch.object(SystemParameters, "_get_dbutils", return_value=None), \
             mock.patch.object(sys, "argv", ["prog", "--count=42"]):
            cfg = Config.from_environment()
        assert cfg.count == 42


# ============================================================================
# NotebookConfig backward-compat alias
# ============================================================================


class TestNotebookConfigAlias:
    def test_alias_identity(self) -> None:
        from yggdrasil.databricks.jobs.config import NotebookConfig
        assert NotebookConfig is SystemParameters

    def test_subclass_works(self) -> None:
        from yggdrasil.databricks.jobs.config import NotebookConfig

        class Config(NotebookConfig):
            count: int = 1

        cfg = Config(argv=["--count=42"], dbutils=None)
        assert cfg.count == 42

    def test_reexport_from_jobs_package(self) -> None:
        from yggdrasil.databricks.jobs import NotebookConfig
        assert NotebookConfig is SystemParameters


# ============================================================================
# Converters: any_to_holder, any_to_path
# ============================================================================


class TestConverters:
    def test_any_to_holder_from_bytes(self) -> None:
        from yggdrasil.data.cast import convert
        from yggdrasil.io import Holder

        h = convert(b"hello", Holder)
        assert isinstance(h, Holder)
        assert h.read_bytes() == b"hello"

    def test_any_to_holder_identity(self) -> None:
        from yggdrasil.data.cast import convert
        from yggdrasil.io import Holder, Memory

        m = Memory(binary=b"x")
        assert convert(m, Holder) is m

    def test_any_to_path_from_str(self) -> None:
        from yggdrasil.data.cast import convert
        from yggdrasil.io.path.path import Path

        p = convert("/tmp/test.txt", Path)
        assert isinstance(p, Path)
        assert str(p).endswith("/tmp/test.txt")

    def test_any_to_path_identity(self) -> None:
        from yggdrasil.data.cast import convert
        from yggdrasil.io.path.path import Path
        from yggdrasil.io.path.local_path import LocalPath

        p = LocalPath("/tmp/x")
        assert convert(p, Path) is p

    def test_path_field_in_subclass(self) -> None:
        class Config(SystemParameters):
            data_dir: Path = None  # type: ignore[assignment]

        cfg = Config(argv=["--data_dir=/tmp/data"], dbutils=None)
        assert isinstance(cfg.data_dir, Path)
        assert str(cfg.data_dir).endswith("/tmp/data")

    def test_holder_field_in_subclass(self) -> None:
        class Config(SystemParameters):
            blob: Holder = None  # type: ignore[assignment]

        cfg = Config({"blob": b"payload"}, argv=None, dbutils=None)
        assert isinstance(cfg.blob, Holder)
        assert cfg.blob.read_bytes() == b"payload"
