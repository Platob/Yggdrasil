"""Tests for :mod:`yggdrasil.databricks.jobs.inputs`.

Validates the argv parser shape and that ``read_widgets`` /
``read_job_parameters`` both raise loudly when ``dbutils`` is
not on the path. Live dbutils paths are exercised against fakes
so the suite stays runnable outside a Databricks runtime.
"""
from __future__ import annotations

import builtins
import unittest
from unittest.mock import MagicMock

from yggdrasil.databricks.jobs.inputs import (
    get_dbutils,
    read_argv,
    read_job_parameters,
    read_widgets,
)


class TestReadArgv(unittest.TestCase):

    def test_long_form_equals(self):
        self.assertEqual(
            read_argv(["--name=alice", "--count=9"]),
            {"name": "alice", "count": "9"},
        )

    def test_long_form_split(self):
        self.assertEqual(
            read_argv(["--name", "alice", "--count", "9"]),
            {"name": "alice", "count": "9"},
        )

    def test_mixed_forms(self):
        self.assertEqual(
            read_argv(["--name=alice", "--count", "9", "--flag"]),
            {"name": "alice", "count": "9", "flag": "true"},
        )

    def test_bare_positional_tokens_are_skipped(self):
        # Stray positional inputs (no ``--``) are silently dropped.
        self.assertEqual(
            read_argv(["positional", "--key=value", "another"]),
            {"key": "value"},
        )

    def test_bare_flag_at_tail_defaults_to_true(self):
        # Trailing ``--flag`` with no following token keeps the bool semantics.
        self.assertEqual(read_argv(["--flag"]), {"flag": "true"})

    def test_default_pulls_from_sys_argv(self):
        import sys
        original = sys.argv
        try:
            sys.argv = ["script.py", "--name=bob", "--n=3"]
            self.assertEqual(read_argv(), {"name": "bob", "n": "3"})
        finally:
            sys.argv = original


class _FakeWidgets:
    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    def get(self, name: str) -> str:
        return self._mapping[name]


class _FakeDbutils:
    def __init__(self, mapping: dict[str, str], bindings: dict[str, str] | None = None) -> None:
        self.widgets = _FakeWidgets(mapping)
        # notebook.entry_point.getCurrentBindings() shape.
        bindings_obj = MagicMock()
        bindings_obj.getCurrentBindings.return_value = dict(bindings or mapping)
        notebook_obj = MagicMock()
        notebook_obj.entry_point = bindings_obj
        self.notebook = notebook_obj


class _DbutilsInBuiltins:
    """Context manager that injects a fake ``dbutils`` into ``builtins``."""

    def __init__(self, dbutils: object) -> None:
        self.dbutils = dbutils

    def __enter__(self) -> object:
        builtins.dbutils = self.dbutils  # type: ignore[attr-defined]
        return self.dbutils

    def __exit__(self, *exc: object) -> None:
        del builtins.dbutils  # type: ignore[attr-defined]


class TestGetDbutils(unittest.TestCase):

    def test_returns_none_outside_databricks(self):
        # Test env doesn't have dbutils on builtins.
        self.assertIsNone(get_dbutils())

    def test_picks_up_builtins_injection(self):
        sentinel = object()
        with _DbutilsInBuiltins(sentinel):
            self.assertIs(get_dbutils(), sentinel)


class TestReadWidgets(unittest.TestCase):

    def test_reads_named_widgets(self):
        fake = _FakeDbutils({"name": "alice", "count": "9", "extra": "skip"})
        with _DbutilsInBuiltins(fake):
            self.assertEqual(
                read_widgets("name", "count"),
                {"name": "alice", "count": "9"},
            )

    def test_raises_when_dbutils_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            read_widgets("anything")
        self.assertIn("dbutils.widgets is not available", str(ctx.exception))


class TestReadJobParameters(unittest.TestCase):

    def test_returns_all_bindings(self):
        fake = _FakeDbutils(
            {"name": "alice"},
            bindings={"name": "alice", "count": "9"},
        )
        with _DbutilsInBuiltins(fake):
            self.assertEqual(
                read_job_parameters(),
                {"name": "alice", "count": "9"},
            )

    def test_raises_when_dbutils_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            read_job_parameters()
        self.assertIn("dbutils is not available", str(ctx.exception))

    def test_raises_when_getCurrentBindings_fails(self):
        fake = MagicMock()
        fake.notebook.entry_point.getCurrentBindings.side_effect = RuntimeError("nope")
        with _DbutilsInBuiltins(fake):
            with self.assertRaises(RuntimeError) as ctx:
                read_job_parameters()
            self.assertIn("getCurrentBindings()", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
