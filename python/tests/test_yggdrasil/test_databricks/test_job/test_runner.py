"""Unit tests for the ``ygg-run`` CLI — target resolution + signature coercion.

These exercise the on-cluster side of the transparent dispatch without a live
workspace (no ``--payload`` / ``--result`` round-trip)."""
from __future__ import annotations

import sys
from datetime import date

from yggdrasil.databricks.job import flow, task
from yggdrasil.databricks.job import runner


def test_resolve_target_imports_module_attr():
    obj = runner.resolve_target(f"{__name__}:_demo_flow")
    assert obj is _demo_flow


def test_bind_and_convert_coerces_strings_to_signature():
    def f(day: date, n: int, label: str):
        return (day, n, label)

    bound = runner.bind_and_convert(f, ("2024-01-02", "7", "hello"), {})
    assert bound.args == (date(2024, 1, 2), 7, "hello")   # str → date / int / str


def test_bind_and_convert_is_identity_for_typed_values():
    def f(n: int):
        return n

    bound = runner.bind_and_convert(f, (5,), {})
    assert bound.args == (5,)


def test_bind_and_convert_applies_defaults_and_skips_unannotated():
    def f(a, b: int = 3):
        return a + b

    bound = runner.bind_and_convert(f, ("x",), {})
    assert bound.arguments["a"] == "x"        # unannotated → untouched
    assert bound.arguments["b"] == 3          # default applied


def test_main_runs_target_with_string_params(capsys):
    # No --payload/--result → positional string params, coerced to the signature.
    argv = [f"{__name__}:_adder", "2", "40"]
    assert runner.main(argv) == 0


# Targets used by the tests above (must be importable module attributes).
@flow(name="demo")
def _demo_flow(x: int) -> int:
    return x


@task
def _adder(a: int, b: int) -> int:
    assert a + b == 42
    return a + b


def test_main_injects_databricks_client_and_sets_current():
    import sys
    from unittest.mock import MagicMock, patch
    module = sys.modules[__name__]
    if hasattr(module, "databricks"):       # a prior runner call may have injected one
        del module.databricks
    sentinel = MagicMock(name="client")
    with patch("yggdrasil.databricks.client.DatabricksClient") as DC:
        DC.return_value = sentinel
        rc = runner.main([f"{__name__}:_adder", "2", "40"])
    assert rc == 0
    DC.set_current.assert_called_once_with(sentinel)
    # injected as a module global so a bare `databricks` in the body resolves
    assert module.databricks is sentinel
