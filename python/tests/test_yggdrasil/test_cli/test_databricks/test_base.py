"""Tests for the :class:`DatabricksCLI` abstract base.

Concrete sub-services pull in :class:`DatabricksCLI` and only declare
their own argparse group + ``run`` method. The base owns the shared
``--host`` / ``--token`` / … client flag group and the
:class:`DatabricksClient` construction handshake — those are what we
exercise here, in isolation from any particular sub-service.
"""
from __future__ import annotations

import argparse
from unittest.mock import patch

from yggdrasil.cli.databricks.base import CLIENT_FLAGS, DatabricksCLI
from yggdrasil.databricks.tests import DatabricksTestCase


class _StubCLI(DatabricksCLI):
    """Minimal subclass — adds a single service flag and a trivial ``run``."""

    prog = "ygg-stub"
    description = "stub for tests"

    last_instance: "_StubCLI | None" = None

    @classmethod
    def add_service_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--foo", dest="foo", default=None)

    def run(self) -> int:
        # Record self for the parse_and_run round-trip tests.
        type(self).last_instance = self
        return 0


class TestParser(DatabricksTestCase):
    def test_client_flag_group_present(self):
        parser = _StubCLI.build_parser()
        dests = {a.dest for a in parser._actions}
        for _flag, dest, _meta in CLIENT_FLAGS:
            self.assertIn(dest, dests)

    def test_service_arguments_attached(self):
        parser = _StubCLI.build_parser()
        dests = {a.dest for a in parser._actions}
        self.assertIn("foo", dests)

    def test_debug_flag_present(self):
        parser = _StubCLI.build_parser()
        dests = {a.dest for a in parser._actions}
        self.assertIn("debug", dests)

    def test_client_kwargs_filters_none_values(self):
        args = _StubCLI.build_parser().parse_args([
            "--host", "x.databricks.com",
            "--token", "tok",
            "--profile", "DEFAULT",
            "--foo", "bar",
        ])
        kwargs = _StubCLI.client_kwargs(args)
        self.assertEqual(kwargs, {
            "host": "x.databricks.com",
            "token": "tok",
            "profile": "DEFAULT",
        })
        # ``foo`` is a *service* flag — must not leak into client kwargs.
        self.assertNotIn("foo", kwargs)

    def test_client_kwargs_empty_when_no_flags(self):
        args = _StubCLI.build_parser().parse_args([])
        self.assertEqual(_StubCLI.client_kwargs(args), {})


class TestRunHandshake(DatabricksTestCase):
    def test_parse_and_run_builds_client_and_invokes_run(self):
        _StubCLI.last_instance = None
        client = self.client
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = _StubCLI.parse_and_run([
                "--host", "x.databricks.com", "--token", "tok",
            ])
        self.assertEqual(rc, 0)
        self.assertIsNotNone(_StubCLI.last_instance)
        self.assertIs(_StubCLI.last_instance.client, client)
        # The parsed argparse namespace is held on the instance.
        self.assertEqual(_StubCLI.last_instance.args.host, "x.databricks.com")

    def test_parse_and_run_returns_2_on_client_failure(self):
        import io

        err = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", side_effect=RuntimeError("nope")), \
             patch("sys.stderr", err):
            rc = _StubCLI.parse_and_run(["--host", "x.databricks.com"])
        self.assertEqual(rc, 2)
        self.assertIn("nope", err.getvalue())
        self.assertIn("ygg-stub", err.getvalue())

    def test_debug_flag_sets_yggdrasil_logger_level(self):
        import logging

        client = self.client
        # Drop any prior level so we can assert the flag re-set it.
        logging.getLogger("yggdrasil").setLevel(logging.WARNING)
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            _StubCLI.parse_and_run([
                "--host", "x.databricks.com", "--token", "tok", "--debug",
            ])
        self.assertEqual(logging.getLogger("yggdrasil").level, logging.DEBUG)


class TestAbstractEnforcement(DatabricksTestCase):
    def test_cannot_instantiate_without_subclass(self):
        # ABC: instantiating the base raises immediately.
        with self.assertRaises(TypeError):
            DatabricksCLI(client=self.client, args=argparse.Namespace())  # type: ignore[abstract]
