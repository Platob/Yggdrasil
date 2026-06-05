"""Dispatch tests for ``ygg databricks configure`` (temp config + session files)."""
from __future__ import annotations

import configparser
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from yggdrasil.databricks.cli import main
from yggdrasil.databricks.cli.services.configure import ConfigureCommand


@contextlib.contextmanager
def _sandbox():
    """A temp ``~/.databrickscfg`` + a temp remembered-session file."""
    d = tempfile.mkdtemp()
    cfg = os.path.join(d, "databrickscfg")
    session = Path(d) / "ygg-session.json"
    client = MagicMock()
    client.workspace_client.return_value.current_user.me.return_value.user_name = "me@co.com"
    client.product = "yggdrasil"
    client.product_version = "9.9"
    client.account_id = None
    client.workspace_id = 42
    # ``set_current`` is a classmethod — calls land on the class mock, not the
    # instance — so expose the class via ``client._cls`` for assertions.
    with patch("yggdrasil.databricks.client.DatabricksClient") as dbc, \
         patch.object(ConfigureCommand, "_session_file", staticmethod(lambda: session)), \
         patch("yggdrasil.cli.style.print_logo"), \
         contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        dbc.return_value = client
        client._cls = dbc
        yield cfg, session, client


class TestConfigureHelp(unittest.TestCase):
    def test_configure_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["configure", "--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_configure_list_help_exits_zero(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["configure", "list", "--help"])
        self.assertEqual(ctx.exception.code, 0)


class TestConfigureWrite(unittest.TestCase):
    def test_writes_default_profile_and_session(self):
        with _sandbox() as (cfg, session, client):
            rc = main(["configure", "--config-file", cfg,
                       "--host", "my-ws.cloud.databricks.com", "--token", "dapi-xyz"])
        self.assertEqual(rc, 0)

        parser = configparser.ConfigParser()
        parser.read(cfg)
        # Default profile lands in configparser's DEFAULT section; host is
        # normalised to https:// and the token is persisted.
        self.assertEqual(parser["DEFAULT"]["host"], "https://my-ws.cloud.databricks.com")
        self.assertEqual(parser["DEFAULT"]["token"], "dapi-xyz")

        # Session snapshot dumped, with no secret material.
        meta = json.loads(session.read_text())
        self.assertEqual(meta["profile"], "DEFAULT")
        self.assertEqual(meta["host"], "https://my-ws.cloud.databricks.com")
        self.assertEqual(meta["user"], "me@co.com")
        self.assertEqual(meta["auth_type"], "pat")
        self.assertNotIn("token", meta)
        client._cls.set_current.assert_called_once_with(client)

    def test_named_profile_preserves_existing(self):
        with _sandbox() as (cfg, session, client):
            Path(cfg).write_text("[prod]\nhost = https://prod\ntoken = old\n")
            rc = main(["configure", "--config-file", cfg, "--profile", "staging",
                       "--host", "https://staging", "--token", "stg-tok"])
        self.assertEqual(rc, 0)
        parser = configparser.ConfigParser()
        parser.read(cfg)
        # New section added, the pre-existing one untouched.
        self.assertEqual(parser["staging"]["host"], "https://staging")
        self.assertEqual(parser["prod"]["host"], "https://prod")

    def test_oauth_writes_client_id_secret_not_token(self):
        with _sandbox() as (cfg, session, client):
            rc = main(["configure", "--config-file", cfg, "--profile", "sp",
                       "--host", "https://ws", "--client-id", "cid", "--client-secret", "csecret"])
        self.assertEqual(rc, 0)
        parser = configparser.ConfigParser()
        parser.read(cfg)
        self.assertEqual(parser["sp"]["client_id"], "cid")
        self.assertEqual(parser["sp"]["client_secret"], "csecret")
        self.assertNotIn("token", parser["sp"])
        self.assertEqual(json.loads(session.read_text())["auth_type"], "oauth")

    def test_no_session_skips_remember(self):
        with _sandbox() as (cfg, session, client):
            rc = main(["configure", "--config-file", cfg, "--no-session",
                       "--host", "https://ws", "--token", "t"])
        self.assertEqual(rc, 0)
        self.assertFalse(session.exists())
        client._cls.set_current.assert_not_called()

    def test_no_verify_skips_workspace_call(self):
        with _sandbox() as (cfg, session, client):
            rc = main(["configure", "--config-file", cfg, "--no-verify",
                       "--host", "https://ws", "--token", "t"])
        self.assertEqual(rc, 0)
        client.workspace_client.assert_not_called()

    def test_verification_failure_still_saves_profile(self):
        with _sandbox() as (cfg, session, client):
            client.workspace_client.return_value.current_user.me.side_effect = RuntimeError("401")
            rc = main(["configure", "--config-file", cfg,
                       "--host", "https://ws", "--token", "bad"])
        # Profile is written regardless; configure itself succeeded.
        self.assertEqual(rc, 0)
        parser = configparser.ConfigParser()
        parser.read(cfg)
        self.assertEqual(parser["DEFAULT"]["token"], "bad")
        # Session still remembered, but with no resolved user.
        self.assertIsNone(json.loads(session.read_text())["user"])

    def test_missing_host_noninteractive_returns_one(self):
        with _sandbox() as (cfg, session, client), \
             patch("sys.stdin") as stdin:
            stdin.isatty.return_value = False
            rc = main(["configure", "--config-file", cfg, "--token", "t"])
        self.assertEqual(rc, 1)


class TestConfigureListAndSession(unittest.TestCase):
    def test_list_shows_profiles_and_marks_current(self):
        buf = io.StringIO()
        with _sandbox() as (cfg, session, client):
            main(["configure", "--config-file", cfg, "--profile", "prod",
                  "--host", "https://prod", "--token", "t"])
            with contextlib.redirect_stdout(buf):
                rc = main(["configure", "list", "--config-file", cfg])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("prod", out)
        self.assertIn("https://prod", out)

    def test_session_prints_remembered_metadata(self):
        buf = io.StringIO()
        with _sandbox() as (cfg, session, client):
            main(["configure", "--config-file", cfg,
                  "--host", "https://ws", "--token", "t"])
            with contextlib.redirect_stdout(buf):
                rc = main(["configure", "session"])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["host"], "https://ws")

    def test_session_without_record_returns_one(self):
        d = tempfile.mkdtemp()
        missing = Path(d) / "nope.json"
        with patch.object(ConfigureCommand, "_session_file", staticmethod(lambda: missing)), \
             patch("yggdrasil.cli.style.print_logo"), \
             contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            rc = main(["configure", "session"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
