"""Tests for ``yggdrasil.cli.databricks.genie`` — the ``ygg-genie`` CLI.

Network is mocked: every test uses the Genie API mock from
:class:`DatabricksTestCase` so the REPL can run end-to-end without
touching a workspace. The REPL is driven via injectable
``input_fn`` / ``output_fn`` so we never need a TTY.
"""
from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

from databricks.sdk.service._internal import Wait
from databricks.sdk.service.dashboards import (
    GenieAttachment,
    GenieMessage,
    GenieQueryAttachment,
    MessageStatus,
    TextAttachment,
)

from yggdrasil.cli.databricks.genie import GenieCLI, main
from yggdrasil.databricks.tests import DatabricksTestCase


def _completed(
    *,
    space_id: str = "space-1",
    conversation_id: str = "conv-1",
    message_id: str = "msg-1",
    text: str | None = "ok",
    query: str | None = None,
) -> GenieMessage:
    attachments = [
        GenieAttachment(
            attachment_id="att-1",
            text=TextAttachment(id="tid", content=text) if text else None,
            query=GenieQueryAttachment(id="qid", query=query, statement_id="stmt") if query else None,
        )
    ]
    return GenieMessage(
        space_id=space_id,
        conversation_id=conversation_id,
        content="q",
        message_id=message_id,
        id=message_id,
        status=MessageStatus.COMPLETED,
        attachments=attachments,
    )


# ---------------------------------------------------------------------------
# Parser + defaults wiring (no client / REPL needed)
# ---------------------------------------------------------------------------
class TestParser(DatabricksTestCase):
    def test_parser_has_genie_and_agent_flags(self):
        parser = GenieCLI.build_parser()
        dests = {a.dest for a in parser._actions}
        for name in (
            # client (inherited from base)
            "host", "token", "profile",
            # genie
            "space_id", "warehouse_id", "auto_create_space",
            # agent
            "output_dir", "auto_save", "auto_save_format",
            # repl
            "question", "color",
            # base
            "debug",
        ):
            self.assertIn(name, dests)

    def test_managed_space_tables_collected(self):
        args = GenieCLI.build_parser().parse_args([
            "--managed-space-table", "main.sales.orders",
            "--managed-space-table", "main.sales.returns",
        ])
        self.assertEqual(args.managed_space_tables, ["main.sales.orders", "main.sales.returns"])

    def test_defaults_from_args_pass_through_when_no_flags(self):
        from yggdrasil.databricks.genie import GenieDefaults

        base = GenieDefaults()
        args = GenieCLI.build_parser().parse_args([])
        self.assertIs(GenieCLI.defaults_from_args(args, base), base)

    def test_defaults_from_args_applies_supplied_overrides(self):
        from yggdrasil.databricks.genie import GenieDefaults

        base = GenieDefaults()
        args = GenieCLI.build_parser().parse_args([
            "--space-id", "S",
            "--warehouse-id", "W",
            "--auto-save",
            "--auto-save-format", "csv",
            "--managed-space-table", "main.t.x",
        ])
        merged = GenieCLI.defaults_from_args(args, base)
        self.assertEqual(merged.space_id, "S")
        self.assertEqual(merged.warehouse_id, "W")
        self.assertTrue(merged.agent_auto_save)
        self.assertEqual(merged.agent_auto_save_format, "csv")
        self.assertEqual(merged.managed_space_tables, ("main.t.x",))
        # Untouched fields keep their base value.
        self.assertEqual(merged.space_name, base.space_name)


# ---------------------------------------------------------------------------
# REPL via injected input/output (no TTY)
# ---------------------------------------------------------------------------
class CLIBase(DatabricksTestCase):
    def setUp(self):
        super().setUp()
        self.genie_api = self.workspace_client.genie
        self.client.genie.defaults = replace(
            self.client.genie.defaults, space_id="space-1",
        )
        import tempfile

        self.tmpdir = tempfile.mkdtemp(prefix="ygg-genie-cli-")
        self.addCleanup(self._rmtree, self.tmpdir)
        self.client.genie.defaults = replace(
            self.client.genie.defaults, agent_output_dir=self.tmpdir,
        )

        self.captured: list[str] = []
        self.inputs: list[str] = []
        self.cli = GenieCLI(
            client=self.client,
            color=False,
            input_fn=self._pop_input,
            output_fn=self.captured.append,
        )

    @staticmethod
    def _rmtree(path: str) -> None:
        import shutil

        shutil.rmtree(path, ignore_errors=True)

    def _pop_input(self, _prompt: str) -> str:
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def _out(self) -> str:
        return "\n".join(self.captured)

    def _stub_start(self, message: GenieMessage) -> None:
        self.genie_api.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            space_id=message.space_id,
        )

    def _stub_create(self, message: GenieMessage) -> None:
        self.genie_api.create_message.return_value = Wait(
            waiter=lambda **kwargs: message,
            response=MagicMock(),
            conversation_id=message.conversation_id,
            message_id=message.message_id,
            space_id=message.space_id,
        )


class TestREPLAsk(CLIBase):
    def test_plain_input_routes_to_agent(self):
        self._stub_start(_completed(text="hello"))
        self.inputs = ["how many orders?", "/quit"]
        self.assertEqual(self.cli.run_repl(), 0)
        self.assertIn("hello", self._out())
        self.assertEqual(self.cli.conversation_id, "conv-1")

    def test_follow_up_reuses_conversation(self):
        self._stub_start(_completed(text="first"))
        self._stub_create(_completed(message_id="m2", text="second"))
        self.inputs = ["q1", "q2", "/quit"]
        self.cli.run_repl()
        self.genie_api.create_message.assert_called_once_with(
            space_id="space-1", conversation_id="conv-1", content="q2",
        )

    def test_eof_exits(self):
        self.assertEqual(self.cli.run_repl(), 0)
        self.assertIn("bye.", self._out())

    def test_ask_once(self):
        self._stub_start(_completed(text="answered"))
        self.assertEqual(self.cli.ask_once("how many?"), 0)
        self.assertIn("answered", self._out())

    def test_ask_renders_sql(self):
        self._stub_start(_completed(text="here", query="SELECT 1"))
        self.inputs = ["q", "/quit"]
        self.cli.run_repl()
        self.assertIn("SELECT 1", self._out())

    def test_ask_surfaces_errors(self):
        self.genie_api.start_conversation.side_effect = RuntimeError("boom")
        self.inputs = ["q", "/quit"]
        self.cli.run_repl()
        self.assertIn("boom", self._out())


class TestSlashCommands(CLIBase):
    def test_help_lists_commands(self):
        self.inputs = ["/help", "/quit"]
        self.cli.run_repl()
        out = self._out()
        for needed in ("/help", "/save", "/sql", "/spaces", "/feedback"):
            self.assertIn(needed, out)

    def test_unknown_slash(self):
        self.inputs = ["/nope", "/quit"]
        self.cli.run_repl()
        self.assertIn("unknown command", self._out())

    def test_reset_clears_conversation_id(self):
        self._stub_start(_completed(text="a"))
        self.inputs = ["q", "/reset", "/quit"]
        self.cli.run_repl()
        self.assertIsNone(self.cli.conversation_id)

    def test_history(self):
        self._stub_start(_completed(text="hello"))
        self.inputs = ["q", "/history", "/quit"]
        self.cli.run_repl()
        self.assertIn("hello", self._out())

    def test_last_reprints(self):
        self._stub_start(_completed(text="my reply"))
        self.inputs = ["q", "/last", "/quit"]
        self.cli.run_repl()
        self.assertGreaterEqual(self._out().count("my reply"), 2)

    def test_save_without_last(self):
        self.inputs = ["/save", "/quit"]
        self.cli.run_repl()
        self.assertIn("nothing to save", self._out())

    def test_save_json_writes_file(self):
        self._stub_start(_completed(text="hi", query=None))
        self.inputs = ["q", "/save json", "/quit"]
        self.cli.run_repl()
        self.assertIn("saved", self._out())

    def test_sql_command(self):
        self._stub_start(_completed(text="hi", query="SELECT count(*) FROM t"))
        self.inputs = ["q", "/sql", "/quit"]
        self.cli.run_repl()
        self.assertGreaterEqual(self._out().count("SELECT count(*) FROM t"), 2)

    def test_url(self):
        self._stub_start(_completed(text="hi"))
        self.inputs = ["q", "/url", "/quit"]
        self.cli.run_repl()
        self.assertIn("genie/rooms/space-1", self._out())

    def test_space_switch(self):
        self._stub_start(_completed(text="hi"))
        self.inputs = ["q", "/space space-9", "/quit"]
        self.cli.run_repl()
        self.assertEqual(self.client.genie.defaults.space_id, "space-9")
        self.assertIsNone(self.cli.conversation_id)

    def test_spaces_lists(self):
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[
            MagicMock(space_id="aa", title="Alpha"),
            MagicMock(space_id="bb", title="Beta"),
        ])
        self.inputs = ["/spaces", "/quit"]
        self.cli.run_repl()
        self.assertIn("Alpha", self._out())

    def test_cleanup(self):
        self.genie_api.list_spaces.return_value = MagicMock(spaces=[])
        self.inputs = ["/cleanup", "/quit"]
        self.cli.run_repl()
        self.assertIn("no duplicates", self._out())

    def test_tools_listing(self):
        self.inputs = ["/tools", "/quit"]
        self.cli.run_repl()
        out = self._out()
        self.assertIn("save_parquet", out)
        self.assertIn("polars", out)

    def test_defaults_command(self):
        self.inputs = ["/defaults", "/quit"]
        self.cli.run_repl()
        self.assertIn("space_id", self._out())

    def test_output_dir(self):
        self.inputs = ["/output-dir", "/quit"]
        self.cli.run_repl()
        self.assertIn(self.tmpdir, self._out())

    def test_feedback_usage_when_no_args(self):
        self.inputs = ["/feedback", "/quit"]
        self.assertEqual(self.cli.run_repl(), 0)
        self.assertIn("usage:", self._out())


# ---------------------------------------------------------------------------
# main() entry point — uses the shared parse_and_run from the base
# ---------------------------------------------------------------------------
class TestMainEntry(DatabricksTestCase):
    def test_main_one_shot(self):
        client = self.client
        client.genie.defaults = replace(client.genie.defaults, space_id="space-1")
        self.workspace_client.genie.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _completed(text="one-shot"),
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )

        import io

        buf = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             patch("sys.stdout", buf):
            rc = main([
                "--host", "x.databricks.com",
                "--token", "tok",
                "-q", "one shot",
            ])
        self.assertEqual(rc, 0)
        self.assertIn("one-shot", buf.getvalue())

    def test_main_failure_returns_2(self):
        import io

        err = io.StringIO()
        with patch("yggdrasil.databricks.client.DatabricksClient", side_effect=RuntimeError("nope")), \
             patch("sys.stderr", err):
            rc = main(["--host", "x.databricks.com", "-q", "test"])
        self.assertEqual(rc, 2)
        self.assertIn("nope", err.getvalue())

    def test_main_applies_defaults_overrides(self):
        # Confirms ``GenieCLI.run`` re-applies CLI flags onto the
        # service defaults *before* asking — auto-save format set via
        # the CLI must reach ``GenieDefaults`` by the time the one-shot
        # answer fires.
        client = self.client
        self.workspace_client.genie.start_conversation.return_value = Wait(
            waiter=lambda **kwargs: _completed(text="ok"),
            response=MagicMock(),
            conversation_id="conv-1",
            message_id="msg-1",
            space_id="space-1",
        )
        with patch("yggdrasil.databricks.client.DatabricksClient", return_value=client):
            rc = main([
                "--host", "x.databricks.com",
                "--space-id", "space-1",
                "--auto-save-format", "csv",
                "-q", "test",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(client.genie.defaults.agent_auto_save_format, "csv")


# ---------------------------------------------------------------------------
# --deploy-skills — upload local skills to a workspace path
# ---------------------------------------------------------------------------
class TestDeploySkills(CLIBase):
    """``GenieCLI.deploy_skills`` walks a local dir and writes each .md.

    Network is mocked: the test patches ``WorkspacePath.from_`` to
    return a stub whose ``write_bytes`` / ``mkdir`` calls land on a
    recorded list, so we never touch a real workspace.
    """

    def setUp(self):
        super().setUp()
        import tempfile
        from pathlib import Path

        # Build a throwaway skills dir mirroring the real layout.
        self.skills_root = Path(tempfile.mkdtemp(prefix="ygg-skills-src-"))
        self.addCleanup(self._rmtree, str(self.skills_root))
        (self.skills_root / "user_instructions.md").write_text("# user\n")
        (self.skills_root / ".assistant_workspace_instructions.md").write_text("# workspace\n")
        skills_subdir = self.skills_root / "skills"
        skills_subdir.mkdir()
        (skills_subdir / "ygg-install.md").write_text("# install\n")
        (skills_subdir / "ygg-pitfalls.md").write_text("# pitfalls\n")

    def test_parser_exposes_deploy_flags(self):
        parser = GenieCLI.build_parser()
        dests = {a.dest for a in parser._actions}
        for n in ("deploy_skills", "skills_dir", "deploy_target", "deploy_overwrite"):
            self.assertIn(n, dests)

    def test_resolve_skills_dir_prefers_argument(self):
        # Manually wire the args namespace as the parser would.
        import argparse

        self.cli.args = argparse.Namespace(skills_dir=str(self.skills_root))
        resolved = self.cli._resolve_skills_dir()
        self.assertEqual(resolved, self.skills_root)

    def test_resolve_skills_dir_falls_back_to_env(self):
        import argparse
        import os

        self.cli.args = argparse.Namespace(skills_dir=None)
        prior = os.environ.get("YGG_SKILLS_DIR")
        os.environ["YGG_SKILLS_DIR"] = str(self.skills_root)
        try:
            resolved = self.cli._resolve_skills_dir()
        finally:
            if prior is None:
                os.environ.pop("YGG_SKILLS_DIR", None)
            else:
                os.environ["YGG_SKILLS_DIR"] = prior
        self.assertEqual(resolved, self.skills_root)

    def test_deploy_skills_missing_dir_returns_2(self):
        import argparse

        self.cli.args = argparse.Namespace(
            skills_dir="/path/does/not/exist",
            deploy_target=None,
            deploy_overwrite=True,
        )
        rc = self.cli.deploy_skills()
        self.assertEqual(rc, 2)
        self.assertIn("skills source not found", self._out())

    def test_deploy_skills_uploads_every_md(self):
        import argparse
        from unittest.mock import MagicMock as _MM, patch as _patch

        self.cli.args = argparse.Namespace(
            skills_dir=str(self.skills_root),
            deploy_target="/Workspace/Users/me/.ygg/da",
            deploy_overwrite=True,
        )

        # Patch WorkspacePath.from_ to return a stub that records writes.
        recorded: list[tuple[str, bytes]] = []

        class _StubWP:
            def __init__(self, dest):
                self.dest = dest
                self.parent = _MM()

            def write_bytes(self, data, overwrite=False):
                recorded.append((self.dest, data))

        with _patch(
            "yggdrasil.databricks.fs.workspace_path.WorkspacePath.from_",
            side_effect=lambda dest, **_kw: _StubWP(dest),
        ):
            rc = self.cli.deploy_skills()

        self.assertEqual(rc, 0)
        dests = sorted(d for d, _ in recorded)
        self.assertIn("/Workspace/Users/me/.ygg/da/.assistant_workspace_instructions.md", dests)
        self.assertIn("/Workspace/Users/me/.ygg/da/user_instructions.md", dests)
        self.assertIn("/Workspace/Users/me/.ygg/da/skills/ygg-install.md", dests)
        self.assertIn("/Workspace/Users/me/.ygg/da/skills/ygg-pitfalls.md", dests)
        # Body bytes round-trip the source content.
        body_by_dest = dict(recorded)
        self.assertEqual(
            body_by_dest["/Workspace/Users/me/.ygg/da/skills/ygg-install.md"],
            b"# install\n",
        )

    def test_deploy_skills_continues_on_per_file_failure(self):
        import argparse
        from unittest.mock import MagicMock as _MM, patch as _patch

        self.cli.args = argparse.Namespace(
            skills_dir=str(self.skills_root),
            deploy_target="/Workspace/Shared/.ygg/da",
            deploy_overwrite=True,
        )

        recorded: list[str] = []

        class _StubWP:
            def __init__(self, dest):
                self.dest = dest
                self.parent = _MM()

            def write_bytes(self, data, overwrite=False):
                # Make one of the writes fail; the rest still go through.
                if "ygg-install" in self.dest:
                    raise RuntimeError("upstream 5xx")
                recorded.append(self.dest)

        with _patch(
            "yggdrasil.databricks.fs.workspace_path.WorkspacePath.from_",
            side_effect=lambda dest, **_kw: _StubWP(dest),
        ):
            rc = self.cli.deploy_skills()

        # Some files uploaded → 0 (the failure is logged, not fatal).
        self.assertEqual(rc, 0)
        # Failure was reported …
        out = self._out()
        self.assertIn("upstream 5xx", out)
        self.assertIn("ygg-install.md", out)
        # … but the other three still landed.
        self.assertEqual(len(recorded), 3)

    def test_deploy_skills_empty_dir_returns_1(self):
        import argparse
        import tempfile
        from pathlib import Path

        empty = Path(tempfile.mkdtemp(prefix="ygg-skills-empty-"))
        self.addCleanup(self._rmtree, str(empty))

        self.cli.args = argparse.Namespace(
            skills_dir=str(empty),
            deploy_target="/Workspace/x",
            deploy_overwrite=True,
        )
        rc = self.cli.deploy_skills()
        self.assertEqual(rc, 1)
        self.assertIn("no .md files found", self._out())

    def test_main_deploy_skills_runs_deploy_path(self):
        # End-to-end through main(argv): parser builds, defaults merge,
        # deploy_skills runs, no REPL.
        import io
        from unittest.mock import MagicMock as _MM, patch as _patch

        client = self.client
        recorded: list[str] = []

        class _StubWP:
            def __init__(self, dest):
                self.dest = dest
                self.parent = _MM()

            def write_bytes(self, data, overwrite=False):
                recorded.append(self.dest)

        buf = io.StringIO()
        with _patch("yggdrasil.databricks.client.DatabricksClient", return_value=client), \
             _patch(
                "yggdrasil.databricks.fs.workspace_path.WorkspacePath.from_",
                side_effect=lambda dest, **_kw: _StubWP(dest),
             ), \
             _patch("sys.stdout", buf):
            rc = main([
                "--host", "x.databricks.com",
                "--deploy-skills",
                "--skills-dir", str(self.skills_root),
                "--deploy-target", "/Workspace/Users/test/.ygg/da",
                "--no-color",
            ])
        self.assertEqual(rc, 0)
        self.assertEqual(len(recorded), 4)
