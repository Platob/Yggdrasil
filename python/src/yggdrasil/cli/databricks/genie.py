"""``ygg-genie`` — conversational CLI on top of :class:`GenieAgent`.

Subclasses :class:`DatabricksCLI` so the ``--host`` / ``--token`` /
``--profile`` / ``--auth-type`` … flag group is shared with every other
``ygg-<service>`` script. Genie-specific flags (space, warehouse,
managed-space template) and agent-specific flags (output dir, auto-save)
live in their own groups added by :meth:`add_service_arguments`.

The REPL itself is intentionally stdlib-only — no ``rich`` /
``prompt_toolkit`` / ``click`` dependency. Plain input lines are sent
to Genie; lines starting with ``/`` are local slash commands routed
through :meth:`GenieCLI.dispatch`.
"""
from __future__ import annotations

import argparse
import logging
import shlex
import sys
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

from yggdrasil.cli.databricks.base import DatabricksCLI

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.genie import GenieAnswer, GenieDefaults

__all__ = ["GenieCLI", "main"]

LOGGER = logging.getLogger(__name__)

#: Slash commands shown in ``/help``. Maps name → one-line summary.
_SLASH_HELP: dict[str, str] = {
    "/help": "Show this list of commands.",
    "/quit, /exit": "Leave the REPL.",
    "/login": "Configure Databricks credentials (host + token).",
    "/clear": "Clear the screen.",
    "/reset": "Start a fresh Genie conversation (does NOT clear local history).",
    "/history": "Print every Q & A from this session.",
    "/last": "Re-print the last Genie answer.",
    "/save [format] [path]": "Save the last answer (default format from --auto-save-format).",
    "/sql": "Show the SQL Genie generated for the last answer.",
    "/url": "Print the workspace URL for the current conversation.",
    "/space [id]": "Show the active space id, or switch to a different one.",
    "/spaces": "List Genie spaces accessible to the current identity.",
    "/cleanup": "Trash duplicate managed-title spaces (calls cleanup_dead_spaces).",
    "/feedback positive|negative [comment]": "Rate the last answer.",
    "/defaults": "Show the active GenieDefaults dataclass.",
    "/tools": "List registered agent tools.",
    "/output-dir": "Print the local output directory.",
}


# ---------------------------------------------------------------------------
# Tiny ANSI helper. Auto-disables when stdout is not a TTY.
# ---------------------------------------------------------------------------
class _Style:
    RESET = "\x1b[0m"
    DIM = "\x1b[2m"
    BOLD = "\x1b[1m"
    CYAN = "\x1b[36m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    RED = "\x1b[31m"
    MAGENTA = "\x1b[35m"

    def __init__(self, enabled: bool):
        self.enabled = enabled and sys.stdout.isatty()

    def __call__(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"{code}{text}{self.RESET}"

    def dim(self, t: str) -> str: return self(self.DIM, t)
    def bold(self, t: str) -> str: return self(self.BOLD, t)
    def cyan(self, t: str) -> str: return self(self.CYAN, t)
    def green(self, t: str) -> str: return self(self.GREEN, t)
    def yellow(self, t: str) -> str: return self(self.YELLOW, t)
    def red(self, t: str) -> str: return self(self.RED, t)
    def magenta(self, t: str) -> str: return self(self.MAGENTA, t)


class GenieCLI(DatabricksCLI):
    """Conversational CLI on top of :class:`yggdrasil.databricks.genie.GenieAgent`."""

    prog = "ygg-genie"
    description = (
        "Conversational CLI on top of Databricks Genie. Plain input goes to "
        "Genie; lines starting with '/' are local commands. See '/help' once "
        "inside the REPL for the full slash-command list."
    )
    epilog = """\
        Examples:
          ygg-genie --profile DEFAULT
          ygg-genie --host my.cloud.databricks.com --token "$DBX_TOKEN"
          ygg-genie --profile DEFAULT --space-id 01ef... --auto-save
          ygg-genie --profile DEFAULT -q "How many orders last month?"
    """

    # ------------------------------------------------------------------ #
    # Construction — accepts the optional test seams the previous module
    # exposed (``color``, ``input_fn``, ``output_fn``) so tests can drive
    # the REPL without a TTY.
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        client: Any,
        args: Optional[argparse.Namespace] = None,
        *,
        color: bool = True,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        super().__init__(client=client, args=args or argparse.Namespace())
        # ``args`` is None when callers build a ``GenieCLI`` directly
        # without going through ``parse_and_run`` (tests + library use).
        color_from_args = getattr(self.args, "color", True)
        self.style = _Style(enabled=color and color_from_args)
        self.input_fn = input_fn
        self.output_fn = output_fn
        self.conversation_id: Optional[str] = None
        self.commands: dict[str, Callable[[list[str]], Optional[bool]]] = {
            "help": self._cmd_help,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
            "login": self._cmd_login,
            "clear": self._cmd_clear,
            "reset": self._cmd_reset,
            "history": self._cmd_history,
            "last": self._cmd_last,
            "save": self._cmd_save,
            "sql": self._cmd_sql,
            "url": self._cmd_url,
            "space": self._cmd_space,
            "spaces": self._cmd_spaces,
            "cleanup": self._cmd_cleanup,
            "feedback": self._cmd_feedback,
            "defaults": self._cmd_defaults,
            "tools": self._cmd_tools,
            "output-dir": self._cmd_output_dir,
        }

    # ------------------------------------------------------------------ #
    # Argparse
    # ------------------------------------------------------------------ #
    @classmethod
    def add_service_arguments(cls, parser: argparse.ArgumentParser) -> None:
        genie_grp = parser.add_argument_group("Genie defaults")
        genie_grp.add_argument(
            "--space-id", dest="space_id", default=None,
            help="Genie space id (skips auto-pick).",
        )
        genie_grp.add_argument(
            "--space-name", dest="space_name", default=None,
            help="Bias space auto-pick by exact title match.",
        )
        genie_grp.add_argument(
            "--warehouse-id", dest="warehouse_id", default=None,
            help="SQL warehouse id used to materialise Genie's query results.",
        )
        genie_grp.add_argument(
            "--auto-pick-space", dest="auto_pick_space", default=None,
            action=argparse.BooleanOptionalAction,
            help="Allow Genie to pick a space when --space-id is unset (default on).",
        )
        genie_grp.add_argument(
            "--auto-create-space", dest="auto_create_space", default=None,
            action=argparse.BooleanOptionalAction,
            help="Create a default Genie space when none resolves (off by default).",
        )
        genie_grp.add_argument(
            "--cleanup-dead-spaces", dest="cleanup_dead_spaces", default=None,
            action=argparse.BooleanOptionalAction,
            help="Trash duplicate managed-title spaces on ensure_space() (off by default).",
        )
        genie_grp.add_argument(
            "--managed-space-title", dest="managed_space_title", default=None,
            help="Title used for auto-created and cleanup-tracked spaces.",
        )
        genie_grp.add_argument(
            "--managed-space-table", dest="managed_space_tables",
            action="append", default=None, metavar="CATALOG.SCHEMA.TABLE",
            help="Table to expose in an auto-created space (repeatable).",
        )

        agent_grp = parser.add_argument_group("Agent")
        agent_grp.add_argument(
            "--output-dir", dest="output_dir", default=None,
            help="Where to save artifacts (default: $XDG_CACHE_HOME/yggdrasil/genie).",
        )
        agent_grp.add_argument(
            "--auto-save", dest="auto_save", default=None,
            action=argparse.BooleanOptionalAction,
            help="Save Genie's query result after every reply.",
        )
        agent_grp.add_argument(
            "--auto-save-format", dest="auto_save_format", default=None,
            choices=["parquet", "csv", "arrow", "json", "text"],
            help="Format used by --auto-save (default parquet).",
        )

        repl_grp = parser.add_argument_group("REPL")
        repl_grp.add_argument(
            "-q", "--question", dest="question", default=None,
            help="Ask one question, print the reply, exit (no REPL).",
        )
        repl_grp.add_argument(
            "--no-color", dest="color", default=True, action="store_false",
            help="Disable ANSI colors.",
        )

        deploy_grp = parser.add_argument_group("Workspace skill deployment")
        deploy_grp.add_argument(
            "--deploy-skills", dest="deploy_skills", default=False,
            action="store_true",
            help="Upload assistant skills + instructions to the workspace, then exit.",
        )
        deploy_grp.add_argument(
            "--skills-dir", dest="skills_dir", default=None,
            help="Source directory for skills (default: $YGG_SKILLS_DIR, "
                 "else ./databricks-assistant).",
        )
        deploy_grp.add_argument(
            "--deploy-target", dest="deploy_target", default=None,
            help="Workspace target path (default: /Workspace/Users/<me>/.ygg/databricks-assistant).",
        )
        deploy_grp.add_argument(
            "--deploy-overwrite", dest="deploy_overwrite", default=True,
            action=argparse.BooleanOptionalAction,
            help="Overwrite existing files at the target (default on).",
        )

    # ------------------------------------------------------------------ #
    # Defaults wiring
    # ------------------------------------------------------------------ #
    @staticmethod
    def defaults_from_args(args: argparse.Namespace, base: "GenieDefaults") -> "GenieDefaults":
        """Merge CLI overrides into a base :class:`GenieDefaults`."""
        updates: dict[str, Any] = {}
        if getattr(args, "space_id", None) is not None:
            updates["space_id"] = args.space_id
        if getattr(args, "space_name", None) is not None:
            updates["space_name"] = args.space_name
        if getattr(args, "warehouse_id", None) is not None:
            updates["warehouse_id"] = args.warehouse_id
        if getattr(args, "auto_pick_space", None) is not None:
            updates["auto_pick_space"] = args.auto_pick_space
        if getattr(args, "auto_create_space", None) is not None:
            updates["auto_create_space"] = args.auto_create_space
        if getattr(args, "cleanup_dead_spaces", None) is not None:
            updates["cleanup_dead_spaces"] = args.cleanup_dead_spaces
        if getattr(args, "managed_space_title", None) is not None:
            updates["managed_space_title"] = args.managed_space_title
        if getattr(args, "managed_space_tables", None) is not None:
            updates["managed_space_tables"] = tuple(args.managed_space_tables)
        if getattr(args, "output_dir", None) is not None:
            updates["agent_output_dir"] = args.output_dir
        if getattr(args, "auto_save", None) is not None:
            updates["agent_auto_save"] = args.auto_save
        if getattr(args, "auto_save_format", None) is not None:
            updates["agent_auto_save_format"] = args.auto_save_format
        return _dc_replace(base, **updates) if updates else base

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    @property
    def genie(self):
        return self.client.genie

    @property
    def agent(self):
        return self.genie.agent

    @property
    def defaults(self):
        return self.genie.defaults

    # ------------------------------------------------------------------ #
    # Output helpers
    # ------------------------------------------------------------------ #
    def out(self, text: str = "") -> None:
        self.output_fn(text)

    def info(self, text: str) -> None:
        self.out(self.style.dim(text))

    def warn(self, text: str) -> None:
        self.out(self.style.yellow(text))

    def error(self, text: str) -> None:
        self.out(self.style.red(text))

    def success(self, text: str) -> None:
        self.out(self.style.green(text))

    # ------------------------------------------------------------------ #
    # Banner / entry
    # ------------------------------------------------------------------ #
    def print_banner(self) -> None:
        host = getattr(self.client, "host", None) or "?"
        self.out("")
        self.out(self.style.bold("  Yggdrasil Genie") + self.style.dim(f"  ·  {host}"))
        space_id = self.defaults.space_id
        if space_id:
            self.info(f"  space: {space_id}")
        else:
            self.info("  space: (auto-pick on first ask)")
        self.info(f"  output: {self.agent.output_dir}")
        self.info("  type a question · '/help' for commands · '/quit' to exit")
        self.out("")

    def run(self) -> int:
        """Apply CLI defaults to the Genie service, then drop into the REPL.

        When ``--deploy-skills`` is on, upload the local skills /
        instructions to the workspace and exit. When ``--question/-q``
        was supplied, fall back to a single-shot ask + exit (no REPL
        prompt). Otherwise drop into the REPL.
        """
        if not self._ensure_connected():
            return 2

        self.genie.defaults = self.defaults_from_args(self.args, self.genie.defaults)

        if getattr(self.args, "deploy_skills", False):
            return self.deploy_skills()

        question = getattr(self.args, "question", None)
        if question is not None:
            return self.ask_once(question)
        return self.run_repl()

    # ------------------------------------------------------------------ #
    # Credential setup
    # ------------------------------------------------------------------ #
    def _ensure_connected(self) -> bool:
        """Validate the client can reach Databricks, prompting on failure."""
        try:
            self.client.make_config()
            return True
        except Exception:
            pass
        self.warn("  Databricks credentials not configured.")
        return self._prompt_credentials()

    def _prompt_credentials(self) -> bool:
        """Interactively ask for host (+ optional token) and rebuild the client."""
        from yggdrasil.databricks.client import DatabricksClient

        try:
            host = self.input_fn(
                self.style.cyan("  host ")
                + self.style.dim("(e.g. https://adb-123.7.azuredatabricks.net): "),
            ).strip()
            if not host:
                self.error("  host is required.")
                return False
            token = self.input_fn(
                self.style.cyan("  token ")
                + self.style.dim("(PAT, empty for browser SSO): "),
            ).strip() or None
        except (EOFError, KeyboardInterrupt):
            self.out("")
            return False

        try:
            self.client = DatabricksClient(host=host, token=token)
            self.client.make_config()
            self.success(f"  connected to {host}")
            return True
        except Exception as exc:
            self.error(f"  failed: {exc}")
            return False

    # ------------------------------------------------------------------ #
    # Workspace skill deployment
    # ------------------------------------------------------------------ #
    def deploy_skills(self) -> int:
        """Upload ``databricks-assistant/`` to the workspace.

        Reads from ``--skills-dir`` (or ``$YGG_SKILLS_DIR``, then
        ``./databricks-assistant``) and uploads every ``*.md`` to
        ``--deploy-target`` (default
        ``/Workspace/Users/<me>/.ygg/databricks-assistant``). Sub-paths
        ``skills/<file>.md`` and the two top-level instruction files
        (``.assistant_workspace_instructions.md`` /
        ``user_instructions.md``) are preserved relative to the source
        root.

        Returns 0 on success, non-zero when the source directory is
        missing or no files were uploaded.
        """
        import os
        from pathlib import Path

        source = self._resolve_skills_dir()
        if source is None or not source.is_dir():
            self.error(
                f"  skills source not found: {source}. "
                "Pass --skills-dir or set YGG_SKILLS_DIR."
            )
            return 2

        target_root = (
            getattr(self.args, "deploy_target", None)
            or os.environ.get("YGG_SKILLS_TARGET")
            or self._default_deploy_target()
        )
        overwrite = bool(getattr(self.args, "deploy_overwrite", True))

        # Collect (source_file, relative_path) pairs so the workspace
        # layout mirrors the local tree (skills/ subfolder preserved).
        candidates: list[tuple[Path, str]] = []
        for path in sorted(source.rglob("*.md")):
            rel = path.relative_to(source)
            candidates.append((path, str(rel).replace(os.sep, "/")))

        if not candidates:
            self.warn(f"  no .md files found under {source}")
            return 1

        self.info(f"  deploying {len(candidates)} file(s) from {source}")
        self.info(f"  target: {target_root}")

        uploaded = 0
        for src, rel in candidates:
            dest = f"{target_root.rstrip('/')}/{rel}"
            try:
                self._upload_one(src, dest, overwrite=overwrite)
            except Exception as exc:
                self.error(f"  ✗ {rel}: {type(exc).__name__}: {exc}")
                continue
            self.success(f"  ✓ {rel}")
            uploaded += 1

        if uploaded == 0:
            self.error("  no files uploaded")
            return 1
        self.success(f"  done — {uploaded}/{len(candidates)} uploaded.")
        return 0

    def _resolve_skills_dir(self):
        """Pick the source dir for ``--deploy-skills``."""
        import os
        from pathlib import Path

        arg = getattr(self.args, "skills_dir", None)
        if arg:
            return Path(arg).expanduser()
        env = os.environ.get("YGG_SKILLS_DIR")
        if env:
            return Path(env).expanduser()
        return Path.cwd() / "databricks-assistant"

    def _default_deploy_target(self) -> str:
        """Default workspace target: ``/Workspace/Users/<me>/.ygg/databricks-assistant``.

        Falls back to ``/Workspace/Shared/.ygg/databricks-assistant``
        when the current identity can't be resolved (service principal
        or notebook context unavailable).
        """
        try:
            current_user = self.client.workspace_client().current_user.me()
            user_name = getattr(current_user, "user_name", None)
            if user_name:
                return f"/Workspace/Users/{user_name}/.ygg/databricks-assistant"
        except Exception:
            LOGGER.debug("Could not resolve current user; falling back to /Workspace/Shared",
                         exc_info=True)
        return "/Workspace/Shared/.ygg/databricks-assistant"

    def _upload_one(self, src, dest: str, *, overwrite: bool) -> None:
        """Upload one file to the workspace path.

        Uses :class:`WorkspacePath` so retry / cache-invalidation /
        parent-creation behave like every other lifecycle op in the
        codebase. ``overwrite=True`` is the default — re-deploying
        replaces the existing file; ``False`` raises if it already
        exists.
        """
        from yggdrasil.databricks import WorkspacePath

        wp = WorkspacePath.from_(dest, service=self.client.workspaces)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_bytes(src.read_bytes(), overwrite=overwrite)

    def run_repl(self) -> int:
        """REPL loop until the user quits. Returns a process exit code."""
        self.print_banner()
        while True:
            try:
                line = self.input_fn(self.style.cyan("› "))
            except (EOFError, KeyboardInterrupt):
                self.out("")
                self.info("bye.")
                return 0
            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                stop = self.dispatch(line)
                if stop:
                    return 0
                continue
            self.ask(line)

    def ask_once(self, question: str) -> int:
        """One-shot ask used by ``-q / --question``."""
        self.ask(question)
        return 0

    def ask(self, question: str) -> None:
        self.info("  asking Genie…")
        try:
            answer = self.agent.run(question, conversation_id=self.conversation_id)
        except KeyboardInterrupt:
            self.warn("  interrupted.")
            return
        except ValueError as exc:
            if "auto-pick" in str(exc) and self._prompt_space():
                self.ask(question)
                return
            self.error(f"  error: {exc}")
            LOGGER.debug("ask failed", exc_info=True)
            return
        except Exception as exc:
            self.error(f"  error: {type(exc).__name__}: {exc}")
            LOGGER.debug("ask failed", exc_info=True)
            return
        self.conversation_id = answer.conversation_id or self.conversation_id
        self._render_answer(answer)

    def _prompt_space(self) -> bool:
        """List available Genie spaces and let the user pick one."""
        try:
            spaces = list(self.genie.list_spaces())
        except Exception:
            spaces = []

        if spaces:
            self.out("")
            self.info("  Available Genie spaces:")
            for i, sp in enumerate(spaces, 1):
                title = getattr(sp._details, "title", None) if sp._details else None
                label = f"{sp.space_id}"
                if title:
                    label += f"  {title}"
                self.out(f"  {self.style.dim(f'{i:>3}.')} {label}")
            self.out("")
            try:
                choice = self.input_fn(
                    self.style.cyan("  pick a space ")
                    + self.style.dim(f"(1-{len(spaces)}): "),
                ).strip()
            except (EOFError, KeyboardInterrupt):
                self.out("")
                return False
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(spaces):
                    picked = spaces[idx]
                    self.genie.defaults = _dc_replace(
                        self.defaults, space_id=picked.space_id,
                    )
                    title = getattr(picked._details, "title", None) if picked._details else None
                    self.success(f"  using space {picked.space_id}" + (f" ({title})" if title else ""))
                    return True
            except ValueError:
                pass
            self.error(f"  invalid choice: {choice!r}")
            return False
        else:
            self.warn("  no Genie spaces found — creating one.")
            return self._prompt_create_space()

    def _prompt_create_space(self) -> bool:
        """Prompt for a warehouse, auto-create a Genie space."""
        try:
            warehouses = list(self.client.warehouses.list())
        except Exception:
            warehouses = []

        wh_id = self.defaults.warehouse_id
        if not wh_id:
            if warehouses:
                self.out("")
                self.info("  Available warehouses:")
                for i, wh in enumerate(warehouses, 1):
                    name = getattr(wh, "name", None) or getattr(wh, "warehouse_id", "?")
                    self.out(f"  {self.style.dim(f'{i:>3}.')} {name}")
                self.out("")
                try:
                    choice = self.input_fn(
                        self.style.cyan("  pick a warehouse ")
                        + self.style.dim(f"(1-{len(warehouses)}): "),
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    self.out("")
                    return False
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(warehouses):
                        wh_id = warehouses[idx].warehouse_id
                except (ValueError, AttributeError):
                    pass
            if not wh_id:
                try:
                    wh_id = self.input_fn(
                        self.style.cyan("  warehouse id: "),
                    ).strip() or None
                except (EOFError, KeyboardInterrupt):
                    self.out("")
                    return False
            if not wh_id:
                self.error("  warehouse id is required to create a space.")
                return False

        self.genie.defaults = _dc_replace(
            self.defaults,
            warehouse_id=wh_id,
            auto_create_space=True,
        )
        self.info("  creating space…")
        return True

    # ------------------------------------------------------------------ #
    # Slash dispatch
    # ------------------------------------------------------------------ #
    def dispatch(self, line: str) -> Optional[bool]:
        try:
            tokens = shlex.split(line[1:])
        except ValueError as exc:
            self.error(f"  could not parse command: {exc}")
            return None
        if not tokens:
            return None
        name, *rest = tokens
        name = name.lower()
        handler = self.commands.get(name)
        if handler is None:
            self.error(f"  unknown command '/{name}'. Type '/help'.")
            return None
        try:
            return handler(rest)
        except SystemExit:
            raise
        except Exception as exc:  # never let a slash command kill the REPL
            self.error(f"  /{name} failed: {type(exc).__name__}: {exc}")
            LOGGER.debug("slash command failed", exc_info=True)
            return None

    # ------------------------------------------------------------------ #
    # Slash commands
    # ------------------------------------------------------------------ #
    def _cmd_help(self, _args: list[str]) -> None:
        self.out("")
        for name, summary in _SLASH_HELP.items():
            self.out(f"  {self.style.cyan(name):<48s} {self.style.dim(summary)}")
        self.out("")

    def _cmd_quit(self, _args: list[str]) -> bool:
        self.info("bye.")
        return True

    def _cmd_login(self, _args: list[str]) -> None:
        self._prompt_credentials()

    def _cmd_clear(self, _args: list[str]) -> None:
        if self.style.enabled:
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.flush()

    def _cmd_reset(self, _args: list[str]) -> None:
        self.conversation_id = None
        self.info("  conversation reset — next question starts a new thread.")

    def _cmd_history(self, _args: list[str]) -> None:
        if not self.agent.history:
            self.info("  (empty)")
            return
        for i, ans in enumerate(self.agent.history, 1):
            preview = (ans.text or "").splitlines()[0] if ans.text else "(no text)"
            self.out(f"  {self.style.dim(f'{i:>2}.')} {preview}")

    def _cmd_last(self, _args: list[str]) -> None:
        last = self.agent.last()
        if last is None:
            self.info("  (no answer yet)")
            return
        self._render_answer(last)

    def _cmd_save(self, args: list[str]) -> None:
        last = self.agent.last()
        if last is None:
            self.warn("  nothing to save — ask a question first.")
            return
        fmt = args[0] if len(args) >= 1 else None
        path = args[1] if len(args) >= 2 else None
        result = self.agent.save(last, format=fmt, path=path)
        if result is None:
            self.warn("  no query attachment to save — try '/save json' for metadata only.")
        else:
            self.success(f"  saved → {result}")

    def _cmd_sql(self, _args: list[str]) -> None:
        last = self.agent.last()
        if last is None or not last.query:
            self.info("  (no SQL on the last answer)")
            return
        self.out("")
        for line in last.query.splitlines():
            self.out("    " + self.style.yellow(line))
        self.out("")

    def _cmd_url(self, _args: list[str]) -> None:
        last = self.agent.last()
        if last is None:
            self.info("  (no conversation yet)")
            return
        self.out(f"  {last.url()}")

    def _cmd_space(self, args: list[str]) -> None:
        if not args:
            self.info(f"  active space: {self.defaults.space_id or '(auto-pick)'}")
            return
        new_id = args[0]
        self.genie.defaults = _dc_replace(self.defaults, space_id=new_id)
        self.conversation_id = None
        self.success(f"  switched to space {new_id} (conversation reset)")

    def _cmd_spaces(self, _args: list[str]) -> None:
        count = 0
        for sp in self.genie.list_spaces():
            title = getattr(sp._details, "title", None) if sp._details else None
            self.out(f"  {self.style.cyan(sp.space_id)}  {title or ''}")
            count += 1
        if count == 0:
            self.info("  (none)")

    def _cmd_cleanup(self, _args: list[str]) -> None:
        trashed = self.genie.cleanup_dead_spaces()
        if not trashed:
            self.info("  no duplicates to clean.")
        else:
            for sid in trashed:
                self.success(f"  trashed {sid}")

    def _cmd_feedback(self, args: list[str]) -> None:
        if not args:
            self.warn("  usage: /feedback positive|negative [comment]")
            return
        rating = args[0]
        comment = " ".join(args[1:]) or None
        last = self.agent.last()
        if last is None:
            self.warn("  nothing to rate yet.")
            return
        last.feedback(rating, comment=comment)
        self.success(f"  feedback recorded ({rating})")

    def _cmd_defaults(self, _args: list[str]) -> None:
        from dataclasses import asdict

        for k, v in asdict(self.defaults).items():
            self.out(f"  {self.style.dim(f'{k:>32s}')}  {v!r}")

    def _cmd_tools(self, _args: list[str]) -> None:
        for name in sorted(self.agent.tools):
            self.out(f"  {self.style.cyan(name)}")

    def _cmd_output_dir(self, _args: list[str]) -> None:
        self.out(f"  {self.agent.output_dir}")

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_answer(self, answer: "GenieAnswer") -> None:
        self.out("")
        text = answer.text or ""
        if text:
            for line in text.splitlines() or [text]:
                self.out(f"  {line}")
            self.out("")
        if answer.query:
            self.out(self.style.dim("  SQL:"))
            for line in answer.query.splitlines():
                self.out("    " + self.style.yellow(line))
            self.out("")
        if answer.is_failed:
            err = answer.error
            self.error(f"  status: {getattr(answer.status, 'name', answer.status)}")
            if err:
                self.error(f"  {err}")
        else:
            status = getattr(answer.status, "name", str(answer.status))
            self.info(f"  status: {status}  ·  msg: {answer.message_id}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``ygg-genie`` entry point."""
    return GenieCLI.parse_and_run(argv)


if __name__ == "__main__":  # pragma: no cover - manual entry
    raise SystemExit(main())
