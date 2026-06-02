"""``ygg databricks genie`` — a rich, self-driving Databricks Genie console.

A Claude-CLI-style experience: a branded logo, a live spinner while Genie
thinks, answers in rounded panels, results in bordered tables. It drives
the *whole* Databricks surface from one prompt — not just Genie:

    ygg databricks genie spaces
    ygg databricks genie ask "How many renewable sites are there?" --space 01ef…
    ygg databricks genie agent "top 3 sites by capacity" --space 01ef… \
        --planner databricks-claude-sonnet-4
    ygg databricks genie console --space 01ef…     # interactive

Inside the interactive console: plain text asks Genie; ``/agent <goal>``
runs the autonomous agent; ``/sql <query>`` runs raw SQL; ``/tables`` /
``/catalogs`` / ``/warehouses`` / ``/spaces`` browse; ``/use`` switches
space; ``/new`` / ``/help`` / ``/quit``.

Space id falls back to ``$YGG_GENIE_SPACE``; the agent's planner LLM to
``$YGG_GENIE_PLANNER`` (default ``databricks-claude-sonnet-4``).
"""
from __future__ import annotations

import os
import textwrap
from typing import Any, Optional

from yggdrasil.cli import style

#: Prose wrap width — keeps answer panels readable on narrow terminals
#: (SQL is left unwrapped so its formatting survives).
_WRAP = 88


def _wrap(text: str) -> str:
    """Wrap prose to :data:`_WRAP` columns, preserving blank-line paragraphs."""
    out = []
    for para in text.strip().split("\n"):
        out.append(textwrap.fill(para, width=_WRAP) if para.strip() else "")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_answer(answer: Any, *, with_data: bool = True) -> None:
    """Print a GenieAnswer: prose panel, SQL panel, result table."""
    if answer.failed:
        style.fail(answer.error or "Genie could not answer that.")
        return
    if answer.text:
        style.out(style.panel(_wrap(answer.text), title="answer") + "\n")
    if answer.questions and not answer.text:
        body = "\n".join(f"{style.brand('•')} {q}" for q in answer.questions)
        style.out(style.panel(body, title="genie suggests") + "\n")
    if answer.sql:
        style.out(style.panel(answer.sql.strip(), title="sql", color="38;5;42") + "\n")
    if with_data and answer.has_query:
        _render_result(answer)


def _render_result(answer: Any, limit: int = 20) -> None:
    try:
        table = answer.to_arrow()
    except Exception as exc:  # noqa: BLE001 — result fetch is best-effort
        style.warn(f"could not fetch result: {exc}")
        return
    _render_arrow(table, limit)


def _render_arrow(table: Any, limit: int = 30) -> None:
    if table is None or table.num_rows == 0:
        style.out("  " + style.muted("(no rows)") + "\n")
        return
    rows = [[r.get(c) for c in table.column_names] for r in table.to_pylist()]
    style.out(style.table(table.column_names, rows, max_rows=limit) + "\n")


def _render_transcript(run: Any, *, with_data: bool = True) -> None:
    for i, turn in enumerate(run.turns, 1):
        who = style.good("agent") if turn.autonomous else style.brand("goal")
        style.out(f"  {style.muted(f'{i}.')} {who}  {turn.question}\n")
        if turn.answer.text:
            style.out("     " + _wrap(turn.answer.text).replace("\n", "\n     ") + "\n")
    data = run.data_answer
    if with_data and data is not None:
        if data.sql:
            style.out("\n" + style.panel(data.sql.strip(), title="sql", color="38;5;42") + "\n")
        _render_result(data)
    style.ok(f"done in {len(run.turns)} turn(s)")


def _resolve_space(args: Any) -> Optional[str]:
    return getattr(args, "space_id", None) or os.environ.get("YGG_GENIE_SPACE") or None


# ---------------------------------------------------------------------------
# Interactive console
# ---------------------------------------------------------------------------


class GenieConsole:
    """The interactive REPL — owns the active space + conversation."""

    def __init__(self, client: Any, space_id: str, *, planner: Any = None, max_turns: int = 4):
        self.client = client
        self.space_id = space_id
        self.planner = planner
        self.max_turns = max_turns
        self.space = client.genie.space(space_id)
        self.conversation = None
        self._space_cache: list[Any] = []

    @property
    def _prompt(self) -> str:
        tag = "✦" if self.conversation is None else "↳"
        return f"  {style.brand(tag)} {style.bold('genie')} {style.muted('❯')} "

    def banner(self) -> None:
        style.print_logo("GENIE")
        lines = [
            f"{style.muted('space')}    {style.bold(self._space_title())}  "
            f"{style.muted(self.space_id)}",
            f"{style.muted('planner')}  {self._planner_label()}",
            "",
            f"{style.muted('Ask anything, or')}  "
            f"{style.brand('/agent')} {style.muted('·')} {style.brand('/sql')} "
            f"{style.muted('·')} {style.brand('/tables')} {style.muted('·')} "
            f"{style.brand('/spaces')} {style.muted('·')} {style.brand('/help')} "
            f"{style.muted('·')} {style.brand('/quit')}",
        ]
        style.out(style.panel("\n".join(lines), title="Databricks Genie") + "\n\n")

    def _space_title(self) -> str:
        try:
            return self.space.title or "(untitled)"
        except Exception:  # noqa: BLE001 — banner must never crash
            return "(unknown)"

    def _planner_label(self) -> str:
        if not self.planner:
            return style.muted("heuristic (Genie suggestions)")
        return f"{style.good(str(self.planner))} {style.muted('(autonomous)')}"

    # -- loop --------------------------------------------------------------
    def run(self) -> int:
        self.banner()
        while True:
            try:
                line = input(self._prompt).strip()
            except (EOFError, KeyboardInterrupt):
                style.out("\n  " + style.muted("bye.") + "\n")
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if self._command(line):
                    return 0
                continue
            self._ask(line)

    def _command(self, line: str) -> bool:
        """Handle a slash command. Returns ``True`` to exit the REPL."""
        cmd, _, arg = line.partition(" ")
        arg, cmd = arg.strip(), cmd.lower()
        if cmd in ("/quit", "/exit", "/q"):
            style.out("  " + style.muted("bye.") + "\n")
            return True
        if cmd in ("/help", "/h", "/?"):
            self._help()
        elif cmd == "/new":
            self.conversation = None
            style.ok("started a new conversation")
        elif cmd == "/spaces":
            self._spaces()
        elif cmd == "/use":
            self._use(arg)
        elif cmd == "/agent":
            self._agent(arg)
        elif cmd == "/sql":
            self._sql(arg)
        elif cmd == "/tables":
            self._sql(f"SHOW TABLES IN {arg}" if arg else "SHOW TABLES")
        elif cmd == "/catalogs":
            self._sql("SHOW CATALOGS")
        elif cmd == "/warehouses":
            self._warehouses()
        else:
            style.warn(f"unknown command {cmd} — try /help")
        return False

    def _help(self) -> None:
        rows = [
            ["<text>", "ask Genie in the current conversation"],
            ["/agent <goal>", "run the autonomous agent (LLM plans, Genie executes)"],
            ["/sql <query>", "run raw SQL on the warehouse, rendered as a table"],
            ["/tables [c.s]", "SHOW TABLES in a schema"],
            ["/catalogs", "SHOW CATALOGS"],
            ["/warehouses", "list SQL warehouses"],
            ["/spaces", "list Genie spaces"],
            ["/use <id|#>", "switch the active Genie space"],
            ["/new", "start a fresh conversation"],
            ["/quit", "leave"],
        ]
        style.out(style.table(["command", "what it does"], rows) + "\n")

    def _ask(self, text: str) -> None:
        sp = style.Spinner("Genie is thinking…", color="38;5;209").start()
        try:
            if self.conversation is None:
                self.conversation, answer = self.space.start_conversation(text)
            else:
                answer = self.conversation.ask(text)
        except Exception as exc:  # noqa: BLE001 — REPL keeps going
            sp.stop()
            style.fail(str(exc))
            return
        sp.stop()
        _render_answer(answer)

    def _agent(self, goal: str) -> None:
        if not goal:
            style.warn("usage: /agent <goal>")
            return
        agent = self.client.genie.agent(
            space_id=self.space_id, planner=self.planner, max_turns=self.max_turns,
        )
        style.step(f"agent investigating: {goal}")
        sp = style.Spinner("planning…", color="38;5;209").start()
        try:
            run = agent.run(goal)
        except Exception as exc:  # noqa: BLE001
            sp.stop()
            style.fail(str(exc))
            return
        sp.stop()
        _render_transcript(run)

    def _sql(self, query: str) -> None:
        if not query:
            style.warn("usage: /sql <query>")
            return
        sp = style.Spinner("running on warehouse…", color="38;5;42").start()
        try:
            table = self.client.sql.execute(query).to_arrow_table()
        except Exception as exc:  # noqa: BLE001
            sp.stop()
            style.fail(str(exc))
            return
        sp.stop()
        _render_arrow(table)

    def _warehouses(self) -> None:
        try:
            rows = [
                [wh.warehouse_id, wh.warehouse_name]
                for wh in self.client.warehouses.list_warehouses()
            ]
        except Exception as exc:  # noqa: BLE001
            style.fail(str(exc))
            return
        style.out(style.table(["id", "name"], rows or [["—", "—"]]) + "\n")

    def _spaces(self) -> None:
        self._space_cache = list(self.client.genie.list_spaces())
        rows = [[str(i), s.space_id, s.title or ""] for i, s in enumerate(self._space_cache, 1)]
        style.out(style.table(["#", "space_id", "title"], rows or [["—", "—", "—"]]) + "\n")
        style.out("  " + style.muted("switch with /use <#> or /use <space_id>") + "\n")

    def _use(self, arg: str) -> None:
        if not arg:
            style.warn("usage: /use <#|space_id>")
            return
        target = arg
        if arg.isdigit() and self._space_cache and 1 <= int(arg) <= len(self._space_cache):
            target = self._space_cache[int(arg) - 1].space_id
        self.space_id = target
        self.space = self.client.genie.space(target)
        self.conversation = None
        style.ok(f"now on space {style.bold(self._space_title())} ({target})")


# ---------------------------------------------------------------------------
# ``ygg databricks genie`` sub-command
# ---------------------------------------------------------------------------


class GenieCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "genie", help="Genie conversational analytics (ask / agent / console).",
        )
        sub = parser.add_subparsers(dest="genie_action")

        ls = sub.add_parser("spaces", help="List Genie spaces.")
        ls.set_defaults(handler=cls._spaces)

        ask = sub.add_parser("ask", help="Ask a one-shot question against a space.")
        ask.add_argument("question", help="Natural-language question.")
        ask.add_argument("--space", dest="space_id", default=None,
                         help="Genie space id (env: YGG_GENIE_SPACE).")
        ask.add_argument("--no-data", action="store_true",
                         help="Skip materialising the query result table.")
        ask.set_defaults(handler=cls._ask)

        agent = sub.add_parser(
            "agent", help="Let the Genie agent drive a multi-turn investigation.",
        )
        agent.add_argument("goal", help="What you want the agent to figure out.")
        agent.add_argument("--space", dest="space_id", default=None,
                           help="Genie space id (env: YGG_GENIE_SPACE).")
        agent.add_argument("--planner", nargs="?", const=True, default=None,
                           help="Model Serving endpoint that plans the agent's steps "
                                "(env: YGG_GENIE_PLANNER). Bare --planner uses the "
                                "default; omitted → heuristic (Genie suggestions).")
        agent.add_argument("--max-turns", type=int, default=4,
                           help="Max conversational turns (default: 4).")
        agent.add_argument("--no-data", action="store_true",
                           help="Skip materialising the final result table.")
        agent.set_defaults(handler=cls._agent)

        for name in ("console", "repl"):
            con = sub.add_parser(name, help="Interactive Genie console.")
            con.add_argument("--space", dest="space_id", default=None,
                             help="Genie space id (env: YGG_GENIE_SPACE).")
            con.add_argument("--planner", nargs="?", const=True, default=None,
                             help="Model Serving endpoint that plans /agent steps.")
            con.add_argument("--max-turns", type=int, default=4)
            con.set_defaults(handler=cls._console)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # -- handlers ----------------------------------------------------------
    @classmethod
    def _spaces(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        rows = [[s.space_id, s.title or ""] for s in client.genie.list_spaces()]
        style.out(style.table(["space_id", "title"], rows or [["—", "—"]]) + "\n")
        return 0

    @classmethod
    def _ask(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        answer = client.genie.ask(args.question, space_id=_resolve_space(args))
        _render_answer(answer, with_data=not args.no_data)
        return 0 if not answer.failed else 1

    @classmethod
    def _agent(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        agent = client.genie.agent(
            space_id=_resolve_space(args), planner=args.planner, max_turns=args.max_turns,
        )
        style.step(f"agent investigating: {args.goal}")
        run = agent.run(args.goal)
        _render_transcript(run, with_data=not args.no_data)
        return 0

    @classmethod
    def _console(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        space_id = _resolve_space(args)
        if not space_id:
            style.fail("no space id — pass --space <id> or set YGG_GENIE_SPACE.")
            return 2
        return GenieConsole(
            client, space_id, planner=args.planner, max_turns=args.max_turns,
        ).run()
