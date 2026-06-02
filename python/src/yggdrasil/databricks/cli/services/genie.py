"""``ygg databricks genie`` — converse with Genie spaces (+ autonomous agent)."""
from __future__ import annotations

import sys
from typing import Any, Optional


class GenieCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "genie", help="Genie conversational analytics (ask / agent / spaces).",
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
        agent.add_argument("--max-turns", type=int, default=4,
                           help="Max conversational turns (default: 4).")
        agent.add_argument("--no-data", action="store_true",
                           help="Skip materialising the final result table.")
        agent.set_defaults(handler=cls._agent)

        repl = sub.add_parser("repl", help="Interactive Genie conversation.")
        repl.add_argument("--space", dest="space_id", default=None,
                          help="Genie space id (env: YGG_GENIE_SPACE).")
        repl.set_defaults(handler=cls._repl)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    # ------------------------------------------------------------------ #
    # Handlers
    # ------------------------------------------------------------------ #
    @classmethod
    def _spaces(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        for space in client.genie.list_spaces():
            sys.stdout.write(f"{space.space_id}\t{space.title or ''}\n")
        return 0

    @classmethod
    def _ask(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        space_id = cls._resolve_space(args)
        answer = client.genie.ask(args.question, space_id=space_id)
        cls._print_answer(answer, with_data=not args.no_data)
        return 0 if not answer.failed else 1

    @classmethod
    def _agent(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        space_id = cls._resolve_space(args)
        agent = client.genie.agent(space_id=space_id, max_turns=args.max_turns)
        run = agent.run(args.goal)
        sys.stdout.write(run.summary() + "\n")
        if not args.no_data:
            data = run.data_answer
            if data is not None:
                sys.stdout.write("\n--- result ---\n")
                cls._print_table(data)
        return 0

    @classmethod
    def _repl(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        space_id = cls._resolve_space(args)
        space = client.genie.space(space_id)
        sys.stdout.write(
            f"Genie REPL on space {space_id}. "
            f"Type a question, /new to reset, /quit to exit.\n"
        )
        conversation = None
        while True:
            try:
                line = input("genie> ").strip()
            except (EOFError, KeyboardInterrupt):
                sys.stdout.write("\n")
                return 0
            if not line:
                continue
            if line in ("/quit", "/exit", "/q"):
                return 0
            if line == "/new":
                conversation = None
                sys.stdout.write("(new conversation)\n")
                continue
            try:
                if conversation is None:
                    conversation, answer = space.start_conversation(line)
                else:
                    answer = conversation.ask(line)
            except Exception as exc:  # noqa: BLE001 — REPL keeps going
                sys.stderr.write(f"error: {exc}\n")
                continue
            cls._print_answer(answer, with_data=True)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_space(args: Any) -> Optional[str]:
        import os

        return getattr(args, "space_id", None) or os.environ.get("YGG_GENIE_SPACE") or None

    @classmethod
    def _print_answer(cls, answer: Any, *, with_data: bool) -> None:
        if answer.text:
            sys.stdout.write(answer.text.rstrip() + "\n")
        if answer.failed and answer.error:
            sys.stderr.write(f"error: {answer.error}\n")
            return
        if answer.sql:
            sys.stdout.write(f"\nSQL:\n{answer.sql.strip()}\n")
        if answer.questions and not answer.text:
            sys.stdout.write("Suggested: " + "; ".join(answer.questions) + "\n")
        if with_data and answer.has_query:
            cls._print_table(answer)

    @staticmethod
    def _print_table(answer: Any, limit: int = 20) -> None:
        try:
            table = answer.to_arrow()
        except Exception as exc:  # noqa: BLE001 — data fetch is best-effort
            sys.stderr.write(f"(could not fetch result: {exc})\n")
            return
        if table is None or table.num_rows == 0:
            sys.stdout.write("(no rows)\n")
            return
        rows = table.to_pylist()
        cols = table.column_names
        sys.stdout.write("\t".join(cols) + "\n")
        for row in rows[:limit]:
            sys.stdout.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        if table.num_rows > limit:
            sys.stdout.write(f"… {table.num_rows - limit} more rows\n")
