"""``ygg-genie`` — the autonomous Genie agent console script.

A thin :class:`~yggdrasil.cli.databricks.base.DatabricksCLI` subclass that
points the self-driving :class:`~yggdrasil.databricks.genie.GenieAgent` at a
space and prints the transcript + final result. Three shapes:

- ``ygg-genie --space <id> "why did Q3 revenue dip?"`` — agent mode: drive
  a multi-turn investigation on its own and print the transcript.
- ``ygg-genie --space <id> --ask "top 5 customers"`` — one-shot: a single
  question, no autonomous follow-ups.
- ``ygg-genie --space <id>`` — interactive REPL.

The space id falls back to ``$YGG_GENIE_SPACE`` so it can be omitted.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from .base import DatabricksCLI


__all__ = ["GenieCLI", "main"]


class GenieCLI(DatabricksCLI):
    prog = "ygg-genie"
    description = "Autonomous Databricks Genie agent — ask, investigate, converse."
    epilog = """
        Examples:
          ygg-genie --space 01ef… "why did Q3 revenue dip?"
          ygg-genie --space 01ef… --ask "top 5 customers by revenue"
          ygg-genie --space 01ef… --max-turns 6 "break down churn by plan"
          YGG_GENIE_SPACE=01ef… ygg-genie                 # interactive REPL
    """

    @classmethod
    def add_service_arguments(cls, parser: argparse.ArgumentParser) -> None:
        grp = parser.add_argument_group("Genie")
        grp.add_argument(
            "prompt", nargs="?", default=None,
            help="Goal/question. Omit for an interactive REPL.",
        )
        grp.add_argument(
            "--space", dest="space_id", default=None,
            help="Genie space id (env: YGG_GENIE_SPACE).",
        )
        grp.add_argument(
            "--ask", action="store_true",
            help="One-shot ask (no autonomous follow-ups).",
        )
        grp.add_argument(
            "--max-turns", type=int, default=4,
            help="Max conversational turns the agent will drive (default: 4).",
        )
        grp.add_argument(
            "--no-data", action="store_true",
            help="Skip materialising the result table.",
        )

    # ------------------------------------------------------------------ #
    def run(self) -> int:
        space_id = self.args.space_id or os.environ.get("YGG_GENIE_SPACE") or None
        if not space_id:
            sys.stderr.write(
                "ygg-genie: no space id — pass --space <id> or set YGG_GENIE_SPACE.\n"
            )
            return 2

        genie = self.client.genie
        prompt = self.args.prompt

        if prompt is None:
            return self._repl(genie, space_id)

        if self.args.ask:
            answer = genie.ask(prompt, space_id=space_id)
            self._print_answer(answer, with_data=not self.args.no_data)
            return 0 if not answer.failed else 1

        # Default: autonomous agent.
        run = genie.agent(space_id=space_id, max_turns=self.args.max_turns).run(prompt)
        sys.stdout.write(run.summary() + "\n")
        if not self.args.no_data and run.data_answer is not None:
            sys.stdout.write("\n--- result ---\n")
            self._print_table(run.data_answer)
        return 0

    # ------------------------------------------------------------------ #
    def _repl(self, genie, space_id: str) -> int:
        space = genie.space(space_id)
        sys.stdout.write(
            f"Genie REPL on space {space_id}. "
            f"/agent <goal> to investigate, /new to reset, /quit to exit.\n"
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
            if line.startswith("/agent "):
                run = space.agent(max_turns=self.args.max_turns).run(line[len("/agent "):])
                sys.stdout.write(run.summary() + "\n")
                if run.data_answer is not None:
                    self._print_table(run.data_answer)
                continue
            try:
                if conversation is None:
                    conversation, answer = space.start_conversation(line)
                else:
                    answer = conversation.ask(line)
            except Exception as exc:  # noqa: BLE001 — REPL keeps going
                sys.stderr.write(f"error: {exc}\n")
                continue
            self._print_answer(answer, with_data=True)

    # ------------------------------------------------------------------ #
    @classmethod
    def _print_answer(cls, answer, *, with_data: bool) -> None:
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
    def _print_table(answer, limit: int = 20) -> None:
        try:
            table = answer.to_arrow()
        except Exception as exc:  # noqa: BLE001 — best-effort
            sys.stderr.write(f"(could not fetch result: {exc})\n")
            return
        if table is None or table.num_rows == 0:
            sys.stdout.write("(no rows)\n")
            return
        cols = table.column_names
        sys.stdout.write("\t".join(cols) + "\n")
        for row in table.to_pylist()[:limit]:
            sys.stdout.write("\t".join(str(row.get(c, "")) for c in cols) + "\n")
        if table.num_rows > limit:
            sys.stdout.write(f"… {table.num_rows - limit} more rows\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return GenieCLI.parse_and_run(argv)
