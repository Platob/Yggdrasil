"""Live-integration smoke test for the Databricks Genie surface.

Validates the Genie chain against a real workspace, end to end:

- :meth:`Genie.list_spaces` enumerates the workspace's Genie spaces;
- a space (``$YGG_GENIE_SPACE`` or the first discovered) reports its
  title + warehouse via :attr:`GenieSpace.infos`;
- :meth:`GenieSpace.ask` round-trips a question to a ``COMPLETED``
  :class:`GenieAnswer` carrying natural-language text and/or a generated
  SQL query, and — when a query ran — materialises the inline result to
  a :class:`pyarrow.Table`;
- the autonomous :class:`GenieAgent` drives a multi-turn investigation
  and returns a transcript with at least the opening turn.

Skipped wholesale unless ``DATABRICKS_HOST`` is set, and per-test when no
Genie space is reachable. Only *reads* / *asks* — it creates no spaces,
so there is nothing to clean up. Permission / availability failures
degrade to ``unittest.SkipTest``. Override the target space via
``YGG_GENIE_SPACE``.
"""
from __future__ import annotations

import os
import unittest

from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied
from databricks.sdk.errors.sdk import OperationFailed

from yggdrasil.databricks.genie import AgentRun, GenieAnswer, GenieSpace

from .. import DatabricksIntegrationCase

# Genie raises these when a space simply can't answer a question (wrong
# domain, no matching tables, model declines) — a data/space condition,
# not a client bug, so the smoke test treats them as "try the next space".
_ASK_ERRORS = (OperationFailed, DatabricksError, PermissionDenied)

#: A data-shaped question most analytics spaces can answer; exercises the
#: text + SQL + inline-result path when the space has tables.
_QUESTION = os.environ.get(
    "YGG_GENIE_QUESTION", "How many rows are in the largest table?",
)


class TestGenieIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.genie = cls.client.genie

    def _candidate_spaces(self) -> list[GenieSpace]:
        """Genie spaces to try: the env-pinned one, else everything visible."""
        env_space = os.environ.get("YGG_GENIE_SPACE", "").strip()
        if env_space:
            return [self.genie.space(env_space)]
        try:
            spaces = list(self.genie.list_spaces())
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"no permission to list Genie spaces: {exc}")
        if not spaces:
            raise unittest.SkipTest(
                "no Genie spaces visible in this workspace; set YGG_GENIE_SPACE."
            )
        return spaces

    def _a_space(self) -> GenieSpace:
        return self._candidate_spaces()[0]

    def _ask_any_space(self) -> tuple[GenieSpace, GenieAnswer]:
        """Ask ``_QUESTION`` against candidate spaces; return the first that answers."""
        last_error: Exception | None = None
        for space in self._candidate_spaces():
            try:
                answer = space.ask(_QUESTION)
            except _ASK_ERRORS as exc:
                last_error = exc
                continue
            if not answer.failed:
                return space, answer
        raise unittest.SkipTest(
            f"no Genie space could answer {_QUESTION!r}"
            + (f" (last error: {last_error})" if last_error else "")
        )

    def test_list_spaces(self):
        try:
            spaces = list(self.genie.list_spaces())
        except PermissionDenied as exc:
            raise unittest.SkipTest(f"no permission to list Genie spaces: {exc}")
        for sp in spaces:
            self.assertIsInstance(sp, GenieSpace)
            self.assertTrue(sp.space_id)

    def test_space_infos(self):
        space = self._a_space()
        # ``title`` triggers a get_space round-trip; tolerate an empty title.
        self.assertIsInstance(space.title, (str, type(None)))
        self.assertTrue(space.space_id)

    def test_ask_round_trip(self):
        _space, answer = self._ask_any_space()
        self.assertIsInstance(answer, GenieAnswer)
        # A completed answer carries text, a query, and/or suggestions.
        self.assertTrue(
            answer.text or answer.has_query or answer.questions,
            "expected text, a query, or suggested questions",
        )
        # When Genie ran a query, the inline result materialises to Arrow.
        if answer.has_query:
            self.assertTrue(answer.sql)
            table = answer.to_arrow()
            self.assertIsNotNone(table)
            self.assertEqual(table.num_rows, len(answer.rows()))

    def test_agent_investigation(self):
        # Find a space that answers, then let the agent drive it. Genie is
        # non-deterministic — a question that answered a moment ago can come
        # back FAILED — so retry a couple of times before degrading to skip.
        space, _answer = self._ask_any_space()
        agent = space.agent(max_turns=3)
        run = None
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                run = agent.run(_QUESTION)
                break
            except _ASK_ERRORS as exc:
                last_error = exc
        if run is None:
            raise unittest.SkipTest(f"Genie agent run failed: {last_error}")

        self.assertIsInstance(run, AgentRun)
        self.assertGreaterEqual(len(run.turns), 1)
        self.assertEqual(run.turns[0].question, _QUESTION)
        self.assertFalse(run.turns[0].autonomous)
        # The transcript renders without blowing up.
        self.assertIn("Goal:", run.summary())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
