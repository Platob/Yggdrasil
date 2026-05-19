"""Live-integration tests for :class:`GenieAgent`.

Skipped unless ``DATABRICKS_HOST`` (plus matching credentials) is
exported — see :class:`DatabricksIntegrationCase`. Tests rely on
:attr:`GenieDefaults.auto_pick_space` (on by default) so they run
against whichever Genie space the workspace returns first; no env
gate is required beyond what the SDK itself needs. The whole class
skips when the workspace has zero Genie spaces.

Assertions deliberately target API mechanics (a completed
:class:`GenieAnswer`, a readable file on disk, a stable
``conversation_id`` across turns) rather than specific Genie reply
text — that way the same prompts work against any auto-picked space
without becoming brittle.

Environment knobs
-----------------
``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` / ``DATABRICKS_CONFIG_PROFILE``
    Standard SDK credentials read by :class:`DatabricksClient`.
``DATABRICKS_GENIE_SPACE_ID``
    Optional override. When set, pins every test to that space id;
    otherwise auto-pick runs.
``DATABRICKS_GENIE_WAREHOUSE_ID``
    Optional warehouse override for query materialisation. Falls back
    to the workspace default warehouse.
``DATABRICKS_GENIE_QUESTION_TEXT``
    Optional natural-language question used by the text-only ask. The
    default ("Hello, what data do you have access to?") is a generic
    prompt every Genie space should respond to.
``DATABRICKS_GENIE_QUESTION_SQL``
    Optional natural-language question used by the SQL-producing ask.
    Defaults to a generic count question; tests that need a SQL
    attachment self-skip when Genie returns text only.

Cleanup
-------
Each test writes artifacts under a per-test ``tempfile.mkdtemp(...)``
which ``addCleanup`` deletes on the way out, so no files survive the
run. Genie spaces, conversations, and messages are never deleted —
those are workspace state and the test would race other consumers.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, Optional

from yggdrasil.databricks.genie import GenieAgent, GenieAnswer, GenieSpace

from .. import DatabricksIntegrationCase


# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------
_DEFAULT_TEXT_QUESTION = "Hello, what data do you have access to?"
_DEFAULT_SQL_QUESTION = "How many rows are in the largest table you can see?"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


class GenieIntegrationCase(DatabricksIntegrationCase):
    """Common setup: configure the Genie agent + per-test tmp output dir.

    Subclasses inherit the live :class:`DatabricksClient` (``cls.client``)
    from :class:`DatabricksIntegrationCase` and access the agent via
    :attr:`agent`. No env-var gate beyond what the SDK requires —
    ``auto_pick_space`` resolves a target on the first ask. When the
    workspace has zero Genie spaces, the whole case skips (there's
    literally nothing to test against).
    """

    SPACE_ID_ENV: ClassVar[str] = "DATABRICKS_GENIE_SPACE_ID"
    WAREHOUSE_ID_ENV: ClassVar[str] = "DATABRICKS_GENIE_WAREHOUSE_ID"

    def setUp(self) -> None:
        super().setUp()

        # Honor an explicit env override for reproducibility, but never
        # require it — ``auto_pick_space`` (default True) handles the
        # unset case by walking ``list_spaces``.
        space_id = _env(self.SPACE_ID_ENV)
        warehouse_id = _env(self.WAREHOUSE_ID_ENV)

        if space_id is None:
            # Probe once so the whole case skips cleanly on workspaces
            # with no Genie spaces, instead of failing every test with
            # "Could not auto-pick a Genie space".
            first = next(iter(self.client.genie.list_spaces()), None)
            if first is None:
                raise unittest.SkipTest(
                    "Workspace has no Genie spaces — nothing to test "
                    "against. Create at least one space or set "
                    f"{self.SPACE_ID_ENV} to a known id."
                )

        self.tmpdir = Path(tempfile.mkdtemp(prefix="ygg-genie-int-"))
        self.addCleanup(shutil.rmtree, self.tmpdir, ignore_errors=True)

        self.client.genie.defaults = replace(
            self.client.genie.defaults,
            space_id=space_id,  # None lets auto_pick_space resolve on first ask
            warehouse_id=warehouse_id,
            agent_output_dir=str(self.tmpdir),
            # Auto-save off here — every test explicitly opts in so the
            # assertion shape stays obvious.
            agent_auto_save=False,
        )
        # Reset agent history between tests so /history doesn't leak.
        self.client.genie._agent = None

    @property
    def agent(self) -> GenieAgent:
        return self.client.genie.agent


# ---------------------------------------------------------------------------
# Read-only space metadata
# ---------------------------------------------------------------------------
class TestGenieIntegrationSpace(GenieIntegrationCase):
    """Space lookup + metadata — cheap, no chat traffic."""

    def test_ensure_space_returns_a_space(self):
        space = self.client.genie.ensure_space()
        self.assertIsInstance(space, GenieSpace)
        self.assertTrue(space.space_id)
        # When the user pinned ``DATABRICKS_GENIE_SPACE_ID``, ensure_space
        # honors it; otherwise auto-pick returned something real.
        pinned = _env(self.SPACE_ID_ENV)
        if pinned is not None:
            self.assertEqual(space.space_id, pinned)

    def test_list_spaces_is_non_empty(self):
        spaces = list(self.client.genie.list_spaces())
        self.assertGreater(len(spaces), 0)

    def test_space_details_load(self):
        space = self.client.genie.space()
        # ``details`` is lazy — pulls the GET get_space payload.
        self.assertIsNotNone(space.details)
        self.assertTrue(space.title or space.space_id)


# ---------------------------------------------------------------------------
# Agent ask / chat
# ---------------------------------------------------------------------------
class TestGenieAgentAsk(GenieIntegrationCase):
    """End-to-end Genie ask through the agent."""

    def test_agent_run_returns_answer(self):
        question = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        answer = self.agent.run(question, save=False)
        self.assertIsInstance(answer, GenieAnswer)
        self.assertTrue(answer.is_completed, msg=f"non-terminal status: {answer.status!r}")
        # At least one of (text, query) is populated for any non-empty
        # space — otherwise Genie returned nothing actionable and the
        # space configuration is the problem to debug.
        self.assertTrue(
            (answer.text or "").strip() or answer.query,
            msg="empty answer — check space configuration",
        )

    def test_agent_history_grows_with_each_run(self):
        question = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        before = len(self.agent.history)
        self.agent.run(question, save=False)
        self.assertEqual(len(self.agent.history), before + 1)
        self.assertIs(self.agent.last(), self.agent.history[-1])

    def test_agent_chat_reuses_conversation(self):
        q1 = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        q2 = "Tell me one more interesting thing."
        answers = self.agent.chat(q1, q2, save=False, max_steps=2)
        self.assertEqual(len(answers), 2)
        # Second message lives on the same conversation thread.
        self.assertEqual(
            answers[0].conversation_id,
            answers[1].conversation_id,
            msg=f"conversation ids drifted: {[a.conversation_id for a in answers]!r}",
        )


# ---------------------------------------------------------------------------
# Agent save (SQL-bearing answer)
# ---------------------------------------------------------------------------
class TestGenieAgentSave(GenieIntegrationCase):
    """Materialise + persist Genie's SQL result via the agent.

    Skipped per-test when the question Genie generates doesn't carry a
    query attachment — the agent surface is verified, but the save
    assertion only fires when Genie actually returned SQL.
    """

    def setUp(self) -> None:
        super().setUp()
        question = _env("DATABRICKS_GENIE_QUESTION_SQL", _DEFAULT_SQL_QUESTION)
        self.answer = self.agent.run(question, save=False)
        if not self.answer.query:
            self.skipTest(
                "Genie did not return a SQL attachment for the integration "
                f"prompt ({question!r}); set DATABRICKS_GENIE_QUESTION_SQL to "
                "something this space answers with SQL."
            )

    def test_save_parquet_writes_readable_file(self):
        import pyarrow.parquet as pq

        path = self.agent.save(self.answer, format="parquet")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        # Confirm Arrow can read it back — quickest end-to-end check.
        table = pq.read_table(str(path))
        self.assertGreater(table.num_columns, 0)

    def test_save_csv(self):
        path = self.agent.save(self.answer, format="csv")
        self.assertIsNotNone(path)
        self.assertTrue(path.exists())
        # CSV header is the column names — anything but an empty file works.
        self.assertGreater(path.stat().st_size, 0)

    def test_save_arrow_ipc_roundtrip(self):
        from pyarrow import feather

        path = self.agent.save(self.answer, format="arrow")
        self.assertIsNotNone(path)
        round_trip = feather.read_table(str(path))
        self.assertGreater(round_trip.num_columns, 0)

    def test_save_json_envelope(self):
        from yggdrasil.pickle import json as ygg_json

        path = self.agent.save(self.answer, format="json")
        self.assertIsNotNone(path)
        payload = ygg_json.loads(path.read_bytes())
        # Envelope shape is stable across SDK versions.
        for key in ("space_id", "conversation_id", "message_id", "text", "query", "url"):
            self.assertIn(key, payload)

    def test_arrow_table_matches_save(self):
        # The cached statement-result drives both calls — a save followed
        # by a fresh arrow_table() must agree on row count.
        from pyarrow import feather

        path = self.agent.save(self.answer, format="arrow")
        self.assertIsNotNone(path)
        from_arrow = self.answer.arrow_table()
        from_disk = feather.read_table(str(path))
        self.assertEqual(from_disk.num_rows, from_arrow.num_rows)
        self.assertEqual(from_disk.column_names, from_arrow.column_names)


# ---------------------------------------------------------------------------
# Auto-save sanity check
# ---------------------------------------------------------------------------
class TestGenieAgentAutoSave(GenieIntegrationCase):
    """``GenieDefaults.agent_auto_save`` writes after every run."""

    def setUp(self) -> None:
        super().setUp()
        self.client.genie.defaults = replace(
            self.client.genie.defaults,
            agent_auto_save=True,
            agent_auto_save_format="json",
        )

    def test_auto_save_writes_after_run(self):
        question = _env("DATABRICKS_GENIE_QUESTION_TEXT", _DEFAULT_TEXT_QUESTION)
        self.agent.run(question)
        # Auto-save with format=json always writes (envelope-only) — no
        # query attachment required. A tabular format would need the SQL
        # path which is space-dependent, so json keeps this test stable.
        written = list(self.tmpdir.rglob("*.json"))
        self.assertEqual(len(written), 1, msg=f"unexpected files: {written!r}")
