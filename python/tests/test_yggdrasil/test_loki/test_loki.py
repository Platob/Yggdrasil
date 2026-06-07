"""Unit tests for the global yggdrasil agent (Loki)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import Loki
from yggdrasil.loki.skill import REGISTRY, LokiSkill, register
from yggdrasil.loki.capability import Backend


class _Dummy(LokiSkill):
    name = "dummy-test"
    description = "echo kwargs"

    def run(self, agent, **kwargs):
        return {"agent": agent.name, **kwargs}


class _NeedsDatabricks(LokiSkill):
    name = "needs-dbx-test"
    requires = "databricks"

    def run(self, agent, **kwargs):
        return "ran"


class TestBehaviorRegistry(unittest.TestCase):
    def setUp(self):
        self._saved = dict(REGISTRY)

    def tearDown(self):
        REGISTRY.clear()
        REGISTRY.update(self._saved)

    def test_register_and_dispatch(self):
        register(_Dummy)
        loki = Loki()
        self.assertIn("dummy-test", [b.name for b in loki.skills()])
        self.assertEqual(loki.run("dummy-test", x=1), {"agent": "loki", "x": 1})

    def test_unknown_behavior_raises(self):
        with self.assertRaises(KeyError):
            Loki().run("nope-not-real")

    def test_unavailable_behavior_raises(self):
        register(_NeedsDatabricks)
        loki = Loki()
        loki._backends = [Backend("databricks", available=False)]
        with self.assertRaises(RuntimeError):
            loki.run("needs-dbx-test")

    def test_available_behavior_runs(self):
        register(_NeedsDatabricks)
        loki = Loki()
        loki._backends = [Backend("databricks", available=True)]
        self.assertEqual(loki.run("needs-dbx-test"), "ran")


class TestAgent(unittest.TestCase):
    def test_identity_is_stable_int64(self):
        loki = Loki()
        self.assertIsInstance(loki.agent_id, int)
        self.assertEqual(loki.agent_id, Loki().agent_id)
        self.assertLess(loki.agent_id, 2**63)

    def test_backend_lookup_and_has(self):
        loki = Loki()
        loki._backends = [Backend("databricks", True, {"host": "h"}), Backend("local", True)]
        self.assertTrue(loki.has("databricks"))
        self.assertFalse(loki.has("node"))
        self.assertEqual(loki.backend("databricks").detail["host"], "h")

    def test_databricks_token_provider(self):
        loki = Loki()
        loki._backends = [Backend("databricks", True, {"host": "https://w", "auth_type": "pat"})]
        info = loki.token_info()
        self.assertTrue(info["available"])
        self.assertEqual(info["host"], "https://w")

        fake_client = MagicMock()
        with patch("yggdrasil.databricks.DatabricksClient") as DC:
            DC.current.return_value = fake_client
            self.assertIs(loki.databricks, fake_client)

    def test_no_databricks_means_no_token(self):
        loki = Loki()
        loki._backends = [Backend("databricks", False), Backend("local", True)]
        self.assertIsNone(loki.databricks)
        self.assertFalse(loki.token_info()["available"])

    def test_card_shape(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        card = loki.card()
        self.assertEqual(card["agent"], "loki")
        self.assertIn("backends", card)
        self.assertIn("skills", card)
        self.assertIn("token", card)

    def test_can_run_local_probe_is_cached(self):
        # The hardware probe imports torch (slow first call) and is asked on
        # every turn by select() — resolve it once per process.
        loki = Loki()
        with patch("yggdrasil.loki.resources.can_run_local", return_value=True) as crl:
            self.assertTrue(loki.can_run_local())
            self.assertTrue(loki.can_run_local())
        crl.assert_called_once()

    def test_specialist_resolution_is_cached(self):
        # The REPL asks for the databricks specialist every databricks turn;
        # the import + singleton lookup + backend check resolves once (here,
        # with no databricks SDK/session, to None) and is then memoized.
        loki = Loki()
        first = loki.specialist("databricks")
        self.assertIn("databricks", loki._specialists)
        self.assertIs(loki.specialist("databricks"), first)
        # The cache is authoritative — a seeded value short-circuits resolution.
        sentinel = object()
        loki._specialists["databricks"] = sentinel
        self.assertIs(loki.specialist("databricks"), sentinel)


class TestPythonProjectBehavior(unittest.TestCase):
    def test_runs_anywhere_and_executes_supplied_code(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        out = loki.run(
            "python_project",
            project="My Demo",
            code="def main():\n    print('hi from', 7 * 6)\n\nif __name__ == '__main__':\n    main()\n",
        )
        self.assertEqual(out["package"], "my_demo")           # slugged
        self.assertEqual(out["returncode"], 0)
        self.assertIn("hi from 42", out["stdout"])
        self.assertIn("pyproject.toml", out["files"])
        self.assertIn("my_demo/main.py", out["files"])

    def test_reasons_code_from_task_and_strips_fences(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        loki.reason = lambda prompt, *, system=None, **_: (
            "```python\nprint('reasoned ok')\n```"
        )
        out = loki.run("python_project", project="r", task="say hello")
        self.assertEqual(out["returncode"], 0)
        self.assertIn("reasoned ok", out["stdout"])

    def test_scaffold_only_without_run(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        out = loki.run("python_project", project="noexec", code="print('x')\n", run=False)
        self.assertNotIn("returncode", out)
        self.assertTrue(out["project_dir"].endswith("noexec"))

    def test_requires_code_or_task(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        with self.assertRaises(ValueError):
            loki.run("python_project", project="empty")


class _ScriptedEngine:
    """A TokenEngine stand-in that replays a fixed list of JSON replies."""

    name = "scripted"
    model = "scripted-1"

    def __init__(self, replies):
        self._replies = list(replies)
        self.seen = []

    def available(self):
        return True

    def complete(self, messages, *, system=None, max_tokens=4000, **_):
        self.seen.append(messages[-1]["content"])
        from yggdrasil.loki.engine import Completion

        return Completion(text=self._replies.pop(0))


class TestAgentAct(unittest.TestCase):
    """The autonomous reason→act→observe loop over a confined toolbox."""

    def setUp(self):
        import tempfile

        self.dir = tempfile.mkdtemp(prefix="ygg-act-")

    def _loki(self, replies):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        eng = _ScriptedEngine(replies)
        loki.engine = lambda name=None: eng  # force our scripted brain
        loki.select = lambda text=None, *, tier=None, base=None, confirm=None: eng  # …on the auto path too
        return loki, eng

    def test_act_writes_a_file_and_reports_change(self):
        import json
        import pathlib

        loki, _ = self._loki([
            json.dumps({"thought": "create it", "tool": "write_file",
                        "args": {"path": "hi.txt", "content": "hello loki"}}),
            json.dumps({"thought": "done", "done": True, "answer": "wrote hi.txt"}),
        ])
        result = loki.act("create hi.txt", root=self.dir, max_steps=5)
        self.assertTrue(result["completed"])
        self.assertEqual(result["answer"], "wrote hi.txt")
        self.assertEqual(result["files_changed"], ["hi.txt"])
        self.assertEqual(len(result["steps"]), 1)
        self.assertEqual((pathlib.Path(self.dir) / "hi.txt").read_text(), "hello loki")

    def test_act_can_smoke_test_a_change_as_a_checkpoint(self):
        import json
        import pathlib

        loki, _ = self._loki([
            json.dumps({"thought": "write it", "tool": "write_file",
                        "args": {"path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}}),
            json.dumps({"thought": "checkpoint", "tool": "smoke",
                        "args": {"code": "from calc import add; assert add(2, 3) == 5; print('ok')"}}),
            json.dumps({"thought": "verified", "done": True, "answer": "added add(); smoke passed"}),
        ])
        result = loki.act("add add() and validate it", root=self.dir, max_steps=6)
        self.assertTrue(result["completed"])
        # The smoke checkpoint actually ran the assertion against the new file.
        smoke_step = next(s for s in result["steps"] if s["tool"] == "smoke")
        self.assertIn("exit=0", smoke_step["observation"])
        self.assertIn("ok", smoke_step["observation"])
        self.assertEqual((pathlib.Path(self.dir) / "calc.py").exists(), True)

    def test_act_stops_when_over_budget(self):
        import json
        from unittest.mock import patch

        from yggdrasil.loki.usage import METER

        loki, eng = self._loki([
            json.dumps({"tool": "list_dir", "args": {"path": "."}}),
            json.dumps({"done": True, "answer": "should not reach"}),
        ])
        # Budget already exceeded → the loop stops before the first model call.
        with patch.object(METER, "over_budget", return_value=True):
            result = loki.act("do work", root=self.dir, max_steps=5)
        self.assertFalse(result["completed"])
        self.assertIn("budget", result["answer"])
        self.assertEqual(eng.seen, [])                    # the engine was never called

    def test_act_feeds_observations_back(self):
        import json

        loki, eng = self._loki([
            json.dumps({"tool": "list_dir", "args": {"path": "."}}),
            json.dumps({"done": True, "answer": "looked around"}),
        ])
        loki.act("inspect", root=self.dir, max_steps=5)
        # The second turn must have received the list_dir observation.
        self.assertTrue(any("Observation from list_dir" in m for m in eng.seen))

    def test_act_tolerates_fenced_json(self):
        loki, _ = self._loki([
            "```json\n{\"done\": true, \"answer\": \"ok\"}\n```",
        ])
        result = loki.act("noop", root=self.dir, max_steps=3)
        self.assertTrue(result["completed"])
        self.assertEqual(result["answer"], "ok")

    def test_act_reprompts_on_garbage_then_finishes(self):
        import json

        loki, eng = self._loki([
            "I will now think about this carefully.",  # no JSON
            json.dumps({"done": True, "answer": "recovered"}),
        ])
        result = loki.act("noop", root=self.dir, max_steps=5)
        self.assertTrue(result["completed"])
        self.assertEqual(result["answer"], "recovered")

    def test_act_stops_at_max_steps(self):
        import json

        loki, _ = self._loki([
            json.dumps({"tool": "list_dir", "args": {}}) for _ in range(10)
        ])
        result = loki.act("loop forever", root=self.dir, max_steps=3)
        self.assertFalse(result["completed"])
        self.assertIn("max_steps", result["answer"])
        self.assertEqual(len(result["steps"]), 3)

    def test_read_only_act_cannot_write(self):
        import json

        loki, _ = self._loki([
            json.dumps({"tool": "write_file", "args": {"path": "x", "content": "y"}}),
            json.dumps({"done": True, "answer": "tried"}),
        ])
        result = loki.act("write", root=self.dir, read_only=True, max_steps=5)
        self.assertEqual(result["files_changed"], [])
        self.assertIn("unknown tool", result["steps"][0]["observation"])

    def test_act_via_behavior_dispatch(self):
        import json

        loki, _ = self._loki([
            json.dumps({"done": True, "answer": "behavior path works"}),
        ])
        result = loki.run("agent", task="noop", root=self.dir, max_steps=3)
        self.assertEqual(result["answer"], "behavior path works")

    def test_act_without_engine_raises(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        loki.select = lambda text=None, *, tier=None, base=None, confirm=None: None
        with self.assertRaises(RuntimeError):
            loki.act("anything", root=self.dir)

    def test_reason_stream_yields_chunks(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        eng = _ScriptedEngine([""])
        eng.stream = lambda messages, **k: iter(["A", "B", "C"])
        eng.generate_stream = lambda prompt, **k: eng.stream([], **k)
        loki.select = lambda text=None, *, tier=None, base=None, confirm=None: eng
        self.assertEqual("".join(loki.reason_stream("hi")), "ABC")

    def test_reason_stream_without_engine_raises(self):
        loki = Loki()
        loki._backends = [Backend("local", True)]
        loki.select = lambda text=None, *, tier=None, base=None, confirm=None: None
        with self.assertRaises(RuntimeError):
            list(loki.reason_stream("x"))


class TestReasoningRouter(unittest.TestCase):
    """Loki categorizes a request and picks a solution path / specialist."""

    def test_url_routes_to_web_with_extracted_url(self):
        plan = Loki().route("fetch https://example.com/data.csv please")
        self.assertEqual(plan["category"], "web")
        self.assertEqual(plan["action"], "web")
        self.assertEqual(plan["url"], "https://example.com/data.csv")

    def test_web_verb_without_url_still_routes_web(self):
        plan = Loki().route("browse the latest news")
        self.assertEqual(plan["category"], "web")
        self.assertIsNone(plan["url"])

    def test_databricks_signal_routes_to_specialist(self):
        plan = Loki().route("how do I size a Databricks SQL warehouse?")
        self.assertEqual(plan["category"], "databricks")
        self.assertEqual(plan["specialist"], "databricks")

    def test_genie_signal_picks_genie_action(self):
        self.assertEqual(Loki().route("ask genie for revenue by region")["action"], "genie")

    def test_file_signal_routes_to_act(self):
        plan = Loki().route("fix the bug in calc.py and add a test")
        self.assertEqual(plan["category"], "files")
        self.assertEqual(plan["action"], "act")
        self.assertIsNone(plan["specialist"])

    def test_plain_request_is_chat_reason(self):
        plan = Loki().route("what is the capital of France?")
        self.assertEqual(plan["category"], "chat")
        self.assertEqual(plan["action"], "reason")

    def test_specialist_unknown_is_none(self):
        self.assertIsNone(Loki().specialist("nope"))


class TestReplCommands(unittest.TestCase):
    """The interactive session's slash commands and budget prompt."""

    def setUp(self):
        from yggdrasil.loki.usage import METER

        self.METER = METER
        self._saved = dict(METER._rows)
        self._limit = METER.cost_limit
        METER.reset()

    def tearDown(self):
        self.METER.reset()
        self.METER._rows.update(self._saved)
        self.METER.cost_limit = self._limit

    def test_tier_and_budget_commands(self):
        from yggdrasil.cli import style
        from yggdrasil.loki import cli

        loki = Loki()
        state = {"tier": None, "root": "."}
        self.assertTrue(cli._repl_command(loki, style, state, "/tier deep"))
        self.assertEqual(state["tier"], "deep")
        self.assertTrue(cli._repl_command(loki, style, state, "/tier auto"))
        self.assertIsNone(state["tier"])
        cli._repl_command(loki, style, state, "/budget 5")
        self.assertEqual(self.METER.cost_limit, 5.0)
        cli._repl_command(loki, style, state, "/budget +2")
        self.assertEqual(self.METER.cost_limit, 7.0)
        cli._repl_command(loki, style, state, "/budget off")
        self.assertIsNone(self.METER.cost_limit)

    def test_budget_prompt_raises_step_on_enter(self):
        from yggdrasil.cli import style
        from yggdrasil.loki import cli

        self.METER.set_limit(1.0)
        with patch("builtins.input", return_value=""):
            self.assertTrue(cli._budget_prompt(style))
        self.assertAlmostEqual(self.METER.cost_limit, 1.0 + self.METER.cost_step)

    def test_budget_prompt_stop_returns_false(self):
        from yggdrasil.cli import style
        from yggdrasil.loki import cli

        self.METER.set_limit(1.0)
        with patch("builtins.input", return_value="s"):
            self.assertFalse(cli._budget_prompt(style))

    def test_select_engine_auto_picks_available(self):
        from yggdrasil.cli import style
        from yggdrasil.loki import cli

        loki = Loki()
        loki._backends = [Backend("local", True)]
        eng = MagicMock(name="claude"); eng.name = "claude"; eng.available.return_value = True
        eng.model_label = "claude-opus-4-8 (adaptive)"
        loki.engines = lambda: [eng]
        loki.available_engines = lambda **kw: {"claude": eng}
        loki.engine = lambda name=None: eng
        state = {"tier": None, "root": ".", "engine": None}
        cli._select_engine(loki, style, state)
        self.assertEqual(state["engine"], "claude")

    def test_engine_command_switches_when_available(self):
        from yggdrasil.cli import style
        from yggdrasil.loki import cli

        loki = Loki()
        c = MagicMock(); c.name = "claude"; c.available.return_value = True
        d = MagicMock(); d.name = "databricks"; d.available.return_value = True
        loki.engines = lambda: [c, d]
        loki.engine = lambda name=None: {"claude": c, "databricks": d}.get(name, c)
        state = {"tier": None, "root": ".", "engine": "claude"}
        cli._repl_command(loki, style, state, "/engine databricks")
        self.assertEqual(state["engine"], "databricks")

    def test_json_helper_returns_str_not_bytes(self):
        # Regression: orjson emits bytes; the CLI --json path must decode.
        from yggdrasil.loki import cli

        out = cli._json({"a": 1, "b": ["x", "y"]})
        self.assertIsInstance(out, str)
        self.assertIn('"a"', out)


class _BoomEngine:
    """An available engine whose generation always errors."""
    def __init__(self, name="claude"):
        self.name = name; self.local = False
    def available(self): return True
    def generate(self, *a, **k): raise RuntimeError("403 invalid access token")
    def generate_stream(self, *a, **k):
        raise RuntimeError("403 invalid access token")
        yield  # pragma: no cover


class _OkEngine:
    """An available engine that answers."""
    def __init__(self, name="openai", text="fallback reply"):
        self.name = name; self.local = False; self._text = text
    def available(self): return True
    def generate(self, *a, **k): return self._text
    def generate_stream(self, *a, **k):
        for w in self._text.split(" "):
            yield w + " "


class TestEngineFallback(unittest.TestCase):
    """A failing engine is logged and skipped to the next available one."""

    def _loki(self, engines):
        loki = Loki(); loki._backends = []
        self._p1 = patch.object(Loki, "_engine_instances", return_value=engines)
        self._p2 = patch.object(Loki, "can_run_local", return_value=False)
        self._p1.start(); self._p2.start()
        self.addCleanup(self._p1.stop); self.addCleanup(self._p2.stop)
        return loki

    def test_reason_skips_failing_engine(self):
        loki = self._loki({"claude": _BoomEngine(), "openai": _OkEngine()})
        with self.assertLogs("yggdrasil.loki.agent", level="WARNING") as logs:
            self.assertEqual(loki.reason("hi"), "fallback reply")
        self.assertTrue(any("claude" in m and "skipping" in m for m in logs.output))

    def test_reason_stream_skips_failing_engine_before_first_token(self):
        loki = self._loki({"claude": _BoomEngine(), "openai": _OkEngine()})
        with self.assertLogs("yggdrasil.loki.agent", level="WARNING"):
            self.assertEqual("".join(loki.reason_stream("hi")).strip(), "fallback reply")

    def test_stream_does_not_switch_after_output_started(self):
        # An engine that yields then errors must not silently fail over — the
        # partial output is already on screen.
        class _PartialThenBoom(_OkEngine):
            def generate_stream(self, *a, **k):
                yield "half "
                raise RuntimeError("mid-stream drop")
        loki = self._loki({"claude": _PartialThenBoom(name="claude"), "openai": _OkEngine()})
        out = []
        with self.assertRaises(RuntimeError):
            for chunk in loki.reason_stream("hi"):
                out.append(chunk)
        self.assertEqual(out, ["half "])

    def test_all_engines_failing_raises(self):
        loki = self._loki({"claude": _BoomEngine("claude"), "openai": _BoomEngine("openai")})
        with self.assertRaises(RuntimeError), \
             self.assertLogs("yggdrasil.loki.agent", level="WARNING"):
            loki.reason("hi")

    def test_repeated_identical_failure_is_logged_once(self):
        # A wedged engine fails the same way every turn (e.g. a model that can't
        # download). The first skip warns; later turns must NOT re-dump it.
        loki = self._loki({"claude": _BoomEngine(), "openai": _OkEngine()})
        with self.assertLogs("yggdrasil.loki.agent", level="WARNING") as first:
            loki.reason("turn one")
        self.assertTrue(any("claude" in m and "skipping" in m for m in first.output))
        # Subsequent identical failures are suppressed at WARNING (demoted to
        # DEBUG), so the same error doesn't spam the session log.
        with self.assertLogs("yggdrasil.loki.agent", level="DEBUG") as again:
            loki.reason("turn two")
            loki.reason("turn three")
        warnings = [m for m in again.output if m.startswith("WARNING")]
        suppressed = [m for m in again.output if "suppressed" in m and m.startswith("DEBUG")]
        self.assertEqual(warnings, [])
        self.assertTrue(suppressed)


if __name__ == "__main__":
    unittest.main()
