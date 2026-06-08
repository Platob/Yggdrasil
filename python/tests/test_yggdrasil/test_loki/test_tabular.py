"""Tests for the data/timeseries path — classifier, routing, TabularSkill."""
from __future__ import annotations

import functools
import http.server
import tempfile
import threading
import unittest
from pathlib import Path

from yggdrasil.loki import Loki
from yggdrasil.loki.capability import Backend

try:
    import polars  # noqa: F401

    from yggdrasil.loki import web  # noqa: F401  (HTTPResponse → io handlers)

    _HAVE_STACK = True
except Exception:  # pragma: no cover
    _HAVE_STACK = False


def _loki():
    loki = Loki()
    loki._backends = [Backend("local", True)]
    return loki


class TestClassifyAndRoute(unittest.TestCase):
    def test_classify_data_and_timeseries(self):
        loki = _loki()
        self.assertTrue(loki.classify_data("EUR/USD rate over the last 2 weeks")["timeseries"])
        self.assertTrue(loki.classify_data("show me the iris dataset csv")["data"])
        self.assertFalse(loki.classify_data("fix the bug in app.py")["data"])

    def test_data_url_routes_to_tabular(self):
        plan = _loki().route(
            "get https://api.x.io/v1/rates over the last 2 weeks of exchange rates")
        self.assertEqual(plan["category"], "data")
        self.assertEqual(plan["action"], "tabular")
        self.assertTrue(plan["timeseries"])
        self.assertEqual(plan["url"], "https://api.x.io/v1/rates")

    def test_plain_url_still_routes_to_web(self):
        plan = _loki().route("fetch https://example.com/about")
        self.assertEqual(plan["action"], "web")

    def test_local_data_file_routes_to_tabular(self):
        plan = _loki().route("analyze /tmp/cities.csv and summarize the trend")
        self.assertEqual((plan["action"], plan["category"]), ("tabular", "data"))
        self.assertEqual(plan["url"], "/tmp/cities.csv")

    def test_code_file_still_routes_to_act_not_tabular(self):
        plan = _loki().route("fix the bug in app.py")
        self.assertNotEqual(plan["action"], "tabular")


@unittest.skipUnless(_HAVE_STACK, "requires the polars/io stack")
class TestTabularBehavior(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="ygg-tab-")
        d = Path(cls.dir)
        (d / "data.csv").write_text("city,pop\nParis,2161\nTokyo,13960\n")
        # A JSON array of records — the io JSON handler tabularizes it directly
        # (HTTPResponse.to_polars), no custom normalization needed.
        (d / "ts.json").write_text(
            '[{"date":"2026-05-22","value":1.1595},{"date":"2026-05-23","value":1.1643}]')
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=cls.dir)
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.cache = tempfile.mkdtemp(prefix="ygg-cache-")

    def test_fetch_csv_caches_parquet_and_proposes(self):
        res = _loki().run("tabular", url=f"{self.base}/data.csv",
                          cache_dir=self.cache, key="cities")
        self.assertEqual(res["rows"], 2)
        self.assertEqual(res["columns"], ["city", "pop"])
        cached = Path(res["cached_to"])
        self.assertTrue(cached.is_file() and cached.suffix == ".parquet")
        self.assertTrue(any("reuse" in s for s in res["next_steps"]))

    def test_fetch_json_array_via_io(self):
        # JSON → frame is the io layer's job (HTTPResponse.to_polars), not a
        # bespoke normalizer: an array of records lands as proper columns.
        res = _loki().run("tabular", url=f"{self.base}/ts.json",
                          cache_dir=self.cache, key="ts")
        self.assertEqual(res["columns"], ["date", "value"])
        self.assertEqual(res["rows"], 2)

    def test_reuse_cache_then_store(self):
        first = _loki().run("tabular", url=f"{self.base}/data.csv",
                            cache_dir=self.cache, key="cities")
        out = Path(self.cache) / "exported.csv"
        again = _loki().run("tabular", cache=first["cached_to"],
                            cache_dir=self.cache, store=str(out))
        self.assertEqual(again["rows"], 2)
        self.assertEqual(again["stored"], str(out))
        self.assertTrue(out.is_file())

    def test_reads_local_path_via_io_not_http(self):
        # A bare local path goes through the io handlers (IO.from_), not HTTP.
        local = Path(self.dir) / "data.csv"
        res = _loki().run("tabular", url=str(local), cache_dir=self.cache, key="local")
        self.assertEqual(res["rows"], 2)
        self.assertEqual(res["columns"], ["city", "pop"])

    def test_requires_url_or_cache(self):
        with self.assertRaises(ValueError):
            _loki().run("tabular")

    def test_transform_casts_field_types(self):
        first = _loki().run("tabular", url=f"{self.base}/data.csv",
                            cache_dir=self.cache, key="cities")
        t = _loki().run("transform", cache=first["cached_to"], cache_dir=self.cache,
                        cast={"pop": "float64"}, key="cities-typed")
        self.assertEqual(t["schema"]["pop"], "Float64")
        self.assertTrue(Path(t["cached_to"]).is_file())

    def test_transform_select_and_rename(self):
        first = _loki().run("tabular", url=f"{self.base}/ts.json",
                            cache_dir=self.cache, key="ts")
        t = _loki().run("transform", cache=first["cached_to"], cache_dir=self.cache,
                        cast={"value": "float64", "date": "date"},
                        rename={"value": "usd"}, select=["date", "usd"], key="ts2")
        self.assertEqual(t["columns"], ["date", "usd"])
        self.assertEqual(t["schema"]["date"], "Date")
        self.assertEqual(t["schema"]["usd"], "Float64")

    def test_preview_is_tabular_display(self):
        res = _loki().run("tabular", url=f"{self.base}/data.csv",
                          cache_dir=self.cache, key="cities")
        # The preview is Tabular.display() — an aligned table (header + rows),
        # not a hand-rolled serialization.
        self.assertIn("city", res["preview"])
        self.assertIn("pop", res["preview"])
        self.assertIn("Paris", res["preview"])


class TestPlanning(unittest.TestCase):
    def test_agentplan_is_mapping_compatible_and_typed(self):
        from yggdrasil.loki.planning import AgentPlan

        p = _loki().plan("analyze the iris dataset csv trends")
        self.assertIsInstance(p, AgentPlan)
        self.assertEqual(p["category"], p.category)          # mapping + attr
        self.assertEqual(p.get("nope", "x"), "x")
        self.assertIn("category", p.to_dict())

    def test_persona_classification(self):
        loki = _loki()
        self.assertEqual(loki.plan("refactor the bug in this function").persona, "software-engineer")
        self.assertEqual(loki.plan("quote the stock market price, spread and volatility").persona, "trader")
        self.assertEqual(loki.plan("build an ETL pipeline into a delta warehouse").persona, "data-engineer")
        self.assertEqual(loki.plan("I need to confess something I feel guilty about").persona, "confessor")
        self.assertEqual(loki.plan("what's the weather like").persona, "assistant")

    def test_required_skills_filled(self):
        p = _loki().plan("fetch https://x/rates.csv over the last 2 weeks")
        self.assertIn("tabular", p.required_skills)
        self.assertTrue(p.persona_prompt())  # data-analyst persona has a prompt

    def test_autonomous_routing_to_scaffold_and_delegate(self):
        loki = _loki()
        # Plain-language intents route to the new actions — no slash command.
        self.assertEqual(loki.plan("create a new python project called acme").action, "scaffold")
        self.assertEqual(loki.plan("scaffold a rust + go cli from scratch").action, "scaffold")
        self.assertEqual(loki.plan("do these in parallel: add tests and fix lint").action, "delegate")
        self.assertEqual(loki.plan("swarm: refactor each module").action, "delegate")
        # A plain question with no scaffold/delegate/data/databricks signal reasons.
        self.assertEqual(loki.plan("what is the capital of France").action, "reason")


if __name__ == "__main__":
    unittest.main()
