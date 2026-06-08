"""Tests for the ENTSO-E energy-data path — yggdrasil.loki.entsoe + EntsoeSkill."""
from __future__ import annotations

import datetime as dt
import os
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import entsoe

_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"


def _prices_xml() -> str:
    # Two hourly points in one period — the A44 day-ahead-prices shape.
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Publication_MarketDocument xmlns="{_NS}"><mRID>x</mRID>'
        f'<TimeSeries><mRID>1</mRID>'
        f'<currency_Unit.name>EUR</currency_Unit.name>'
        f'<price_Measure_Unit.name>MWH</price_Measure_Unit.name>'
        f'<Period><timeInterval><start>2024-01-01T00:00Z</start>'
        f'<end>2024-01-01T02:00Z</end></timeInterval><resolution>PT60M</resolution>'
        f'<Point><position>1</position><price.amount>40.5</price.amount></Point>'
        f'<Point><position>2</position><price.amount>42.0</price.amount></Point>'
        f'</Period></TimeSeries></Publication_MarketDocument>'
    )


def _load_xml() -> str:
    return (
        f'<Publication_MarketDocument xmlns="{_NS}"><mRID>x</mRID>'
        f'<TimeSeries><mRID>1</mRID>'
        f'<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>'
        f'<Period><timeInterval><start>2024-01-01T00:00Z</start>'
        f'<end>2024-01-01T01:00Z</end></timeInterval><resolution>PT60M</resolution>'
        f'<Point><position>1</position><quantity>50000</quantity></Point>'
        f'</Period></TimeSeries></Publication_MarketDocument>'
    )


class TestZoneResolution(unittest.TestCase):
    def test_alias_resolves_to_eic(self):
        self.assertEqual(entsoe.resolve_zone("DE_LU"), "10Y1001A1001A82H")

    def test_alias_is_separator_and_case_insensitive(self):
        self.assertEqual(entsoe.resolve_zone("de-lu"), "10Y1001A1001A82H")
        self.assertEqual(entsoe.resolve_zone(" fr "), entsoe.ZONES["FR"])

    def test_eic_passthrough(self):
        self.assertEqual(entsoe.resolve_zone("10YFR-RTE------C"), "10YFR-RTE------C")

    def test_unknown_alias_raises(self):
        with self.assertRaises(KeyError):
            entsoe.resolve_zone("ZZ")


class TestBuildQuery(unittest.TestCase):
    def test_prices_set_both_domains(self):
        q = entsoe.build_query("day_ahead_prices", "DE_LU",
                               "2024-01-01", "2024-01-02", security_token="tok")
        self.assertEqual(q["documentType"], "A44")
        self.assertEqual(q["in_Domain"], q["out_Domain"], "10Y1001A1001A82H")
        self.assertEqual(q["periodStart"], "202401010000")
        self.assertEqual(q["periodEnd"], "202401020000")
        self.assertEqual(q["securityToken"], "tok")

    def test_load_uses_out_bidding_zone_and_process(self):
        q = entsoe.build_query("load", "FR", "2024-01-01", "2024-01-02")
        self.assertEqual(q["documentType"], "A65")
        self.assertEqual(q["processType"], "A16")
        self.assertEqual(q["outBiddingZone_Domain"], entsoe.ZONES["FR"])
        self.assertNotIn("in_Domain", q)
        self.assertNotIn("securityToken", q)          # omitted when not provided

    def test_generation_uses_in_domain(self):
        q = entsoe.build_query("generation", "NL", "2024-01-01", "2024-01-02")
        self.assertEqual(q["documentType"], "A75")
        self.assertEqual(q["in_Domain"], entsoe.ZONES["NL"])

    def test_datetime_with_tz_normalized_to_utc(self):
        start = dt.datetime(2024, 1, 1, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=2)))
        q = entsoe.build_query("day_ahead_prices", "DE_LU", start, "2024-01-02")
        self.assertEqual(q["periodStart"], "202312312300")   # 01:00+02:00 → 23:00Z

    def test_unknown_series_raises(self):
        with self.assertRaises(KeyError):
            entsoe.build_query("nope", "DE_LU", "2024-01-01", "2024-01-02")


class TestParseTimeseries(unittest.TestCase):
    def test_parses_prices_with_timestamps(self):
        rows = entsoe.parse_timeseries_xml(_prices_xml())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["timestamp"],
                         dt.datetime(2024, 1, 1, 0, 0, tzinfo=dt.timezone.utc))
        self.assertEqual(rows[1]["timestamp"],
                         dt.datetime(2024, 1, 1, 1, 0, tzinfo=dt.timezone.utc))
        self.assertEqual(rows[0]["value"], 40.5)
        self.assertEqual(rows[0]["currency"], "EUR")
        self.assertEqual(rows[0]["unit"], "MWH")

    def test_parses_load_quantity(self):
        rows = entsoe.parse_timeseries_xml(_load_xml())
        self.assertEqual(rows[0]["value"], 50000.0)
        self.assertEqual(rows[0]["unit"], "MAW")

    def test_15min_resolution_steps_quarter_hour(self):
        xml = _prices_xml().replace("PT60M", "PT15M")
        rows = entsoe.parse_timeseries_xml(xml)
        self.assertEqual(rows[1]["timestamp"],
                         dt.datetime(2024, 1, 1, 0, 15, tzinfo=dt.timezone.utc))

    def test_acknowledgement_means_no_data(self):
        ack = ('<Acknowledgement_MarketDocument><Reason><code>999</code>'
               '<text>No matching data found</text></Reason></Acknowledgement_MarketDocument>')
        self.assertEqual(entsoe.parse_timeseries_xml(ack), [])


class TestToFrame(unittest.TestCase):
    def test_frame_is_sorted_and_tagged(self):
        try:
            import polars as pl  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        df = entsoe.to_frame(_prices_xml(), zone="DE_LU", series="day_ahead_prices")
        self.assertEqual(df.height, 2)
        self.assertEqual(set(["timestamp", "value", "zone", "series"]) - set(df.columns), set())
        self.assertEqual(df["zone"].unique().to_list(), ["DE_LU"])
        self.assertEqual(df["value"].to_list(), [40.5, 42.0])

    def test_empty_document_yields_empty_typed_frame(self):
        try:
            import polars as pl
        except Exception:
            self.skipTest("polars not installed")
        df = entsoe.to_frame("<Acknowledgement_MarketDocument/>", zone="DE_LU")
        self.assertEqual(df.height, 0)
        self.assertIn("timestamp", df.columns)


class TestFetchFrame(unittest.TestCase):
    def test_missing_token_raises_clear_error(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ENTSOE_API_TOKEN", None)
            os.environ.pop("ENTSOE_SECURITY_TOKEN", None)
            with self.assertRaises(ValueError) as ctx:
                entsoe.fetch_frame("day_ahead_prices", "DE_LU", "2024-01-01", "2024-01-02")
        self.assertIn("ENTSOE_API_TOKEN", str(ctx.exception))

    def test_fetch_uses_httpsession_and_parses(self):
        try:
            import polars as pl  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        resp = MagicMock(text=_prices_xml())
        sess = MagicMock()
        sess.get.return_value = resp
        with patch("yggdrasil.http_.HTTPSession", return_value=sess):
            df = entsoe.fetch_frame("day_ahead_prices", "DE_LU",
                                    "2024-01-01", "2024-01-02", security_token="tok")
        self.assertEqual(df.height, 2)
        # The request went to the Transparency API with the resolved params.
        call = sess.get.call_args
        self.assertEqual(call.args[0], entsoe.ENTSOE_API)
        self.assertEqual(call.kwargs["params"]["documentType"], "A44")
        self.assertEqual(call.kwargs["params"]["securityToken"], "tok")


class TestInferAndRouting(unittest.TestCase):
    def test_infer_series_and_zone(self):
        self.assertEqual(entsoe.infer_query("german day-ahead power prices"),
                         {"series": "day_ahead_prices", "zone": "DE_LU"})
        self.assertEqual(entsoe.infer_query("french electricity demand")["series"], "load")
        self.assertEqual(entsoe.infer_query("spanish power generation"),
                         {"series": "generation", "zone": "ES"})

    def test_two_letter_alias_not_matched_in_words(self):
        # "es" in "prices" must NOT resolve to Spain (word-boundary matching).
        self.assertEqual(entsoe.infer_query("power prices please")["zone"], "DE_LU")

    def test_autonomous_routing_to_entsoe_skill(self):
        from yggdrasil.loki import Loki

        loki = Loki()
        p = loki.plan("german day-ahead power prices for last week")
        self.assertEqual((p.action, p.skill), ("skill", "entsoe"))
        self.assertEqual(p.skill_kwargs["zone"], "DE_LU")
        # A definitional question is not a data ask → plain reasoning.
        self.assertEqual(loki.plan("what is electricity").action, "reason")


class TestEntsoeRendering(unittest.TestCase):
    def _render(self, res):
        import io
        from contextlib import redirect_stdout

        from yggdrasil.cli import style
        from yggdrasil.loki import cli
        style.force_color(False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._print_entsoe(style, res)
        return style.strip(buf.getvalue())

    def test_online_result_renders_aligned_frame(self):
        out = self._render({"available": True, "series": "day_ahead_prices", "zone": "FR",
                            "eic": "10YFR-RTE------C", "rows": 2,
                            "preview": "timestamp │ value\n──────────┼──────\n00:00 │ 40.5",
                            "cached_to": "/c/x.parquet", "stored": None,
                            "next_steps": ["reuse: …"]})
        self.assertIn("day_ahead_prices · FR", out)
        self.assertIn("2 rows", out)
        self.assertIn("cached", out)
        self.assertIn("│", out)                          # the frame preview, indented

    def test_offline_shows_hint(self):
        out = self._render({"available": False, "hint": "set ENTSOE_API_TOKEN"})
        self.assertIn("ENTSOE_API_TOKEN", out)


class TestEntsoeSkill(unittest.TestCase):
    def test_offline_safe_without_token(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.skills import EntsoeSkill

        with patch.object(entsoe, "token", return_value=None):
            res = EntsoeSkill().run(Loki(), series="day_ahead_prices", zone="DE_LU")
        self.assertFalse(res["available"])
        self.assertIn("ENTSOE_API_TOKEN", res["hint"])

    def test_fetches_caches_and_reports(self):
        try:
            import polars as pl  # noqa: F401
        except Exception:
            self.skipTest("polars not installed")
        import tempfile

        from yggdrasil.loki import Loki
        from yggdrasil.loki.skills import EntsoeSkill

        df = entsoe.to_frame(_prices_xml(), zone="DE_LU", series="day_ahead_prices")
        with tempfile.TemporaryDirectory() as d, \
                patch.object(entsoe, "token", return_value="tok"), \
                patch.object(entsoe, "fetch_frame", return_value=df) as fetch:
            res = EntsoeSkill().run(Loki(), series="day_ahead_prices", zone="DE_LU",
                                    days=2, cache_dir=d)
        self.assertTrue(res["available"])
        self.assertEqual(res["rows"], 2)
        self.assertEqual(res["eic"], "10Y1001A1001A82H")
        self.assertTrue(res["cached_to"].endswith(".parquet"))
        self.assertTrue(any("reuse" in s for s in res["next_steps"]))
        fetch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
