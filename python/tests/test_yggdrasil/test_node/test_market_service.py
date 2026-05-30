"""Market watchlist persistence (file-backed) + graceful yfinance fallback."""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from yggdrasil.node.api.services.market import MarketService
from yggdrasil.node.config import Settings


def _svc(home: Path) -> MarketService:
    return MarketService(Settings(node_id="t", node_home=home, front_home=home))


class TestWatchlist(unittest.TestCase):
    def test_empty_by_default(self):
        with tempfile.TemporaryDirectory() as d:
            res = _svc(Path(d)).get_watchlist()
            self.assertEqual(res["tickers"], [])
            self.assertEqual(res["node_id"], "t")

    def test_add_normalizes_and_dedupes(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d))
            svc.add_to_watchlist(" aapl ")
            res = svc.add_to_watchlist("AAPL")          # dupe after upper+strip
            self.assertEqual(res["tickers"], ["AAPL"])

    def test_add_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            _svc(home).add_to_watchlist("msft")
            self.assertEqual(_svc(home).get_watchlist()["tickers"], ["MSFT"])

    def test_remove(self):
        with tempfile.TemporaryDirectory() as d:
            svc = _svc(Path(d))
            svc.add_to_watchlist("AAPL")
            svc.add_to_watchlist("MSFT")
            res = svc.remove_from_watchlist("aapl")
            self.assertEqual(res["tickers"], ["MSFT"])


class TestQuoteFallback(unittest.TestCase):
    def test_quote_when_yfinance_missing(self):
        # yfinance isn't a runtime dep; the service degrades to available=False
        # rather than raising, so the endpoint never 500s.
        with tempfile.TemporaryDirectory() as d:
            try:
                import yfinance  # noqa: F401
                self.skipTest("yfinance installed; fallback path not exercised")
            except ImportError:
                pass
            res = asyncio.run(_svc(Path(d)).get_quote("AAPL"))
            self.assertEqual(res["ticker"], "AAPL")
            self.assertFalse(res["available"])
            self.assertIn("yfinance", res["error"])


if __name__ == "__main__":
    unittest.main()
