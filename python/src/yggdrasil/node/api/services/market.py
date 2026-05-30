"""Simple market data service — yfinance for quotes/history + local watchlist."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool

from ...config import Settings


class MarketService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._watchlist_path = Path(settings.node_home) / "watchlist.json"

    def _load_watchlist(self) -> list[str]:
        if not self._watchlist_path.exists():
            return []
        return json.loads(self._watchlist_path.read_text())

    def _save_watchlist(self, tickers: list[str]) -> None:
        self._watchlist_path.parent.mkdir(parents=True, exist_ok=True)
        self._watchlist_path.write_text(json.dumps(tickers))

    def get_watchlist(self) -> dict:
        return {"node_id": self.settings.node_id, "tickers": self._load_watchlist()}

    def add_to_watchlist(self, ticker: str) -> dict:
        tickers = self._load_watchlist()
        t = ticker.upper().strip()
        if t not in tickers:
            tickers.append(t)
            self._save_watchlist(tickers)
        return {"node_id": self.settings.node_id, "tickers": tickers}

    def remove_from_watchlist(self, ticker: str) -> dict:
        tickers = [t for t in self._load_watchlist() if t != ticker.upper().strip()]
        self._save_watchlist(tickers)
        return {"node_id": self.settings.node_id, "tickers": tickers}

    async def get_quote(self, ticker: str) -> dict:
        return await run_in_threadpool(self._fetch_quote, ticker)

    async def get_history(self, ticker: str, period: str = "1mo", interval: str = "1d") -> dict:
        return await run_in_threadpool(self._fetch_history, ticker, period, interval)

    def _fetch_quote(self, ticker: str) -> dict[str, Any]:
        try:
            import yfinance as yf  # type: ignore
            t = yf.Ticker(ticker)
            info = t.fast_info
            return {
                "ticker": ticker.upper(),
                "price": getattr(info, "last_price", None),
                "previous_close": getattr(info, "previous_close", None),
                "change": None,
                "change_pct": None,
                "volume": getattr(info, "last_volume", None),
                "timestamp": None,
                "available": True,
            }
        except ImportError:
            return {"ticker": ticker.upper(), "price": None, "available": False, "error": "yfinance not installed"}
        except Exception as e:
            return {"ticker": ticker.upper(), "price": None, "available": False, "error": str(e)}

    def _fetch_history(self, ticker: str, period: str, interval: str) -> dict[str, Any]:
        try:
            import yfinance as yf  # type: ignore
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            if df.empty:
                return {"ticker": ticker.upper(), "x": [], "open": [], "high": [], "low": [], "close": [], "volume": [], "available": True}
            # yfinance returns a MultiIndex column frame for multi-ticker calls;
            # flatten to the field name so a single-ticker download reads cleanly.
            if hasattr(df.columns, "levels"):
                df.columns = df.columns.get_level_values(0)
            x = [str(i) for i in df.index.tolist()]

            def _col(name):
                col = df.get(name)
                if col is None:
                    return [None] * len(x)
                return [float(v) if v == v else None for v in col.tolist()]

            return {
                "ticker": ticker.upper(),
                "period": period,
                "interval": interval,
                "x": x,
                "open": _col("Open"),
                "high": _col("High"),
                "low": _col("Low"),
                "close": _col("Close"),
                "volume": _col("Volume"),
                "available": True,
            }
        except ImportError:
            return {"ticker": ticker.upper(), "x": [], "open": [], "high": [], "low": [], "close": [], "volume": [], "available": False, "error": "yfinance not installed"}
        except Exception as e:
            return {"ticker": ticker.upper(), "x": [], "open": [], "high": [], "low": [], "close": [], "volume": [], "available": False, "error": str(e)}
