import unittest

from yggdrasil.databricks import Workspace


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.workspace = Workspace().connect()
        cls.loki = cls.workspace.loki()

    def test_trading_chat_structured(self):
        # Trading analytics chat (stateful) + structured JSON output
        trade = self.loki.new_trading_chat()

        # Keep context factual + compact (no extra system messages; folded into single system prompt)
        trade.add_context(
            "tables",
            {
                "trades": "uc.schema.trades",
                "curves": "uc.schema.curves",
                "notes": "curves has forward marks by instrument+delivery and an asof timestamp/date",
            },
        )
        trade.add_context(
            "columns_hint",
            "trades(book, instrument, delivery, qty, trade_price, trade_ts), "
            "curves(instrument, delivery, asof, px)",
        )

        resp = trade.chat(
            "Build a PnL explain template for yesterday by book: realized vs unrealized, and curve move impact. "
            "Return SQL + any key assumptions.",
            structured=True,
            temperature=0.1,
        )

        assert isinstance(resp, dict)
        assert "sql" in resp
        assert resp["sql"] is None or isinstance(resp["sql"], str)

        # Optional debug prints
        print(resp.get("final_answer", ""))
        print(resp.get("sql", ""))

    def test_sql_chat_generates_sql_only(self):
        # SQL engine chat (stateful) -> should return SQL text only
        sqlc = self.loki.new_sql_chat()

        # Provide schema/table mapping up front to reduce guessing
        sqlc.add_context(
            "tables",
            {"trades": "uc.schema.trades", "curves": "uc.schema.curves"},
        )
        sqlc.add_context(
            "schema",
            "uc.schema.trades(book, instrument, delivery, qty, trade_price, trade_ts), "
            "uc.schema.curves(instrument, delivery, asof, px)",
        )

        sql = sqlc.generate_sql(
            "Compute daily unrealized PnL by book for the last 30 days using end-of-day curve marks. "
            "Join trades to curves on instrument+delivery. "
            "Assume curve.asof is a timestamp; use the latest mark per day.",
            sql_only=True,
            temperature=0.0,
        )

        assert isinstance(sql, str)
        assert len(sql) > 0
        # Basic sanity: no markdown fences if sql_only=True
        assert "```" not in sql

        print(sql)
