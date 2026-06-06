"""Tests for the token-efficient LLM data protocol.

These *measure* the candidate encodings (the "test it yourself" part) and pin
the conclusion: CSV is the most token-efficient legible format for sharing a
table with a model, Arrow IPC is the smallest binary wire but token-hostile in
a prompt, and the IPC channel round-trips losslessly.
"""
from __future__ import annotations

import unittest

try:
    import polars as pl

    from yggdrasil.loki import dataproto

    _HAVE = True
except Exception:  # pragma: no cover
    _HAVE = False


def _frame(n: int = 200) -> "pl.DataFrame":
    return pl.DataFrame({
        "date": [f"2026-01-{(i % 28) + 1:02d}" for i in range(n)],
        "symbol": ["EURUSD"] * n,
        "open": [1.10 + i * 1e-4 for i in range(n)],
        "close": [1.11 + i * 1e-4 for i in range(n)],
        "volume": [1000 + i for i in range(n)],
    })


@unittest.skipUnless(_HAVE, "requires polars")
class TestDataProto(unittest.TestCase):
    def test_encode_has_schema_header_and_csv_body(self):
        df = _frame(5)
        out = dataproto.encode(df)
        self.assertTrue(out.startswith("# 5 rows × 5 cols |"))
        self.assertIn("date:str", out)        # short dtype tags
        self.assertIn("volume:int", out)
        self.assertIn("date,symbol,open,close,volume", out)  # CSV header line

    def test_encode_truncates_with_a_note(self):
        out = dataproto.encode(_frame(500), max_rows=20)
        self.assertIn("# 500 rows × 5 cols", out)
        self.assertIn("480 more rows", out)
        # Body is the sample only: header + 20 data rows + 1 truncation note.
        self.assertEqual(out.count("\n"), 1 + 20 + 1)

    def test_csv_is_the_most_token_efficient_text_format(self):
        # The measured conclusion the protocol is built on.
        measured = dataproto.compare(_frame(100))
        text = {k: v["tokens"] for k, v in measured.items() if k != "ipc_b64"}
        self.assertEqual(min(text, key=text.get), "csv")
        self.assertEqual(dataproto.best_text_format(_frame(100)), "csv")
        # Markdown and JSON are markedly heavier than CSV (punctuation tax).
        self.assertLess(text["csv"], text["markdown"])
        self.assertLess(text["csv"], text["json"])

    def test_arrow_ipc_is_smallest_bytes_but_token_hostile(self):
        measured = dataproto.compare(_frame(200))
        # Compressed Arrow IPC is the smallest *byte* payload …
        self.assertLess(measured["ipc_b64"]["bytes"], measured["json"]["bytes"])
        # … yet base64'd binary costs *more tokens* than CSV — so it stays out
        # of prompts and serves the binary inter-agent channel instead.
        self.assertGreater(measured["ipc_b64"]["tokens"], measured["csv"]["tokens"])

    def test_ipc_round_trips_losslessly(self):
        df = _frame(50)
        back = dataproto.from_ipc(dataproto.to_ipc(df))
        self.assertEqual(back.shape, df.shape)
        self.assertEqual(back.columns, df.columns)
        self.assertEqual(back["volume"].sum(), df["volume"].sum())

    def test_compare_accepts_a_real_tokenizer(self):
        # A caller can plug a real tokenizer; here a word-splitter stands in.
        measured = dataproto.compare(_frame(10), tokenizer=lambda s: len(s.split()))
        self.assertTrue(all(v["tokens"] > 0 for v in measured.values()))


if __name__ == "__main__":
    unittest.main()
