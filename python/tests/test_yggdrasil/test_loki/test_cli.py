"""Tests for ``ygg loki`` CLI helpers."""
from __future__ import annotations

import json
import unittest

from yggdrasil.loki import cli


class TestJsonable(unittest.TestCase):
    """``--json`` must survive whatever a skill returns (frames included)."""

    def test_passthrough_basics(self):
        self.assertEqual(cli._jsonable({"a": 1, "b": [True, None, "x"]}),
                         {"a": 1, "b": [True, None, "x"]})

    def test_polars_frame_becomes_records(self):
        try:
            import polars as pl
        except Exception:
            self.skipTest("polars not installed")
        df = pl.DataFrame({"n": [1, 2], "city": ["Paris", "Tokyo"]})
        out = cli._jsonable({"rows": df, "row_count": 2})
        self.assertEqual(out["rows"], [{"n": 1, "city": "Paris"}, {"n": 2, "city": "Tokyo"}])
        # And the whole thing is now orjson-encodable.
        self.assertIn("Paris", cli._json({"rows": df}))

    def test_unknown_object_falls_back_to_str(self):
        class Weird:
            def __str__(self):
                return "weird!"

        self.assertEqual(cli._jsonable({"x": Weird()}), {"x": "weird!"})

    def test_model_like_to_dict_is_used(self):
        class Model:
            def to_dict(self):
                return {"k": "v"}

        self.assertEqual(cli._jsonable(Model()), {"k": "v"})
        # End-to-end: valid JSON out.
        self.assertEqual(json.loads(cli._json(Model())), {"k": "v"})


if __name__ == "__main__":
    unittest.main()
