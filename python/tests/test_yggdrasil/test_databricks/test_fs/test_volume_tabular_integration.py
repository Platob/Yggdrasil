"""Live tabular round-trips + call efficiency over a Unity Catalog volume.

Writes and reads tabular data (Parquet / CSV) through a real :class:`VolumePath`
— no mocks — as Arrow and polars, asserting both **content** round-trips and the
**number of backend calls** (a whole-file table write is a single PUT thanks to
the overwrite fast path; a read is a single GET).

Skipped unless ``DATABRICKS_HOST`` is set; reads the volume from
``DATABRICKS_INTEGRATION_CATALOG`` / ``_SCHEMA`` / ``_VOLUME`` (default
``trading`` / ``unittest`` / ``tmp``); permission errors degrade to a skip.
"""
from __future__ import annotations

import collections
import os
import secrets
import unittest
from contextlib import contextmanager

import pyarrow as pa
from databricks.sdk.errors import DatabricksError
from databricks.sdk.errors.platform import PermissionDenied

from yggdrasil.enums import MimeTypes

from .. import DatabricksIntegrationCase


class TestVolumeTabularIntegration(DatabricksIntegrationCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cat = os.environ.get("DATABRICKS_INTEGRATION_CATALOG", "trading")
        sch = os.environ.get("DATABRICKS_INTEGRATION_SCHEMA", "unittest")
        vol = os.environ.get("DATABRICKS_INTEGRATION_VOLUME", "tmp")
        base = f"/Volumes/{cat}/{sch}/{vol}/_ygg_tab_{secrets.token_hex(4)}"
        cls.base = cls.client.path(base)
        try:
            cls.base.mkdir(parents=True, exist_ok=True)
        except (DatabricksError, PermissionDenied) as exc:
            raise unittest.SkipTest(f"cannot write to {base}: {exc}") from exc
        cls.table = pa.table({
            "id": pa.array([1, 2, 3], pa.int64()),
            "v": pa.array([1.5, 2.5, 3.5], pa.float64()),
            "g": pa.array(["a", "b", "a"], pa.string()),
        })

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.base.remove(recursive=True)
        except Exception:
            pass
        super().tearDownClass()

    @contextmanager
    def _count(self):
        from yggdrasil.http_.session import HTTPSession
        calls: collections.Counter = collections.Counter()
        orig = HTTPSession.fetch
        HTTPSession.fetch = lambda s, m, u, **k: (calls.update([m]), orig(s, m, u, **k))[1]
        try:
            yield calls
        finally:
            HTTPSession.fetch = orig

    def _fresh(self, name: str):
        p = self.base / name
        p.invalidate_singleton()
        return p

    def test_parquet_write_is_one_put_read_is_one_get(self):
        with self._count() as calls:
            self._fresh("t.parquet").write_table(self.table)
        self.assertEqual(calls.get("PUT"), 1, dict(calls))      # single overwrite PUT
        self.assertEqual(calls.get("GET", 0), 0, dict(calls))

        with self._count() as calls:
            got = self._fresh("t.parquet").read_arrow_table()
        self.assertEqual(calls.get("GET"), 1, dict(calls))      # single fetch
        self.assertEqual(got.num_rows, 3)
        self.assertEqual(got.column("id").to_pylist(), [1, 2, 3])

    def test_parquet_roundtrip_to_polars(self):
        self._fresh("p.parquet").write_table(self.table)
        frame = self._fresh("p.parquet").read_polars_frame()
        self.assertEqual(frame.shape, (3, 3))
        self.assertEqual(sorted(frame["id"].to_list()), [1, 2, 3])

    def test_csv_roundtrip(self):
        self._fresh("t.csv").as_media(MimeTypes.CSV).write_arrow_table(self.table)
        got = self._fresh("t.csv").as_media(MimeTypes.CSV).read_arrow_table()
        self.assertEqual(got.num_rows, 3)
        self.assertEqual(sorted(got.column("g").to_pylist()), ["a", "a", "b"])

    def test_schema_via_read(self):
        self._fresh("s.parquet").write_table(self.table)
        schema = self._fresh("s.parquet").read_arrow_table().schema
        self.assertEqual(schema.names, ["id", "v", "g"])
