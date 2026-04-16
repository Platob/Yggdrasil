"""Unittest base class for Apache Arrow tests.

Quick start
-----------
::

    from yggdrasil.testing import ArrowTestCase
    import pyarrow as pa

    class TestMyCodec(ArrowTestCase):
        def test_roundtrip(self):
            tbl = pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]})
            out = self.tmp_path / "t.parquet"
            self.write_parquet(tbl, out)
            self.assertFrameEqual(self.read_parquet(out), tbl)

Auto-install
------------
The class uses :func:`yggdrasil.environ.runtime_import_module` to load
``pyarrow``. Auto-install is opt-in: set ``auto_install = True`` on the
subclass or export ``YGG_TEST_AUTO_INSTALL=1``. Otherwise a missing
``pyarrow`` skips the class with an install hint.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from yggdrasil.environ import runtime_import_module

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = ["ArrowTestCase"]


def _auto_install(class_flag: bool | None) -> bool:
    if class_flag is not None:
        return class_flag
    return os.environ.get("YGG_TEST_AUTO_INSTALL", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


class ArrowTestCase(unittest.TestCase):
    """Base class for pyarrow integration tests.

    Attributes
    ----------
    pa : module
        The imported ``pyarrow`` module. Populated by ``setUpClass``.
    pq : module
        The imported ``pyarrow.parquet`` module (if ``require_parquet``).
    tmp_path : pathlib.Path
        Per-test scratch directory.

    Class attributes
    ----------------
    auto_install : bool | None
        Override the global auto-install behaviour for this class. ``None``
        means "follow ``YGG_TEST_AUTO_INSTALL``".
    require_parquet : bool
        If True (default), also import ``pyarrow.parquet``.
    """

    auto_install: ClassVar[bool | None] = None
    require_parquet: ClassVar[bool] = True

    pa: ClassVar[Any]          # pyarrow module
    pq: ClassVar[Any] = None   # pyarrow.parquet, if available
    tmp_path: Path

    # --- lifecycle ------------------------------------------------------
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        install = _auto_install(cls.auto_install)
        try:
            cls.pa = runtime_import_module("pyarrow", install=install)
            if cls.require_parquet:
                cls.pq = runtime_import_module(
                    "pyarrow.parquet", pip_name="pyarrow", install=install,
                )
        except ImportError:
            raise unittest.SkipTest(
                "'pyarrow' is not installed. "
                "Install it with: pip install pyarrow  "
                "or: pip install 'ygg[arrow]'  "
                "(or set YGG_TEST_AUTO_INSTALL=1 to auto-install)"
            )

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="ygg-arrow-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_path, ignore_errors=True)
        super().tearDown()

    # --- convenience constructors --------------------------------------
    def table(self, data: dict[str, Any], schema: Any = None) -> "pa.Table":
        """Shorthand for ``pa.table(data, schema=schema)``."""
        return self.pa.table(data, schema=schema)

    def record_batch(
        self,
        data: dict[str, Any],
        schema: Any = None,
    ) -> "pa.RecordBatch":
        """Shorthand for ``pa.record_batch(data, schema=schema)``."""
        return self.pa.record_batch(data, schema=schema)

    # --- I/O helpers ---------------------------------------------------
    def write_parquet(self, table: "pa.Table", path: Path | str) -> Path:
        """Write ``table`` to ``path`` as Parquet and return the path."""
        path = Path(path)
        self.pq.write_table(table, path)
        return path

    def read_parquet(self, path: Path | str) -> "pa.Table":
        """Read a Parquet file as a ``pa.Table``."""
        return self.pq.read_table(Path(path))

    def write_ipc(self, table: "pa.Table", path: Path | str) -> Path:
        """Write ``table`` as Arrow IPC (file format) and return the path."""
        path = Path(path)
        with self.pa.OSFile(str(path), "wb") as sink:
            with self.pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
        return path

    def read_ipc(self, path: Path | str) -> "pa.Table":
        """Read an Arrow IPC file as a ``pa.Table``."""
        with self.pa.memory_map(str(Path(path)), "r") as src:
            return self.pa.ipc.open_file(src).read_all()

    # --- assertions ----------------------------------------------------
    def assertFrameEqual(
        self,
        actual: "pa.Table",
        expected: "pa.Table",
        *,
        check_schema: bool = True,
        check_metadata: bool = False,
    ) -> None:
        """Assert two Arrow tables are equal.

        ``check_metadata=False`` ignores schema and field metadata, which
        is the usual "does the data match" check. Set it True for
        round-trip tests where metadata preservation matters.
        """
        if check_schema:
            schema_ok = actual.schema.equals(
                expected.schema, check_metadata=check_metadata,
            )
            if not schema_ok:
                self.fail(
                    f"Arrow schemas differ.\n"
                    f"--- expected ---\n{expected.schema}\n"
                    f"--- actual ---\n{actual.schema}"
                )

        if not actual.equals(expected, check_metadata=check_metadata):
            lines = ["Arrow tables differ."]
            for name in expected.column_names:
                if name not in actual.column_names:
                    lines.append(f"  column missing in actual: {name!r}")
                    continue
                if not actual[name].equals(expected[name]):
                    lines.append(f"  column differs: {name!r}")
                    lines.append(f"    expected: {expected[name].to_pylist()!r}")
                    lines.append(f"    actual:   {actual[name].to_pylist()!r}")
            for name in actual.column_names:
                if name not in expected.column_names:
                    lines.append(f"  unexpected column in actual: {name!r}")
            self.fail("\n".join(lines))

    def assertSchemaEqual(
        self,
        actual: "pa.Table | pa.Schema",
        expected_fields: list[tuple[str, Any]],
    ) -> None:
        """Assert a table/schema has exactly the given ``(name, type)`` fields."""
        schema = actual.schema if hasattr(actual, "schema") else actual
        got = [(f.name, str(f.type)) for f in schema]
        want = [(n, str(t)) for n, t in expected_fields]
        self.assertEqual(got, want, "Arrow schemas differ")