"""``Tabular.from_`` / ``Tabular.new`` dispatch to in-memory holders.

In-memory engine frames (arrow / polars / pandas / spark) route to their
native Tabular; path/URL-shaped sources fall back to the IO layer.
"""

from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.arrow.tabular import ArrowTabular
from yggdrasil.io.tabular.base import Tabular


def test_from_arrow_table_and_batch() -> None:
    assert type(Tabular.from_(pa.table({"a": [1]}))).__name__ == "ArrowTabular"
    assert type(Tabular.from_(pa.record_batch({"a": [1]}))).__name__ == "ArrowTabular"


def test_from_polars_frame() -> None:
    pl = pytest.importorskip("polars")
    assert type(Tabular.from_(pl.DataFrame({"a": [1]}))).__name__ == "PolarsTabular"
    # LazyFrame is collected into a PolarsTabular
    t = Tabular.from_(pl.LazyFrame({"a": [1, 2]}))
    assert type(t).__name__ == "PolarsTabular"
    assert t.read_arrow_table().num_rows == 2


def test_from_pandas_frame() -> None:
    pd = pytest.importorskip("pandas")
    assert type(Tabular.from_(pd.DataFrame({"a": [1]}))).__name__ == "PandasTabular"


def test_from_tabular_passthrough() -> None:
    at = ArrowTabular(pa.table({"x": [1]}))
    assert Tabular.from_(at) is at
    assert Tabular.new(at) is at


def test_from_path_falls_back_to_io() -> None:
    # Path-shaped string is serialized-data territory → IO layer, not a frame.
    resolved = Tabular.from_("data.parquet")
    assert "Path" in type(resolved).__name__


def test_from_dict_is_not_a_source() -> None:
    # A dict is neither a frame nor serialized data — from_ declines it.
    assert Tabular.from_({"a": [1]}, default="X") == "X"


def test_new_builds_inmemory_from_frames_and_data() -> None:
    pl = pytest.importorskip("polars")
    assert type(Tabular.new(pa.table({"a": [1]}))).__name__ == "ArrowTabular"
    assert type(Tabular.new(pl.DataFrame({"a": [1]}))).__name__ == "PolarsTabular"
    # dict / list / None → in-memory ArrowTabular
    assert type(Tabular.new({"a": [1, 2]})).__name__ == "ArrowTabular"
    assert Tabular.new([{"a": 1}, {"a": 2}]).read_arrow_table().num_rows == 2
    assert Tabular.new(None).read_arrow_table().num_rows == 0
