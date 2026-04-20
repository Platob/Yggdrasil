"""Unit tests for :class:`yggdrasil.io.buffer.path_io.PathIO`.

Covers:

* options validation
* filter normalization — both ds.Expression (pushdown) and pc mask (fallback)
* parallel-path equivalence (same spec → same rows, both paths)
* roundtrips on dataset-capable (Parquet) and fallback (XLSX) formats
* column projection (including filter-column augmentation on fallback)
* partition value injection (Hive + directory-name)
* schema collection (single file, directory, full vs. first-only)
* count_rows (dataset fast path + fallback)
* cast symmetry between read_arrow_table and read_arrow_batches
* abstract-method enforcement
* read_dataset NotImplementedError on non-dataset formats
"""
from __future__ import annotations

import operator
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from yggdrasil.io import MimeTypes
from yggdrasil.io.buffer.bytes_io import BytesIO
from yggdrasil.io.buffer.media_io import MediaIO
from yggdrasil.io.buffer.path_io import PathIO, PathOptions
from yggdrasil.io.enums import MediaType, MimeType


# =====================================================================
# Deps probes
# =====================================================================

def _has(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_OPENPYXL = _has("openpyxl")
HAS_PYARROW_PARQUET = _has("pyarrow.parquet")


# =====================================================================
# Minimal concrete subclass for testing
# =====================================================================

@dataclass(slots=True)
class _LocalPathIO(PathIO):
    """Trivial local-filesystem PathIO for exercising the abstract base."""

    @classmethod
    def make(
        cls,
        path: str | Path,
        media: MediaType | MimeType | str | None = None,
    ) -> "_LocalPathIO":
        p = Path(path)
        media_type = (
            media if isinstance(media, MediaType)
            else MediaType(media) if isinstance(media, MimeType)
            else MediaType.parse(media, default=None) if isinstance(media, str)
            else None
        )
        # PathIO requires a holder BytesIO per the MediaIO base contract,
        # but PathIO reads directly from the filesystem and never touches
        # it. A fresh empty BytesIO is a safe placeholder.
        holder = BytesIO()

        # For a single file, parse the extension. For a directory, leave
        # media_type as None so PathIO.__post_init__ infers it from the
        # first file discovered via iter_files() — parsing the directory
        # path itself would resolve to OCTET_STREAM and short-circuit.
        if media_type is None and p.is_file():
            media_type = MediaType.parse(p, default=None)

        return cls(media_type=media_type, holder=holder, path=p)

    def iter_files(
        self,
        recursive: bool = True,
        *,
        include_hidden: bool = False,
        supported_only: bool = True,
        mime_type: MimeType | str | None = None,
    ) -> Iterator["_LocalPathIO"]:
        if self.path.is_file():
            yield self
            return

        if not self.path.is_dir():
            return

        pattern = "**/*" if recursive else "*"
        for candidate in sorted(self.path.glob(pattern)):
            if not candidate.is_file():
                continue

            name = candidate.name
            if not include_hidden and (name.startswith(".") or name.startswith("_")):
                continue

            file_mime = MimeType.parse(candidate, default=None)
            if supported_only and file_mime is None:
                continue

            if mime_type is not None:
                wanted = (
                    mime_type if isinstance(mime_type, MimeType)
                    else MimeType.parse(mime_type, default=None)
                )
                if file_mime is not wanted:
                    continue

            yield _LocalPathIO(
                media_type=MediaType(file_mime) if file_mime is not None else MediaType.parse(candidate, default=None),
                holder=BytesIO(),
                path=candidate,
            )


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture()
def sample_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["alice", "bob", "carol", "dave", "eve"],
            "status": ["active", "inactive", "active", "active", "inactive"],
            "score": [3.14, 2.71, 1.41, 4.20, 0.0],
        }
    )


@pytest.fixture()
def parquet_dir(tmp_path: Path, sample_table: pa.Table) -> Path:
    """Write sample_table as three partitioned Parquet files."""
    pytest.importorskip("pyarrow.parquet")
    d = tmp_path / "dataset"
    d.mkdir()
    # Partition by status hive-style.
    (d / "status=active").mkdir()
    (d / "status=inactive").mkdir()

    active = sample_table.filter(pc.equal(sample_table.column("status"), "active"))
    inactive = sample_table.filter(pc.equal(sample_table.column("status"), "inactive"))

    pq.write_table(active, d / "status=active" / "part-0.parquet")
    pq.write_table(inactive, d / "status=inactive" / "part-0.parquet")
    return d


@pytest.fixture()
def parquet_file(tmp_path: Path, sample_table: pa.Table) -> Path:
    pytest.importorskip("pyarrow.parquet")
    p = tmp_path / "data.parquet"
    pq.write_table(sample_table, p)
    return p


@pytest.fixture()
def xlsx_file(tmp_path: Path, sample_table: pa.Table) -> Path:
    """Write sample_table as a single XLSX file via the XLSX MediaIO."""
    if not HAS_OPENPYXL:
        pytest.skip("openpyxl required for fallback path tests")
    p = tmp_path / "data.xlsx"
    buf = BytesIO()
    MediaIO.make(buf, MimeTypes.XLSX).write_arrow_table(
        sample_table, engine="fallback"
    )
    p.write_bytes(buf.to_bytes())
    return p


# =====================================================================
# Options
# =====================================================================

class TestPathOptions:
    def test_defaults(self):
        opt = PathOptions()
        assert opt.filter is None
        assert opt.recursive is True
        assert opt.include_hidden is False
        assert opt.supported_only is True
        assert opt.format is None
        assert opt.partitioning == "hive"
        assert opt.partition_base_dir is None
        assert opt.ignore_prefixes == (".", "_")
        assert opt.batch_readahead == 16
        assert opt.fragment_readahead == 4

    def test_recursive_must_be_bool(self):
        with pytest.raises(TypeError, match="recursive"):
            PathOptions(recursive="yes")  # type: ignore[arg-type]

    def test_partition_base_dir_normalized_to_path(self, tmp_path: Path):
        opt = PathOptions(partition_base_dir=str(tmp_path))
        assert isinstance(opt.partition_base_dir, Path)

    def test_ignore_prefixes_must_be_sequence_not_string(self):
        with pytest.raises(TypeError, match="ignore_prefixes"):
            PathOptions(ignore_prefixes=".dotfile")  # type: ignore[arg-type]

    def test_ignore_prefixes_must_be_strings(self):
        with pytest.raises(TypeError, match="ignore_prefixes"):
            PathOptions(ignore_prefixes=[".", 42])  # type: ignore[list-item]

    def test_readahead_must_be_non_negative(self):
        with pytest.raises(ValueError, match="batch_readahead"):
            PathOptions(batch_readahead=-1)

    def test_readahead_none_becomes_zero(self):
        opt = PathOptions(batch_readahead=None, fragment_readahead=None)
        assert opt.batch_readahead == 0
        assert opt.fragment_readahead == 0

    def test_readahead_rejects_bool(self):
        """bool is a subclass of int — the validator should catch that."""
        with pytest.raises(TypeError, match="batch_readahead"):
            PathOptions(batch_readahead=True)  # type: ignore[arg-type]


# =====================================================================
# Filter normalization — dataset (Expression) path
# =====================================================================

class TestFilterToExpression:
    def test_none_returns_none(self):
        assert PathIO._normalize_filter_to_expression(None) is None

    def test_passthrough_expression(self):
        expr = ds.field("x") > 5
        assert PathIO._normalize_filter_to_expression(expr) is expr

    def test_dict_filter(self):
        e = PathIO._normalize_filter_to_expression({"x": 5})
        assert isinstance(e, ds.Expression)

    def test_dict_with_none_is_null(self):
        e = PathIO._normalize_filter_to_expression({"x": None})
        assert isinstance(e, ds.Expression)

    def test_dict_with_list_becomes_in(self):
        e = PathIO._normalize_filter_to_expression({"x": [1, 2, 3]})
        assert isinstance(e, ds.Expression)

    def test_empty_in_matches_nothing(self):
        e = PathIO._normalize_filter_to_expression({"x": []})
        assert isinstance(e, ds.Expression)

    def test_tuple_shorthand(self):
        e = PathIO._normalize_filter_to_expression(("x", ">", 5))
        assert isinstance(e, ds.Expression)

    def test_list_of_tuples(self):
        e = PathIO._normalize_filter_to_expression([
            ("x", ">", 5),
            ("y", "=", "foo"),
        ])
        assert isinstance(e, ds.Expression)

    def test_rejects_malformed_tuple(self):
        with pytest.raises(TypeError, match="filter"):
            PathIO._normalize_filter_to_expression([("x",)])  # 1-tuple invalid

    def test_rejects_unknown_operator(self):
        with pytest.raises(ValueError, match="operator"):
            PathIO._normalize_filter_to_expression([("x", "matches", "foo")])

    @pytest.mark.parametrize("op", [
        "=", "==", "eq",
        "!=", "<>", "ne",
        ">", "gt",
        ">=", "gte", "ge",
        "<", "lt",
        "<=", "lte", "le",
        "in", "not in",
        "is", "is not",
    ])
    def test_all_operators_accepted(self, op):
        value = [1, 2] if op in {"in", "not in"} else 1
        e = PathIO._normalize_filter_to_expression([("x", op, value)])
        assert isinstance(e, ds.Expression)


# =====================================================================
# Filter normalization — fallback (pc mask) path
# =====================================================================

class TestFilterToMask:
    @pytest.fixture()
    def batch(self) -> pa.RecordBatch:
        return pa.RecordBatch.from_arrays(
            [
                pa.array([1, 2, 3, 4, 5]),
                pa.array(["a", "b", "c", "d", "e"]),
                pa.array([None, 2, None, 4, 5]),
            ],
            names=["x", "y", "z"],
        )

    def test_dict_equality(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch, filter_spec={"x": 3}, pc=pc
        )
        assert mask.to_pylist() == [False, False, True, False, False]

    def test_tuple_gt(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch, filter_spec=[("x", ">", 2)], pc=pc
        )
        assert mask.to_pylist() == [False, False, True, True, True]

    def test_tuple_in(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch,
            filter_spec=[("x", "in", [1, 3, 5])],
            pc=pc,
        )
        assert mask.to_pylist() == [True, False, True, False, True]

    def test_tuple_not_in(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch,
            filter_spec=[("x", "not in", [1, 3, 5])],
            pc=pc,
        )
        assert mask.to_pylist() == [False, True, False, True, False]

    def test_is_null(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch, filter_spec={"z": None}, pc=pc
        )
        assert mask.to_pylist() == [True, False, True, False, False]

    def test_is_not_null(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch,
            filter_spec=[("z", "is not", None)],
            pc=pc,
        )
        assert mask.to_pylist() == [False, True, False, True, True]

    def test_empty_in_matches_nothing(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch, filter_spec={"x": []}, pc=pc
        )
        assert all(v is False or v is None for v in mask.to_pylist())

    def test_dict_multi_key_is_anded(self, batch: pa.RecordBatch):
        mask = PathIO._build_filter_mask(
            table_or_batch=batch,
            filter_spec={"x": [1, 2, 3], "y": "b"},
            pc=pc,
        )
        assert mask.to_pylist() == [False, True, False, False, False]


# =====================================================================
# Parallel-path equivalence
# =====================================================================
# Both filter implementations should produce the same rows for the
# same filter spec. If they diverge on some edge case, tests break
# and we know immediately.

class TestFilterPathEquivalence:
    @pytest.fixture()
    def table(self) -> pa.Table:
        return pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "category": ["a", "b", "a", "c", "b"],
                "value": [10, None, 30, 40, None],
            }
        )

    @pytest.mark.parametrize("filter_spec", [
        {"id": 3},
        {"category": ["a", "c"]},
        [("id", ">", 2)],
        [("id", ">=", 2), ("id", "<=", 4)],
        [("value", "is not", None)],
        ("id", "in", [1, 3]),
    ])
    def test_both_paths_produce_same_rows(self, table: pa.Table, filter_spec):
        # Dataset path (via explicit expression).
        expr = PathIO._normalize_filter_to_expression(filter_spec)
        mask_from_expr_path = ds.dataset(table).to_table(filter=expr)

        # Fallback path (via pc mask).
        pc_mask = PathIO._build_filter_mask(
            table_or_batch=table, filter_spec=filter_spec, pc=pc
        )
        mask_from_fallback = table.filter(pc_mask)

        # Compare row ID sets — both paths return identical rows.
        dataset_ids = set(mask_from_expr_path.column("id").to_pylist())
        fallback_ids = set(mask_from_fallback.column("id").to_pylist())
        assert dataset_ids == fallback_ids, (
            f"divergence on filter={filter_spec!r}: "
            f"dataset={dataset_ids} vs fallback={fallback_ids}"
        )


# =====================================================================
# Filter column extraction
# =====================================================================

class TestFilterColumns:
    def test_none(self):
        assert PathIO._filter_columns(None) == []

    def test_dict(self):
        cols = PathIO._filter_columns({"a": 1, "b": 2})
        assert set(cols) == {"a", "b"}

    def test_tuple_shorthand(self):
        cols = PathIO._filter_columns(("a", ">", 5))
        assert cols == ["a"]

    def test_list_of_tuples(self):
        cols = PathIO._filter_columns([("a", "=", 1), ("b", ">", 2)])
        assert cols == ["a", "b"]

    def test_expression_returns_empty(self):
        """We can't introspect ds.Expression from Python easily — return []."""
        expr = ds.field("x") > 5
        # Expression is not a dict or standard Sequence-of-tuples, so we
        # return [] — dataset path handles it via pushdown anyway.
        assert PathIO._filter_columns(expr) == []


# =====================================================================
# Dataset-path reads (Parquet)
# =====================================================================

@pytest.mark.skipif(not HAS_PYARROW_PARQUET, reason="pyarrow.parquet required")
class TestDatasetPathReads:
    def test_read_single_parquet_file(self, parquet_file: Path, sample_table):
        pio = _LocalPathIO.make(parquet_file)
        out = pio.read_arrow_table()
        assert out.num_rows == sample_table.num_rows
        assert set(out.column_names) >= set(sample_table.column_names)

    def test_read_parquet_directory(self, parquet_dir: Path, sample_table):
        pio = _LocalPathIO.make(parquet_dir)
        out = pio.read_arrow_table()
        assert out.num_rows == sample_table.num_rows

    def test_read_with_filter_pushdown(self, parquet_dir: Path):
        pio = _LocalPathIO.make(parquet_dir)
        out = pio.read_arrow_table(filter={"status": "active"})
        assert out.num_rows == 3  # alice, carol, dave
        assert set(out.column("name").to_pylist()) == {"alice", "carol", "dave"}

    def test_read_with_projection(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        out = pio.read_arrow_table(columns=["id", "name"])
        assert out.column_names == ["id", "name"]

    def test_read_with_projection_and_filter(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        # Filter on status, project only id + name — filter col doesn't
        # need to survive in the output.
        out = pio.read_arrow_table(
            columns=["id", "name"],
            filter=[("status", "=", "inactive")],
        )
        assert out.column_names == ["id", "name"]
        assert out.num_rows == 2  # bob, eve

    def test_batch_size_respected(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        batches = list(pio.read_arrow_batches(batch_size=2))
        total = sum(b.num_rows for b in batches)
        assert total == 5

    def test_read_dataset_returns_dataset(self, parquet_dir: Path):
        pio = _LocalPathIO.make(parquet_dir)
        dataset = pio.read_dataset()
        assert isinstance(dataset, ds.Dataset)
        assert dataset.count_rows() == 5


# =====================================================================
# Fallback-path reads (XLSX)
# =====================================================================

@pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl required")
class TestFallbackPathReads:
    def test_read_xlsx_file(self, xlsx_file: Path, sample_table):
        pio = _LocalPathIO.make(xlsx_file)
        out = pio.read_arrow_table(engine="fallback")
        assert out.num_rows == sample_table.num_rows

    def test_read_dataset_raises_for_xlsx(self, xlsx_file: Path):
        pio = _LocalPathIO.make(xlsx_file)
        with pytest.raises(NotImplementedError, match="read_dataset"):
            pio.read_dataset()

    def test_fallback_filter_applied(self, xlsx_file: Path):
        """Filter works on the fallback path via pc masks."""
        pio = _LocalPathIO.make(xlsx_file)
        out = pio.read_arrow_table(
            engine="fallback",
            filter={"status": "active"},
        )
        assert out.num_rows == 3

    def test_fallback_projection_augments_filter_columns(
        self, xlsx_file: Path,
    ):
        """Projecting a subset that excludes the filter column still works."""
        pio = _LocalPathIO.make(xlsx_file)
        out = pio.read_arrow_table(
            engine="fallback",
            columns=["id", "name"],            # status not in projection
            filter=[("status", "=", "active")],
        )
        # Final projection drops status, but filter was applied.
        assert out.column_names == ["id", "name"]
        assert out.num_rows == 3


# =====================================================================
# Cast symmetry
# =====================================================================

@pytest.mark.skipif(not HAS_PYARROW_PARQUET, reason="pyarrow.parquet required")
class TestCastSymmetry:
    def test_cast_applied_via_batches(self, parquet_file: Path):
        """read_arrow_batches applies cast (it didn't in the old code)."""
        try:
            from yggdrasil.data.cast.options import CastOptions
        except ImportError:
            pytest.skip("CastOptions not importable")

        # Build a cast that forces id: int64 → float64.
        target = pa.schema([
            pa.field("id", pa.float64()),
            pa.field("name", pa.string()),
            pa.field("status", pa.string()),
            pa.field("score", pa.float64()),
        ])
        try:
            cast = CastOptions(target_field=target)
        except TypeError:
            pytest.skip("CastOptions(target_field=...) signature mismatch")

        pio = _LocalPathIO.make(parquet_file)
        batches = list(pio.read_arrow_batches(cast=cast))
        assert batches, "expected at least one batch"
        for b in batches:
            assert b.schema.field("id").type == pa.float64(), (
                f"cast not applied on batch path: got {b.schema.field('id').type}"
            )

    def test_table_and_batches_produce_same_types(self, parquet_file: Path):
        """read_arrow_table and read_arrow_batches produce same column types."""
        pio = _LocalPathIO.make(parquet_file)

        table = pio.read_arrow_table()
        batches = list(pio.read_arrow_batches())
        from_batches = pa.Table.from_batches(batches)

        # Schema-level equality: same names, same types, same order.
        assert table.schema == from_batches.schema


# =====================================================================
# Partition values
# =====================================================================

class TestPartitionValues:
    def test_hive_extraction(self):
        values = PathIO._partition_values(
            file_path=Path("/root/year=2024/month=01/data.parquet"),
            partitioning="hive",
            partition_base_dir=Path("/root"),
        )
        assert values == {"year": "2024", "month": "01"}

    def test_hive_with_non_hive_segments_ignored(self):
        """Non-key=value segments are silently skipped."""
        values = PathIO._partition_values(
            file_path=Path("/root/some_subdir/year=2024/data.parquet"),
            partitioning="hive",
            partition_base_dir=Path("/root"),
        )
        assert values == {"year": "2024"}

    def test_directory_name_partitioning(self):
        values = PathIO._partition_values(
            file_path=Path("/root/2024/01/data.parquet"),
            partitioning=["year", "month"],
            partition_base_dir=Path("/root"),
        )
        assert values == {"year": "2024", "month": "01"}

    def test_no_partitioning_returns_empty(self):
        values = PathIO._partition_values(
            file_path=Path("/root/year=2024/data.parquet"),
            partitioning=None,
            partition_base_dir=Path("/root"),
        )
        assert values == {}

    def test_flat_file_has_no_partitions(self):
        """A file directly in the base dir has no partition values."""
        values = PathIO._partition_values(
            file_path=Path("/root/data.parquet"),
            partitioning="hive",
            partition_base_dir=Path("/root"),
        )
        assert values == {}


# =====================================================================
# Schema collection
# =====================================================================

@pytest.mark.skipif(not HAS_PYARROW_PARQUET, reason="pyarrow.parquet required")
class TestSchema:
    def test_single_file_schema(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        schema = pio._collect_arrow_schema()
        assert set(schema.names) == {"id", "name", "status", "score"}

    def test_directory_schema_first_file_only(
        self, tmp_path: Path, sample_table,
    ):
        """full=False inspects only the first file."""
        d = tmp_path / "mixed"
        d.mkdir()
        # File 1: just id + name
        pq.write_table(sample_table.select(["id", "name"]), d / "a.parquet")
        # File 2: all columns
        pq.write_table(sample_table, d / "b.parquet")

        pio = _LocalPathIO.make(d)
        schema = pio._collect_arrow_schema(full=False)
        assert set(schema.names) == {"id", "name"}

    def test_directory_schema_full_unifies(
        self, tmp_path: Path, sample_table,
    ):
        """full=True unifies schemas across files."""
        d = tmp_path / "mixed"
        d.mkdir()
        pq.write_table(sample_table.select(["id", "name"]), d / "a.parquet")
        pq.write_table(sample_table.select(["id", "score"]), d / "b.parquet")

        pio = _LocalPathIO.make(d)
        schema = pio._collect_arrow_schema(full=True)
        assert set(schema.names) >= {"id", "name", "score"}


# =====================================================================
# count_rows
# =====================================================================

@pytest.mark.skipif(not HAS_PYARROW_PARQUET, reason="pyarrow.parquet required")
class TestCountRows:
    def test_dataset_fast_path(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        # No filter → should hit dataset.count_rows().
        assert pio.count_rows() == 5

    def test_with_filter(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        # Filter → falls back to materializing.
        assert pio.count_rows(filter={"status": "active"}) == 3

    @pytest.mark.skipif(not HAS_OPENPYXL, reason="openpyxl required")
    def test_fallback_format(self, xlsx_file: Path):
        pio = _LocalPathIO.make(xlsx_file)
        assert pio.count_rows(engine="fallback") == 5


# =====================================================================
# Path probes
# =====================================================================

class TestPathProbes:
    def test_exists(self, tmp_path: Path):
        pio = _LocalPathIO.make(tmp_path)
        assert pio.exists is True

    def test_is_dir(self, tmp_path: Path):
        pio = _LocalPathIO.make(tmp_path)
        assert pio.is_dir is True
        assert pio.is_file is False

    def test_is_file(self, tmp_path: Path):
        (tmp_path / "empty.json").write_text("[]")
        pio = _LocalPathIO.make(tmp_path / "empty.json")
        assert pio.is_file is True
        assert pio.is_dir is False


# =====================================================================
# Abstract / unsupported behavior
# =====================================================================

class TestAbstractBehavior:
    def test_pathio_is_abstract(self):
        """Instantiating PathIO directly should fail — abstract methods present."""
        # PathIO has abstract methods (make, iter_files), so direct
        # construction should fail. The exact TypeError message varies
        # (missing holder arg vs abstract methods) depending on which
        # check fires first, so we just assert some TypeError.
        with pytest.raises(TypeError):
            PathIO(media_type=MediaType.parse(".parquet"), holder=BytesIO(), path=Path("."))  # type: ignore[abstract]

    def test_write_raises_not_implemented(self, tmp_path: Path):
        pio = _LocalPathIO.make(tmp_path / "out.parquet")
        with pytest.raises(NotImplementedError, match="does not support writes"):
            pio.write_arrow_table(pa.table({"a": [1]}))


# =====================================================================
# iter_files behavior (tests the test subclass, but verifies contract)
# =====================================================================

@pytest.mark.skipif(not HAS_PYARROW_PARQUET, reason="pyarrow.parquet required")
class TestIterFiles:
    def test_iter_files_single_file_yields_self(self, parquet_file: Path):
        pio = _LocalPathIO.make(parquet_file)
        files = list(pio.iter_files())
        assert len(files) == 1
        assert files[0].path == parquet_file

    def test_iter_files_directory_yields_all_parquet(self, parquet_dir: Path):
        pio = _LocalPathIO.make(parquet_dir)
        files = list(pio.iter_files())
        assert len(files) == 2
        assert all(f.path.suffix == ".parquet" for f in files)

    def test_iter_files_non_recursive(self, tmp_path: Path):
        """recursive=False skips subdirectories."""
        (tmp_path / "top.parquet").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.parquet").touch()

        pio = _LocalPathIO.make(tmp_path)
        files = list(pio.iter_files(recursive=False))
        names = {f.path.name for f in files}
        assert "top.parquet" in names
        assert "nested.parquet" not in names

    def test_iter_files_hidden_skipped_by_default(self, tmp_path: Path):
        (tmp_path / "visible.parquet").touch()
        (tmp_path / ".hidden.parquet").touch()
        (tmp_path / "_underscore.parquet").touch()

        pio = _LocalPathIO.make(tmp_path)
        files = list(pio.iter_files(include_hidden=False))
        names = {f.path.name for f in files}
        assert names == {"visible.parquet"}