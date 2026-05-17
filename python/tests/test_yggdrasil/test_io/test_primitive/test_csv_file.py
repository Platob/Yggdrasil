"""Behavior tests for :class:`yggdrasil.io.primitive.csv_file.CSVFile`."""
from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.csv as pa_csv
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.data.options import CastOptions
from yggdrasil.data.schema import Schema
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.csv_file import CSVFile, CsvOptions
from yggdrasil.io.holder import Holder
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})


@pytest.fixture
def deep_table() -> pa.Table:
    """Deep-nested fixture covering every shape ``_is_nested_arrow_type``
    handles — ``list``, ``list<list>``, ``struct``, ``list<struct>``,
    ``struct<list<struct>>``, ``map``, ``dictionary``."""
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field("tags", pa.list_(pa.int64())),
        pa.field("matrix", pa.list_(pa.list_(pa.int32()))),
        pa.field("flat", pa.struct([
            ("x", pa.int64()),
            ("y", pa.float64()),
            ("label", pa.string()),
        ])),
        pa.field("events", pa.list_(pa.struct([
            ("id", pa.int64()),
            ("tag", pa.string()),
        ]))),
        pa.field("profile", pa.struct([
            ("name", pa.string()),
            ("scores", pa.list_(pa.struct([
                ("k", pa.string()),
                ("v", pa.float64()),
            ]))),
        ])),
        pa.field("attrs", pa.map_(pa.string(), pa.int64())),
        pa.field("kind", pa.dictionary(pa.int32(), pa.string())),
    ])
    return pa.table(
        {
            "id": [1, 2, 3],
            "name": ["a", "b", "c"],
            "tags": [[1, 2], [3], None],
            "matrix": [[[1, 2], [3]], [[7]], None],
            "flat": [
                {"x": 10, "y": 1.5, "label": "alpha"},
                {"x": 20, "y": 2.5, "label": "beta"},
                None,
            ],
            "events": [
                [{"id": 10, "tag": "x"}, {"id": 11, "tag": "y"}],
                [],
                [{"id": 99, "tag": "z"}],
            ],
            "profile": [
                {"name": "u1", "scores": [{"k": "a", "v": 0.5}]},
                {"name": "u2", "scores": []},
                None,
            ],
            "attrs": [
                [("a", 1), ("b", 2)],
                [],
                [("only", 99)],
            ],
            "kind": pa.array(["alpha", "beta", "alpha"]).dictionary_encode(),
        },
        schema=schema,
    )


class TestRegistration:

    def test_mime_type_is_csv(self) -> None:
        assert CSVFile.mime_type is MimeTypes.CSV

    def test_registry(self) -> None:
        assert Holder.class_for_media_type(MimeTypes.CSV) is CSVFile


class TestRoundTrip:

    def test_round_trip_arrow(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.equals(table)

    def test_csv_text_shape(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        # The pyarrow CSV writer quotes string columns by default.
        text = io.to_bytes().decode("utf-8")
        lines = text.strip().splitlines()
        assert lines[0] == '"id","name"'
        assert lines[1] == '1,"a"'

    def test_collect_schema(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"id", "name"}

    def test_pandas_round_trip(self, table) -> None:
        pd = pytest.importorskip("pandas")
        io = CSVFile()
        io.write_arrow_table(table)
        pd.testing.assert_frame_equal(io.read_pandas_frame(), table.to_pandas())

    def test_polars_round_trip(self, table) -> None:
        pl = pytest.importorskip("polars")
        io = CSVFile()
        io.write_arrow_table(table)
        assert io.read_polars_frame().equals(pl.from_arrow(table))


class TestEmpty:

    def test_read_empty(self) -> None:
        assert list(CSVFile().read_arrow_batches()) == []

    def test_collect_schema_empty(self) -> None:
        from yggdrasil.data.schema import Schema
        assert CSVFile().collect_schema() == Schema.empty()


class TestTargetSchemaCast:
    """``target_field`` reshapes every batch on read and write so the
    encoder/decoder sees the caller's schema. Without a target the
    code path is a passthrough — covered by :class:`TestRoundTrip`."""

    def _target_field(self):
        from yggdrasil.data.data_field import Field
        return Field.from_(pa.schema([
            pa.field("id", pa.int64()),
            pa.field("v", pa.float64()),
        ]))

    def test_read_casts_to_target_schema(self) -> None:
        # CSV always carries text bytes; the reader infers types from
        # column values, so an explicit target lets the caller pin a
        # narrower or wider type.
        io = CSVFile()
        io.write_arrow_table(pa.table({
            "id": ["1", "2", "3"], "v": ["1.5", "2.5", "3.5"],
        }))
        casted = io.read_arrow_table(target=self._target_field())
        assert casted.schema.field("id").type == pa.int64()
        assert casted.schema.field("v").type == pa.float64()

    def test_write_casts_to_target_schema(self) -> None:
        # The casted values are what get persisted as text.
        io = CSVFile()
        io.write_arrow_table(
            pa.table({"id": ["1", "2"], "v": ["1.5", "2.5"]}),
            target=self._target_field(),
        )
        raw = io.read_arrow_table()
        assert raw.schema.field("id").type == pa.int64()
        assert raw.schema.field("v").type == pa.float64()


class TestModes:

    def test_overwrite_replaces(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        smaller = pa.table({"id": [9], "name": ["z"]})
        io.write_arrow_table(smaller, options=CsvOptions(mode=Mode.OVERWRITE))
        assert io.read_arrow_table().equals(smaller)

    def test_append_concatenates_without_extra_header(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        more = pa.table({"id": [4], "name": ["d"]})
        io.write_arrow_batches(more.to_batches(), options=CsvOptions(mode=Mode.APPEND))

        text = io.to_bytes().decode("utf-8")
        # Header should appear exactly once.
        assert text.count('"id","name"') == 1
        loaded = io.read_arrow_table()
        assert loaded.num_rows == table.num_rows + more.num_rows

    def test_append_on_empty_writes_header(self, table) -> None:
        io = CSVFile()
        io.write_arrow_batches(table.to_batches(), options=CsvOptions(mode=Mode.APPEND))
        text = io.to_bytes().decode("utf-8")
        assert text.startswith('"id","name"')

    def test_ignore_skips_when_non_empty(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        before = io.size
        io.write_arrow_batches(
            pa.table({"id": [9], "name": ["z"]}).to_batches(),
            options=CsvOptions(mode=Mode.IGNORE),
        )
        assert io.size == before

    def test_error_if_exists_raises(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(), options=CsvOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestKeyedMerge:
    """``options.match_by`` drives key-aware APPEND / UPSERT."""

    def test_append_with_keys_drops_incoming_duplicates(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=CsvOptions(mode=Mode.APPEND, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3, 4]
        assert loaded.column("name").to_pylist() == ["a", "b", "c", "d"]

    def test_upsert_with_keys_replaces_existing(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table)
        more = pa.table({"id": [2, 4], "name": ["X", "d"]})
        io.write_arrow_batches(
            more.to_batches(),
            options=CsvOptions(mode=Mode.UPSERT, match_by=["id"]),
        )
        loaded = io.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 3, 2, 4]
        assert loaded.column("name").to_pylist() == ["a", "c", "X", "d"]


class TestDelimiter:

    def test_tsv_round_trip(self, table) -> None:
        io = CSVFile()
        io.write_arrow_table(table, options=CsvOptions(delimiter="\t"))
        assert "\t" in io.to_bytes().decode("utf-8")
        loaded = io.read_arrow_table(options=CsvOptions(delimiter="\t"))
        assert loaded.equals(table)


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "data.csv"))
        io = CSVFile(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        # Vanilla read.
        text = target.read_text()
        assert "id" in text and "name" in text

        reader = CSVFile(holder=target, owns_holder=False)
        assert reader.read_arrow_table().equals(table)


class TestExternalWriterPattern:

    def test_pandas_to_csv_then_read_arrow(self, tmp_path, table) -> None:
        pd = pytest.importorskip("pandas")
        target = LocalPath(str(tmp_path / "data.csv"))
        with target.open("wb") as bio:
            table.to_pandas().to_csv(bio, index=False)

        # Read back through CSVFile. pandas writes unquoted strings by
        # default, so the reader does the inference.
        reader = CSVFile(holder=target, owns_holder=False)
        loaded = reader.read_arrow_table()
        assert loaded.column("id").to_pylist() == [1, 2, 3]
        assert loaded.column("name").to_pylist() == ["a", "b", "c"]

    def test_polars_native_path_round_trip(self, tmp_path, table) -> None:
        pl = pytest.importorskip("polars")
        target = LocalPath(str(tmp_path / "data.csv"))
        df = pl.from_arrow(table)
        with target.open("wb") as bio:
            df.write_csv(bio)

        reader = CSVFile(holder=target, owns_holder=False)
        out = reader.read_polars_frame()
        assert out.equals(df)


class TestNestedAsJson:
    """Nested columns serialize to JSON cells on write.

    pyarrow's CSV writer rejects ``list`` / ``struct`` / ``map`` /
    ``dictionary`` types with ``ArrowInvalid: Unsupported Type``;
    :class:`CSVFile` flattens those columns to JSON strings before they
    reach the encoder. The cells must round-trip through stdlib
    :mod:`json` parsing without loss for downstream consumers.
    """

    def test_pyarrow_csv_writer_rejects_nested(self) -> None:
        """Document the invariant that motivates the encode step —
        if pyarrow ever starts accepting nested types natively, this
        test is the canary that says we can drop the helper."""
        t = pa.table({"tags": [[1, 2], [3]]})
        with pytest.raises(pa.ArrowInvalid, match="Unsupported Type"):
            pa_csv.write_csv(t, pa.BufferOutputStream())

    def test_list_column_written_as_json(self) -> None:
        t = pa.table({"id": [1, 2, 3], "tags": [[1, 2], [3], None]})
        io = CSVFile()
        io.write_table(t)

        rows = [
            row.split(",", 1)
            for row in io.to_bytes().decode("utf-8").strip().splitlines()[1:]
        ]
        # null list stays null (empty CSV cell), populated lists become
        # JSON arrays quoted by the CSV writer.
        assert rows[0][1] == '"[1,2]"'
        assert rows[1][1] == '"[3]"'
        assert rows[2][1] == ""

    def test_struct_column_written_as_json(self) -> None:
        t = pa.table({
            "id": [1, 2],
            "flat": pa.array(
                [{"x": 10, "y": 1.5}, {"x": 20, "y": 2.5}],
                type=pa.struct([("x", pa.int64()), ("y", pa.float64())]),
            ),
        })
        io = CSVFile()
        io.write_table(t)
        text = io.to_bytes().decode("utf-8")

        line = text.splitlines()[1]
        # Strip the leading ``1,`` then unquote the JSON cell.
        json_cell = line.split(",", 1)[1].strip('"').replace('""', '"')
        assert json.loads(json_cell) == {"x": 10, "y": 1.5}

    def test_map_column_written_as_json(self) -> None:
        t = pa.table(
            {"id": [1, 2], "attrs": [[("a", 1), ("b", 2)], [("only", 99)]]},
            schema=pa.schema([
                pa.field("id", pa.int64()),
                pa.field("attrs", pa.map_(pa.string(), pa.int64())),
            ]),
        )
        io = CSVFile()
        io.write_table(t)

        # Map cells survive as JSON. Arrow's ``Array.to_pylist`` hands
        # map entries back as ``[(k, v), ...]`` (positional pairs, not
        # dicts), so the JSON shape is a list-of-pairs.
        import csv as stdlib_csv
        rows = list(stdlib_csv.DictReader(io.to_bytes().decode("utf-8").splitlines()))
        assert json.loads(rows[0]["attrs"]) == [["a", 1], ["b", 2]]
        assert json.loads(rows[1]["attrs"]) == [["only", 99]]

    def test_dictionary_column_written_as_json(self) -> None:
        t = pa.table({
            "id": [1, 2, 3],
            "kind": pa.array(["alpha", "beta", "alpha"]).dictionary_encode(),
        })
        io = CSVFile()
        io.write_table(t)

        # Dictionary-encoded columns surface to ``to_pylist`` as their
        # decoded scalar values — strings here — so the JSON cell is a
        # JSON-encoded string literal. Pull cells through stdlib csv
        # so the quote-escaping bookkeeping is somebody else's problem.
        import csv as stdlib_csv
        rows = list(stdlib_csv.DictReader(io.to_bytes().decode("utf-8").splitlines()))
        assert json.loads(rows[0]["kind"]) == "alpha"
        assert json.loads(rows[1]["kind"]) == "beta"
        assert json.loads(rows[2]["kind"]) == "alpha"

    def test_deep_nested_round_trip_via_json_parse(self, deep_table) -> None:
        """Every nested column survives the write as a JSON string
        whose ``json.loads`` reproduces the original Python value."""
        io = CSVFile()
        io.write_table(deep_table)
        text = io.to_bytes().decode("utf-8")

        # Re-parse rows via stdlib csv to recover JSON-encoded cells.
        import csv as stdlib_csv

        reader = stdlib_csv.DictReader(text.splitlines())
        parsed = list(reader)
        assert len(parsed) == deep_table.num_rows

        # Compare each nested column row-by-row through json.loads.
        nested_cols = ["tags", "matrix", "flat", "events", "profile", "kind"]
        expected_rows = deep_table.to_pylist()
        for col in nested_cols:
            for i, row in enumerate(parsed):
                expected = expected_rows[i][col]
                actual_cell = row[col]
                if expected is None:
                    assert actual_cell == "", (col, i, actual_cell)
                else:
                    assert json.loads(actual_cell) == expected, (col, i)

    def test_opt_out_via_nested_as_json_false_raises(self) -> None:
        """``nested_as_json=False`` opts out of the encoding — callers
        who have already flattened upstream get the writer's native
        ``Unsupported Type`` error so the contract is unambiguous."""
        t = pa.table({"id": [1], "tags": [[1, 2]]})
        io = CSVFile()
        with pytest.raises(pa.ArrowInvalid, match="Unsupported Type"):
            io.write_table(t, options=CsvOptions(nested_as_json=False))

    def test_scalar_only_table_unaffected(self, table) -> None:
        """Scalar-only writes don't pay any extra cost (no nested
        indices → encoder is a passthrough). Smoke-tests that the
        helper doesn't accidentally mangle plain columns."""
        io = CSVFile()
        io.write_table(table)
        # ``read_arrow_table`` reads what was written; if scalar columns
        # were touched the round-trip would diverge.
        assert io.read_arrow_table().equals(table)

    def test_target_schema_cast_then_nested_encode(self) -> None:
        """A bound ``target_field`` reshapes the batch first, then
        nested encoding flattens whatever survives — exercises the
        ``cast → encode`` order so a cast-introduced nested column is
        also caught."""
        # Source has the column as a string already; target widens it
        # to a JSON-decoded list (i.e. structured target). The CSV
        # writer should still get a flat string column after encoding.
        target = pa.schema([
            pa.field("id", pa.int64()),
            pa.field("tags", pa.list_(pa.int64())),
        ])
        target_field = Schema.from_arrow(target).to_field()
        options = CsvOptions(target=target_field)

        source = pa.table({"id": [1, 2], "tags": ['[1, 2]', '[3]']})
        io = CSVFile()
        io.write_table(source, options)
        text = io.to_bytes().decode("utf-8")
        line = text.splitlines()[1]
        assert line.split(",", 1)[1] == '"[1,2]"'

    def test_multiple_batches_share_encoded_schema(self) -> None:
        """Streaming multiple batches must reuse the same encoded
        writer schema across batches — the encoder caches indices
        from the first batch and applies them to every subsequent
        one. A schema mismatch would surface as
        ``Table schema does not match`` from the writer.
        """
        b1 = pa.record_batch({"id": [1], "tags": [[1, 2]]})
        b2 = pa.record_batch({"id": [2], "tags": [[3, 4]]})
        b3 = pa.record_batch({"id": [3], "tags": [None]})
        io = CSVFile()
        io.write_table([b1, b2, b3])

        rows = io.to_bytes().decode("utf-8").strip().splitlines()
        assert len(rows) == 4  # header + 3 rows
        assert rows[1].endswith('"[1,2]"')
        assert rows[2].endswith('"[3,4]"')
        assert rows[3].endswith(",")  # null cell

    def test_pandas_with_nested_object_columns(self) -> None:
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({
            "id": [1, 2],
            "tags": [[1, 2], [3]],
        })
        # No target — pandas → arrow infers list<int64>.
        io = CSVFile()
        io.write_table(df)
        text = io.to_bytes().decode("utf-8")
        assert '"[1,2]"' in text
        assert '"[3]"' in text

    def test_polars_with_nested_columns(self) -> None:
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({
            "id": [1, 2],
            "tags": [[1, 2], [3]],
        })
        io = CSVFile()
        io.write_table(df)
        text = io.to_bytes().decode("utf-8")
        assert '"[1,2]"' in text
        assert '"[3]"' in text
