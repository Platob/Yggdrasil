"""Behavior tests for :class:`yggdrasil.io.primitive.form_urlencoded_io`."""
from __future__ import annotations

import pyarrow as pa
import pytest

from yggdrasil.data.enums import MimeTypes, Mode
from yggdrasil.io.path.local_path import LocalPath
from yggdrasil.io.primitive.form_urlencoded_io import (
    FormUrlencodedIO,
    FormUrlencodedOptions,
)
from yggdrasil.io.tabular import Tabular


@pytest.fixture
def table() -> pa.Table:
    return pa.table({"name": ["alice", "bob"], "city": ["paris", "berlin"]})


class TestRegistration:

    def test_mime_type_is_form_urlencoded(self) -> None:
        assert FormUrlencodedIO.mime_type is MimeTypes.FORM_URLENCODED

    def test_mime_string(self) -> None:
        assert MimeTypes.FORM_URLENCODED.value == "application/x-www-form-urlencoded"

    def test_mime_is_tabular(self) -> None:
        assert MimeTypes.FORM_URLENCODED.is_tabular is True

    def test_registry(self) -> None:
        assert Tabular.class_for_media_type(MimeTypes.FORM_URLENCODED) is FormUrlencodedIO

    def test_resolves_from_mime_string(self) -> None:
        from yggdrasil.data.enums.mime_type import MimeType

        assert MimeType.from_str("application/x-www-form-urlencoded") is MimeTypes.FORM_URLENCODED


class TestRoundTrip:

    def test_arrow_round_trip(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == ["alice", "bob"]
        assert loaded.column("city").to_pylist() == ["paris", "berlin"]

    def test_writes_one_line_per_row(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        text = io.to_bytes().decode("utf-8")
        lines = [ln for ln in text.split("\n") if ln]
        assert lines == ["name=alice&city=paris", "name=bob&city=berlin"]

    def test_collect_schema(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        assert set(io.collect_schema().field_names()) == {"name", "city"}

    def test_url_escapes_special_characters(self) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(pa.table({"q": ["hello world & friends"]}))
        text = io.to_bytes().decode("utf-8")
        assert text.startswith("q=hello+world+%26+friends")
        loaded = io.read_arrow_table()
        assert loaded.column("q").to_pylist() == ["hello world & friends"]

    def test_unicode_round_trip(self) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(pa.table({"city": ["café", "東京"]}))
        loaded = io.read_arrow_table()
        assert loaded.column("city").to_pylist() == ["café", "東京"]


class TestInputShapes:

    def test_reads_single_submission(self) -> None:
        io = FormUrlencodedIO(b"name=alice&city=paris")
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == ["alice"]
        assert loaded.column("city").to_pylist() == ["paris"]

    def test_reads_multiline_submissions(self) -> None:
        io = FormUrlencodedIO(b"name=alice&city=paris\nname=bob&city=berlin\n")
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == ["alice", "bob"]

    def test_keep_blank_values(self) -> None:
        io = FormUrlencodedIO(b"name=&city=paris")
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == [""]
        assert loaded.column("city").to_pylist() == ["paris"]

    def test_drops_blank_values_when_disabled(self) -> None:
        io = FormUrlencodedIO(b"name=&city=paris")
        loaded = io.read_arrow_table(
            options=FormUrlencodedOptions(keep_blank_values=False)
        )
        assert "name" not in loaded.schema.names
        assert loaded.column("city").to_pylist() == ["paris"]

    def test_plus_decodes_as_space(self) -> None:
        io = FormUrlencodedIO(b"q=hello+world")
        loaded = io.read_arrow_table()
        assert loaded.column("q").to_pylist() == ["hello world"]

    def test_percent_decoding(self) -> None:
        io = FormUrlencodedIO(b"q=hello%20world%26friends")
        loaded = io.read_arrow_table()
        assert loaded.column("q").to_pylist() == ["hello world&friends"]

    def test_empty_buffer_yields_no_rows(self) -> None:
        io = FormUrlencodedIO()
        loaded = io.read_arrow_table()
        assert loaded.num_rows == 0


class TestMultiValuePolicy:

    def test_last_wins_by_default(self) -> None:
        io = FormUrlencodedIO(b"tag=a&tag=b&tag=c")
        loaded = io.read_arrow_table()
        assert loaded.column("tag").to_pylist() == ["c"]

    def test_first_wins(self) -> None:
        io = FormUrlencodedIO(b"tag=a&tag=b&tag=c")
        loaded = io.read_arrow_table(
            options=FormUrlencodedOptions(multi_values="first")
        )
        assert loaded.column("tag").to_pylist() == ["a"]

    def test_list_policy(self) -> None:
        io = FormUrlencodedIO(b"tag=a&tag=b&tag=c")
        loaded = io.read_arrow_table(
            options=FormUrlencodedOptions(multi_values="list")
        )
        assert loaded.column("tag").to_pylist() == [["a", "b", "c"]]

    def test_list_column_writes_repeated_pairs(self) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(
            pa.table(
                {"tag": pa.array([["a", "b", "c"]], type=pa.list_(pa.string()))}
            )
        )
        text = io.to_bytes().decode("utf-8")
        assert text.strip() == "tag=a&tag=b&tag=c"


class TestModes:

    def test_overwrite(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        smaller = pa.table({"name": ["zoe"], "city": ["rome"]})
        io.write_arrow_table(
            smaller, options=FormUrlencodedOptions(mode=Mode.OVERWRITE)
        )
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == ["zoe"]

    def test_append_concatenates_rows(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        more = pa.table({"name": ["carol"], "city": ["madrid"]})
        io.write_arrow_batches(
            more.to_batches(), options=FormUrlencodedOptions(mode=Mode.APPEND)
        )
        loaded = io.read_arrow_table()
        assert loaded.column("name").to_pylist() == ["alice", "bob", "carol"]

    def test_error_if_exists(self, table) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(table)
        with pytest.raises(FileExistsError):
            io.write_arrow_batches(
                table.to_batches(),
                options=FormUrlencodedOptions(mode=Mode.ERROR_IF_EXISTS),
            )


class TestNullHandling:

    def test_null_skipped_by_default(self) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(pa.table({"a": ["x"], "b": [None]}))
        text = io.to_bytes().decode("utf-8")
        assert text.strip() == "a=x"

    def test_null_as_blank_when_enabled(self) -> None:
        io = FormUrlencodedIO()
        io.write_arrow_table(
            pa.table({"a": ["x"], "b": [None]}),
            options=FormUrlencodedOptions(emit_null_as_blank=True),
        )
        text = io.to_bytes().decode("utf-8")
        assert text.strip() == "a=x&b="


class TestHolderBacked:

    def test_local_path_round_trip(self, tmp_path, table) -> None:
        target = LocalPath(str(tmp_path / "form.urlencoded"))
        io = FormUrlencodedIO(holder=target, owns_holder=False)
        io.write_arrow_table(table)
        text = target.read_text()
        assert "name=alice" in text
        assert "name=bob" in text

    def test_dispatches_by_extension(self, tmp_path, table) -> None:
        # ``IO(path=...)`` resolves to the registered Tabular leaf
        # via the mime registry — confirms the extension wiring.
        from yggdrasil.io.base import IO

        path = str(tmp_path / "form.urlencoded")
        io = IO(path=path)
        assert isinstance(io, FormUrlencodedIO)
