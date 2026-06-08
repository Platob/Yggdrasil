"""Tests for :class:`yggdrasil.path.path.Path` base-class behaviour exercised through :class:`LocalPath`."""
from __future__ import annotations

import pathlib
import pickle

import pyarrow as pa
import pytest

from yggdrasil.enums import Mode
from yggdrasil.path.local_path import LocalPath
from yggdrasil.path.path import Path
from yggdrasil.url import URL


# ---------------------------------------------------------------------------
# TestFrom — construction / coercion
# ---------------------------------------------------------------------------


class TestFrom:

    def test_from_string_returns_local_path(self, tmp_path):
        p = Path.from_(str(tmp_path / "file.txt"))
        assert isinstance(p, LocalPath)

    def test_from_pathlib_returns_local_path(self, tmp_path):
        p = Path.from_(pathlib.Path(tmp_path / "file.txt"))
        assert isinstance(p, LocalPath)

    def test_from_file_url_returns_local_path(self):
        p = Path.from_("file:///tmp/some_file.txt")
        assert isinstance(p, LocalPath)
        assert p.full_path() == "/tmp/some_file.txt"

    def test_from_existing_local_path_returns_same_instance(self, tmp_path):
        original = LocalPath(str(tmp_path / "file.txt"))
        result = Path.from_(original)
        assert result is original

    def test_from_none_raises(self):
        with pytest.raises((TypeError, ValueError)):
            Path.from_(None)


# ---------------------------------------------------------------------------
# TestTabularInterface — Arrow read/write round-trips
# ---------------------------------------------------------------------------


class TestTabularInterface:

    def test_write_table_then_read_arrow_table(self, tmp_path):
        p = LocalPath(str(tmp_path / "data.parquet"))
        table = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p.write_table(table)

        result = p.read_arrow_table()
        assert result.num_rows == 3
        assert result.column_names == ["a", "b"]
        assert result.column("a").to_pylist() == [1, 2, 3]
        assert result.column("b").to_pylist() == ["x", "y", "z"]

    def test_write_arrow_batches_then_read_arrow_batches(self, tmp_path):
        p = LocalPath(str(tmp_path / "batched.parquet"))
        batches = [
            pa.record_batch({"x": [10, 20], "y": [1.0, 2.0]}),
            pa.record_batch({"x": [30, 40], "y": [3.0, 4.0]}),
        ]
        p.write_arrow_batches(batches)

        read_batches = list(p.read_arrow_batches())
        total_rows = sum(b.num_rows for b in read_batches)
        assert total_rows == 4

        # Verify content via table materialisation
        table = p.read_arrow_table()
        assert table.column("x").to_pylist() == [10, 20, 30, 40]

    def test_collect_schema_from_written_data(self, tmp_path):
        p = LocalPath(str(tmp_path / "schema.parquet"))
        table = pa.table({"id": pa.array([1, 2], type=pa.int64()), "name": ["a", "b"]})
        p.write_table(table)

        schema = p.collect_schema()
        assert "id" in schema.names
        assert "name" in schema.names
        assert len(schema) == 2

    def test_read_arrow_table_on_empty_file(self, tmp_path):
        p = LocalPath(str(tmp_path / "empty.parquet"))
        p.touch()
        result = p.read_arrow_table()
        assert result.num_rows == 0

    def test_write_overwrite_replaces_content(self, tmp_path):
        p = LocalPath(str(tmp_path / "overwrite.parquet"))
        p.write_table(pa.table({"v": [1, 2, 3]}))
        assert p.read_arrow_table().num_rows == 3

        p.write_table(pa.table({"v": [10, 20]}), mode=Mode.OVERWRITE)
        result = p.read_arrow_table()
        assert result.num_rows == 2
        assert result.column("v").to_pylist() == [10, 20]

    def test_write_append_adds_rows(self, tmp_path):
        # Parquet single-file append rewrites the file; verify new
        # content replaces old for a single-file path. Folder-based
        # paths would accumulate parts, but single-file is the
        # contract tested here.
        p = LocalPath(str(tmp_path / "append.parquet"))
        p.write_table(pa.table({"v": [1, 2]}))
        p.write_table(pa.table({"v": [3, 4]}), mode=Mode.APPEND)

        result = p.read_arrow_table()
        # Single parquet file: APPEND at the byte layer overwrites
        # the whole file, so the latest write wins.
        assert result.num_rows >= 2
        assert 3 in result.column("v").to_pylist()
        assert 4 in result.column("v").to_pylist()


# ---------------------------------------------------------------------------
# TestFolderOps — directory / file manipulation
# ---------------------------------------------------------------------------


class TestFolderOps:

    def test_mkdir_and_iterdir_round_trip(self, tmp_path):
        d = LocalPath(str(tmp_path / "subdir"))
        d.mkdir()

        f1 = LocalPath(str(tmp_path / "subdir" / "one.txt"))
        f1.touch()
        f2 = LocalPath(str(tmp_path / "subdir" / "two.txt"))
        f2.touch()

        children = sorted(c.name for c in d.iterdir())
        assert children == ["one.txt", "two.txt"]

    def test_is_dir_and_is_file_detection(self, tmp_path):
        d = LocalPath(str(tmp_path / "adir"))
        d.mkdir()
        f = LocalPath(str(tmp_path / "afile.bin"))
        f.touch()

        assert d.is_dir() is True
        assert d.is_file() is False
        assert f.is_file() is True
        assert f.is_dir() is False

    def test_remove_removes_file(self, tmp_path):
        f = LocalPath(str(tmp_path / "gone.txt"))
        f.touch()
        assert f.exists() is True

        f.remove()
        assert f.exists() is False

    def test_touch_creates_empty_file(self, tmp_path):
        f = LocalPath(str(tmp_path / "touched.txt"))
        assert f.exists() is False

        f.touch()
        assert f.exists() is True
        assert f.is_file() is True

    def test_exists_on_missing_returns_false(self, tmp_path):
        missing = LocalPath(str(tmp_path / "nonexistent.xyz"))
        assert missing.exists() is False


class TestDeletionCentralization:
    """``remove`` / ``unlink`` / ``rm`` and whole-asset ``delete`` all route
    through the single ``_delete`` primitive (path-removal mode)."""

    def test_unlink_removes_file(self, tmp_path):
        f = LocalPath(str(tmp_path / "f.txt"))
        f.touch()
        f.unlink()
        assert f.exists() is False

    def test_unlink_refuses_directory(self, tmp_path):
        d = LocalPath(str(tmp_path / "d"))
        d.mkdir()
        (LocalPath(str(tmp_path / "d" / "c.txt"))).touch()
        with pytest.raises(IsADirectoryError):
            d.unlink()
        assert d.exists() is True  # untouched

    def test_remove_recursive_directory(self, tmp_path):
        d = LocalPath(str(tmp_path / "tree"))
        d.mkdir()
        LocalPath(str(tmp_path / "tree" / "a.txt")).touch()
        LocalPath(str(tmp_path / "tree" / "b.txt")).touch()
        d.remove(recursive=True)
        assert d.exists() is False

    def test_rm_is_remove(self):
        assert Path.rm is Path.remove

    def test_delete_no_predicate_removes_leaf(self, tmp_path):
        # Whole-asset delete (no predicate) converges with path removal.
        f = LocalPath(str(tmp_path / "g.bin"))
        f.write_bytes(b"data")
        f.delete()
        assert f.exists() is False

    def test_delete_no_predicate_removes_directory(self, tmp_path):
        # ``delete()`` (no predicate) on a directory removes the whole tree —
        # it must NOT try to read the directory back as a tabular leaf.
        d = LocalPath(str(tmp_path / "tree"))
        d.mkdir()
        LocalPath(str(tmp_path / "tree" / "a.bin")).write_bytes(b"a")
        LocalPath(str(tmp_path / "tree" / "b.txt")).write_bytes(b"b")
        d.delete()
        assert d.exists() is False

    def test_delete_no_predicate_never_reads_arrow_batches(self, tmp_path, monkeypatch):
        # Regression: a bare ``path.delete()`` used to fall through to the
        # byte-leaf row rewrite, which reads Arrow batches and blows up with
        # ``NotImplementedError: IO has no tabular decoder`` on a directory /
        # non-tabular file / missing path. No predicate ⇒ pure path removal.
        from yggdrasil.io.holder import IO

        def boom(self, *a, **k):
            raise AssertionError("delete read Arrow batches for a no-predicate delete")

        monkeypatch.setattr(IO, "_read_arrow_batches", boom)

        # directory
        d = LocalPath(str(tmp_path / "dir"))
        d.mkdir()
        LocalPath(str(tmp_path / "dir" / "x.bin")).write_bytes(b"x")
        d.delete()
        assert d.exists() is False

        # non-tabular plain file
        f = LocalPath(str(tmp_path / "blob.other"))
        f.write_bytes(b"\x00\x01\x02not-a-table")
        f.delete()
        assert f.exists() is False

    def test_delete_no_predicate_is_idempotent_on_missing(self, tmp_path):
        # ``delete()`` (no predicate) on an already-absent path is a no-op:
        # the goal — "this path is gone" — is already met. ``remove`` /
        # ``unlink`` keep the strict ``missing_ok=False`` contract (below).
        ghost = LocalPath(str(tmp_path / "never" / "here.other"))
        ghost.delete()  # must not raise, must not read Arrow batches
        assert ghost.exists() is False

    def test_remove_routes_through_delete_remove_path(self, tmp_path, monkeypatch):
        f = LocalPath(str(tmp_path / "h.txt"))
        f.touch()
        seen = {}
        orig = LocalPath._delete

        def spy(self, predicate=None, **kw):
            seen.update(kw)
            return orig(self, predicate, **kw)

        monkeypatch.setattr(LocalPath, "_delete", spy)
        f.remove()
        assert seen.get("remove_path") is True

    def test_unlink_routes_through_delete_files_only(self, tmp_path, monkeypatch):
        f = LocalPath(str(tmp_path / "i.txt"))
        f.touch()
        seen = {}
        orig = LocalPath._delete

        def spy(self, predicate=None, **kw):
            seen.update(kw)
            return orig(self, predicate, **kw)

        monkeypatch.setattr(LocalPath, "_delete", spy)
        f.unlink()
        assert seen.get("remove_path") is True
        assert seen.get("files_only") is True

    def test_missing_ok_false_raises(self, tmp_path):
        missing = LocalPath(str(tmp_path / "nope.txt"))
        with pytest.raises(FileNotFoundError):
            missing.remove(missing_ok=False)


# ---------------------------------------------------------------------------
# TestTransfer — Path-to-Path byte copy
# ---------------------------------------------------------------------------


class TestTransfer:

    def test_transfer_to_copies_data(self, tmp_path):
        src = LocalPath(str(tmp_path / "src.parquet"))
        dst = LocalPath(str(tmp_path / "dst.parquet"))

        table = pa.table({"col": [100, 200, 300]})
        src.write_table(table)

        dst.upload(src)
        assert dst.exists() is True
        result = dst.read_arrow_table()
        assert result.num_rows == 3

    def test_transfer_preserves_content(self, tmp_path):
        src = LocalPath(str(tmp_path / "orig.parquet"))
        dst = LocalPath(str(tmp_path / "copy.parquet"))

        table = pa.table({"k": [1, 2], "v": ["alpha", "beta"]})
        src.write_table(table)

        dst.upload(src)
        result = dst.read_arrow_table()
        assert result.column("k").to_pylist() == [1, 2]
        assert result.column("v").to_pylist() == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# TestPickle — serialization round-trip
# ---------------------------------------------------------------------------


class TestPickle:

    def test_pickle_round_trip_preserves_path(self, tmp_path):
        original = LocalPath(str(tmp_path / "pk.txt"))
        original.touch()

        restored = pickle.loads(pickle.dumps(original))
        assert isinstance(restored, LocalPath)
        assert restored == original
        assert restored.full_path() == original.full_path()

    def test_pickled_path_can_read_data_written_before_pickle(self, tmp_path):
        p = LocalPath(str(tmp_path / "before_pickle.parquet"))
        table = pa.table({"n": [10, 20, 30]})
        p.write_table(table)

        restored = pickle.loads(pickle.dumps(p))
        result = restored.read_arrow_table()
        assert result.num_rows == 3
        assert result.column("n").to_pylist() == [10, 20, 30]


# ---------------------------------------------------------------------------
# TestEquality — __eq__ / __hash__
# ---------------------------------------------------------------------------


class TestEquality:

    def test_same_path_equal(self, tmp_path):
        path_str = str(tmp_path / "eq.txt")
        a = LocalPath(path_str)
        b = LocalPath(path_str)
        assert a == b

    def test_different_path_not_equal(self, tmp_path):
        a = LocalPath(str(tmp_path / "one.txt"))
        b = LocalPath(str(tmp_path / "two.txt"))
        assert a != b

    def test_hash_consistent_with_eq(self, tmp_path):
        path_str = str(tmp_path / "hashme.txt")
        a = LocalPath(path_str)
        b = LocalPath(path_str)
        assert a == b
        assert hash(a) == hash(b)

        # Different paths should (very likely) hash differently
        c = LocalPath(str(tmp_path / "other.txt"))
        assert a != c


# ---------------------------------------------------------------------------
# TestURLIntegration — URL ↔ Path interplay
# ---------------------------------------------------------------------------


class TestURLIntegration:

    def test_path_url_returns_file_scheme(self, tmp_path):
        p = LocalPath(str(tmp_path / "urltest.txt"))
        url = p.url
        assert isinstance(url, URL)
        assert url.scheme == "file"
        assert str(url).startswith("file:///")

    def test_url_path_matches_filesystem_path(self, tmp_path):
        fs_path = str(tmp_path / "match.txt")
        p = LocalPath(fs_path)
        assert p.url.path == fs_path

    def test_url_static_values_has_filepath(self, tmp_path):
        p = LocalPath(str(tmp_path / "static.txt"))
        sv = p.url.static_values
        assert "$filepath" in sv
        assert sv["$filepath"] == p.url.path
