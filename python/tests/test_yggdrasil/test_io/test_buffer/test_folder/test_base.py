"""FolderIO core: enumeration, child minting, partition routing."""

from __future__ import annotations

import pytest

from yggdrasil.io.buffer.nested import FolderIO, FolderOptions
from yggdrasil.io.buffer.primitive import ParquetIO
from yggdrasil.io.enums import MimeTypes
from .._helpers import sample_table


class TestFolderBase:
    def test_default_media_type(self):
        assert FolderIO.default_media_type() == MimeTypes.FOLDER

    def test_options_class(self):
        assert FolderIO.options_class() is FolderOptions

    def test_missing_path_is_empty(self, tmp_path):
        assert FolderIO(path=str(tmp_path / "missing")).is_empty()


class TestFolderIterChildren:
    def test_yields_per_file(self, tmp_path):
        # Drop two parquet files into the folder.
        (tmp_path / "a.parquet").touch()
        ParquetIO(path=str(tmp_path / "a.parquet")).write_arrow_table(sample_table())
        ParquetIO(path=str(tmp_path / "b.parquet")).write_arrow_table(sample_table())

        names = sorted(c.path.name for c in FolderIO(path=str(tmp_path))._iter_children(FolderOptions()))
        assert names == ["a.parquet", "b.parquet"]

    def test_yields_dot_prefixed_entries(self, tmp_path):
        # Iteration is unfiltered — dot-prefixed files are yielded
        # alongside everything else. Backends that need to hide
        # specific names (Delta's ``_delta_log/``, YGG's ``.ygg/``)
        # override ``_iter_children`` themselves.
        ParquetIO(path=str(tmp_path / "a.parquet")).write_arrow_table(sample_table())
        (tmp_path / ".hidden").touch()

        names = sorted(
            c.path.name
            for c in FolderIO(path=str(tmp_path))._iter_children(FolderOptions())
        )
        assert names == [".hidden", "a.parquet"]


class TestFolderMakeChild:
    def test_make_child_returns_primitive_io(self, tmp_path):
        folder = FolderIO(path=str(tmp_path))
        child = folder.make_child("part.parquet")
        # Format inferred from extension → ParquetIO.
        assert isinstance(child, ParquetIO)
        assert child.parent is folder

    def test_make_child_rejects_traversal(self, tmp_path):
        folder = FolderIO(path=str(tmp_path))
        with pytest.raises(ValueError):
            folder.make_child("../escape.parquet")
