"""Behaviors of :class:`yggdrasil.data.enums.path_scheme.PathScheme`.

The enum centralizes the URL-scheme tokens every Yggdrasil
filesystem backend uses. Contract:

* members are :class:`str` and carry the canonical scheme token
  (``PathScheme.DBFS == "dbfs"``);
* :meth:`from_` accepts canonical tokens, common aliases
  (``"s3a"`` / ``"local"`` / ``"volume"``), case-insensitive input,
  and a trailing ``://``;
* :meth:`path_class` lazy-imports the matching :class:`Holder`
  subclass — the first call triggers the side-effect import that
  registers the scheme into the runtime ``_HOLDER_SCHEMES``;
* :meth:`resolve` is the ``from_(...) → path_class()`` shortcut.
"""
from __future__ import annotations

import pytest

from yggdrasil.data.enums.path_scheme import PathScheme


class TestCanonicalMembers:

    def test_member_values(self) -> None:
        assert PathScheme.FILE.value == "file"
        assert PathScheme.MEMORY.value == "memory"
        assert PathScheme.DBFS.value == "dbfs"
        assert PathScheme.VOLUMES.value == "volumes"
        assert PathScheme.S3.value == "s3"
        assert PathScheme.HTTP.value == "http"
        assert PathScheme.HTTPS.value == "https"

    def test_workspace_is_not_a_filesystem_scheme(self) -> None:
        """``/Workspace`` is managed through the Workspace API, not a
        filesystem one — :class:`PathScheme` is the FS-only view."""
        assert not hasattr(PathScheme, "WORKSPACE")
        assert not PathScheme.is_valid("workspace")

    def test_str_subclass(self) -> None:
        # Members slot in anywhere a string is expected.
        assert isinstance(PathScheme.DBFS, str)
        assert PathScheme.DBFS == "dbfs"
        assert f"{PathScheme.S3}://bucket/key" == "s3://bucket/key"


class TestFrom:

    def test_passthrough(self) -> None:
        assert PathScheme.from_(PathScheme.DBFS) is PathScheme.DBFS

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("file", PathScheme.FILE),
            ("FILE", PathScheme.FILE),
            ("file://", PathScheme.FILE),
            ("local", PathScheme.FILE),
            ("", PathScheme.FILE),
            ("memory", PathScheme.MEMORY),
            ("mem", PathScheme.MEMORY),
            ("dbfs", PathScheme.DBFS),
            ("DBFS", PathScheme.DBFS),
            ("volumes", PathScheme.VOLUMES),
            ("volume", PathScheme.VOLUMES),
            ("uc", PathScheme.VOLUMES),
            ("s3", PathScheme.S3),
            ("s3a", PathScheme.S3),
            ("s3n", PathScheme.S3),
            ("http", PathScheme.HTTP),
            ("https", PathScheme.HTTPS),
        ],
    )
    def test_string_aliases(self, raw, expected) -> None:
        assert PathScheme.from_(raw) is expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            PathScheme.from_("redis")

    def test_unknown_with_default(self) -> None:
        assert PathScheme.from_("redis", default=None) is None

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            PathScheme.from_(None)

    def test_none_with_default(self) -> None:
        assert PathScheme.from_(None, default=PathScheme.FILE) is PathScheme.FILE

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError):
            PathScheme.from_(42)

    def test_is_valid(self) -> None:
        assert PathScheme.is_valid("dbfs")
        assert PathScheme.is_valid("s3a")
        assert not PathScheme.is_valid("redis")
        assert not PathScheme.is_valid(None)


class TestPathClass:

    def test_resolves_local_path(self) -> None:
        from yggdrasil.io.path.local_path import LocalPath
        assert PathScheme.FILE.path_class() is LocalPath

    def test_resolves_memory(self) -> None:
        from yggdrasil.io.memory import Memory
        assert PathScheme.MEMORY.path_class() is Memory

    def test_resolve_shortcut_routes_through_from(self) -> None:
        from yggdrasil.io.path.local_path import LocalPath
        # Alias goes through :meth:`from_` and out :meth:`path_class`.
        assert PathScheme.resolve("local") is LocalPath
        assert PathScheme.resolve("file://") is LocalPath
