"""Behaviors of :class:`yggdrasil.enums.scheme.Scheme`.

The enum centralizes the URL-scheme tokens every Yggdrasil
:class:`URLBased` backend uses. Contract:

* members are :class:`str` and carry the canonical scheme token
  (``Scheme.DBFS == "dbfs"``,
  ``Scheme.DATABRICKS_VOLUME == "dbfs+volume"``);
* the Databricks family uses the compound ``dbfs+<surface>``
  convention — the bare ``dbfs`` scheme is the abstract dispatcher;
* :meth:`from_` accepts canonical tokens, the small set of common
  aliases (``"s3a"`` / ``"local"`` / ``"memory"`` / ``"databricks"``),
  case-insensitive input, and a trailing ``://``;
* :meth:`path_class` lazy-imports the matching :class:`URLBased`
  subclass and triggers the registration side-effect;
* :meth:`resolve` is the ``from_(...) → path_class()`` shortcut.
"""
from __future__ import annotations

import pytest

from yggdrasil.enums.scheme import Scheme


class TestCanonicalMembers:

    def test_member_values(self) -> None:
        assert Scheme.FILE.value == "file"
        assert Scheme.MEMORY.value == "mem"
        assert Scheme.S3.value == "s3"
        assert Scheme.HTTP.value == "http"
        assert Scheme.HTTPS.value == "https"

    def test_databricks_compound_members(self) -> None:
        assert Scheme.DBFS.value == "dbfs"
        assert Scheme.DATABRICKS_DBFS.value == "dbfs+dbfs"
        assert Scheme.DATABRICKS_VOLUME.value == "dbfs+volume"
        assert Scheme.DATABRICKS_WORKSPACE.value == "dbfs+workspace"

    def test_str_subclass(self) -> None:
        # Members slot in anywhere a string is expected.
        assert isinstance(Scheme.DBFS, str)
        assert Scheme.DBFS == "dbfs"
        assert f"{Scheme.S3}://bucket/key" == "s3://bucket/key"


class TestFrom:

    def test_passthrough(self) -> None:
        assert Scheme.from_(Scheme.DBFS) is Scheme.DBFS

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("file", Scheme.FILE),
            ("FILE", Scheme.FILE),
            ("file://", Scheme.FILE),
            ("local", Scheme.FILE),
            ("", Scheme.FILE),
            ("mem", Scheme.MEMORY),
            ("memory", Scheme.MEMORY),
            ("dbfs", Scheme.DBFS),
            ("DBFS", Scheme.DBFS),
            ("dbfs+dbfs", Scheme.DATABRICKS_DBFS),
            ("dbfs+volume", Scheme.DATABRICKS_VOLUME),
            ("dbfs+workspace", Scheme.DATABRICKS_WORKSPACE),
            ("s3", Scheme.S3),
            ("s3a", Scheme.S3),
            ("s3n", Scheme.S3),
            ("http", Scheme.HTTP),
            ("https", Scheme.HTTPS),
        ],
    )
    def test_string_aliases(self, raw, expected) -> None:
        assert Scheme.from_(raw) is expected

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            Scheme.from_("redis")

    def test_unknown_with_default(self) -> None:
        assert Scheme.from_("redis", default=None) is None

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError):
            Scheme.from_(None)

    def test_none_with_default(self) -> None:
        assert Scheme.from_(None, default=Scheme.FILE) is Scheme.FILE

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError):
            Scheme.from_(42)

    def test_is_valid(self) -> None:
        assert Scheme.is_valid("dbfs")
        assert Scheme.is_valid("dbfs+volume")
        assert Scheme.is_valid("dbfs+workspace")
        assert Scheme.is_valid("s3a")
        assert not Scheme.is_valid("workspace")  # legacy alias dropped
        assert not Scheme.is_valid("volumes")  # legacy alias dropped
        assert not Scheme.is_valid("redis")
        assert not Scheme.is_valid(None)


class TestPathClass:

    def test_resolves_local_path(self) -> None:
        from yggdrasil.path.local_path import LocalPath
        assert Scheme.FILE.path_class() is LocalPath

    def test_resolves_memory(self) -> None:
        from yggdrasil.io.memory import Memory
        assert Scheme.MEMORY.path_class() is Memory

    def test_resolve_shortcut_routes_through_from(self) -> None:
        from yggdrasil.path.local_path import LocalPath
        # Alias goes through :meth:`from_` and out :meth:`path_class`.
        assert Scheme.resolve("local") is LocalPath
        assert Scheme.resolve("file://") is LocalPath
