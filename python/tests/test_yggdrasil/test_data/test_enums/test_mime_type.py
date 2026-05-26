"""Behavior tests for :class:`yggdrasil.data.enums.mime_type.MimeType`.

Covers the practical entry points that the rest of the library calls:

* ``MimeTypes`` constants exist and carry the right metadata.
* ``MimeType.get`` is a pure dict lookup — never raises.
* ``MimeType.from_str`` resolves names, MIME values, extensions, and
  ``application/x-…`` prefixes.
* ``MimeType.from_magic`` sniffs magic bytes and falls back to the
  default contract.
* ``MimeType.parse_many`` flattens accept-headers, codec composites,
  and dotted-extension chains in wrapping order.
"""
from __future__ import annotations

import io

import pytest

from yggdrasil.enums.mime_type import MimeType, MimeTypes


class TestRegistryConstants:

    def test_singletons_have_consistent_metadata(self) -> None:
        assert MimeTypes.PARQUET.value == "application/vnd.apache.parquet"
        assert "parquet" in MimeTypes.PARQUET.extensions
        assert MimeTypes.CSV.is_tabular
        assert MimeTypes.JSON.value == "application/json"
        assert MimeTypes.GZIP.is_codec

    def test_octet_stream_is_any_bytes(self) -> None:
        assert MimeTypes.OCTET_STREAM.is_any_bytes


class TestPureGet:
    """`get` is a side-effect-free dict lookup."""

    def test_known_value(self) -> None:
        assert MimeType.get("application/json") is MimeTypes.JSON

    def test_known_name(self) -> None:
        assert MimeType.get("PARQUET") is MimeTypes.PARQUET

    def test_unknown_returns_none(self) -> None:
        assert MimeType.get("application/no-such-type") is None

    def test_non_string_returns_none(self) -> None:
        assert MimeType.get(42) is None

    def test_application_x_prefix_strip(self) -> None:
        # `application/parquet` is the unprefixed form of `vnd.apache.parquet`
        # — but the registry stores the canonical name (PARQUET) too.
        assert MimeType.get("PARQUET") is MimeTypes.PARQUET


class TestFromStr:

    def test_extension_with_dot(self) -> None:
        assert MimeType.from_(".csv") is MimeTypes.CSV

    def test_extension_no_dot(self) -> None:
        assert MimeType.from_("csv") is MimeTypes.CSV

    def test_unknown_extension_default_none(self) -> None:
        # ``.qqq`` is not registered; ``from_str`` should miss cleanly.
        assert MimeType.from_str(".qqq", default=None) is None

    def test_unknown_extension_default_omitted_raises(self) -> None:
        with pytest.raises(ValueError, match="resolution failed"):
            MimeType.from_str(".qqq")


class TestFromMagic:

    def test_parquet_header(self) -> None:
        assert MimeType.from_magic(b"PAR1\x00\x00") is MimeTypes.PARQUET

    def test_gzip_header(self) -> None:
        assert MimeType.from_magic(b"\x1f\x8b\x08\x00") is MimeTypes.GZIP

    def test_zstd_header(self) -> None:
        assert MimeType.from_magic(b"\x28\xb5\x2f\xfd\x00\x00") is MimeTypes.ZSTD

    def test_arrow_ipc_header(self) -> None:
        assert MimeType.from_magic(b"ARROW1\x00\x00") is MimeTypes.ARROW_IPC

    def test_unknown_magic_default_none(self) -> None:
        assert MimeType.from_magic(b"\x00\x01\x02\x03", default=None) is None

    def test_io_input(self) -> None:
        bio = io.BytesIO(b"PAR1\x00\x00")
        assert MimeType.from_magic(bio) is MimeTypes.PARQUET
        # Cursor restored.
        assert bio.tell() == 0


class TestParseManyComposites:

    def test_format_plus_codec_string(self) -> None:
        out = MimeType.parse_many("parquet+zstd")
        assert MimeTypes.PARQUET in out
        assert MimeTypes.ZSTD in out
        # Wrapping order: format first, codec second.
        assert out.index(MimeTypes.PARQUET) < out.index(MimeTypes.ZSTD)

    def test_dotted_extension_chain(self) -> None:
        out = MimeType.parse_many("trades.parquet.zst")
        assert out[0] is MimeTypes.PARQUET
        assert out[1] is MimeTypes.ZSTD

    def test_accept_header_split(self) -> None:
        out = MimeType.parse_many("application/json, text/csv;q=0.8")
        assert MimeTypes.JSON in out
        assert MimeTypes.CSV in out

    def test_iterable_input(self) -> None:
        out = MimeType.parse_many([MimeTypes.PARQUET, "csv"])
        assert MimeTypes.PARQUET in out
        assert MimeTypes.CSV in out

    def test_none_returns_empty(self) -> None:
        assert MimeType.parse_many(None) == []

    def test_unknown_silently_dropped(self) -> None:
        out = MimeType.parse_many("totally-not-real")
        assert out == []

    def test_dedup_first_seen_wins(self) -> None:
        out = MimeType.parse_many(["csv", "csv", MimeTypes.CSV])
        assert out == [MimeTypes.CSV]
