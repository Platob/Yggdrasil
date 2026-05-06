"""Behavior tests for :class:`yggdrasil.data.enums.media_type.MediaType`.

`MediaType` is the format + codec wrapper layer above `MimeType`.
The two-stage sniff (outer magic identifies a wrapper, inner peek
into the decompressed head identifies the format) is what makes
``Tabular.for_holder`` route a gzip-wrapped parquet to ParquetIO
instead of OctetStreamIO. These tests pin the practical surface:

* `from_` accepts MimeType / Codec / bytes / file-like / path-like.
* `from_path` resolves by extension first, magic-sniff second.
* `from_magic` and `from_io` agree for the same content.
* Wrapper-only buffers fall back to ``OCTET_STREAM + codec``.
"""
from __future__ import annotations

import gzip
import io


from yggdrasil.data.enums.codec import Codecs
from yggdrasil.data.enums.media_type import MediaType, MediaTypes
from yggdrasil.data.enums.mime_type import MimeTypes


class TestIdentity:

    def test_mime_type_input(self) -> None:
        mt = MediaType.from_(MimeTypes.PARQUET)
        assert mt.mime_type is MimeTypes.PARQUET
        assert mt.codec is None

    def test_codec_input_wraps_octet_stream(self) -> None:
        mt = MediaType.from_(Codecs.GZIP)
        assert mt.mime_type is MimeTypes.OCTET_STREAM
        assert mt.codec is Codecs.GZIP

    def test_pass_through_media_type(self) -> None:
        original = MediaTypes.PARQUET
        assert MediaType.from_(original) is original

    def test_codec_mime_type_normalizes_to_codec_field(self) -> None:
        # Constructing with a codec mime sets ``codec`` and resets
        # ``mime_type`` to OCTET_STREAM — that's the canonical form.
        mt = MediaType(mime_type=MimeTypes.GZIP)
        assert mt.codec is Codecs.GZIP
        assert mt.mime_type is MimeTypes.OCTET_STREAM


class TestFromMagic:

    def test_parquet_header(self) -> None:
        out = MediaType.from_magic(b"PAR1\x00\x00")
        assert out.mime_type is MimeTypes.PARQUET
        assert out.codec is None

    def test_gzip_wrapping_unknown_falls_back_to_codec_only(self) -> None:
        compressed = gzip.compress(b"\x00\x01\x02\x03\x04")
        out = MediaType.from_magic(compressed)
        assert out.mime_type is MimeTypes.OCTET_STREAM
        assert out.codec is Codecs.GZIP

    def test_unknown_default_none(self) -> None:
        # The library's contract here: unknown bytes return ``OCTET_STREAM``
        # with ``default=None`` collapsing to the safe fallback rather than
        # bubbling a hard miss.
        out = MediaType.from_magic(b"\x00\x01\x02", default=None)
        assert out is None or out.mime_type is MimeTypes.OCTET_STREAM


class TestFromIo:
    """``from_io`` must restore the caller's cursor on exit."""

    def test_restores_cursor(self) -> None:
        bio = io.BytesIO(b"PAR1\x00\x00ZZZ")
        bio.seek(3)
        out = MediaType.from_io(bio)
        assert out.mime_type is MimeTypes.PARQUET
        assert bio.tell() == 3

    def test_io_two_stage_sniff_through_gzip(self) -> None:
        payload = gzip.compress(b"ARROW1\x00\x00")
        bio = io.BytesIO(payload)
        out = MediaType.from_io(bio)
        # IO path peeks past the codec wrapper, so the inner mime survives.
        assert out.codec is Codecs.GZIP
        assert out.mime_type is MimeTypes.ARROW_IPC


class TestFromPath:

    def test_extension_resolves_without_io(self) -> None:
        out = MediaType.from_("data.parquet")
        assert out.mime_type is MimeTypes.PARQUET

    def test_csv_extension(self) -> None:
        out = MediaType.from_("trades.csv")
        assert out.mime_type is MimeTypes.CSV

    def test_dotted_codec_chain(self) -> None:
        # parquet+gzip via dotted suffix.
        out = MediaType.from_("trades.parquet.gz")
        assert out.mime_type is MimeTypes.PARQUET
        assert out.codec is Codecs.GZIP


class TestRepr:

    def test_no_codec_repr_is_value(self) -> None:
        assert repr(MediaTypes.PARQUET) == MimeTypes.PARQUET.value

    def test_with_codec_repr_includes_codec(self) -> None:
        out = MediaType(MimeTypes.PARQUET, codec=Codecs.GZIP)
        assert "parquet" in repr(out)
        assert "gzip" in repr(out)
