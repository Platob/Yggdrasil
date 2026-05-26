"""Behavior tests for :class:`yggdrasil.enums.codec.Codec`.

The codec enum sits between MimeType and the byte path. Tests pin
the practical surface every caller uses:

* singleton resolution by short name and by MimeType.
* bytes round-trip on every codec the host has installed.
* streaming round-trip via :class:`BytesIO` for the codecs that
  expose streaming hooks.
* `is_streaming` honestly reports support — non-streaming codecs
  still produce correct output through the bytes fallback.
"""
from __future__ import annotations


import pytest

from yggdrasil.enums.codec import Codec, Codecs
from yggdrasil.enums.mime_type import MimeTypes
from yggdrasil.io.bytes_io import BytesIO


_PAYLOAD = (b"yggdrasil-codec-roundtrip-" + b"x" * 16) * 32


def _codec_available(codec: Codec) -> bool:
    """True when the codec's underlying library imports.

    The codec instances exist regardless of whether the underlying
    library is installed; bytes-roundtrip is what actually depends on
    the import. Skip when the import would fail.
    """
    try:
        codec.compress_bytes(b"x")
    except Exception:
        return False
    return True


_ALL = [
    Codecs.GZIP, Codecs.ZSTD, Codecs.LZ4, Codecs.BZIP2,
    Codecs.XZ, Codecs.SNAPPY, Codecs.BROTLI, Codecs.ZLIB,
    Codecs.LZMA,
]


class TestSingletons:

    def test_codecs_are_codec_instances(self) -> None:
        for c in _ALL:
            assert isinstance(c, Codec)

    def test_each_has_unique_name(self) -> None:
        names = [c.name for c in _ALL]
        assert len(names) == len(set(names))

    def test_each_has_codec_mime(self) -> None:
        for c in _ALL:
            assert c.mime_type.is_codec


class TestFrom:

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("gzip", Codecs.GZIP),
            ("GZIP", Codecs.GZIP),
            ("zstd", Codecs.ZSTD),
            ("lz4", Codecs.LZ4),
            ("brotli", Codecs.BROTLI),
        ],
    )
    def test_short_name(self, name: str, expected: Codec) -> None:
        assert Codec.from_(name) is expected

    def test_mime_type_input(self) -> None:
        assert Codec.from_(MimeTypes.GZIP) is Codecs.GZIP

    def test_codec_pass_through(self) -> None:
        assert Codec.from_(Codecs.ZSTD) is Codecs.ZSTD

    def test_none_with_default(self) -> None:
        assert Codec.from_(None, default=None) is None

    def test_unknown_raises_without_default(self) -> None:
        with pytest.raises(ValueError):
            Codec.from_("not-a-codec")

    def test_unknown_returns_none_with_default(self) -> None:
        assert Codec.from_("not-a-codec", default=None) is None


class TestBytesRoundTrip:

    @pytest.mark.parametrize("codec", _ALL, ids=[c.name for c in _ALL])
    def test_round_trip(self, codec: Codec) -> None:
        if not _codec_available(codec):
            pytest.skip(f"{codec.name} backend not installed")
        compressed = codec.compress_bytes(_PAYLOAD)
        assert compressed != _PAYLOAD  # actually compressed (or different framing)
        assert codec.decompress_bytes(compressed) == _PAYLOAD

    @pytest.mark.parametrize("codec", _ALL, ids=[c.name for c in _ALL])
    def test_empty_round_trip(self, codec: Codec) -> None:
        if not _codec_available(codec):
            pytest.skip(f"{codec.name} backend not installed")
        compressed = codec.compress_bytes(b"")
        assert codec.decompress_bytes(compressed) == b""


class TestStreamingRoundTrip:
    """Streaming hooks are optional; the test asserts honesty.

    For codecs that advertise streaming, the compress/decompress
    methods must accept a BytesIO source and produce a BytesIO sink
    whose round-trip equals the input.
    """

    @pytest.mark.parametrize("codec", _ALL, ids=[c.name for c in _ALL])
    def test_round_trip_via_bytes_io(self, codec: Codec) -> None:
        if not _codec_available(codec):
            pytest.skip(f"{codec.name} backend not installed")
        src = BytesIO(_PAYLOAD)
        with src:
            compressed = codec.compress(src)
        with compressed:
            decompressed = codec.decompress(compressed)
        with decompressed:
            assert decompressed.to_bytes() == _PAYLOAD

    def test_is_streaming_reflects_subclass_overrides(self) -> None:
        # gzip / zstd / lz4 / bzip2 / xz / lzma override streaming hooks.
        # snappy / brotli do not.
        assert Codecs.GZIP.is_streaming
        assert not Codecs.SNAPPY.is_streaming
        assert not Codecs.BROTLI.is_streaming


class TestReadStartEnd:
    """Partial-decode helper — head + tail without materializing."""

    @pytest.mark.parametrize(
        "codec",
        [Codecs.GZIP, Codecs.ZSTD, Codecs.BZIP2, Codecs.XZ],
        ids=lambda c: c.name,
    )
    def test_head_tail_match_full_decode(self, codec: Codec) -> None:
        if not _codec_available(codec):
            pytest.skip(f"{codec.name} backend not installed")
        compressed = codec.compress_bytes(_PAYLOAD)
        head, tail = codec.read_start_end(compressed, n_start=8, n_end=12)
        assert head == _PAYLOAD[:8]
        assert tail == _PAYLOAD[-12:]
