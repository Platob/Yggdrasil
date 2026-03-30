"""Unit tests for MediaType / MimeType / Codec pickle serializers."""
from __future__ import annotations

import pytest

from yggdrasil.io.enums.codec import Codec, GZIP, ZSTD, LZ4, BROTLI
from yggdrasil.io.enums.media_type import MediaType
from yggdrasil.io.enums.mime_type import MimeType
from yggdrasil.pickle.ser.media import (
    CodecSerialized,
    MediaTypeSerialized,
    MimeTypeSerialized,
)
from yggdrasil.pickle.ser.serialized import Serialized
from yggdrasil.pickle.ser.tags import Tags


# ===========================================================================
# Tags
# ===========================================================================

class TestTags:

    def test_media_type_tag_in_framework_range(self):
        assert Tags.is_framework(Tags.MEDIA_TYPE)

    def test_mime_type_tag_in_framework_range(self):
        assert Tags.is_framework(Tags.MIME_TYPE)

    def test_codec_tag_in_framework_range(self):
        assert Tags.is_framework(Tags.CODEC)

    def test_tags_are_distinct(self):
        tags = {Tags.MEDIA_TYPE, Tags.MIME_TYPE, Tags.CODEC}
        assert len(tags) == 3

    def test_tag_names_registered(self):
        assert Tags.get_name(Tags.MEDIA_TYPE) == "MEDIA_TYPE"
        assert Tags.get_name(Tags.MIME_TYPE) == "MIME_TYPE"
        assert Tags.get_name(Tags.CODEC) == "CODEC"

    def test_classes_registered(self):
        assert Tags.get_class(Tags.MEDIA_TYPE) is MediaTypeSerialized
        assert Tags.get_class(Tags.MIME_TYPE) is MimeTypeSerialized
        assert Tags.get_class(Tags.CODEC) is CodecSerialized


# ===========================================================================
# MimeTypeSerialized
# ===========================================================================

class TestMimeTypeSerialized:

    @pytest.mark.parametrize("mt", [
        MimeTypes.JSON,
        MimeTypes.PARQUET,
        MimeTypes.CSV,
        MimeTypes.ARROW_IPC,
        MimeTypes.OCTET_STREAM,
        MimeTypes.GZIP,
        MimeTypes.ZSTD,
        MimeType.PNG,
        MimeTypes.XML,
    ])
    def test_roundtrip(self, mt: MimeType):
        ser = MimeTypeSerialized.from_python_object(mt)
        assert ser is not None
        assert ser.tag == Tags.MIME_TYPE
        restored = ser.as_python()
        assert restored is mt

    def test_not_mime_type_returns_none(self):
        assert MimeTypeSerialized.from_python_object("not a mime") is None

    def test_value_property(self):
        ser = MimeTypeSerialized.from_python_object(MimeTypes.JSON)
        assert ser.value is MimeTypes.JSON

    def test_via_generic_from_python_object(self):
        ser = Serialized.from_python_object(MimeTypes.PARQUET)
        assert isinstance(ser, MimeTypeSerialized)
        assert ser.as_python() is MimeTypes.PARQUET

    def test_binary_roundtrip(self):
        """Serialize → bytes → deserialize."""
        original = MimeTypes.CSV
        ser = MimeTypeSerialized.from_python_object(original)
        buf = ser.write_to()
        restored_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(restored_ser, MimeTypeSerialized)
        assert restored_ser.as_python() is original


# ===========================================================================
# CodecSerialized
# ===========================================================================

class TestCodecSerialized:

    @pytest.mark.parametrize("codec", [GZIP, ZSTD, LZ4, BROTLI])
    def test_roundtrip(self, codec: Codec):
        ser = CodecSerialized.from_python_object(codec)
        assert ser is not None
        assert ser.tag == Tags.CODEC
        restored = ser.as_python()
        assert restored.name == codec.name

    def test_not_codec_returns_none(self):
        assert CodecSerialized.from_python_object(42) is None

    def test_value_property(self):
        ser = CodecSerialized.from_python_object(GZIP)
        assert ser.value.name == "gzip"

    def test_via_generic_from_python_object(self):
        ser = Serialized.from_python_object(ZSTD)
        assert isinstance(ser, CodecSerialized)
        assert ser.as_python().name == "zstd"

    def test_binary_roundtrip(self):
        ser = CodecSerialized.from_python_object(GZIP)
        buf = ser.write_to()
        restored_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(restored_ser, CodecSerialized)
        assert restored_ser.as_python().name == "gzip"


# ===========================================================================
# MediaTypeSerialized
# ===========================================================================

class TestMediaTypeSerialized:

    def test_roundtrip_mime_only(self):
        mt = MediaType(mime_type=MimeTypes.JSON)
        ser = MediaTypeSerialized.from_python_object(mt)
        assert ser is not None
        assert ser.tag == Tags.MEDIA_TYPE
        restored = ser.as_python()
        assert restored.mime_type is MimeTypes.JSON
        assert restored.codec is None

    def test_roundtrip_mime_with_codec(self):
        mt = MediaType(mime_type=MimeTypes.PARQUET, codec=GZIP)
        ser = MediaTypeSerialized.from_python_object(mt)
        assert ser is not None
        restored = ser.as_python()
        assert restored.mime_type is MimeTypes.PARQUET
        assert restored.codec is not None
        assert restored.codec.name == "gzip"

    @pytest.mark.parametrize("mt", [
        MediaType(MimeTypes.JSON),
        MediaType(MimeTypes.CSV),
        MediaType(MimeTypes.PARQUET, GZIP),
        MediaType(MimeTypes.PARQUET, ZSTD),
        MediaType(MimeTypes.ARROW_IPC),
        MediaType(MimeTypes.OCTET_STREAM, LZ4),
    ])
    def test_roundtrip_parametric(self, mt: MediaType):
        ser = MediaTypeSerialized.from_python_object(mt)
        restored = ser.as_python()
        assert restored.mime_type is mt.mime_type
        if mt.codec is not None:
            assert restored.codec.name == mt.codec.name
        else:
            assert restored.codec is None

    def test_not_media_type_returns_none(self):
        assert MediaTypeSerialized.from_python_object("nope") is None

    def test_via_generic_from_python_object(self):
        mt = MediaType(MimeTypes.PARQUET, GZIP)
        ser = Serialized.from_python_object(mt)
        assert isinstance(ser, MediaTypeSerialized)
        restored = ser.as_python()
        assert restored.mime_type is MimeTypes.PARQUET

    def test_binary_roundtrip(self):
        original = MediaType(MimeTypes.JSON, ZSTD)
        ser = MediaTypeSerialized.from_python_object(original)
        buf = ser.write_to()
        restored_ser = Serialized.read_from(buf, pos=0)
        assert isinstance(restored_ser, MediaTypeSerialized)
        restored = restored_ser.as_python()
        assert restored.mime_type is MimeTypes.JSON
        assert restored.codec.name == "zstd"

    def test_payload_format_no_codec(self):
        mt = MediaType(MimeTypes.JSON)
        ser = MediaTypeSerialized.from_python_object(mt)
        payload = ser.decode()
        assert payload == b"JSON"

    def test_payload_format_with_codec(self):
        mt = MediaType(MimeTypes.PARQUET, GZIP)
        ser = MediaTypeSerialized.from_python_object(mt)
        payload = ser.decode()
        assert payload == b"PARQUET+gzip"


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:

    def test_unknown_mime_name_raises(self):
        """Build a serialized with invalid MimeType name."""
        ser = MimeTypeSerialized.build(
            tag=Tags.MIME_TYPE,
            data=b"NONEXISTENT_FORMAT",
        )
        with pytest.raises(ValueError, match="Unknown MimeType"):
            ser.as_python()

    def test_unknown_codec_name_raises(self):
        ser = CodecSerialized.build(
            tag=Tags.CODEC,
            data=b"nonexistent_codec",
        )
        with pytest.raises(ValueError, match="Unknown Codec"):
            ser.as_python()

    def test_unknown_media_mime_raises(self):
        ser = MediaTypeSerialized.build(
            tag=Tags.MEDIA_TYPE,
            data=b"NONEXISTENT+gzip",
        )
        with pytest.raises(ValueError, match="Unknown MimeType"):
            ser.as_python()

    def test_unknown_media_codec_raises(self):
        ser = MediaTypeSerialized.build(
            tag=Tags.MEDIA_TYPE,
            data=b"JSON+nonexistent",
        )
        with pytest.raises(ValueError, match="Unknown Codec"):
            ser.as_python()

