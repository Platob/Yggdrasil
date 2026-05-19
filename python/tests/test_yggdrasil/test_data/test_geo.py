"""Tests for :mod:`yggdrasil.data.types.geo` — the ``GeoPoint`` shape.

Covers the contract documented in the module: singleton ``StructType``
with non-nullable ``lat`` + ``lon`` ``float64`` children, factory
that splices ``comment`` into ``metadata[b"comment"]``, and the
``is_geo_point_type`` matcher.
"""
from __future__ import annotations

import pyarrow as pa

from yggdrasil.data import (
    GEO_POINT_FIELDS,
    GEO_POINT_TYPE,
    DataType,
    Field,
    Schema,
    geo_point,
    is_geo_point_type,
)
from yggdrasil.data.types.nested.struct import StructType
from yggdrasil.data.types.primitive.numeric.floating_point import Float64Type


# ---------------------------------------------------------------------------
# Canonical type
# ---------------------------------------------------------------------------
class TestGeoPointType:

    def test_is_struct_with_two_children(self):
        assert isinstance(GEO_POINT_TYPE, StructType)
        assert len(GEO_POINT_TYPE.fields) == 2

    def test_children_are_lat_lon_float64(self):
        names = [f.name for f in GEO_POINT_TYPE.fields]
        assert names == ["lat", "lon"]
        for child in GEO_POINT_TYPE.fields:
            assert isinstance(child.dtype, Float64Type)
            assert child.nullable is False

    def test_module_constant_lists_field_names(self):
        # GEO_POINT_FIELDS mirrors the children — downstream encoders
        # that fan the struct apart can grab the names from one place.
        assert GEO_POINT_FIELDS == ("lat", "lon")

    def test_is_a_singleton(self):
        # Two ``geo_point(...)`` calls reuse the same type instance —
        # critical for the cast-registry exact-match fast path.
        f1 = geo_point("a")
        f2 = geo_point("b")
        assert f1.dtype is GEO_POINT_TYPE
        assert f2.dtype is GEO_POINT_TYPE


# ---------------------------------------------------------------------------
# geo_point() factory
# ---------------------------------------------------------------------------
class TestGeoPointFactory:

    def test_default_field_is_nullable(self):
        f = geo_point("position")
        assert f.name == "position"
        assert f.nullable is True
        assert f.dtype is GEO_POINT_TYPE

    def test_can_disable_nullable(self):
        f = geo_point("position", nullable=False)
        assert f.nullable is False

    def test_comment_lands_in_metadata(self):
        f = geo_point("position", comment="WGS84 pickup point.")
        # Comment is stored as bytes-keyed UTF-8 in metadata, exposed
        # through Field.comment as a decoded str.
        assert f.comment == "WGS84 pickup point."
        assert f.metadata.get(b"comment") == b"WGS84 pickup point."

    def test_explicit_metadata_merges_with_comment(self):
        f = geo_point(
            "position",
            comment="Pickup point.",
            metadata={b"origin": b"vendor-feed-v3"},
        )
        assert f.comment == "Pickup point."
        assert f.metadata.get(b"origin") == b"vendor-feed-v3"

    def test_explicit_comment_in_metadata_wins(self):
        # When the caller already put a comment into ``metadata`` and
        # then passes a ``comment=`` kwarg, the explicit metadata entry
        # wins — _merge_comment uses setdefault.
        f = geo_point(
            "position",
            comment="from kwarg",
            metadata={b"comment": b"from metadata dict"},
        )
        assert f.comment == "from metadata dict"

    def test_tags_pass_through(self):
        f = geo_point("position", tags={"renderable": "true"})
        # Tags merge into metadata via the Field constructor's
        # _normalize_metadata with the ``t:`` prefix (see TAG_PREFIX).
        assert f.metadata.get(b"t:renderable") == b"true"


# ---------------------------------------------------------------------------
# is_geo_point_type matcher
# ---------------------------------------------------------------------------
class TestIsGeoPointType:

    def test_matches_canonical(self):
        assert is_geo_point_type(GEO_POINT_TYPE)

    def test_matches_independently_constructed_equivalent(self):
        # Same shape built by hand should still match.
        rolled = StructType(fields=(
            Field("lat", Float64Type(), nullable=False),
            Field("lon", Float64Type(), nullable=False),
        ))
        assert is_geo_point_type(rolled)

    def test_rejects_wrong_field_names(self):
        wrong = StructType(fields=(
            Field("latitude",  Float64Type(), nullable=False),
            Field("longitude", Float64Type(), nullable=False),
        ))
        assert is_geo_point_type(wrong) is False

    def test_rejects_extra_children(self):
        extra = StructType(fields=(
            Field("lat", Float64Type(), nullable=False),
            Field("lon", Float64Type(), nullable=False),
            Field("alt", Float64Type(), nullable=False),
        ))
        assert is_geo_point_type(extra) is False

    def test_rejects_wrong_child_dtype(self):
        from yggdrasil.data.types.primitive.numeric.floating_point import Float32Type
        wrong_type = StructType(fields=(
            Field("lat", Float32Type(), nullable=False),
            Field("lon", Float64Type(), nullable=False),
        ))
        assert is_geo_point_type(wrong_type) is False

    def test_rejects_non_struct(self):
        assert is_geo_point_type(Float64Type()) is False
        assert is_geo_point_type(None) is False
        assert is_geo_point_type("lat,lon") is False


# ---------------------------------------------------------------------------
# Integration — geo_point inside a Schema, Arrow round-trip
# ---------------------------------------------------------------------------
class TestGeoPointInSchema:

    def test_schema_carries_geo_point_field(self):
        schema = Schema.from_fields([
            Field("trip_id", DataType.from_("string"), nullable=False),
            geo_point("origin",      comment="Pickup."),
            geo_point("destination", comment="Drop-off."),
        ])
        names = [f.name for f in schema.fields]
        assert names == ["trip_id", "origin", "destination"]
        assert is_geo_point_type(schema.field_by(name="origin").dtype)

    def test_arrow_round_trip(self):
        """``geo_point`` field → Arrow struct → back to ``Field``.

        Confirms the type survives the Arrow boundary intact (Arrow's
        own struct → :func:`Field.from_arrow_field` → ``StructType``
        with two float64 children).
        """
        f = geo_point("position", nullable=False)
        arrow_field = f.to_arrow()
        assert pa.types.is_struct(arrow_field.type)
        child_names = [arrow_field.type.field(i).name for i in range(arrow_field.type.num_fields)]
        assert child_names == ["lat", "lon"]
        for i in range(arrow_field.type.num_fields):
            assert arrow_field.type.field(i).type == pa.float64()

        # Round-trip back into the yggdrasil universe.
        round_trip = Field.from_arrow_field(arrow_field)
        assert is_geo_point_type(round_trip.dtype)
