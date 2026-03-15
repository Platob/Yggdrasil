from __future__ import annotations

from mongoengine.fields import *  # noqa: F403
from mongoengine.fields import (
    BinaryField as MongoBinaryField,
    BooleanField as MongoBooleanField,
    CachedReferenceField as MongoCachedReferenceField,
    ComplexDateTimeField as MongoComplexDateTimeField,
    DateField as MongoDateField,
    DateTimeField as MongoDateTimeField,
    Decimal128Field as MongoDecimal128Field,
    DecimalField as MongoDecimalField,
    DictField as MongoDictField,
    DynamicField as MongoDynamicField,
    EmailField as MongoEmailField,
    EmbeddedDocumentField as MongoEmbeddedDocumentField,
    EmbeddedDocumentListField as MongoEmbeddedDocumentListField,
    EnumField as MongoEnumField,
    FileField as MongoFileField,
    FloatField as MongoFloatField,
    GenericEmbeddedDocumentField as MongoGenericEmbeddedDocumentField,
    GenericLazyReferenceField as MongoGenericLazyReferenceField,
    GenericReferenceField as MongoGenericReferenceField,
    GeoPointField as MongoGeoPointField,
    ImageField as MongoImageField,
    IntField as MongoIntField,
    LazyReferenceField as MongoLazyReferenceField,
    LineStringField as MongoLineStringField,
    ListField as MongoListField,
    LongField as MongoLongField,
    MapField as MongoMapField,
    MultiLineStringField as MongoMultiLineStringField,
    MultiPointField as MongoMultiPointField,
    MultiPolygonField as MongoMultiPolygonField,
    ObjectIdField as MongoObjectIdField,
    PointField as MongoPointField,
    PolygonField as MongoPolygonField,
    ReferenceField as MongoReferenceField,
    SequenceField as MongoSequenceField,
    SortedListField as MongoSortedListField,
    StringField as MongoStringField,
    URLField as MongoURLField,
    UUIDField as MongoUUIDField,
)

__all__ = [
    "StringField",
    "URLField",
    "EmailField",
    "IntField",
    "LongField",
    "FloatField",
    "DecimalField",
    "BooleanField",
    "DateTimeField",
    "DateField",
    "ComplexDateTimeField",
    "EmbeddedDocumentField",
    "ObjectIdField",
    "GenericEmbeddedDocumentField",
    "DynamicField",
    "ListField",
    "SortedListField",
    "EmbeddedDocumentListField",
    "DictField",
    "MapField",
    "ReferenceField",
    "CachedReferenceField",
    "LazyReferenceField",
    "GenericLazyReferenceField",
    "GenericReferenceField",
    "BinaryField",
    "FileField",
    "ImageField",
    "GeoPointField",
    "PointField",
    "LineStringField",
    "PolygonField",
    "SequenceField",
    "UUIDField",
    "EnumField",
    "MultiPointField",
    "MultiLineStringField",
    "MultiPolygonField",
    "Decimal128Field",
]


def _pa():
    import pyarrow as pa
    return pa


class _ArrowFieldMixin:
    def arrow_type(self):
        return _pa().string()


class StringField(_ArrowFieldMixin, MongoStringField):
    def arrow_type(self):
        return _pa().string()


class URLField(_ArrowFieldMixin, MongoURLField):
    def arrow_type(self):
        return _pa().string()


class EmailField(_ArrowFieldMixin, MongoEmailField):
    def arrow_type(self):
        return _pa().string()


class IntField(_ArrowFieldMixin, MongoIntField):
    def arrow_type(self):
        return _pa().int64()


class LongField(_ArrowFieldMixin, MongoLongField):
    def arrow_type(self):
        return _pa().int64()


class FloatField(_ArrowFieldMixin, MongoFloatField):
    def arrow_type(self):
        return _pa().float64()


class DecimalField(_ArrowFieldMixin, MongoDecimalField):
    def arrow_type(self):
        pa = _pa()
        scale = getattr(self, "precision", None) or 9
        return pa.decimal128(38, scale)


class Decimal128Field(_ArrowFieldMixin, MongoDecimal128Field):
    def arrow_type(self):
        return _pa().decimal128(38, 9)


class BooleanField(_ArrowFieldMixin, MongoBooleanField):
    def arrow_type(self):
        return _pa().bool_()


class DateTimeField(_ArrowFieldMixin, MongoDateTimeField):
    def arrow_type(self):
        return _pa().timestamp("us")


class DateField(_ArrowFieldMixin, MongoDateField):
    def arrow_type(self):
        return _pa().date32()


class ComplexDateTimeField(_ArrowFieldMixin, MongoComplexDateTimeField):
    def arrow_type(self):
        return _pa().string()


class ObjectIdField(_ArrowFieldMixin, MongoObjectIdField):
    def arrow_type(self):
        return _pa().string()


class UUIDField(_ArrowFieldMixin, MongoUUIDField):
    def arrow_type(self):
        return _pa().string()


class BinaryField(_ArrowFieldMixin, MongoBinaryField):
    def arrow_type(self):
        return _pa().binary()


class SequenceField(_ArrowFieldMixin, MongoSequenceField):
    def arrow_type(self):
        return _pa().int64()


class EnumField(_ArrowFieldMixin, MongoEnumField):
    def arrow_type(self):

        pa = _pa()
        first = None
        try:
            first = next(iter(self._enum_cls))
        except StopIteration:
            return pa.string()

        value = first.value
        if isinstance(value, bool):
            return pa.bool_()
        if isinstance(value, int):
            return pa.int64()
        if isinstance(value, float):
            return pa.float64()
        if isinstance(value, str):
            return pa.string()
        return pa.string()


class DynamicField(_ArrowFieldMixin, MongoDynamicField):
    def arrow_type(self):
        return _pa().string()


class DictField(_ArrowFieldMixin, MongoDictField):
    def arrow_type(self):
        pa = _pa()
        if getattr(self, "field", None) is not None:
            return pa.map_(pa.string(), self.field.arrow_type())
        return pa.map_(pa.string(), pa.string())


class MapField(_ArrowFieldMixin, MongoMapField):
    def arrow_type(self):
        pa = _pa()
        return pa.map_(pa.string(), self.field.arrow_type())


class ListField(_ArrowFieldMixin, MongoListField):
    def arrow_type(self):
        pa = _pa()
        inner = self.field.arrow_type() if getattr(self, "field", None) is not None else pa.string()
        return pa.list_(inner)


class SortedListField(_ArrowFieldMixin, MongoSortedListField):
    def arrow_type(self):
        pa = _pa()
        inner = self.field.arrow_type() if getattr(self, "field", None) is not None else pa.string()
        return pa.list_(inner)


class EmbeddedDocumentField(_ArrowFieldMixin, MongoEmbeddedDocumentField):
    def arrow_type(self):
        import pyarrow as pa
        schema = self.document_type.arrow_schema()
        return pa.struct(list(schema))


class EmbeddedDocumentListField(_ArrowFieldMixin, MongoEmbeddedDocumentListField):
    def arrow_type(self):
        import pyarrow as pa
        return pa.list_(self.field.arrow_type())


class GenericEmbeddedDocumentField(_ArrowFieldMixin, MongoGenericEmbeddedDocumentField):
    def arrow_type(self):
        pa = _pa()
        return pa.struct([
            pa.field("_cls", pa.string(), nullable=True),
        ])


class ReferenceField(_ArrowFieldMixin, MongoReferenceField):
    def arrow_type(self):
        return _pa().string()


class CachedReferenceField(_ArrowFieldMixin, MongoCachedReferenceField):
    def arrow_type(self):
        pa = _pa()

        ref_doc = self.document_type
        fields = [
            pa.field("_id", pa.string(), nullable=False),
        ]

        for name in getattr(self, "fields", []):
            field = ref_doc._fields.get(name)
            if field is None:
                continue
            arrow_type = field.arrow_type() if hasattr(field, "arrow_type") else pa.string()
            fields.append(
                pa.field(
                    name,
                    arrow_type,
                    nullable=not getattr(field, "required", False),
                )
            )

        return pa.struct(fields)


class LazyReferenceField(_ArrowFieldMixin, MongoLazyReferenceField):
    def arrow_type(self):
        return _pa().string()


class GenericReferenceField(_ArrowFieldMixin, MongoGenericReferenceField):
    def arrow_type(self):
        pa = _pa()
        return pa.struct([
            pa.field("_cls", pa.string(), nullable=True),
            pa.field("_ref", pa.string(), nullable=True),
        ])


class GenericLazyReferenceField(_ArrowFieldMixin, MongoGenericLazyReferenceField):
    def arrow_type(self):
        pa = _pa()
        return pa.struct([
            pa.field("_cls", pa.string(), nullable=True),
            pa.field("_ref", pa.string(), nullable=True),
        ])


class FileField(_ArrowFieldMixin, MongoFileField):
    def arrow_type(self):
        return _pa().string()


class ImageField(_ArrowFieldMixin, MongoImageField):
    def arrow_type(self):
        return _pa().string()


class GeoPointField(_ArrowFieldMixin, MongoGeoPointField):
    def arrow_type(self):
        return _pa().list_(_pa().float64())


class PointField(_ArrowFieldMixin, MongoPointField):
    def arrow_type(self):
        return _pa().string()


class LineStringField(_ArrowFieldMixin, MongoLineStringField):
    def arrow_type(self):
        return _pa().string()


class PolygonField(_ArrowFieldMixin, MongoPolygonField):
    def arrow_type(self):
        return _pa().string()


class MultiPointField(_ArrowFieldMixin, MongoMultiPointField):
    def arrow_type(self):
        return _pa().string()


class MultiLineStringField(_ArrowFieldMixin, MongoMultiLineStringField):
    def arrow_type(self):
        return _pa().string()


class MultiPolygonField(_ArrowFieldMixin, MongoMultiPolygonField):
    def arrow_type(self):
        return _pa().string()
