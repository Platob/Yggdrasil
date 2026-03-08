from __future__ import annotations

import os
from collections.abc import Iterator

import mongoengine
import mongoengine.document as document_mod
from mongoengine import Document as MongoDocument

__all__ = [
    "Document",
]


class _DualMethod:
    def __init__(self, inst_func, cls_func):
        self._inst_func = inst_func
        self._cls_func = cls_func

    def __get__(self, obj, owner):
        if obj is None:
            def _bound_cls(*args, **kwargs):
                return self._cls_func(owner, *args, **kwargs)
            return _bound_cls

        def _bound_inst(*args, **kwargs):
            return self._inst_func(obj, *args, **kwargs)
        return _bound_inst


class Document(MongoDocument):
    meta = {
        "abstract": True,
        "allow_inheritance": True,
    }

    @staticmethod
    def _normalize_value(value):
        if value is None:
            return None

        t = type(value)
        name = t.__name__

        if name in {"ObjectId", "UUID"}:
            return str(value)

        if name == "DBRef":
            return str(value.id)

        if isinstance(value, dict):
            return {k: Document._normalize_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [Document._normalize_value(v) for v in value]

        return value

    @classmethod
    def arrow_schema(cls):
        import pyarrow as pa

        arrow_fields = []
        for name in cls._fields_ordered:
            if name == "_cls":
                continue

            field = cls._fields[name]
            arrow_type = (
                field.arrow_type()
                if hasattr(field, "arrow_type")
                else pa.string()
            )

            db_name = "_id" if name == "id" else name

            arrow_fields.append(
                pa.field(
                    db_name,
                    arrow_type,
                    nullable=not getattr(field, "required", False),
                )
            )

        return pa.schema(arrow_fields)

    def to_dict(self, normalize: bool = True) -> dict:
        data = self.to_mongo().to_dict()

        data.pop("_cls", None)

        if "_id" not in data:
            data["_id"] = os.urandom(16).hex()

        if not normalize:
            return data

        return {k: self._normalize_value(v) for k, v in data.items()}

    # -------------------------
    # instance implementations
    # -------------------------

    def _to_arrow_table_instance(self):
        import pyarrow as pa

        schema = type(self).arrow_schema()
        row = {name: self.to_dict(normalize=True).get(name) for name in schema.names}
        return pa.Table.from_pylist([row], schema=schema)

    def _to_arrow_batch_instance(self):
        return self._to_arrow_table_instance().to_batches()[0]

    def _to_arrow_batches_instance(self) -> Iterator:
        yield self._to_arrow_batch_instance()

    def _to_pandas_instance(self):
        df = self._to_arrow_table_instance().to_pandas()

        if "_id" in df.columns:
            df["_id"] = df["_id"].astype("string")

        return df

    # -------------------------
    # class/query implementations
    # -------------------------

    @classmethod
    def _to_arrow_table_class(
        cls,
        *args,
        **kwargs,
    ):
        return cls.objects.to_arrow_table(*args, **kwargs)

    @classmethod
    def _to_arrow_batch_class(
        cls,
        *args,
        **kwargs,
    ):
        batches = cls.objects.to_arrow_batches(*args, **kwargs)
        return next(iter(batches))

    @classmethod
    def _to_arrow_batches_class(
        cls,
        *args,
        **kwargs,
    ):
        return cls.objects.to_arrow_batches(*args, **kwargs)

    @classmethod
    def _to_pandas_class(
        cls,
        *args,
        **kwargs,
    ):
        return cls.objects.to_pandas(*args, **kwargs)

    # -------------------------
    # dual-use public API
    # -------------------------

    to_arrow_table = _DualMethod(_to_arrow_table_instance, _to_arrow_table_class)
    to_arrow_batch = _DualMethod(_to_arrow_batch_instance, _to_arrow_batch_class)
    to_arrow_batches = _DualMethod(_to_arrow_batches_instance, _to_arrow_batches_class)
    to_pandas = _DualMethod(_to_pandas_instance, _to_pandas_class)


document_mod.Document = Document
mongoengine.Document = Document