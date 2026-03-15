from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

__all__ = ["Tags"]

if TYPE_CHECKING:
    from yggdrasil.pickle.ser.serialized import Serialized


class Tags:
    """Wire tags for serialized payload kinds.

    Keep these values stable. They are part of the binary protocol.
    """

    # Category labels
    CLASSES: ClassVar[dict[int, type["Serialized[object]"]]] = {}

    CATEGORY_UNKNOWN: ClassVar[str] = "unknown"
    CATEGORY_PRIMITIVE: ClassVar[str] = "primitive"
    CATEGORY_COLLECTION: ClassVar[str] = "collection"
    CATEGORY_SYSTEM: ClassVar[str] = "system"
    CATEGORY_FRAMEWORK: ClassVar[str] = "framework"
    CATEGORY_ARROW: ClassVar[str] = "arrow"
    CATEGORY_PANDAS: ClassVar[str] = "pandas"
    CATEGORY_POLARS: ClassVar[str] = "polars"

    # Primitives
    NONE: int = 0
    UTF8_STRING: int = 1
    LATIN1_STRING: int = 2
    BYTES: int = 3
    UINT8: int = 4
    INT8: int = 5
    UINT16: int = 6
    INT16: int = 7
    UINT32: int = 8
    INT32: int = 9
    UINT64: int = 10
    INT64: int = 11
    FLOAT16: int = 12
    FLOAT32: int = 13
    FLOAT64: int = 14
    BOOL: int = 15
    DECIMAL: int = 16
    DATETIME: int = 17
    DATE: int = 18
    TIME: int = 19

    # Collections
    ARRAY: int = 100
    LIST: int = 101
    MAPPING: int = 102
    TUPLE: int = 103
    SET: int = 104
    GENERATOR: int = 105
    ITERATOR: int = 106

    LARGE_ARRAY: int = 160
    LARGE_LIST: int = 161
    LARGE_MAPPING: int = 162
    LARGE_TUPLE: int = 163
    LARGE_SET: int = 164
    LARGE_GENERATOR: int = 165
    LARGE_ITERATOR: int = 166

    # System
    MODULE: int = 200
    CLASS: int = 201
    FUNCTION: int = 202
    BASE_EXCEPTION: int = 203
    PATH: int = 204
    IO: int = 205
    PICKLE: int = 206
    DILL: int = 207
    CLOUDPICKLE: int = 208
    DATACLASS: int = 209

    # Arrow
    ARROW_TABLE: int = 400
    ARROW_RECORD_BATCH: int = 401
    ARROW_STREAM: int = 402
    ARROW_DATASET: int = 403
    ARROW_SCHEMA: int = 404
    ARROW_FIELD: int = 405
    ARROW_DATA_TYPE: int = 406
    ARROW_ARRAY: int = 407
    ARROW_CHUNKED_ARRAY: int = 408
    ARROW_SCALAR: int = 409
    ARROW_TENSOR: int = 410

    # Pandas
    PANDAS_DATAFRAME: int = 500
    PANDAS_SERIES: int = 501
    PANDAS_INDEX: int = 502

    # Polars
    POLARS_DATAFRAME: int = 600
    POLARS_SERIES: int = 601
    POLARS_LAZYFRAME: int = 602
    POLARS_EXPR: int = 603
    POLARS_SCHEMA: int = 604
    POLARS_DATATYPE: int = 605

    @classmethod
    def get_category(cls, tag: int) -> str:
        if 0 <= tag < 100:
            return cls.CATEGORY_PRIMITIVE
        if 100 <= tag < 200:
            return cls.CATEGORY_COLLECTION
        if 200 <= tag < 300:
            return cls.CATEGORY_SYSTEM
        if 300 <= tag < 400:
            return cls.CATEGORY_FRAMEWORK
        if 400 <= tag < 500:
            return cls.CATEGORY_ARROW
        if 500 <= tag < 600:
            return cls.CATEGORY_PANDAS
        if 600 <= tag < 700:
            return cls.CATEGORY_POLARS
        return cls.CATEGORY_UNKNOWN

    @classmethod
    def is_primitive(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_PRIMITIVE

    @classmethod
    def is_collection(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_COLLECTION

    @classmethod
    def is_system(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_SYSTEM

    @classmethod
    def is_framework(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_FRAMEWORK

    @classmethod
    def is_arrow(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_ARROW

    @classmethod
    def is_pandas(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_PANDAS

    @classmethod
    def is_polars(cls, tag: int) -> bool:
        return cls.get_category(tag) == cls.CATEGORY_POLARS

    @classmethod
    def get_name(cls, tag: int) -> str | None:
        for name, value in vars(cls).items():
            if name.startswith("_"):
                continue
            if not name.isupper():
                continue
            if isinstance(value, int) and value == tag:
                return name
        return None

    @classmethod
    def get_class(cls, tag: int) -> type["Serialized[object]"] | None:
        """
        Resolve the concrete Serialized subclass for a tag.

        Imports are routed by category first to reduce circular import risk.
        """
        existing = cls.CLASSES.get(tag)
        if existing is not None:
            return existing

        cat = cls.get_category(tag)

        if cat == cls.CATEGORY_PRIMITIVE:
            from yggdrasil.pickle.ser.primitives import PrimitiveSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        elif cat == cls.CATEGORY_COLLECTION:
            from yggdrasil.pickle.ser.collections import CollectionSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        elif cat == cls.CATEGORY_SYSTEM:
            from yggdrasil.pickle.ser.complexs import ComplexSerialized  # noqa: F401
            from yggdrasil.pickle.ser.ios import IOSerialized  # noqa: F401
            from yggdrasil.pickle.ser.paths import PathSerialized  # noqa: F401
            from yggdrasil.pickle.ser.pickles import PickleSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        elif cat == cls.CATEGORY_ARROW:
            from yggdrasil.pickle.ser.pyarrow import ArrowSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        elif cat == cls.CATEGORY_PANDAS:
            from yggdrasil.pickle.ser.pandas import PandasSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        elif cat == cls.CATEGORY_POLARS:
            from yggdrasil.pickle.ser.polars import PolarsSerialized  # noqa: F401
            existing = cls.CLASSES.get(tag)
        else:
            existing = None

        if existing is None:
            raise NotImplementedError(
                f"Tag {tag} is not registered with a Serialized class"
            )
        return existing

    @classmethod
    def resolve_class(cls, tag: int) -> type["Serialized[object]"] | None:
        return cls.get_class(tag)

    @classmethod
    def is_known(cls, tag: int) -> bool:
        name = cls.get_name(tag)
        if name is not None:
            return True
        return tag in cls.CLASSES

    @classmethod
    def register_class(
        cls,
        serialized_cls: type["Serialized[object]"],
        *,
        tag: int | None = None,
    ) -> None:
        if tag is None:
            tag = getattr(serialized_cls, "TAG", None)

        if isinstance(tag, int) and tag >= 0:
            cls.CLASSES[tag] = serialized_cls