from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

__all__ = ["Tags"]

if TYPE_CHECKING:
    from yggdrasil.pickle.ser.serialized import Serialized


class Tags:
    """
    Stable wire tags for serialized payload kinds.

    These integers are part of the binary protocol and must remain stable.

    Layout by reserved ranges
    -------------------------
    0..99    : primitive / logical scalars
    100..199 : collections
    200..299 : system / complex / runtime objects  (logging: 216–219)
    300..399 : framework-specific internal objects
    400..499 : pyarrow
    500..599 : pandas
    600..699 : polars
    700..799 : pyspark
    800..899 : databricks
    """

    # ------------------------------------------------------------------
    # runtime registries
    # ------------------------------------------------------------------

    CLASSES: ClassVar[dict[int, type["Serialized[object]"]]] = {}
    TYPES: ClassVar[dict[type, type["Serialized[object]"]]] = {}
    _IMPORTED_CATEGORIES: ClassVar[set[int]] = set()

    TAG_TO_NAME: ClassVar[dict[int, str]]

    # ------------------------------------------------------------------
    # category labels
    # ------------------------------------------------------------------

    CATEGORY_UNKNOWN: ClassVar[str] = "unknown"
    CATEGORY_PRIMITIVE: ClassVar[str] = "primitive"
    CATEGORY_COLLECTION: ClassVar[str] = "collection"
    CATEGORY_SYSTEM: ClassVar[str] = "system"
    CATEGORY_FRAMEWORK: ClassVar[str] = "framework"
    CATEGORY_ARROW: ClassVar[str] = "arrow"
    CATEGORY_PANDAS: ClassVar[str] = "pandas"
    CATEGORY_POLARS: ClassVar[str] = "polars"
    CATEGORY_PYSPARK: ClassVar[str] = "pyspark"
    CATEGORY_DATABRICKS: ClassVar[str] = "databricks"

    # ------------------------------------------------------------------
    # category ranges
    # ------------------------------------------------------------------

    PRIMITIVE_BASE: ClassVar[int] = 0
    COLLECTION_BASE: ClassVar[int] = 100
    SYSTEM_BASE: ClassVar[int] = 200
    FRAMEWORK_BASE: ClassVar[int] = 300
    ARROW_BASE: ClassVar[int] = 400
    PANDAS_BASE: ClassVar[int] = 500
    POLARS_BASE: ClassVar[int] = 600
    PYSPARK_BASE: ClassVar[int] = 700
    DATABRICKS_BASE: ClassVar[int] = 800
    CATEGORY_SIZE: ClassVar[int] = 100

    # ------------------------------------------------------------------
    # primitives / logical scalars
    # ------------------------------------------------------------------

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
    TIMEDELTA: int = 20
    TIMEZONE: int = 21
    UUID: int = 22
    COMPLEX: int = 23
    IPADDRESS: int = 24

    # ------------------------------------------------------------------
    # collections
    # ------------------------------------------------------------------

    ARRAY: int = 100
    LIST: int = 101
    MAPPING: int = 102
    MAPPING_PROXY: int = 103
    TUPLE: int = 104
    SET: int = 105
    GENERATOR: int = 106
    ITERATOR: int = 107
    FROZENSET: int = 108
    DEQUE: int = 109

    LARGE_ARRAY: int = 160
    LARGE_LIST: int = 161
    LARGE_MAPPING: int = 162
    LARGE_MAPPING_PROXY: int = 163
    LARGE_TUPLE: int = 164
    LARGE_SET: int = 165
    LARGE_GENERATOR: int = 166
    LARGE_ITERATOR: int = 167
    LARGE_FROZENSET: int = 168
    LARGE_DEQUE: int = 169

    # ------------------------------------------------------------------
    # system / complex
    # ------------------------------------------------------------------

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
    IO_BINARY = 210
    IO_TEXT = 211
    IO_BYTES_BUFFER = 212
    IO_STRING_BUFFER = 213
    URL: int = 214
    METHOD: int = 215
    LOGGING_LOGGER: int = 216
    LOGGING_HANDLER: int = 217
    LOGGING_FORMATTER: int = 218
    LOGGING_LOG_RECORD: int = 219
    GENERIC_OBJECT: int = 220
    RUNTIME_RESOURCE: int = 221
    PREPARED_REQUEST: int = 222
    RESPONSE: int = 223

    # ------------------------------------------------------------------
    # framework-specific internal objects
    # ------------------------------------------------------------------

    MEDIA_TYPE: int = 300
    MIME_TYPE: int = 301
    CODEC: int = 302
    YGG_FIELD: int = 303
    YGG_SCHEMA: int = 304

    # ------------------------------------------------------------------
    # arrow
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # pandas
    # ------------------------------------------------------------------

    PANDAS_DATAFRAME: int = 500
    PANDAS_SERIES: int = 501
    PANDAS_INDEX: int = 502

    # ------------------------------------------------------------------
    # polars
    # ------------------------------------------------------------------

    POLARS_DATAFRAME: int = 600
    POLARS_SERIES: int = 601
    POLARS_LAZYFRAME: int = 602
    POLARS_EXPR: int = 603
    POLARS_SCHEMA: int = 604
    POLARS_DATATYPE: int = 605

    # ------------------------------------------------------------------
    # pyspark
    # ------------------------------------------------------------------

    PYSPARK_DATAFRAME: int = 700
    PYSPARK_ROW: int = 701
    PYSPARK_SCHEMA: int = 702
    PYSPARK_DATATYPE: int = 703
    PYSPARK_COLUMN: int = 704
    PYSPARK_RDD: int = 705
    PYSPARK_SESSION: int = 706

    # ------------------------------------------------------------------
    # databricks
    # ------------------------------------------------------------------

    DATABRICKS_CONFIG: int = 800
    DATABRICKS_WORKSPACE_CLIENT: int = 801
    DATABRICKS_ACCOUNT_CLIENT: int = 802

    # ------------------------------------------------------------------
    # category helpers
    # ------------------------------------------------------------------

    @classmethod
    def _category_id(cls, tag: int) -> int:
        return -1 if tag < 0 else tag // cls.CATEGORY_SIZE

    @classmethod
    def get_category(cls, tag: int) -> str:
        cid = cls._category_id(tag)

        if cid == 0:
            return cls.CATEGORY_PRIMITIVE
        if cid == 1:
            return cls.CATEGORY_COLLECTION
        if cid == 2:
            return cls.CATEGORY_SYSTEM
        if cid == 3:
            return cls.CATEGORY_FRAMEWORK
        if cid == 4:
            return cls.CATEGORY_ARROW
        if cid == 5:
            return cls.CATEGORY_PANDAS
        if cid == 6:
            return cls.CATEGORY_POLARS
        if cid == 7:
            return cls.CATEGORY_PYSPARK
        if cid == 8:
            return cls.CATEGORY_DATABRICKS
        return cls.CATEGORY_UNKNOWN

    @classmethod
    def is_primitive(cls, tag: int) -> bool:
        return 0 <= tag < 100

    @classmethod
    def is_collection(cls, tag: int) -> bool:
        return 100 <= tag < 200

    @classmethod
    def is_system(cls, tag: int) -> bool:
        return 200 <= tag < 300

    @classmethod
    def is_framework(cls, tag: int) -> bool:
        return 300 <= tag < 400

    @classmethod
    def is_arrow(cls, tag: int) -> bool:
        return 400 <= tag < 500

    @classmethod
    def is_pandas(cls, tag: int) -> bool:
        return 500 <= tag < 600

    @classmethod
    def is_polars(cls, tag: int) -> bool:
        return 600 <= tag < 700

    @classmethod
    def is_pyspark(cls, tag: int) -> bool:
        return 700 <= tag < 800

    @classmethod
    def is_databricks(cls, tag: int) -> bool:
        return 800 <= tag < 900

    # ------------------------------------------------------------------
    # lookup helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls, tag: int) -> str | None:
        return cls.TAG_TO_NAME.get(tag)

    @classmethod
    def is_known(cls, tag: int) -> bool:
        return tag in cls.TAG_TO_NAME or tag in cls.CLASSES

    # ------------------------------------------------------------------
    # lazy import routing
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_category_imported(cls, tag: int) -> None:
        """
        Import serializer modules lazily by tag category.

        Idempotent: a class-level set tracks which category IDs have already
        been imported so that repeated calls (e.g., on every TYPES cache miss)
        pay only a single ``set.__contains__`` check rather than re-entering
        the import machinery each time.

        Parameters
        ----------
        tag:
            Any integer tag whose category should be imported.  The category
            is derived as ``tag // CATEGORY_SIZE`` (e.g., tag 101 → cid 1 →
            collections).  Callers may also pass a category base directly
            (``cid * CATEGORY_SIZE``); both forms are equivalent.
        """
        cid = cls._category_id(tag)

        if cid in cls._IMPORTED_CATEGORIES:
            return

        if cid == 0:
            from yggdrasil.pickle.ser.primitives import PrimitiveSerialized  # noqa: F401
            from yggdrasil.pickle.ser.logicals import LogicalSerialized  # noqa: F401
        elif cid == 1:
            from yggdrasil.pickle.ser.collections import CollectionSerialized  # noqa: F401
        elif cid == 2:
            from yggdrasil.pickle.ser.complexs import ComplexSerialized  # noqa: F401
            from yggdrasil.pickle.ser.ios import IOSerialized  # noqa: F401
            from yggdrasil.pickle.ser.pickles import PickleSerialized  # noqa: F401
            from yggdrasil.pickle.ser.logicals import PathSerialized  # noqa: F401
            from yggdrasil.pickle.ser.logging import LoggingSerialized  # noqa: F401
            from yggdrasil.pickle.ser.http_ import HttpSerialized  # noqa: F401
        elif cid == 3:
            from yggdrasil.pickle.ser.media import MediaTypeSerialized  # noqa: F401
            from yggdrasil.pickle.ser.data import DataSerialized  # noqa: F401
        elif cid == 4:
            from yggdrasil.pickle.ser.pyarrow import ArrowSerialized  # noqa: F401
        elif cid == 5:
            from yggdrasil.pickle.ser.pandas import PandasSerialized  # noqa: F401
        elif cid == 6:
            from yggdrasil.pickle.ser.polars import PolarsSerialized  # noqa: F401
        elif cid == 7:
            from yggdrasil.pickle.ser.pyspark import PySparkSerialized  # noqa: F401
        elif cid == 8:
            from yggdrasil.pickle.ser.databricks import DatabricksSerialized  # noqa: F401

        cls._IMPORTED_CATEGORIES.add(cid)

    # ------------------------------------------------------------------
    # class resolution
    # ------------------------------------------------------------------

    @classmethod
    def get_class(cls, tag: int) -> type["Serialized[object]"]:
        """
        Resolve the concrete Serialized subclass registered for a wire tag.
        """
        existing = cls.CLASSES.get(tag)
        if existing is not None:
            return existing

        cls._ensure_category_imported(tag)
        existing = cls.CLASSES.get(tag)

        if existing is None:
            raise NotImplementedError(
                f"Tag {tag} ({cls.get_name(tag) or 'UNKNOWN'}) is not registered "
                f"with a Serialized class"
            )
        return existing

    @classmethod
    def get_class_from_type(cls, pytype: type) -> type["Serialized[object]"] | None:
        """
        Resolve the serializer class for a Python runtime type.

        Fast path:
            direct TYPES lookup

        Slow path:
            force-load all core categories, then retry.
            This is safe because Python's import machinery is idempotent —
            already-imported modules are served from sys.modules instantly.
            The old guard ``if not cls.TYPES`` was broken: if collections.py
            was imported first it populated TYPES with 8 entries, so the guard
            was never triggered and primitives (including NoneType) were never
            registered, causing None to fall through to PickleSerialized.
        """
        existing = cls.TYPES.get(pytype)
        if existing is not None:
            return existing

        for cid in (0, 1, 2, 4, 8):
            cls._ensure_category_imported(cid * cls.CATEGORY_SIZE)

        return cls.TYPES.get(pytype)

    @classmethod
    def resolve_class(cls, tag: int) -> type["Serialized[object]"]:
        return cls.get_class(tag)

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    @classmethod
    def register_class(
        cls,
        serialized_cls: type["Serialized[object]"],
        *,
        tag: int | None = None,
        pytype: type | None = None,
    ) -> None:
        """
        Register a Serialized subclass by wire tag and/or Python type.
        """
        if tag is None:
            tag = getattr(serialized_cls, "TAG", None)

        if isinstance(tag, int) and tag >= 0:
            cls.CLASSES[tag] = serialized_cls

        if pytype is not None:
            existing = cls.TYPES.get(pytype)

            if existing is None:
                cls.TYPES[pytype] = serialized_cls


def _build_tag_to_name() -> dict[int, str]:
    out: dict[int, str] = {}
    for name, value in vars(Tags).items():
        if name.startswith("_"):
            continue
        if not name.isupper():
            continue
        if isinstance(value, int):
            out[value] = name
    return out


Tags.TAG_TO_NAME = _build_tag_to_name()