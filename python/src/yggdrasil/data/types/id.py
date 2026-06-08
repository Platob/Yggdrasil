from __future__ import annotations

from enum import IntEnum

__all__ = [
    "DataTypeId",
]


class DataTypeId(IntEnum):
    """Stable numeric tag for every :class:`DataType` subclass.

    The integer values are clustered by family so a plain
    ``sorted(DataTypeId)`` walk visits related types together — and
    so range-based property checks (``is_integer`` / ``is_temporal``
    / …) read as one inequality rather than a hand-maintained set.
    Placeholder gaps inside each family leave room for future
    additions (e.g. ``INT128``, ``BFLOAT16``, ``TIMETZ``) without
    re-numbering anything else.

    Layout:

    * ``0–9``    — sentinel / opaque (``OBJECT``, ``NULL``).
    * ``10–19``  — boolean.
    * ``20–39``  — integer family (abstract ``INTEGER`` + sized
      signed ``INT8…INT64`` + unsigned ``UINT8…UINT64``).
    * ``40–49``  — floating-point family (abstract ``FLOAT`` + sized
      ``FLOAT16…FLOAT64``).
    * ``50–59``  — decimal.
    * ``60–69``  — temporal (``DATE`` / ``TIME`` / ``TIMESTAMP`` /
      ``DURATION``).
    * ``70–79``  — bytes (``BINARY`` / ``STRING``).
    * ``80–99``  — extensions (``DICTIONARY`` / ``ENUM`` /
      ``STR_ENUM`` / ``INT_ENUM`` / ``UNION`` / ``SJSON`` / ``BJSON``).
    * ``100+``   — nested (``ARRAY`` / ``MAP`` / ``STRUCT``).
    """

    # ── Sentinel / opaque ────────────────────────────────────────────────
    OBJECT = 0  # variant / unknown — caller treats it as a black box
    NULL = 1

    # ── Boolean ──────────────────────────────────────────────────────────
    BOOL = 10

    # ── Integer family ───────────────────────────────────────────────────
    # Abstract ``INTEGER`` first, then signed widths ascending, then
    # unsigned widths ascending. The gap between signed (24) and
    # unsigned (26) leaves room for ``INT128`` / ``UINT128`` later.
    INTEGER = 20
    INT8 = 21
    INT16 = 22
    INT32 = 23
    INT64 = 24
    UINT8 = 26
    UINT16 = 27
    UINT32 = 28
    UINT64 = 29

    # ── Floating-point family ────────────────────────────────────────────
    # Abstract first, then sized widths ascending. ``FLOAT8`` is the
    # 1-byte tag used by ML frameworks (E4M3 / E5M2 FP8 variants);
    # we don't model the format choice here — the storage width is
    # what threads through Arrow / Polars / Spark.
    FLOAT = 40
    FLOAT8 = 41
    FLOAT16 = 42
    FLOAT32 = 43
    FLOAT64 = 44

    # ── Decimal ──────────────────────────────────────────────────────────
    DECIMAL = 50

    # ── Temporal family ──────────────────────────────────────────────────
    # Sorted so DATE → TIME → TIMESTAMP captures coarse-to-fine
    # wall-clock precision; DURATION trails because it's an interval,
    # not an instant.
    DATE = 60
    TIME = 61
    TIMESTAMP = 62
    DURATION = 63

    # ── Bytes ────────────────────────────────────────────────────────────
    BINARY = 70
    STRING = 71

    # ── Extensions ───────────────────────────────────────────────────────
    DICTIONARY = 80
    ENUM = 82
    UNION = 83
    SJSON = 84  # JSON encoded as a UTF-8 string (text JSON)
    BJSON = 85  # JSON encoded as bytes (binary / packed JSON)
    STR_ENUM = 86  # str-valued enum (subclass of ENUM)
    INT_ENUM = 87  # int-valued enum (subclass of ENUM)

    # ── Nested ───────────────────────────────────────────────────────────
    ARRAY = 100
    MAP = 101
    STRUCT = 102

    # ─────────────────────────────────────────────────────────────────────
    # Range-based predicates — match the family layout above.
    # ─────────────────────────────────────────────────────────────────────

    @property
    def is_any_or_null(self) -> bool:
        return self.value <= 1

    @property
    def is_boolean(self) -> bool:
        return 10 <= self.value < 20

    @property
    def is_integer(self) -> bool:
        return 20 <= self.value < 40

    @property
    def is_signed_integer(self) -> bool:
        return 21 <= self.value <= 24

    @property
    def is_unsigned_integer(self) -> bool:
        return 26 <= self.value <= 29

    @property
    def is_floating_point(self) -> bool:
        return 40 <= self.value < 50

    @property
    def is_decimal(self) -> bool:
        return 50 <= self.value < 60

    @property
    def is_numeric(self) -> bool:
        return 20 <= self.value < 60

    @property
    def is_temporal(self) -> bool:
        return 60 <= self.value < 70

    @property
    def is_bytes_like(self) -> bool:
        return 70 <= self.value < 80

    @property
    def is_extension(self) -> bool:
        return 80 <= self.value < 100

    @property
    def is_json(self) -> bool:
        # Storage-pinned JSON variants — used by the cross-engine cast
        # paths to recognize a "JSON column" without caring whether the
        # bytes are UTF-8 text or a binary packing.
        return self.value in (84, 85)

    @property
    def is_dictionary_like(self) -> bool:
        # ``DICTIONARY`` / ``ENUM`` / ``STR_ENUM`` / ``INT_ENUM`` all
        # share the dict-encoded categorical contract (fixed value-set
        # with int32 indices on the Arrow side, ``pl.Enum`` on Polars).
        # Used by the cast pipeline to short-circuit re-encoding when
        # the source already carries the right shape.
        return self.value in (80, 82, 86, 87)

    @property
    def is_nested(self) -> bool:
        return self.value >= 100

    @property
    def is_scalar(self) -> bool:
        # Anything non-nested + non-sentinel — covers booleans,
        # numerics, temporal, bytes, and the extension wrappers that
        # ultimately resolve to a scalar payload.
        return 0 < self.value < 100
