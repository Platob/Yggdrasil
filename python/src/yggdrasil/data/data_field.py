from __future__ import annotations

import dataclasses
import itertools
import json
import os
import pathlib
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Optional,
    Iterator,
    Generator,
    AnyStr,
    overload,
)

import pyarrow as pa

import yggdrasil.pickle.json as json_module
from yggdrasil.data.base_meta import (
    BaseChildrenFields,
    _merge_metadata_and_tags,
    _normalize_metadata,
    _to_bytes,
)
from yggdrasil.data.constants import (
    ALIAS_KEY,
    DEFAULT_VALUE_KEY,
    DEFAULT_FIELD_NAME,
    MEDIA_TYPE_METADATA_KEY,
    POSITION_KEY,
    TAG_PREFIX,
)
from yggdrasil.data.enums import Mode
from yggdrasil.data.types.id import DataTypeId
from yggdrasil.data.types.parser import ParsedDataType
from yggdrasil.exceptions import CastError
from yggdrasil.lazy_imports import pandas_module, polars_module, spark_sql_module
from yggdrasil.lazy_imports import path_class, schema_class, spark_column_classes, spark_dataframe_classes
from yggdrasil.pickle.serde import ObjectSerde
from .cast.registry import register_converter
from .data_utils import get_cast_options_class, safe_constraint_name
from .types import NullType, ObjectType
from .types.base import DataType
from .types.nested import StructType

if TYPE_CHECKING:
    import pandas as pd
    import polars
    import pyspark.sql as ps
    import pyspark.sql.types as pst
    from yggdrasil.data.options import CastOptions
    from yggdrasil.data.schema import Schema


__all__ = [
    "Field",
    "field",
    "_normalize_metadata",
    "_to_bytes",
    "_merge_metadata_and_tags",
]


# ======================================================================
# Module-level constants & private helpers
# ======================================================================

_TYPE_JSON_METADATA_KEY = b"type_json"
#: ``str`` form of the ``type_json`` key. Spark stores metadata
#: ``dict[str, Any]`` (Java ``Metadata`` view), so engine-side
#: lookups use this key; the ``bytes`` form is what Arrow uses.
#: Pre-decoded once at import so ``from_spark_field`` doesn't pay a
#: ``.decode("utf-8")`` per call.
_TYPE_JSON_METADATA_KEY_STR = _TYPE_JSON_METADATA_KEY.decode("utf-8")
_NONE_TYPE = type(None)


def _attach_type_json_metadata(
    arrow_type: pa.DataType,
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes]:
    out = dict(metadata or {})

    dtype = DataType.from_arrow_type(arrow_type)
    if isinstance(dtype, type) and issubclass(dtype, DataType):
        dtype = dtype()

    out[_TYPE_JSON_METADATA_KEY] = json_module.dumps(
        dtype.to_dict(),
        to_bytes=True,
        safe=False,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return out


def _strip_internal_metadata(
    metadata: dict[bytes, bytes] | dict[str, str] | None,
) -> dict | None:
    """Drop yggdrasil-internal metadata keys.

    Currently only :data:`_TYPE_JSON_METADATA_KEY` (the dtype JSON
    round-trip blob written into engine metadata for the lossy paths,
    e.g. Spark Map/Array). Accepts both bytes-keyed (Arrow) and
    string-keyed (Spark) metadata maps.
    """
    if not metadata:
        return None

    out = {
        key: value
        for key, value in metadata.items()
        if key != _TYPE_JSON_METADATA_KEY and key != _TYPE_JSON_METADATA_KEY_STR
    }
    return out or None


def _render_spark_column_sql(column: "Any") -> "str | None":
    """Pull the SQL expression string off a ``pyspark.sql.Column``.

    Three probes, in order:

    1. ``column._jc.toString()`` — classic Spark's JVM Column. Spark
       Connect's ``Column`` raises
       ``PySparkAttributeError(JVM_ATTRIBUTE_NOT_SUPPORTED)`` on
       ``_jc`` access, so this is wrapped in ``try/except``.
    2. ``column._expr.__repr__()`` — Spark Connect's ``Column`` stores
       the expression at ``_expr``; ``Column.__repr__`` itself is
       just ``"Column<'%s'>" % self._expr.__repr__()``.
    3. Parse ``repr(column)`` and strip the ``Column<'…'>`` wrapper
       — last-resort for any future PySpark whose slot name changes.

    Returns ``None`` when none of the probes yields a non-empty
    string; the caller raises with a usable message.
    """
    try:
        jc = column._jc
    except Exception:
        jc = None
    if jc is not None:
        try:
            rendered = jc.toString()
        except Exception:
            rendered = None
        if rendered:
            return str(rendered)

    expr = getattr(column, "_expr", None)
    if expr is not None:
        try:
            rendered = expr.__repr__()
        except Exception:
            rendered = None
        if rendered:
            return str(rendered)

    try:
        repr_text = repr(column)
    except Exception:
        return None
    # Strip ``Column<'<sql>'>`` to recover ``<sql>``. Match greedily on the
    # inner ``'…'>`` close so nested quotes inside the SQL don't truncate.
    if repr_text.startswith("Column<'") and repr_text.endswith("'>"):
        return repr_text[len("Column<'"):-len("'>")]
    return repr_text or None


def _parse_spark_column_sql(sql: str) -> "tuple[str, DataType | None]":
    """Best-effort name + dtype extract from a Spark Column's SQL form.

    Spark renders Columns as SQL strings via ``_jc.toString()`` — the
    same surface ``Column.__repr__`` uses (``Column<'<sql>'>``):

    * ``id`` — bare leaf reference. Name = ``"id"``, dtype = ``None``
      (caller picks a fallback).
    * ``CAST(id AS STRING)`` — name = ``"id"``, dtype = ``StringType``.
    * ``id AS user_id`` — name = ``"user_id"``, dtype recurses on
      ``id`` (so a cast inside the alias keeps its dtype).
    * ``CAST(id AS DECIMAL(10,2)) AS price`` — name = ``"price"``,
      dtype = ``DecimalType(10, 2)``.
    * Anything else (arithmetic, function calls, complex subqueries)
      falls back to ``(<full sql>, None)`` so the caller can decide
      whether to keep the SQL as a name or substitute the default.

    Public-shape-stable: this is a string-grammar helper, so we
    operate on the rendered token stream rather than the JVM
    expression tree, which has moved between PySpark releases.
    """
    text = sql.strip()
    if not text:
        return "", None

    # Strip outer backticks Spark wraps around qualified identifiers.
    if len(text) >= 2 and text[0] == "`" and text[-1] == "`" and " " not in text:
        return text[1:-1], None

    # Alias form: ``<expr> AS <alias>``. The alias is always the
    # tail; the inner expression keeps its own dtype.
    # Split on the rightmost top-level " AS " so we don't trip on
    # ``CAST(x AS T)`` nested inside the expression.
    alias_split = _split_top_level_as(text)
    if alias_split is not None:
        inner_sql, alias = alias_split
        # Alias token may itself be backtick-quoted.
        alias = alias.strip()
        if len(alias) >= 2 and alias[0] == "`" and alias[-1] == "`":
            alias = alias[1:-1]
        _, inner_dtype = _parse_spark_column_sql(inner_sql)
        return alias, inner_dtype

    # Cast form: ``CAST(<expr> AS <dtype>)`` — the dtype token is a
    # Spark DDL fragment our parser already understands.
    upper = text.upper()
    if upper.startswith("CAST(") and text.endswith(")"):
        # Walk to the matching closing paren so nested CAST inside
        # CAST doesn't trip us. ``text[5:-1]`` is the body.
        body = text[5:-1]
        cast_split = _split_top_level_as(body)
        if cast_split is not None:
            inner_sql, dtype_token = cast_split
            try:
                dtype = DataType.from_str(dtype_token.strip())
            except Exception:
                dtype = None
            # Name follows the inner leaf, but ``CAST(...)`` has no
            # natural name slot — return the inner expression's
            # parsed name so chains like ``df["x"].cast("string")``
            # land on ``name="x"``.
            inner_name, _ = _parse_spark_column_sql(inner_sql)
            return inner_name or text, dtype

    # Plain identifier — name is the text itself.
    return text, None


def _split_top_level_as(text: str) -> "tuple[str, str] | None":
    """Split *text* on the rightmost top-level ``" AS "`` token.

    "Top-level" means not nested inside parentheses or backticks.
    Returns ``(left, right)`` on a match, ``None`` otherwise.
    Case-insensitive on the ``AS`` keyword to match Spark's
    ``simpleString`` / ``toString`` rendering, which uses uppercase.
    """
    upper = text.upper()
    n = len(text)
    depth = 0
    in_backtick = False
    # Walk right-to-left so chained casts split on the outermost
    # ``AS`` (``CAST(CAST(x AS T1) AS T2) AS y``). Depth must track
    # every paren on the way back, so we start at the end and only
    # attempt the 4-char ``" AS "`` match once there's room.
    for i in range(n - 1, -1, -1):
        ch = text[i]
        if ch == "`":
            in_backtick = not in_backtick
            continue
        if in_backtick:
            continue
        if ch == ")":
            depth += 1
        elif ch == "(":
            depth -= 1
        elif (
            depth == 0
            and i + 4 <= n
            and upper[i:i + 4] == " AS "
            and (i == 0 or text[i - 1] != " ")
        ):
            return text[:i].strip(), text[i + 4:].strip()
    return None


def _safe_issubclass(obj: object, class_or_tuple: object) -> bool:
    return isinstance(obj, type) and issubclass(obj, class_or_tuple)


# The canonical Python type-hint resolution helpers live on
# :class:`DataType` — :meth:`DataType.strip_annotated`,
# :meth:`DataType.unwrap_newtype`, :meth:`DataType.normalize_hint`,
# :meth:`DataType.unwrap_nullable_hint`. The module-level wrappers
# below stay as thin forwarders so existing in-module callers
# (``Field.from_pytype`` and friends) don't have to spell out the
# class prefix on every line, but new code should reach for the
# ``DataType`` classmethods directly.


def _strip_annotated(hint: object) -> object:
    return DataType.strip_annotated(hint)


def _unwrap_newtype(hint: object) -> object:
    return DataType.unwrap_newtype(hint)


def _unwrap_nullable_hint(hint: Any) -> tuple[Any, bool]:
    return DataType.unwrap_nullable_hint(hint)

    return hint, False


def _is_typed_dict_type(hint: object) -> bool:
    return (
        isinstance(hint, type)
        and issubclass(hint, dict)
        and hasattr(hint, "__annotations__")
        and hasattr(hint, "__total__")
    )


def _default_name(value: Any) -> str:
    if isinstance(value, type):
        return getattr(value, "__name__", DEFAULT_FIELD_NAME)
    return getattr(type(value), "__name__", DEFAULT_FIELD_NAME)


def _strip_matching_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"', "`", "´"}:
        return text[1:-1]
    return text


def _split_field_shorthand(value: str) -> tuple[str, str]:
    text = value.strip()
    quote: str | None = None
    depth_paren = 0
    depth_brack = 0
    depth_angle = 0

    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue

        if char in {"'", '"', "`", "´"}:
            quote = char
            continue

        if char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "[":
            depth_brack += 1
        elif char == "]":
            depth_brack -= 1
        elif char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == ":" and depth_paren == 0 and depth_brack == 0 and depth_angle == 0:
            left = text[:index].strip()
            right = text[index + 1 :].strip()
            if not left or not right:
                break
            return left, right

    return DEFAULT_FIELD_NAME, text


def _parse_field_name_token(value: str) -> tuple[str, bool | None]:
    token = _strip_matching_quotes(value.strip())

    if token.endswith("?"):
        return token[:-1].strip(), True
    if token.endswith("!"):
        return token[:-1].strip(), False
    return token, None


def _normalize_inner_fields(value: Any) -> list["Field"]:
    """Coerce *value* into an ordered list of :class:`Field` children.

    Accepts ``None`` (empty), a single ``Field``, a mapping
    (``{name: field}`` — names are forced to match), or any iterable
    of field-like inputs.
    """
    if value is None:
        return []
    if isinstance(value, Field):
        return [value]
    if isinstance(value, Mapping):
        out: list[Field] = []
        for key, raw in value.items():
            f = Field.from_any(raw)
            if f.name != key:
                f = f.copy(name=key)
            out.append(f)
        return out
    return [Field.from_any(_) for _ in value]


def _peel_name_nullable(metadata: Any) -> tuple[Any, str | None, bool | None]:
    """Pop ``b"name"`` / ``b"nullable"`` out of *metadata* if present.

    Schema-shaped Fields stash ``name`` / ``nullable`` as first-class
    attributes, but legacy callers commonly thread them via the
    metadata dict. Strip the keys when they exist so they don't appear
    twice; return *metadata* unchanged when neither key is present.
    """
    if not isinstance(metadata, Mapping):
        return metadata, None, None
    if (
        b"name" not in metadata
        and "name" not in metadata
        and b"nullable" not in metadata
        and "nullable" not in metadata
    ):
        return metadata, None, None

    cleaned = dict(metadata)
    raw_name = cleaned.pop(b"name", None)
    if raw_name is None:
        raw_name = cleaned.pop("name", None)
    name: str | None = None
    if raw_name is not None:
        name = (
            raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        )

    raw_nullable = cleaned.pop(b"nullable", None)
    if raw_nullable is None:
        raw_nullable = cleaned.pop("nullable", None)
    nullable: bool | None = None
    if raw_nullable is not None:
        if isinstance(raw_nullable, bytes):
            nullable = raw_nullable.startswith(b"t") or raw_nullable.startswith(b"T")
        elif isinstance(raw_nullable, bool):
            nullable = raw_nullable
        else:
            nullable = str(raw_nullable).lower().startswith("t")

    return cleaned, name, nullable


# ======================================================================
# Public factory — `field(...)` shorthand constructor
# ======================================================================


def field(
    name: str,
    dtype: DataType | type[DataType] | pa.DataType | None = None,
    *,
    arrow_type: pa.DataType | None = None,
    nullable: bool = True,
    metadata: dict[bytes | str, bytes | str | object] | None = None,
    tags: dict[bytes | str, bytes | str | object] | None = None,
    default: Any = None,
) -> "Field":
    return Field(
        name=name,
        dtype=DataType.from_any(arrow_type if dtype is None else dtype),
        nullable=nullable,
        metadata=_normalize_metadata(metadata, tags=tags, default_value=default),
    )


@dataclass(frozen=True, slots=True, init=False, repr=False, eq=False)
class Field(BaseChildrenFields):
    name: str
    dtype: DataType
    nullable: bool = True
    metadata: dict[bytes, bytes] | None = None
    # Back-pointer to the field this one is nested under (struct
    # member, map key/value, list item, schema child). ``None`` for
    # top-level fields. Used by the cache layer to bubble
    # invalidations up the tree when a child mutates.
    parent: "Field | None" = dataclasses.field(default=None, repr=False, compare=False)
    # Lazily-populated engine-flavoured projections. Populated on
    # first access via :meth:`to_arrow_field` / :meth:`to_arrow_schema`
    # / :meth:`to_polars_field` / :meth:`to_pyspark_field`; cleared by
    # :meth:`_invalidate_cache` on any structural mutation (and by
    # :attr:`parent` cascades).
    _arrow_field: "pa.Field | None" = dataclasses.field(
        default=None, repr=False, compare=False
    )
    _arrow_schema: "pa.Schema | None" = dataclasses.field(
        default=None, repr=False, compare=False
    )
    _polars_field: Any = dataclasses.field(default=None, repr=False, compare=False)
    _polars_schema: Any = dataclasses.field(default=None, repr=False, compare=False)
    _spark_field: Any = dataclasses.field(default=None, repr=False, compare=False)
    _spark_schema: Any = dataclasses.field(default=None, repr=False, compare=False)
    # Cached ``{name: index}`` view of ``children`` for O(1)
    # lookup in :meth:`field_by`. Built on demand by
    # :meth:`_ensure_field_name_map`; cleared whenever the dtype
    # changes (which is the only way the children change).
    _field_name_map: dict[str, int] | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    _field_name_fold_map: dict[str, int] | None = dataclasses.field(
        default=None, repr=False, compare=False
    )
    # Cached :meth:`__hash__` result. ``Field`` is widely used as a
    # dict key / set member (``CastOptions`` target, schema rebuilds,
    # field-name lookup); the underlying hash walks ``self.dtype``
    # plus a sorted view of ``self.metadata`` so the saved cost on
    # repeat hashing of the same instance is significant — especially
    # for struct-shaped fields where the dtype hash recurses into
    # children.
    _hash: int | None = dataclasses.field(default=None, repr=False, compare=False)

    def __repr__(self):
        return self.pretty_format()

    def __str__(self):
        return self.pretty_format()

    def __eq__(self, other: Any) -> bool:
        if other is self:
            return True
        if not isinstance(other, Field):
            return NotImplemented
        # Cheap-first ordering: name (str eq) → nullable (bool eq) →
        # metadata (small ``bytes → bytes`` dict, length-then-pair eq)
        # → dtype (recursive structural compare for nested types).
        # Putting metadata before dtype short-circuits the common
        # "different comment / different tag / different child
        # metadata" case before paying for the deep dtype walk — a
        # 5-column struct dtype eq is ~1.2 us, a metadata-dict eq is
        # ~30 ns. Singleton ``dtype is other.dtype`` covers the
        # steady-state primitive case (one process-wide
        # ``Int64Type()`` etc.).
        return (
            self.name == other.name
            and self.nullable == other.nullable
            and self.metadata == other.metadata
            and (self.dtype is other.dtype or self.dtype == other.dtype)
        )

    def __hash__(self) -> int:
        cached = self._hash
        if cached is not None:
            return cached
        meta_key = tuple(sorted(self.metadata.items())) if self.metadata else None
        h = hash((self.name, self.dtype, self.nullable, meta_key))
        object.__setattr__(self, "_hash", h)
        return h

    @classmethod
    def default(
        cls,
        name: str = "",
        dtype: DataType = ObjectType(),
        nullable: bool = True,
        metadata: dict[bytes, bytes] | None = None,
        tags: dict[bytes, bytes] | None = None,
        default: Any = None,
    ):
        """Build a default-typed Field (``ObjectType()`` unless overridden).

        Convenience constructor for the "I just have a name" path —
        callers passing a plain string into APIs that expect a
        :class:`Field` (e.g. ``CastOptions(match_by=["id"])``) land
        here. The instance-side ``default`` accessor was renamed to
        :attr:`default_value` so this name was free for the
        constructor.
        """
        return cls(
            name=name,
            dtype=dtype,
            nullable=nullable,
            metadata=_normalize_metadata(metadata, tags=tags, default_value=default),
        )

    # Pre-computed full metadata keys for the tag flags below
    # (``TAG_PREFIX + b"<name>"``). The tag accessors are some of the
    # hottest properties on ``Field`` — every ``fields`` filter walks
    # ``constraint_key`` per child, every ``primary_keys`` /
    # ``foreign_keys`` projection walks the matching flag — so building
    # the lookup key once and reading ``self.metadata`` directly saves
    # the per-call ``_to_bytes`` + bytes concat that the generic
    # ``_tag_flag`` path would do.
    _TAG_KEY_PARTITION_BY: ClassVar[bytes] = TAG_PREFIX + b"partition_by"
    _TAG_KEY_CLUSTER_BY: ClassVar[bytes] = TAG_PREFIX + b"cluster_by"
    _TAG_KEY_PRIMARY_KEY: ClassVar[bytes] = TAG_PREFIX + b"primary_key"
    _TAG_KEY_FOREIGN_KEY: ClassVar[bytes] = TAG_PREFIX + b"foreign_key"
    _TAG_KEY_CONSTRAINT_KEY: ClassVar[bytes] = TAG_PREFIX + b"constraint_key"
    _TAG_KEY_SORTED: ClassVar[bytes] = TAG_PREFIX + b"sorted"
    _TAG_KEY_INDEX_KEY: ClassVar[bytes] = TAG_PREFIX + b"index_key"
    _TAG_KEY_INDEX_KEY_LEVEL: ClassVar[bytes] = TAG_PREFIX + b"index_key_level"

    # Tag-flag → short token shown in pretty_format. Order is the
    # display order in the bracketed marker group; only flags whose
    # ``_tag_flag`` is truthy appear, so the output stays scannable
    # for fields with no special role.
    _PRETTY_TAG_FLAGS: ClassVar[tuple[tuple[bytes, str], ...]] = (
        (b"primary_key", "PK"),
        (b"foreign_key", "FK"),
        (b"constraint_key", "CK"),
        (b"partition_by", "partition"),
        (b"cluster_by", "cluster"),
        (b"sorted", "sorted"),
        (b"index_key", "IK"),
    )
    # Tag names handled by ``_PRETTY_TAG_FLAGS``. Anything stored under
    # ``TAG_PREFIX + <name>`` that isn't in this set is treated as a
    # caller-defined tag and rendered as ``name=value`` (or just
    # ``name`` for boolean-truthy values) in the marker group, so the
    # repr no longer silently drops custom metadata like
    # ``tags={"unit": "meters"}``.
    _PRETTY_WELL_KNOWN_TAGS: ClassVar[frozenset[bytes]] = frozenset(
        name for name, _ in _PRETTY_TAG_FLAGS
    )

    def _pretty_markers(self) -> str:
        """Bracketed marker group for :meth:`pretty_format`.

        Surfaces the schema-shaping tags a reader most often cares
        about (primary / foreign / constraint key, partition,
        cluster, sorted), any caller-defined tags stored under
        :data:`TAG_PREFIX`, and the default value if one is set.
        Returns an empty string when nothing is set so the common
        case stays uncluttered.
        """
        tokens: list[str] = [
            label for tag, label in self._PRETTY_TAG_FLAGS if self._tag_flag(tag)
        ]
        # Surface caller-defined tags too — these were previously
        # silent (only the well-known flag set rendered). Custom tag
        # values are stored as bytes; booleans round-trip as
        # ``b"True"``/``b"False"`` so render them as a bare flag, and
        # decode everything else as ``name=value``.
        md = self.metadata
        if md:
            well_known = self._PRETTY_WELL_KNOWN_TAGS
            prefix_len = len(TAG_PREFIX)
            extras: list[tuple[bytes, bytes]] = []
            for key, value in md.items():
                if not key.startswith(TAG_PREFIX):
                    continue
                tag_name = key[prefix_len:]
                if tag_name in well_known:
                    continue
                extras.append((tag_name, value))
            # Deterministic order so the repr is stable across runs.
            extras.sort(key=lambda kv: kv[0])
            for tag_name, value in extras:
                name_str = tag_name.decode("utf-8", errors="replace")
                if value in (b"True", b"true"):
                    tokens.append(name_str)
                    continue
                if value in (b"False", b"false", b""):
                    # Falsey custom tag — skip; matches how
                    # ``_tag_flag`` reads a bool tag.
                    continue
                value_str = value.decode("utf-8", errors="replace")
                # Mirror ``f"x={value!r}"`` shape but on the decoded
                # string so multi-byte text reads cleanly.
                tokens.append(f"{name_str}={value_str!r}")
        if self.has_default:
            try:
                tokens.append(f"default={self.default_value!r}")
            except Exception:
                # Default decoding can fail for malformed metadata —
                # don't let a bad default break repr.
                tokens.append("default=<unparseable>")
        return f" [{', '.join(tokens)}]" if tokens else ""

    def pretty_format(self, indent: int = 2, level: int = 0) -> str:
        """Pretty-print this field with the header on one line and the dtype below.

        Layout is uniform across flat and nested dtypes — every field
        renders as a single ``field: 'name' <dtype>{markers}`` header
        line, with nested dtypes walking their inner fields inline at
        ``level + 1`` so the tree reads as a flat list of consistent
        rows::

            field: 'row' struct
              field: 'id' int64 not null [PK]
              field: 'name' string
              field: 'inner' struct
                field: 'age' int64
                field: 'email' string

        ``indent`` is the per-level step in spaces; ``level`` is the
        current depth. The header carries the dtype kind (``struct`` /
        ``list`` / ``map`` for nested, the primitive pretty-format for
        flat), the ``not null`` marker, the bracketed marker group
        (primary / foreign / constraint key, partition / cluster /
        sorted, any caller-defined tags, default value), and the
        comment.

        Map dtypes flatten the synthetic ``entry`` struct into
        ``field: 'key' …`` / ``field: 'value' …`` lines so the
        key / value framing reads at the same level as a struct's
        own children rather than under an artificial wrapper.

        Examples::

            >>> print(field("id", "int64", nullable=False,
            ...             tags={"primary_key": True}).pretty_format())
            field: 'id' int64 not null [PK]

            >>> print(field("date", "date32",
            ...             tags={"partition_by": True}).pretty_format())
            field: 'date' date32 [partition]

            >>> print(field("user", StructType.from_fields([
            ...     field("id", "int64"),
            ...     field("email", "string"),
            ... ])).pretty_format())
            field: 'user' struct
              field: 'id' int64
              field: 'email' string
        """
        pad = " " * (indent * level)

        suffix = ""
        if not self.nullable:
            suffix += " not null"

        suffix += self._pretty_markers()

        comment = self.comment
        if comment:
            suffix += f" {comment!r}"

        if self.type_id.is_nested:
            dtype_kind = self._nested_pretty_kind()
            header = f"{pad}field: {self.name!r} {dtype_kind}{suffix}"
            children = self._nested_pretty_children()
            if not children:
                return header
            body = "\n".join(
                child.pretty_format(indent=indent, level=level + 1)
                for child in children
            )
            return f"{header}\n{body}"

        dtype_str = self.dtype.pretty_format()
        return f"{pad}field: {self.name!r} {dtype_str}{suffix}"

    def _nested_pretty_kind(self) -> str:
        """Short dtype-kind token displayed on the header line for nested fields.

        Returns ``"struct"`` / ``"list"`` / ``"map"`` for the three
        canonical nested shapes, falling back to the dtype's
        ``type_id.name.lower()`` for any future nested addition.
        """
        tid = self.type_id
        if tid is DataTypeId.STRUCT:
            return "struct"
        if tid is DataTypeId.ARRAY:
            return "list"
        if tid is DataTypeId.MAP:
            return "map"
        return tid.name.lower()

    def _nested_pretty_children(self) -> "Sequence[Field]":
        """Children to walk under the header line for nested pretty_format.

        Mirrors :attr:`dtype.children` except for :class:`MapType`,
        where the synthetic ``entry`` struct gets flattened into the
        ``key_field`` and ``value_field`` pair so the rendered tree
        treats key / value as siblings of a regular struct's columns
        instead of one extra layer of nesting.
        """
        dtype = self.dtype
        tid = dtype.type_id
        if tid is DataTypeId.MAP:
            return (dtype.key_field, dtype.value_field)
        return dtype.children

    # ==================================================================
    # Dunder / identity
    # ==================================================================

    def __new__(cls, *args: Any, **kwargs: Any):
        """Allocate the Field instance, redirecting struct dtypes to :class:`Schema`.

        ``Field(name=..., dtype=struct_t)`` returns a :class:`Schema`
        — a struct-shaped field IS a schema, so callers that hand us
        a :class:`StructType` get the schema-flavoured surface
        (header repr, ``inner_fields`` mapping, struct-aware merges)
        without having to know to call :class:`Schema` instead. Only
        plain ``Field`` calls redirect; explicit ``Schema(...)`` and
        any other subclass go through normal allocation (so
        ``Schema(...)`` keeps its own constructor shape).
        """
        if cls is not Field:
            return object.__new__(cls)
        # Pull dtype out of either positional or kwargs to match
        # Field.__init__'s ``(name, dtype, ...)`` signature.
        dtype = kwargs.get("dtype")
        if dtype is None and len(args) >= 2:
            dtype = args[1]
        if dtype is None:
            return object.__new__(cls)
        # Fast paths: answer the struct-redirect question directly off
        # the input shape so the common non-struct case never runs
        # ``DataType.from_any`` (which then runs a second time inside
        # ``__init__``). Coverage:
        #   - already-resolved ``DataType`` (the dominant case — cached
        #     primitive singletons, ``StructType(...)``, ``MapType(...)``);
        #   - native Arrow ``DataType`` / ``Field`` — ``pa.types.is_struct``
        #     is a single C call;
        #   - everything else (str shorthand, dict, dataclass, …) falls
        #     back through ``from_any``.
        if isinstance(dtype, DataType):
            if dtype.type_id is not DataTypeId.STRUCT:
                return object.__new__(cls)
            resolved = dtype
        elif isinstance(dtype, pa.DataType):
            if not pa.types.is_struct(dtype):
                return object.__new__(cls)
            try:
                resolved = DataType.from_any(dtype)
            except Exception:
                return object.__new__(cls)
        elif isinstance(dtype, pa.Field):
            if not pa.types.is_struct(dtype.type):
                return object.__new__(cls)
            try:
                resolved = DataType.from_any(dtype)
            except Exception:
                return object.__new__(cls)
        else:
            try:
                resolved = DataType.from_any(dtype)
            except Exception:
                return object.__new__(cls)
            if resolved is None or resolved.type_id is not DataTypeId.STRUCT:
                return object.__new__(cls)
        Schema = schema_class()
        instance = object.__new__(Schema)
        # Python's ``type.__call__`` follows up with
        # ``Schema.__init__(instance, *args, **kwargs)``, which loops
        # through ``Field.__init__`` again — the idempotent-init guard
        # there short-circuits the second pass once we've stamped the
        # slots.
        name = (
            kwargs.get("name")
            if "name" in kwargs
            else (args[0] if args else "")
        )
        Field.__init__(
            instance,
            name=name,
            dtype=resolved,
            nullable=kwargs.get("nullable", True),
            metadata=kwargs.get("metadata"),
            tags=kwargs.get("tags"),
            default=kwargs.get("default"),
            parent=kwargs.get("parent"),
        )
        return instance

    def __init__(
        self,
        name: str,
        dtype: DataType | type[DataType] | pa.DataType,
        nullable: bool = True,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
        parent: "Field | None" = None,
    ) -> None:
        # Resolve the dtype up front. ``isinstance`` short-circuit
        # before ``DataType.from_any`` matters — ``from_any`` does an
        # MRO walk + dispatch even when the answer is "give me back
        # what I passed in", and Field construction is hot enough that
        # the saved call shows up.
        resolved_dtype = (
            dtype if isinstance(dtype, DataType) else DataType.from_any(dtype)
        )
        # Idempotent re-init guard for the ``Field.__new__`` struct
        # redirect: when ``__new__`` returns a fully-initialized
        # :class:`Schema`, Python still calls ``Schema.__init__`` on
        # it (which loops back through ``Field.__init__``) with the
        # original args. Detecting that case here keeps us from
        # re-stamping the cache slots a second time. ``self.dtype`` is
        # a slot — accessing it on a fresh instance raises AttributeError
        # which we treat as "not initialized yet" and proceed.
        try:
            existing = self.dtype
        except AttributeError:
            existing = None
        if existing is not None and self.name == name and existing == resolved_dtype:
            return
        setattr_ = object.__setattr__
        setattr_(self, "name", name)
        setattr_(self, "dtype", resolved_dtype)
        setattr_(self, "nullable", bool(nullable))
        # Skip the ``_normalize_metadata`` call entirely when the caller
        # passed nothing — the function would just walk into its own
        # short-circuit and return ``None``, but the dispatch + arg
        # marshalling shows up in the construction profile.
        if metadata is None and tags is None and default is None:
            normalized_meta = None
        else:
            normalized_meta = _normalize_metadata(
                metadata, tags=tags, default_value=default
            )
        setattr_(self, "metadata", normalized_meta)
        setattr_(self, "parent", parent)
        # Initialize cache slots so the dataclass is fully populated
        # — accessing an unset slot would raise AttributeError.
        setattr_(self, "_arrow_field", None)
        setattr_(self, "_arrow_schema", None)
        setattr_(self, "_polars_field", None)
        setattr_(self, "_polars_schema", None)
        setattr_(self, "_spark_field", None)
        setattr_(self, "_spark_schema", None)
        setattr_(self, "_field_name_map", None)
        setattr_(self, "_field_name_fold_map", None)
        setattr_(self, "_hash", None)
        # Adopt children — set their ``parent`` so they bubble cache
        # invalidations back up to us when they mutate. Skip the call
        # entirely on primitive fields where ``dtype.children`` is
        # always empty (saves an attribute lookup + iter() per
        # construction, which is the dominant ``Field`` allocation
        # shape).
        if resolved_dtype.children:
            self._adopt_children()

    # ==================================================================
    # Cache layer — invalidation + child adoption
    # ==================================================================

    def _on_metadata_mutated(self) -> None:
        """Tag flag setter / unsetter touched ``self.metadata`` —
        invalidate cached projections (arrow / polars / spark) so the
        next request rebuilds with the new tags."""
        self._invalidate_cache()

    def _adopt_children(self) -> None:
        """Stamp ``self`` as the parent of every child :class:`Field`.

        Walks ``self.dtype.children`` (the dtype-level view of
        children — works for struct, map, and array). Idempotent:
        calling it on a field whose children already point at ``self``
        is a no-op assignment.
        """
        for child in self.dtype.children:
            if isinstance(child, Field) and child.parent is not self:
                object.__setattr__(child, "parent", self)

    def invalidate_cache(self, *, cascade: bool = True) -> None:
        """Drop cached engine projections, cascading to ancestors by default.

        Public surface over :meth:`_invalidate_cache`. Callers that
        mutate the underlying state outside of the ``with_*`` mutators
        (custom DataType subclass that swaps children in place,
        external code that pokes ``dtype.fields`` directly) should call
        this once to make sure the next ``to_arrow_field`` /
        ``to_polars_field`` / ``to_pyspark_field`` / ``*_schema``
        request rebuilds with the new state. With ``cascade=True``
        (the default) every ancestor reachable via :attr:`parent`
        also drops its cache, so a struct's cached arrow schema gets
        rebuilt after one of its children mutates.
        """
        self._invalidate_cache(cascade=cascade)

    def _invalidate_cache(self, *, cascade: bool = True) -> None:
        """Drop cached arrow / polars / spark projections.

        Called from every mutator (``with_*``, in-place
        :meth:`merge_with`) right after the underlying state changes.
        With ``cascade=True`` (the default), the invalidation walks
        up the :attr:`parent` chain so an ancestor whose cached arrow
        schema embedded this field gets rebuilt next time it's
        requested.
        """
        if (
            self._arrow_field is None
            and self._arrow_schema is None
            and self._polars_field is None
            and self._polars_schema is None
            and self._spark_field is None
            and self._spark_schema is None
            and self._field_name_map is None
            and self._field_name_fold_map is None
            and self._hash is None
        ):
            # Already clean — still cascade so dirty ancestors clear too.
            if cascade and self.parent is not None:
                self.parent._invalidate_cache(cascade=True)
            return
        object.__setattr__(self, "_arrow_field", None)
        object.__setattr__(self, "_arrow_schema", None)
        object.__setattr__(self, "_polars_field", None)
        object.__setattr__(self, "_polars_schema", None)
        object.__setattr__(self, "_spark_field", None)
        object.__setattr__(self, "_spark_schema", None)
        object.__setattr__(self, "_field_name_map", None)
        object.__setattr__(self, "_field_name_fold_map", None)
        object.__setattr__(self, "_hash", None)
        if cascade and self.parent is not None:
            self.parent._invalidate_cache(cascade=True)

    def equals(
        self,
        other: Any,
        check_names: bool = True,
        check_dtypes: bool = True,
        check_nullable: bool = True,
        check_metadata: bool = True,
    ) -> bool:
        """Structural equality check with configurable scope.

        Mirrors :meth:`DataType.equals`. Coerces *other* to a ``Field`` so
        that callers can pass a ``pa.Field`` / dict / etc. without manual
        conversion. Returns ``False`` on coercion failure instead of raising.

        - ``check_names``: compare this field's name and recurse into child
          field names for nested types. For struct-shaped fields the
          comparison is order-independent (children matched by name)
          when ``check_names`` is True, mirroring how Arrow schemas are
          name-keyed.
        - ``check_dtypes``: recurse into the dtype and compare ``nullable``
          (both are structural, schema-defining attributes).
        - ``check_metadata``: compare this field's metadata and recurse.
        """
        if other is self:
            return True
        if other is None:
            return False
        if not isinstance(other, Field):
            try:
                other = Field.from_any(other)
            except Exception:
                return False

        if not self.name:
            self.with_name(other.name, inplace=True)
        elif not other.name:
            other.with_name(self.name, inplace=True)

        if check_names and self.name != other.name:
            return False

        if check_nullable and self.nullable != other.nullable:
            return False

        if check_metadata and self.metadata != other.metadata:
            return False

        if check_dtypes:
            # Struct-shaped fields are name-keyed schemas: when names
            # matter, match children by name (reorder-tolerant); when
            # names don't, fall back to a positional walk so a renamed
            # column can still match its peer.
            if self.type_id is DataTypeId.STRUCT and other.type_id is DataTypeId.STRUCT:
                self_children = self.children
                other_children = other.children
                if len(self_children) != len(other_children):
                    return False
                if check_names:
                    seen: set[str] = set()
                    for sf in self_children:
                        of = other.field_by(name=sf.name, raise_error=False)
                        if of is None:
                            return False
                        if not sf.equals(
                            of,
                            check_names=check_names,
                            check_dtypes=check_dtypes,
                            check_nullable=check_nullable,
                            check_metadata=check_metadata,
                        ):
                            return False
                        seen.add(sf.name)
                    for of in other_children:
                        if of.name not in seen:
                            return False
                else:
                    for sf, of in zip(self_children, other_children):
                        if not sf.equals(
                            of,
                            check_names=check_names,
                            check_dtypes=check_dtypes,
                            check_nullable=check_nullable,
                            check_metadata=check_metadata,
                        ):
                            return False
            elif self.dtype is not other.dtype and not self.dtype.equals(
                other.dtype,
                check_metadata=check_metadata,
            ):
                return False

        return True

    # ==================================================================
    # Properties — dtype projection, defaults, children
    # ==================================================================

    @property
    def alias(self) -> str:
        """Alternate name for this field — falls back to :attr:`name`.

        Stored in :attr:`metadata` under :data:`ALIAS_KEY`. Used by
        :meth:`select_in` (and the engine-specific ``select_in_*``
        helpers) to resolve a column from a frame whose schema
        labels it differently than :attr:`name` — common when a
        source table has been renamed, or when a field carries
        both a wire name and a friendly name.

        Always returns a string: when no alias is configured the
        getter returns :attr:`name` so downstream lookups don't
        have to fork on ``None``. Use :attr:`has_alias` to tell
        "configured alias" from "name fallback".

        ``Field`` is a frozen dataclass — mutating the alias goes
        through :meth:`with_alias` (immutable copy) or
        :meth:`set_alias` (in-place dict mutation, same shape as
        the rest of the metadata setters).
        """
        if self.metadata:
            value = self.metadata.get(ALIAS_KEY)
            if value is not None:
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return self.name

    @property
    def has_alias(self) -> bool:
        """Whether an explicit alias is set in :attr:`metadata`.

        Distinguishes "alias configured" from the
        :attr:`alias` → :attr:`name` fallback so callers that care
        (schema diffs, nested-cast lookup precedence) can branch on
        the real bit.
        """
        return bool(self.metadata and self.metadata.get(ALIAS_KEY))

    def set_alias(self, value: str | None) -> "Field":
        """Set / clear :attr:`alias` on *self* in place.

        Frozen-dataclass property setters trip the auto-generated
        ``__setattr__`` (cpython #44477), so the alias write path
        is a method instead. Returns ``self`` so calls chain.

        Normalization rules:

        * ``value == self.name`` → no-op. Storing the canonical name
          in the alias slot is meaningless — :attr:`alias` already
          falls back to :attr:`name`.
        * Field has no name yet (empty or unset) → promote the
          incoming value to :attr:`name` instead of stashing it in
          metadata, so the field gets a usable identity in one
          step.
        * Otherwise → record under :data:`ALIAS_KEY`.
        """
        if value is None or value == self.name:
            if not self.metadata or ALIAS_KEY not in self.metadata:
                return self
            self.metadata.pop(ALIAS_KEY, None)
            if not self.metadata:
                object.__setattr__(self, "metadata", None)
            self._on_metadata_mutated()
            return self
        if not self.name:
            object.__setattr__(self, "name", value)
            if self.metadata and ALIAS_KEY in self.metadata:
                self.metadata.pop(ALIAS_KEY, None)
                if not self.metadata:
                    object.__setattr__(self, "metadata", None)
            self._on_metadata_mutated()
            return self
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        self.metadata[ALIAS_KEY] = value.encode("utf-8")
        self._on_metadata_mutated()
        return self

    @property
    def position(self) -> int | None:
        """Optional 0-based index this field claims in a parent schema.

        Stored in :attr:`metadata` under :data:`POSITION_KEY`. Used
        by :meth:`select_in_field` (and the engine-specific
        ``select_in_*`` helpers) as the last-resort fallback when
        neither :attr:`name` nor :attr:`alias` matches a child name
        in the receiving schema — the receiver's
        ``children[position]`` (or column at ``position``)
        is then resolved by name and used.

        ``None`` (the default) leaves position-based lookup
        disabled, matching the historical name/alias-only resolver.
        """
        if not self.metadata:
            return None
        value = self.metadata.get(POSITION_KEY)
        if value is None:
            return None
        if isinstance(value, bytes):
            try:
                return int(value.decode("utf-8"))
            except ValueError:
                return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def set_position(self, value: int | None) -> "Field":
        """Set / clear :attr:`position` on *self* in place.

        Negative values are rejected — positions are forward
        indices into the parent schema; if you need a
        last-element fallback, resolve it before calling.
        """
        if value is None:
            if not self.metadata or POSITION_KEY not in self.metadata:
                return self
            self.metadata.pop(POSITION_KEY, None)
            if not self.metadata:
                object.__setattr__(self, "metadata", None)
            self._on_metadata_mutated()
            return self
        if not isinstance(value, int) or value < 0:
            raise ValueError(
                f"Field.position must be a non-negative int (got {value!r})."
            )
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        self.metadata[POSITION_KEY] = str(value).encode("utf-8")
        self._on_metadata_mutated()
        return self

    def with_position(self, position: int | None) -> "Field":
        """Return a copy of this field with :attr:`position` set / cleared."""
        if position == self.position:
            return self
        if position is not None and (not isinstance(position, int) or position < 0):
            raise ValueError(
                f"Field.position must be a non-negative int (got {position!r})."
            )
        metadata = dict(self.metadata) if self.metadata else {}
        if position is None:
            metadata.pop(POSITION_KEY, None)
        else:
            metadata[POSITION_KEY] = str(position).encode("utf-8")
        return self.copy(metadata=metadata or None)

    def with_alias(self, alias: str | None) -> "Field":
        """Return a copy of this field with :attr:`alias` set / cleared.

        Mirrors :meth:`with_default` — immutable shape for callers
        that don't want to mutate the existing instance. Same
        normalization rules as :meth:`set_alias`: no-op on
        ``alias == self.name``; promote to :attr:`name` when the
        receiver has no name yet.
        """
        if alias is None or alias == self.name:
            if not self.has_alias:
                return self
            metadata = dict(self.metadata)
            metadata.pop(ALIAS_KEY, None)
            return self.copy(metadata=metadata or None)
        if not self.name:
            metadata = dict(self.metadata) if self.metadata else {}
            metadata.pop(ALIAS_KEY, None)
            return self.copy(name=alias, metadata=metadata or None)
        metadata = dict(self.metadata) if self.metadata else {}
        metadata[ALIAS_KEY] = alias.encode("utf-8")
        return self.copy(metadata=metadata)

    @property
    def has_default(self) -> bool:
        return (
            self.metadata.get(DEFAULT_VALUE_KEY) is not None if self.metadata else False
        )

    @property
    def default_value(self):
        """Field's default Python value (or the dtype-level default).

        Reads :data:`DEFAULT_VALUE_KEY` from :attr:`metadata` first;
        falls back to ``self.dtype.default_pyobj`` when the metadata
        slot is unset. Renamed from ``default`` so the constructor
        classmethod :meth:`Field.default` can take that name —
        ``field.default`` would otherwise shadow it via descriptor
        lookup.
        """
        if self.metadata is not None:
            default = self.metadata.get(DEFAULT_VALUE_KEY)

            if default is None:
                return self.dtype.default_pyobj(nullable=self.nullable)

            try:
                default = json_module.loads(default, safe=False)
            except Exception as e:
                raise ValueError(
                    f"Could not parse default value {default!r} for {self!r}: {e}"
                ) from e

            return self.dtype.convert_pyobj(default, nullable=self.nullable, safe=False)
        return None

    @property
    def default_arrow_scalar(self) -> pa.Scalar | None:
        if self.metadata is not None:
            default = self.metadata.get(DEFAULT_VALUE_KEY)

            if default is None:
                return self.dtype.default_arrow_scalar(nullable=self.nullable)

            try:
                default = json_module.loads(default, safe=False)
            except Exception as e:
                raise ValueError(
                    f"Could not parse default value {default!r} for {self!r}: {e}"
                ) from e

            return self.dtype.convert_arrow_scalar(
                pa.scalar(default), nullable=self.nullable, safe=False
            )
        return None

    @property
    def type_id(self) -> DataTypeId:
        return self.dtype.type_id

    @property
    def children(self) -> list["Field"]:
        return self.dtype.children

    @property
    def arrow_type(self) -> pa.DataType:
        return self.dtype.to_arrow()

    def _empty_tags(self) -> dict[bytes, bytes]:
        return {}

    # ==================================================================
    # Metadata + tags
    # ==================================================================
    # Inlined from the former ``BaseMetadata`` mixin — :class:`Field`
    # was the only consumer, so the indirection was pure noise.
    # Tag accessors all funnel through ``self.metadata`` (a
    # ``bytes → bytes`` dict whose tag entries are prefixed with
    # :data:`TAG_PREFIX`).

    def _prefixed_metadata(self, prefix: bytes) -> dict[bytes, bytes]:
        if not self.metadata:
            return {}
        return {
            key[len(prefix):]: value
            for key, value in self.metadata.items()
            if key.startswith(prefix)
        }

    def _update_prefixed_metadata(
        self,
        prefix: bytes,
        values: Mapping[bytes | str, bytes | str | object] | None,
    ) -> None:
        if not values:
            return
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        self.metadata.update(
            {
                prefix + _to_bytes(key): _to_bytes(value)
                for key, value in values.items()
                if key and value is not None
            }
        )

    def _tag_value(self, key: bytes | str) -> bytes | None:
        if not self.metadata:
            return None
        return self.metadata.get(TAG_PREFIX + _to_bytes(key))

    def _set_tag_value(self, key: bytes | str, value: Any | None) -> None:
        if value is not None:
            if not self.metadata:
                object.__setattr__(self, "metadata", {})
            self.metadata[TAG_PREFIX + _to_bytes(key)] = _to_bytes(value)
            self._on_metadata_mutated()

    def _unset_tag_value(self, key: bytes | str) -> None:
        if not self.metadata:
            return
        self.metadata.pop(TAG_PREFIX + _to_bytes(key), None)
        if not self.metadata:
            object.__setattr__(self, "metadata", None)
        self._on_metadata_mutated()

    def _tag_flag(self, key: bytes | str) -> bool:
        return bool(self._tag_value(key))

    def _tag_text(self, key: bytes | str) -> str | None:
        value = self._tag_value(key)
        if not value:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    @property
    def tags(self) -> dict[bytes, bytes] | None:
        tags = self._prefixed_metadata(TAG_PREFIX)
        if tags:
            return tags
        return self._empty_tags()

    @tags.setter
    def tags(self, value: "Mapping[AnyStr, AnyStr] | None") -> None:
        self.update_tags(value)

    def update_tags(self, value: "Mapping[AnyStr, AnyStr] | None") -> None:
        self._update_prefixed_metadata(TAG_PREFIX, value)

    @property
    def partition_by(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_PARTITION_BY))

    @property
    def cluster_by(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_CLUSTER_BY))

    @property
    def primary_key(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_PRIMARY_KEY))

    @property
    def foreign_key(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_FOREIGN_KEY))

    @property
    def constraint_key(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_CONSTRAINT_KEY))

    @property
    def sorted(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_SORTED))

    @property
    def index_key(self) -> bool:
        md = self.metadata
        return bool(md and md.get(self._TAG_KEY_INDEX_KEY))

    @property
    def index_key_level(self) -> int | None:
        md = self.metadata
        if not md:
            return None
        raw = md.get(self._TAG_KEY_INDEX_KEY_LEVEL)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @property
    def comment(self) -> str | None:
        if not self.metadata:
            return None
        raw = self.metadata.get(b"comment") or self.metadata.get(b"description")
        if raw:
            return raw.decode("utf-8")
        return None

    @property
    def media_type(self) -> "Any | None":
        """:class:`MediaType` describing how this field's data is stored.

        Decodes the ``b"media_type"`` metadata key — the mime-string
        canonical form (``"application/vnd.apache.arrow.file"``,
        ``"application/vnd.apache.parquet"``, …) round-tripped through
        :meth:`MediaType.from_`. ``None`` when no media-type hint has
        been stamped.

        Populated by :class:`FolderPath._persist_schema` so a schema
        loaded from a folder's ``.ygg/schema.arrow`` sidecar tells the
        reader which on-disk format the rows were last written in
        (Arrow IPC, Parquet, …) without walking the part files.
        Schema-level (top-level :class:`StructField`) is the canonical
        slot, but the accessor lives on :class:`Field` so per-column
        hints (e.g. the response-body field's HTTP ``Content-Type``)
        can use the same property.
        """
        md = self.metadata
        if not md:
            return None
        raw = md.get(MEDIA_TYPE_METADATA_KEY)
        if not raw:
            return None
        from yggdrasil.data.enums.media_type import MediaType
        # ``raw.decode`` can raise ``UnicodeDecodeError`` on a
        # corrupted metadata byte payload; ``MediaType.from_`` with
        # ``default=None`` already short-circuits unknown mime
        # strings without raising, but the underlying parser can
        # still surface ``ValueError`` / ``TypeError`` on truly
        # malformed input. Catch the narrow trio and degrade to
        # "no media type known".
        try:
            return MediaType.from_(
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw,
                default=None,
            )
        except (UnicodeDecodeError, ValueError, TypeError):
            return None

    # ==================================================================
    # Schema-shaped views — meaningful when ``self.dtype`` is a struct;
    # quietly empty otherwise so callers don't have to type-check first.
    # ==================================================================

    @property
    def fields(self) -> list["Field"]:
        """Children excluding constraint-only fields."""
        # Hot path: nearly every nested-field access (arrow_fields,
        # to_arrow_schema, to_polars_schema, to_spark_schema,
        # primary_keys / foreign_keys / constraints) routes through
        # ``fields``. Inline the constraint-key check against the
        # precomputed metadata key so we don't pay the property dispatch
        # + bytes concat on every child.
        children = self.children
        key = self._TAG_KEY_CONSTRAINT_KEY
        return [
            f for f in children
            if not (f.metadata and key in f.metadata)
        ]

    @property
    def inner_fields(self) -> "OrderedDict[str, Field]":
        """Compat view of the children as an ordered ``{name: field}`` map."""
        return OrderedDict((f.name, f) for f in self.children)

    @property
    def arrow_fields(self) -> list[pa.Field]:
        return [f.to_arrow_field() for f in self.fields]

    @property
    def primary_keys(self) -> list["Field"]:
        return [f for f in self.fields if f.primary_key]

    @property
    def foreign_keys(self) -> list["Field"]:
        return [f for f in self.fields if f.foreign_key]

    @property
    def constraints(self) -> list["Field"]:
        return [f for f in self.children if f.constraint_key]

    def is_empty(self) -> bool:
        return len(self.children) == 0

    @classmethod
    def empty(
        cls,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        return cls._make_struct(
            children=(),
            metadata=_normalize_metadata(metadata, tags=tags),
        )

    @classmethod
    def _make_struct(
        cls,
        children: Iterable["Field"] = (),
        *,
        name: str = DEFAULT_FIELD_NAME,
        nullable: bool = False,
        metadata: dict[bytes, bytes] | None = None,
    ) -> "Field":
        """Construct a struct-shaped instance bypassing subclass init shims.

        Used by mapping mutators / set ops / autotag. Going through
        ``__new__`` + ``Field.__init__`` lets the same code construct
        either a ``Field`` or a ``StructField`` based on ``cls``.
        """
        inst = cls.__new__(cls)
        Field.__init__(
            inst,
            name=name,
            dtype=StructType(fields=tuple(children)),
            nullable=bool(nullable),
            metadata=metadata,
        )
        return inst

    def _set_dtype_fields(self, fields: Iterable["Field"]) -> None:
        """Replace the underlying StructType's fields tuple in place.

        Mutating the dtype directly doesn't go through any of Field's
        ``with_*`` mutators, so the cached arrow / polars / spark
        projections (and the field-name lookup map) would otherwise
        miss the change. Invalidate them explicitly and re-adopt the
        new children so they bubble future mutations back through
        ``self.parent``.
        """
        object.__setattr__(self.dtype, "fields", tuple(fields))
        self._invalidate_cache()
        self._adopt_children()

    def _pop_field_name_list(self, key: bytes) -> set[str]:
        """Pop a ``key`` -> field-name-list entry off ``self.metadata``.

        Accepts a JSON array (``'["a","b"]'``) or a dot-separated string
        (``"a.b"``). Returns an empty set when the key is missing or
        empty — and removes the key either way so it does not leak
        through to the engine schemas.
        """
        if not self.metadata:
            return set()
        raw = self.metadata.pop(key, None)
        if not raw:
            return set()
        if raw.startswith(b"[") and raw.endswith(b"]"):
            return set(json.loads(raw))
        return set(raw.decode().split("."))

    # ==================================================================
    # Tag flags — partition_by / cluster_by / primary_key / foreign_key
    # ==================================================================

    def _with_tag_flag(self, key: bytes, value: bool, inplace: bool) -> "Field":
        # Short-circuit when the desired flag state already matches —
        # avoids invalidating cached arrow / polars / spark projections
        # for a no-op tag write.
        if self._tag_flag(key) == bool(value):
            return self
        if inplace:
            if value:
                self._set_tag_value(key, True)
            else:
                self._unset_tag_value(key)
            return self
        else:
            return self.copy()._with_tag_flag(key, value, inplace=True)

    def with_partition_by(self, value: bool = True, inplace: bool = True) -> "Field":
        return self._with_tag_flag(b"partition_by", value, inplace)

    def with_cluster_by(self, value: bool = True, inplace: bool = True) -> "Field":
        return self._with_tag_flag(b"cluster_by", value, inplace)

    def with_primary_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"primary_key", value, inplace)

    def with_foreign_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"foreign_key", value, inplace)

    def with_constraint_key(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"constraint_key", value, inplace)

    def with_sorted(self, value: bool = True, inplace: bool = False) -> "Field":
        return self._with_tag_flag(b"sorted", value, inplace)

    def with_index_key(self, value: bool = True, level: int | None = None, inplace: bool = False) -> "Field":
        result = self._with_tag_flag(b"index_key", value, inplace)
        if value and level is not None:
            result._set_tag_value(b"index_key_level", str(level))
        elif not value:
            result._unset_tag_value(b"index_key_level")
        return result

    # ==================================================================
    # Builders — `with_*` mutators, `copy`, `merge_with`, `autotag`
    # ==================================================================

    def with_name(self, name: str, inplace: bool = False) -> "Field":
        if name == self.name:
            return self

        if inplace:
            object.__setattr__(self, "name", name)
            self._invalidate_cache()
            return self
        return self.copy(name=name)

    def with_dtype(
        self, dtype: DataType | type[DataType] | pa.DataType, inplace: bool = True
    ) -> "Field":
        if dtype == self.dtype:
            return self

        dtype = DataType.from_any(dtype)

        if inplace:
            object.__setattr__(self, "dtype", dtype)
            self._invalidate_cache()
            self._adopt_children()
            return self
        return self.copy(dtype=dtype)

    def with_nullable(self, nullable: bool, inplace: bool = True) -> "Field":
        if nullable == self.nullable:
            return self

        if inplace:
            object.__setattr__(self, "nullable", bool(nullable))
            self._invalidate_cache()
            return self
        return self.copy(nullable=bool(nullable))

    def with_metadata(
        self,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
        inplace: bool = True,
    ):
        if metadata or tags or default is not None:
            normalized = _normalize_metadata(metadata, tags=tags, default_value=default)
            # Skip the cache drop when the normalized payload is
            # byte-for-byte equal to what's already stored.
            if normalized == self.metadata:
                return self
            if inplace:
                object.__setattr__(self, "metadata", normalized)
                self._invalidate_cache()
                return self
            return self.copy(metadata=normalized)
        return self

    def with_default(self, default: Any = None) -> "Field":
        current = self.metadata.get(DEFAULT_VALUE_KEY) if self.metadata else None

        if default is None:
            # Already no default — nothing to invalidate.
            if current is None:
                return self
            metadata = dict(self.metadata)
            metadata.pop(DEFAULT_VALUE_KEY, None)
        else:
            encoded = json_module.dumps(
                default,
                safe=False,
                to_bytes=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if current == encoded:
                return self
            metadata = dict(self.metadata) if self.metadata is not None else {}
            metadata[DEFAULT_VALUE_KEY] = encoded

        object.__setattr__(self, "metadata", metadata or None)
        self._invalidate_cache()
        return self

    def copy(
        self,
        *,
        name: str | None = None,
        dtype: DataType | type[DataType] | pa.DataType | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        fields: Iterable["Field | pa.Field"] | None = None,
    ) -> "Field":
        # ``fields=`` is the schema-shaped escape hatch: build a
        # struct dtype from the given children, deep-copying them so
        # the copy is a fresh tree. Conflicts with ``dtype=`` since
        # both target the dtype slot — fail loudly when both arrive.
        if fields is not None:
            if dtype is not None:
                raise TypeError("Field.copy: pass either fields= or dtype=, not both")
            children = [
                f.copy() if isinstance(f, Field) else Field.from_any(f)
                for f in _normalize_inner_fields(fields)
            ]
            new_dtype: DataType = StructType(fields=tuple(children))
        elif dtype is None:
            # Deep-copy struct children so the copy doesn't share child
            # state with self — mirrors Schema.copy's old behaviour.
            if isinstance(self.dtype, StructType):
                new_dtype = StructType(
                    fields=tuple(c.copy() for c in self.dtype.fields)
                )
            else:
                new_dtype = self.dtype
        else:
            new_dtype = DataType.from_any(dtype)

        meta = (
            (dict(self.metadata) if self.metadata is not None else None)
            if metadata is None and tags is None
            else _normalize_metadata(metadata, tags=tags)
        )

        if name is None:
            name = self.name
        if nullable is None:
            nullable = self.nullable

        inst = type(self).__new__(type(self))
        Field.__init__(
            inst,
            name=name,
            dtype=new_dtype,
            nullable=bool(nullable),
            metadata=meta,
        )
        return inst

    # ==================================================================
    # Children mutators — append / replace / merge a child Field
    # ==================================================================

    def with_field(
        self,
        field: "Field | pa.Field | str",
        *,
        mode: "Mode | str | None" = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> "Field":
        """Return *self* with *field* appended or merged in.

        ``mode`` controls collision behavior when a child with the
        same name already exists. Accepts a :class:`Mode` member or
        any alias :meth:`Mode.from_` understands.

        - :data:`Mode.AUTO` / :data:`Mode.OVERWRITE` — replace the
          existing child verbatim with *field*.
        - :data:`Mode.APPEND` — append a fresh child even if the
          name collides (struct semantics: last-write-wins for
          duplicate names; both entries survive in the children
          tuple).
        - :data:`Mode.IGNORE` — keep the existing child; drop the
          incoming.
        - :data:`Mode.ERROR_IF_EXISTS` — raise :class:`ValueError`
          on collision.
        - :data:`Mode.UPSERT` / :data:`Mode.MERGE` — :meth:`merge_with`
          the existing child against the incoming one (dtype, nullability,
          metadata), keeping the existing child's identity.

        Auto-promotion to struct: when ``self`` isn't a struct
        (a primitive Field, a list/map, …) the call returns a fresh
        struct Field whose first child is the previous ``self``
        (renamed to its current ``name`` so it's addressable) and
        whose second child is *field*. The promoted struct keeps
        ``self``'s name, nullability, and metadata — only the dtype
        changes.

        Bare-string shorthand: ``self.with_field("price")`` reads as
        "make sure a child named 'price' exists." That call goes
        through :meth:`Field.from_any` which infers a sensible
        default dtype.

        ``inplace=True`` (the default) mutates ``self`` and returns
        it. ``inplace=False`` returns a fresh copy.
        """
        if isinstance(field, Field):
            new_field = field
        elif isinstance(field, str):
            # Bare-string shorthand: ``with_field("price")`` reads as
            # "make sure a child named 'price' exists." The dtype
            # defaults to :class:`ObjectType` so the column is typed
            # but un-pinned; callers that care can pass a Field /
            # ``pa.Field`` instead.
            new_field = Field.default(name=field, **kwargs)
        else:
            new_field = Field.from_any(field, **kwargs)

        if not isinstance(self.dtype, StructType):
            return self._promote_to_struct_with(new_field, inplace=inplace)

        action = Mode.from_(mode, default=Mode.AUTO)
        target = self if inplace else self.copy()

        existing_idx: "int | None" = None
        for i, c in enumerate(target.children):
            if c.name == new_field.name:
                existing_idx = i
                break

        if existing_idx is None:
            # No collision — APPEND-style, IGNORE/ERROR are no-ops here.
            new_children = list(target.children)
            new_children.append(new_field)
            target._set_dtype_fields(new_children)
            return target

        if action is Mode.IGNORE:
            return target
        if action is Mode.ERROR_IF_EXISTS:
            raise ValueError(
                f"{type(self).__name__}.with_field: child {new_field.name!r} "
                f"already exists; mode=ERROR_IF_EXISTS refuses overwrite. "
                f"Pass mode='overwrite', 'merge', or 'append' to control "
                f"the collision behavior."
            )
        if action is Mode.APPEND:
            # Honest append: keep the existing child, add a new one
            # with the same name. Struct lookups by name will hit the
            # first match (preserves the existing entry's identity);
            # downstream consumers that care can dedupe.
            new_children = list(target.children)
            new_children.append(new_field)
            target._set_dtype_fields(new_children)
            return target
        if action is Mode.UPSERT or action is Mode.MERGE:
            existing = target.children[existing_idx]
            merged = existing.merge_with(new_field, inplace=False)
            new_children = list(target.children)
            new_children[existing_idx] = merged
            target._set_dtype_fields(new_children)
            return target

        # AUTO / OVERWRITE / TRUNCATE → replace.
        new_children = list(target.children)
        new_children[existing_idx] = new_field
        target._set_dtype_fields(new_children)
        return target

    def with_fields(
        self,
        fields: "Iterable[Field | pa.Field | str]",
        *,
        mode: "Mode | str | None" = None,
        inplace: bool = True,
    ) -> "Field":
        """Apply :meth:`with_field` for every entry in *fields*.

        Same mode semantics as :meth:`with_field`; the loop short-
        circuits :data:`Mode.IGNORE` once any one collision keeps the
        existing child (no global "first one wins, drop the rest"
        gymnastics — collisions are evaluated per name).

        Auto-promotes ``self`` to a struct on the first call when
        needed; subsequent fields land on that struct.
        """
        target = self if inplace else self.copy()
        for f in fields:
            target = target.with_field(f, mode=mode, inplace=True)
        return target

    def _promote_to_struct_with(
        self,
        new_field: "Field",
        *,
        inplace: bool,
    ) -> "Field":
        """Rebuild ``self`` as a struct whose first child is its
        previous self.

        The previous content keeps its name (so ``parent["x"]``
        still resolves to the original), its dtype, its nullability,
        and its metadata. The wrapper inherits ``self``'s name and
        becomes the addressable struct.

        Used by :meth:`with_field` when ``self`` isn't already a
        struct — the call site reads as "stamp a child onto this
        field," and the field auto-grows children-shape semantics
        without the caller having to call :meth:`empty` or
        :meth:`copy(fields=...)` first.
        """
        previous = Field(
            name=self.name or DEFAULT_FIELD_NAME,
            dtype=self.dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )
        new_dtype = StructType(fields=(previous, new_field))

        if inplace:
            object.__setattr__(self, "dtype", new_dtype)
            self._invalidate_cache()
            self._adopt_children()
            return self
        return self.copy(dtype=new_dtype)

    def merge_with(
        self,
        other: "Field",
        *,
        inplace: bool = False,
        mode: Mode | None = None,
        downcast: bool = False,
        upcast: bool = False,
        merge_dtype: bool = True,
        merge_nullable: bool = True,
        merge_metadata: bool = True,
    ):
        other = self.from_any(other)
        if self == other:
            return self

        mode = Mode.from_(mode, default=Mode.AUTO)

        if mode is Mode.AUTO:
            name = self.name or other.name
            nullable = not (not self.nullable or not other.nullable)
            metadata = {
                **(other.metadata or {}),
                **(self.metadata or {}),
            }

            dtype = self.dtype.merge_with(
                other.dtype, mode=mode, downcast=downcast, upcast=upcast
            )

            if inplace:
                object.__setattr__(self, "name", name)
                object.__setattr__(self, "dtype", dtype)
                object.__setattr__(self, "nullable", bool(nullable))
                object.__setattr__(self, "metadata", metadata)
                return self

            return self.copy(
                name=name,
                dtype=dtype,
                nullable=nullable,
                metadata=metadata,
            )
        elif mode is Mode.IGNORE:
            return self
        elif mode is Mode.OVERWRITE:
            if inplace:
                object.__setattr__(self, "dtype", other.dtype)
                object.__setattr__(self, "nullable", other.nullable)
                object.__setattr__(self, "metadata", other.metadata)
                return self
            return self.copy(
                name=other.name or self.name,
                dtype=other.dtype,
                nullable=other.nullable,
                metadata=other.metadata,
            )

        name = self.name or other.name
        nullable = self.nullable or other.nullable if merge_nullable else self.nullable
        metadata = self.metadata if merge_metadata else None

        if merge_metadata:
            metadata = {
                **(self.metadata if self.metadata is not None else {}),
                **(other.metadata if other.metadata is not None else {}),
            }

        if merge_dtype:
            dtype = self.dtype.merge_with(
                other.dtype, mode=mode, downcast=downcast, upcast=upcast
            )
        else:
            dtype = self.dtype

        if inplace:
            object.__setattr__(self, "dtype", dtype)
            object.__setattr__(self, "nullable", bool(nullable))
            object.__setattr__(self, "metadata", metadata)
            return self

        return self.copy(
            name=name or DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=nullable,
            metadata=metadata,
        )

    def autotag(
        self,
        tags: dict[AnyStr, AnyStr] | None = None,
    ) -> "Field":
        """Stamp this field with tags derived from its dtype and name.

        Writes Databricks-friendly auto-tags in place:

        - Everything from :meth:`DataType.autotag` (``kind`` plus dtype
          detail like ``unit`` / ``tz`` / ``precision`` / ``scale`` /
          ``signed`` / ``iso`` / ``srid``).
        - ``nullable`` for data-quality policies.
        - Name-based heuristics for governance: ``role=identifier`` for
          ``*_id`` / ``*_uuid``, ``role=audit_timestamp`` for ``created_at``
          patterns, plus ``pii`` / ``sensitive`` stamps for columns that
          obviously carry personal or credential data.

        For struct-shaped fields (schemas) ``primary_key`` /
        ``partition_by`` / ``cluster_by`` entries on this field's
        metadata get consumed into per-child tags, and each child is
        autotagged in turn — so ``schema.autotag()`` propagates without
        the caller having to walk children manually.

        Returns a new struct-shaped Field for schema-style autotagging,
        or ``self`` for primitive autotagging — both modes also stamp
        in place so existing ``f.autotag()`` chains keep working.
        """
        if self.type_id is DataTypeId.STRUCT:
            primary_key_names = self._pop_field_name_list(b"primary_key")
            partition_by_names = self._pop_field_name_list(b"partition_by")
            cluster_by_names = self._pop_field_name_list(b"cluster_by")

            new_fields: list[Field] = []
            for f in self.children:
                if primary_key_names and f.name in primary_key_names:
                    f.with_primary_key(True, inplace=True)
                if partition_by_names and f.name in partition_by_names:
                    f.with_partition_by(True, inplace=True)
                if cluster_by_names and f.name in cluster_by_names:
                    f.with_cluster_by(True, inplace=True)
                new_fields.append(f.autotag())

            self.update_tags(tags)
            return type(self)._make_struct(
                children=new_fields,
                metadata=self.metadata,
                name=self.name,
                nullable=self.nullable,
            )

        my_tags: dict[bytes, bytes] = dict(self.dtype.autotag())
        if not self.nullable:
            my_tags[b"nullable"] = b"false"

        self.update_tags(my_tags)
        if tags:
            self.update_tags(tags)
        return self

    # ==================================================================
    # Peek — sample an iterable / list for a one-shot field inference
    # ==================================================================

    @classmethod
    def peek_from(cls, obj: Any) -> tuple[Any, "Field"]:
        if isinstance(obj, (Iterator, Generator)):
            first = next(obj, None)

            if first is None:
                return None, cls._make_blank()

            obj = itertools.chain((first,), obj)

            return obj, cls._coerce_to_cls(cls.from_(first))
        elif isinstance(obj, list):
            if not obj:
                return obj, cls._make_blank()
            return obj, cls._coerce_to_cls(cls.from_(obj[0]))
        else:
            return obj, cls._coerce_to_cls(cls.from_(obj))

    @classmethod
    def _make_blank(cls) -> "Field":
        """Empty / null-typed instance — used as the peek fallback.

        Subclasses with schema-shaped ``__init__`` (e.g. ``Schema``)
        can't accept ``Field``'s direct (name, dtype) call; build via
        ``__new__`` + ``Field.__init__`` to bypass the shim.
        """
        inst = cls.__new__(cls)
        Field.__init__(inst, name="", dtype=NullType())
        return inst

    @classmethod
    def _coerce_to_cls(cls, value: "Field") -> "Field":
        """Lift a plain :class:`Field` to ``cls`` if needed.

        Subclass classmethods (e.g. ``Schema.from_arrow``) want a
        ``cls`` instance back. When the inherited factory hands us a
        plain Field, route it through :meth:`from_field` so the result
        matches the caller's expectations.
        """
        if cls is Field or isinstance(value, cls):
            return value
        return cls.from_field(value)

    @classmethod
    def from_field(cls, f: "Field") -> "Field":
        """Lift a :class:`Field` to ``cls``.

        For ``cls is Field`` this is identity. For subclasses (e.g.
        :class:`Schema`) it normalises the input to the subclass shape
        — for struct dtypes we keep the children, for non-struct we
        wrap the field as a single-child struct so the schema-shape
        contract holds.
        """
        if cls is Field or isinstance(f, cls):
            return f
        struct_field = f.to_struct()
        return cls._make_struct(
            children=struct_field.children,
            metadata=struct_field.metadata,
            name=struct_field.name,
            nullable=struct_field.nullable,
        )

    @classmethod
    def from_fields(
        cls,
        fields: Iterable["Field | Any"],
        *,
        name: str = DEFAULT_FIELD_NAME,
        nullable: bool = False,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
    ) -> "Field":
        """Build a struct-shaped instance from a list of fields."""
        return cls._make_struct(
            children=[Field.from_any(f) for f in fields],
            name=name,
            nullable=bool(nullable),
            metadata=_normalize_metadata(metadata, tags=tags),
        )

    # ==================================================================
    # Constructors — generic dispatch entry points
    # ==================================================================

    @classmethod
    def make_constraint_field(
        cls,
        fields: Iterable["Field"],
        name: str = "",
        prefix: str = "",
        default: Any = ...,
        name_limit: int = 256,
    ) -> "Field | None":
        if not fields:
            if default is ...:
                raise ValueError(
                    f"No fields specified as primary key for struct type in {fields!r}"
                )
            return default

        fields = [cls.from_(_) for _ in fields]
        name = name or safe_constraint_name(
            [_.name for _ in fields], prefix=prefix, limit=name_limit
        )

        if len(fields) == 1:
            keep = fields[0].with_name(name)
        else:
            keep = cls(name=name, dtype=StructType.from_fields(fields), nullable=False)

        return keep.with_name(name=name).with_constraint_key(True)

    @classmethod
    def from_any(
        cls, obj: Any, *, name: str | None = None, metadata: dict | None = None
    ) -> "Field":
        if isinstance(obj, cls):
            return obj

        # Cross-cast within the Field hierarchy — caller asked for cls
        # (e.g. Schema) but handed us a sibling Field. Lift through
        # :meth:`from_field` so the result matches the requested class.
        if isinstance(obj, Field):
            return cls.from_field(obj)

        if isinstance(obj, DataType):
            return cls(name=name or DEFAULT_FIELD_NAME, dtype=obj, metadata=metadata)

        ns, _ = ObjectSerde.module_and_name(obj)

        if ns.startswith("yggdrasil"):
            if isinstance(obj, schema_class()):
                return cls.from_field(obj)
        elif ns.startswith("pyarrow"):
            return cls.from_arrow(obj)
        elif ns.startswith("polars"):
            return cls.from_polars(obj)
        elif ns.startswith("pandas"):
            return cls.from_pandas(obj)
        elif ns.startswith("pyspark"):
            return cls.from_spark(obj)

        # Path-like check — only for inputs that could *plausibly* be paths.
        # Routes through ``URL.is_pathish`` (the cross-cutting check) so
        # ``Field.from_any`` doesn't have to know about every Path subtype.
        if isinstance(obj, (pathlib.PurePath, os.PathLike)):
            from yggdrasil.io.url import URL

            if URL.is_pathish(obj):
                try:
                    pc = path_class()
                    return pc.from_(obj).as_media().collect_schema().to_field()
                except Exception:
                    pass

        if isinstance(obj, type):
            return cls.from_pytype(obj)

        if hasattr(obj, "collect_schema"):
            return cls.from_any(obj.collect_schema())

        if callable(obj):
            return cls.from_any(obj())

        if is_dataclass(obj):
            return cls.from_dataclass(obj)

        if isinstance(obj, str):
            return cls.from_str(obj)

        if isinstance(obj, Mapping):
            # ``from_dict`` expects a *schema* dict (``{"name": ..., "dtype": ...}``
            # / ``{"type_text": ...}`` shape) — handing it an *instance* dict
            # like ``{"id": 1, "name": "alice"}`` raises because no schema
            # keys resolve. That's exactly what ``Schema.from_(row_dict)``
            # call sites (e.g. ``Dataset.infer_schema``) hand us. Try
            # the schema-dict route first, fall back to Arrow's struct
            # inference from the row instance.
            field_obj = cls.from_dict(obj, default=None)
            if field_obj is not None:
                return field_obj
            try:
                table = pa.Table.from_pylist([dict(obj)])
            except Exception:
                # Mixed / non-Arrow-compatible values — let the original
                # schema-dict error fire so the caller sees the real reason.
                return cls.from_dict(obj)
            return cls.from_arrow(table.schema)

        if hasattr(obj, "value"):
            return cls.from_any(obj.value)

        if isinstance(obj, (list, tuple)):
            return cls.from_list(obj)

        raise TypeError(f"Cannot build Field from {type(obj).__name__}")

    @classmethod
    def from_(cls, obj: Any, *, name: str | None = None) -> "Field":
        return cls.from_any(obj)

    @classmethod
    def from_list(cls, value: list) -> "Field":
        if not value:
            raise ValueError("Cannot build Field from empty list")

        try:
            table = pa.Table.from_pylist(value)
            return cls.from_arrow(table)
        except Exception:
            pass

        for item in range(min(len(value), 5)):
            try:
                return cls.from_any(value[item])
            except Exception:
                pass

        raise ValueError("Cannot build Field from list with no valid items")

    # ------------------------------------------------------------------
    # Constructors — Python types, dataclasses, strings, dicts, JSON
    # ------------------------------------------------------------------

    @classmethod
    def from_pytype(
        cls,
        hint: Any,
        *,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        if isinstance(hint, str):
            parsed = ParsedDataType.from_(hint)
            resolved_name = name or parsed.name or _default_name(hint)
            resolved_nullable = parsed.nullable if nullable is None else bool(nullable)

            if resolved_nullable is None:
                resolved_nullable = False if parsed.type_id == DataTypeId.NULL else True

            return cls(
                name=resolved_name,
                dtype=DataType.from_parsed(parsed),
                nullable=resolved_nullable,
                metadata=_normalize_metadata(metadata, tags=tags),
                default=default,
            )

        base_hint, inferred_nullable = _unwrap_nullable_hint(hint)
        resolved_nullable = inferred_nullable if nullable is None else bool(nullable)

        return cls(
            name=name or _default_name(base_hint),
            dtype=DataType.from_pytype(base_hint),
            nullable=resolved_nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
            default=default,
        )

    @classmethod
    def from_dataclass(
        cls,
        hint: Any,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default_value: Any = None,
    ) -> "Field":
        if not is_dataclass(hint):
            raise TypeError(f"Unsupported dataclass input: {hint!r}")
        elif not isinstance(hint, type):
            hint = hint.__class__

        dtype = StructType.from_dataclass(hint)

        return cls(
            name=name or hint.__name__,
            dtype=dtype,
            nullable=False if nullable is None else bool(nullable),
            metadata=_normalize_metadata(
                metadata, tags=tags, default_value=default_value
            ),
        )

    @classmethod
    def from_dataclass_field(
        cls,
        value: dataclasses.Field,
        *,
        owner: type | None = None,
    ) -> "Field":
        default = None
        # ``value`` is a ``dataclasses.Field`` here — its ``default``
        # is the stdlib attribute, not our renamed ``default_value``.
        if value.default is not None:
            default = value.default
        elif value.default_factory is not None:  # type: ignore[attr-defined]
            try:
                default = value.default_factory()  # type: ignore[misc]
            except Exception:
                default = None

        resolved_hint = value.type

        if owner is not None:
            try:
                import typing as _typing

                resolved = _typing.get_type_hints(owner, include_extras=True)
                resolved_hint = resolved.get(value.name, value.type)
            except Exception:
                resolved_hint = value.type

        if isinstance(resolved_hint, str):
            parsed = ParsedDataType.from_(resolved_hint)
            inferred_nullable = bool(parsed.nullable)
            if default is None:
                inferred_nullable = True

            return cls(
                name=value.name,
                dtype=DataType.from_parsed(parsed),
                nullable=inferred_nullable,
                metadata=None,
                default=default,
            )

        base_hint, inferred_nullable = _unwrap_nullable_hint(resolved_hint)

        if default is None:
            inferred_nullable = True

        return cls(
            name=value.name,
            dtype=DataType.from_pytype(base_hint),
            nullable=inferred_nullable,
            metadata=None,
            default=default,
        )

    @classmethod
    def from_str(cls, value: str) -> "Field":
        text = str(value).strip()
        if not text:
            raise ValueError("Field string cannot be empty")

        if text.startswith("{") and text.endswith("}"):
            payload = json_module.loads(text)
            if not isinstance(payload, Mapping):
                raise ValueError("Field JSON string must decode to an object")
            return cls.from_dict(payload)

        name_text, type_text = _split_field_shorthand(text)
        parsed = ParsedDataType.from_(type_text)
        parsed_name, name_nullable = _parse_field_name_token(name_text)

        nullable = parsed.nullable
        if name_nullable is not None:
            nullable = name_nullable
        if nullable is None:
            nullable = False if parsed.type_id == DataTypeId.NULL else True

        return cls(
            name=parsed_name,
            dtype=DataType.from_parsed(parsed),
            nullable=nullable,
            metadata=None,
            default=None,
        )

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedDataType,
        *,
        name: str | None = None,
        nullable: bool | None = None,
        metadata: dict[bytes | str, bytes | str | object] | None = None,
        tags: dict[bytes | str, bytes | str | object] | None = None,
        default: Any = None,
    ) -> "Field":
        if not isinstance(parsed, ParsedDataType):
            raise TypeError(
                f"Field.from_parsed expects ParsedDataType; got {type(parsed).__name__}"
            )

        resolved_name = name or parsed.name or DEFAULT_FIELD_NAME
        resolved_nullable = parsed.nullable if nullable is None else bool(nullable)

        if resolved_nullable is None:
            resolved_nullable = False if parsed.type_id == DataTypeId.NULL else True

        return cls(
            name=resolved_name,
            dtype=DataType.from_parsed(parsed),
            nullable=resolved_nullable,
            metadata=_normalize_metadata(metadata, tags=tags),
            default=default,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], default: Any = ...) -> "Field":
        try:
            if not value:
                raise ValueError(
                    f"Cannot build {cls.__name__} from empty dictionary {value!r}"
                )

            name = value.get("name") or ""
            nullable = bool(value.get("nullable", True))
            dtype = (
                value.get(b"dtype")
                or value.get("dtype")
                or value.get(_TYPE_JSON_METADATA_KEY)
                or value.get(_TYPE_JSON_METADATA_KEY.decode())
            )
            if dtype:
                dtype = DataType.from_json(dtype)

            if dtype is None:
                for key in ("type_text", "type_json", "type"):
                    found = value.get(key)
                    if found is not None:
                        try:
                            dtype = DataType.from_any(found)
                            break
                        except Exception:
                            pass

                if dtype is None:
                    raise ValueError(
                        f"Cannot build {cls.__name__} from dictionary without type: {value!r}"
                    )

            metadata = _normalize_metadata(value.get("metadata", {}), tags=None)

            return cls(
                name=name,
                dtype=dtype,
                nullable=nullable,
                metadata=metadata,
            )
        except Exception as e:
            if default is ...:
                raise ValueError(
                    f"Cannot build {cls.__name__} from dictionary: {e}"
                ) from e
            return default

    @classmethod
    def from_json(cls, value: Any) -> "Field":
        if isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value).decode("utf-8")

        if not isinstance(value, str):
            if isinstance(value, Mapping):
                return cls.from_dict(value)
            elif isinstance(value, (list, tuple)):
                return cls.from_list(value)
            raise TypeError(
                f"Field.from_json expects str or bytes-like input; got {type(value).__name__}"
            )

        loaded = json_module.loads(value)

        if isinstance(loaded, Mapping):
            return cls.from_dict(loaded)
        elif isinstance(loaded, str):
            return cls.from_str(loaded)
        else:
            raise TypeError(f"Cannot build Field from {type(loaded).__name__}")

    @classmethod
    def from_path(cls, path: Any) -> "Field":
        path = path_class().from_(path)
        return path.as_media().collect_schema().to_field()

    # ------------------------------------------------------------------
    # Constructors — arrow
    # ------------------------------------------------------------------

    @classmethod
    def from_arrow(
        cls,
        value: pa.Field | pa.Schema | pa.DataType | Any,
        from_metadata: bool = True,
    ) -> "Field":
        if not isinstance(value, pa.Field):
            if isinstance(value, pa.Field):
                return cls.from_arrow_field(value, from_metadata=from_metadata)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value, from_metadata=from_metadata)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None),
                    from_metadata=from_metadata,
                )
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                nullable = value.null_count > 0 or len(value) == 0

                return cls.from_arrow_field(
                    pa.field(
                        DEFAULT_FIELD_NAME, value.type, nullable=nullable, metadata=None
                    ),
                    from_metadata=from_metadata,
                )

            if hasattr(value, "schema"):
                value = value.schema
            elif hasattr(value, "arrow_schema"):
                value = value.arrow_schema
            elif hasattr(value, "arrow_type"):
                value = value.arrow_type

            if callable(value):
                value = value()

            if isinstance(value, pa.Field):
                return cls.from_arrow_field(value, from_metadata=from_metadata)
            if isinstance(value, pa.Schema):
                return cls.from_arrow_schema(value, from_metadata=from_metadata)
            if isinstance(value, pa.DataType):
                return cls.from_arrow_field(
                    pa.field(DEFAULT_FIELD_NAME, value, nullable=True, metadata=None),
                    from_metadata=from_metadata,
                )
            raise TypeError(f"Cannot build Field from {type(value).__name__}")

        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=DataType.from_arrow_type(value.type),
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_arrow_schema(cls, value: pa.Schema, from_metadata: bool = True):
        if from_metadata and value.metadata:
            found = value.metadata.get(_TYPE_JSON_METADATA_KEY)
            if found:
                return cls.from_json(found)

        name = DEFAULT_FIELD_NAME
        if value.metadata:
            name = value.metadata.get(b"name", DEFAULT_FIELD_NAME.encode()).decode(
                "utf-8"
            )

        return cls(
            name=name,
            dtype=DataType.from_arrow_schema(value),
            nullable=False,
            metadata=_strip_internal_metadata(value.metadata),
        )

    @classmethod
    def from_arrow_field(cls, value: pa.Field, from_metadata: bool = True) -> "Field":
        if from_metadata and value.metadata:
            dtype = value.metadata.get(_TYPE_JSON_METADATA_KEY, None)
            if dtype is None:
                dtype = DataType.from_arrow_type(value.type)
            else:
                dtype = DataType.from_json(dtype)
        else:
            dtype = DataType.from_arrow_type(value.type)

        return cls(
            name=value.name or DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=value.nullable,
            metadata=_strip_internal_metadata(value.metadata),
        )

    # ------------------------------------------------------------------
    # Constructors — pandas
    # ------------------------------------------------------------------

    @classmethod
    def from_pandas(cls, obj: Any = None) -> "Field":
        pd = pandas_module()

        if isinstance(obj, pd.DataFrame):
            table = pa.Table.from_pandas(obj)
            raw_meta = (table.schema.metadata or {}).get(b"pandas")
            index_levels: dict[str, int] = {}
            if raw_meta:
                from yggdrasil.pickle import json as ygg_json
                pmeta = ygg_json.loads(raw_meta)
                index_levels = {
                    e: pos
                    for pos, e in enumerate(pmeta.get("index_columns", ()))
                    if isinstance(e, str)
                }
            struct_field = cls.from_arrow_schema(
                table.schema.remove_metadata(),
            )
            if index_levels:
                for child in struct_field.fields:
                    level = index_levels.get(child.name)
                    if level is not None:
                        child.with_index_key(True, level=level, inplace=True)
            return struct_field

        if isinstance(obj, pd.Series):
            nullable = bool(obj.isna().any())
            return cls(
                name=obj.name or DEFAULT_FIELD_NAME,
                dtype=DataType.from_pandas(obj),
                nullable=nullable,
                metadata=None,
            )

        if isinstance(obj, pd.Index):
            nullable = bool(obj.hasnans) if hasattr(obj, "hasnans") else False
            return cls(
                name=obj.name or DEFAULT_FIELD_NAME,
                dtype=DataType.from_pandas(obj),
                nullable=nullable,
                tags={b"index_key": b"true"},
            )

        return cls(
            name=_default_name(obj),
            dtype=DataType.from_pandas(obj),
            nullable=obj is pd.NA,
            metadata=None,
        )

    # ------------------------------------------------------------------
    # Constructors — polars
    # ------------------------------------------------------------------

    @classmethod
    def from_polars(cls, obj: Any = None) -> "Field":
        pl = polars_module()

        if isinstance(obj, pl.Field):
            return cls.from_polars_field(obj)
        if isinstance(obj, pl.Schema):
            return cls.from_polars_schema(obj)
        if isinstance(obj, pl.DataFrame):
            return cls.from_polars_schema(obj.schema)
        if isinstance(obj, pl.LazyFrame):
            schema = obj.collect_schema()
            if isinstance(schema, pl.Schema):
                return cls.from_polars_schema(schema)
        if isinstance(obj, pl.Series):
            return cls.from_polars_series(obj)

        dtype = DataType.from_polars(obj)

        return cls(
            name=DEFAULT_FIELD_NAME,
            dtype=dtype,
            nullable=True,
            metadata={},
        )

    @classmethod
    def from_polars_series(cls, series: "polars.Series") -> "Field":
        dtype = DataType.from_polars_type(series.dtype)
        nullable = bool(series.null_count() > 0)
        return cls(
            name=series.name,
            dtype=dtype,
            nullable=nullable,
            metadata={},
        )

    @classmethod
    def from_polars_field(cls, value: "polars.Field") -> "Field":
        try:
            nullable = getattr(value, "nullable", True)
        except Exception:
            nullable = True

        try:
            metadata = getattr(value, "metadata", None)
            if metadata is not None:
                metadata = _normalize_metadata(metadata, tags=None)
        except Exception:
            metadata = None

        return cls(
            name=value.name,
            dtype=DataType.from_polars_type(value.dtype),
            nullable=nullable,
            metadata=metadata,
        )

    @classmethod
    def from_polars_schema(cls, value: "polars.Schema") -> "Field":
        return cls(
            name=DEFAULT_FIELD_NAME,
            dtype=DataType.from_polars_schema(value),
            nullable=False,
        )

    # ------------------------------------------------------------------
    # Constructors — spark
    # ------------------------------------------------------------------

    @classmethod
    def from_spark(cls, obj: Any = None, from_metadata: bool = True) -> "Field":
        _psql = spark_sql_module()

        if isinstance(obj, _psql.types.StructField):
            return cls.from_spark_field(obj, from_metadata=from_metadata)
        if isinstance(obj, spark_dataframe_classes()):
            obj = _psql.types.StructField(
                DEFAULT_FIELD_NAME,
                obj.schema,
                nullable=False,
                metadata=None,
            )
            return cls.from_spark_field(obj, from_metadata=from_metadata)
        if isinstance(obj, _psql.types.DataType):
            obj = _psql.types.StructField(
                DEFAULT_FIELD_NAME,
                obj,
                nullable=True,
                metadata=None,
            )
            return cls.from_spark_field(obj, from_metadata=from_metadata)
        if isinstance(obj, spark_column_classes()):
            return cls.from_spark_column(obj)

        raise TypeError(
            f"Cannot build {cls.__name__} from {type(obj).__name__}"
        )

    @classmethod
    def from_spark_column(cls, column: "ps.Column") -> "Field":
        """Build a :class:`Field` from a ``pyspark.sql.Column``.

        ``Column`` objects don't expose a typed dtype on the public
        Python surface — we read the SQL-rendered expression instead
        and parse that:

        * ``id`` — bare reference. Name is ``id``, dtype defers
          to the fallback (``ObjectType``) since neither the JVM
          nor the Spark Connect proxy exposes the underlying
          schema on a free-standing column.
        * ``CAST(<expr> AS <dtype>)`` / ``CAST(<expr> AS <dtype>)``
          — name follows the inner ``<expr>``'s leaf, dtype reads
          straight off ``<dtype>`` through
          :meth:`DataType.from_str`. Covers ``df["x"].cast("string")``,
          ``df["x"].astype("decimal(10,2)")``, ``F.col("x").cast(StringType())``.
        * ``<expr> AS <alias>`` — name follows ``<alias>``, dtype
          comes from the inner ``<expr>`` (recurses, so a cast
          inside an alias keeps its dtype).
        * Anything else falls back to the full SQL string as the
          name with :class:`ObjectType` as the dtype, since we
          can't infer the dtype of an arbitrary Catalyst
          expression without binding it through
          :meth:`SparkSession.createDataFrame` (which would be a
          live JVM round trip the caller didn't ask for).

        Source of the SQL string, in order:

        1. Classic Spark: ``column._jc.toString()`` — the JVM Column.
        2. Spark Connect: ``column._expr.__repr__()`` — the proxy
           doesn't have ``_jc`` (accessing it raises
           ``PySparkAttributeError(JVM_ATTRIBUTE_NOT_SUPPORTED)``)
           but ``_expr.__repr__`` is exactly what
           ``Column.__repr__`` wraps as ``"Column<'<sql>'>"``.
        3. ``repr(column)`` stripped of the ``Column<'…'>``
           wrapper — last-resort for any future PySpark whose
           internal slots renamed.

        Use :meth:`Field.from_spark_field` instead when the caller
        already has the resolved ``StructField`` (e.g. from
        ``df.schema.fields[i]``) — that path keeps the precise dtype
        without going through the SQL string.
        """
        from .types.primitive import ObjectType

        rendered = _render_spark_column_sql(column)
        if not rendered:
            raise TypeError(
                f"Cannot read SQL representation from pyspark Column: {column!r}"
            )

        name, dtype = _parse_spark_column_sql(rendered)
        if dtype is None:
            dtype = ObjectType()
        if not name:
            name = DEFAULT_FIELD_NAME
        # Nullability is not part of the Column SQL string. Default
        # ``True`` matches the rest of the from-engine constructors
        # (``from_polars_field`` etc.) and the conservative-Spark
        # ``Expression.nullable`` would default to True anyway for
        # everything but ``isNotNull``-gated expressions.
        return cls(name=name, dtype=dtype, nullable=True)

    @classmethod
    def from_spark_field(
        cls, value: "pst.StructField", from_metadata: bool = True
    ) -> "Field":
        # Spark stores metadata as ``dict[str, Any]`` (a Java
        # ``Metadata`` view). The ``type_json`` round-trip blob — only
        # written for Map/Array dtypes by :meth:`to_pyspark_field` —
        # therefore lives under the str key, not the bytes one. Pull
        # ``value.metadata`` once so the (callable on the Java side)
        # accessor doesn't fire twice; route the ``dataType`` through
        # :meth:`DataType.from_spark_type` directly so we skip the
        # ``from_spark`` isinstance fan-out (we already know it's a
        # ``pst.DataType``) — the cached lookup then collapses the
        # walk for repeated dtype tokens.
        meta = value.metadata
        if from_metadata and meta:
            dtype_json = meta.get(_TYPE_JSON_METADATA_KEY_STR)
            if dtype_json is None:
                dtype_json = meta.get(_TYPE_JSON_METADATA_KEY)
            if dtype_json is None:
                dtype = DataType.from_spark_type(value.dataType)
            else:
                dtype = DataType.from_json(dtype_json)
        else:
            dtype = DataType.from_spark_type(value.dataType)

        # ``_normalize_metadata`` re-builds a fresh dict even when the
        # input is the empty dict that Spark hands back by default —
        # short-circuit so the no-metadata path skips both the
        # normalize + the strip-internal-keys pass.
        if meta:
            normalized = _strip_internal_metadata(
                _normalize_metadata(meta, tags=None)
            )
        else:
            normalized = None

        return cls(
            name=value.name,
            dtype=dtype,
            nullable=value.nullable,
            metadata=normalized,
        )

    # ==================================================================
    # Exporters — dict / JSON / arrow / polars / spark / databricks DDL
    # ==================================================================

    def to_dict(self, dump_parent: bool = False) -> dict[str, Any]:
        """Serialize this field to a JSON-friendly dict.

        ``dump_parent`` (default ``False``) controls whether
        :attr:`parent` — the structural back-pointer to the field
        this one is nested under — is included. Children are still
        emitted via the dtype's ``to_dict`` (a struct field's
        ``dtype`` carries its members), so dropping ``parent``
        prevents the recursion that would otherwise echo the whole
        ancestor chain into every nested field's payload.
        """
        out = dict(
            name=self.name,
            dtype=self.dtype.to_dict(),
            nullable=self.nullable,
        )

        if self.metadata:
            out["metadata"] = {k.decode(): v.decode() for k, v in self.metadata.items()}

        if dump_parent and self.parent is not None:
            out["parent"] = self.parent.to_dict(dump_parent=False)

        return out

    def to_json(self, to_bytes: bool = False, dump_parent: bool = False) -> AnyStr:
        payload = self.to_dict(dump_parent=dump_parent)

        return json_module.dumps(
            payload,
            safe=False,
            to_bytes=to_bytes,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def to_arrow(self):
        return self.to_arrow_field()

    def to_arrow_type(self):
        return self.arrow_type

    def to_arrow_field(self, dump_json: bool = False) -> pa.Field:
        """Project to a :class:`pa.Field`.

        Arrow preserves nested-type structure (struct, list, map)
        with per-field metadata recursively, so the dtype intent
        round-trips natively without us stuffing a ``type_json`` blob
        into the metadata. Only callers that need the exact
        :class:`DataType` subclass back (e.g. Decimal precision /
        Timestamp tz / extension types) should pass
        ``dump_json=True``.

        ``dump_json`` defaults to ``False``; the cached path is the
        canonical (no-blob) shape, which is what every internal caller
        wants now that :meth:`from_arrow_field` falls back through
        :meth:`DataType.from_arrow_type` when the blob is missing.
        """
        if not dump_json and self._arrow_field is not None:
            return self._arrow_field

        metadata = self.metadata.copy() if self.metadata else {}
        if dump_json:
            metadata[_TYPE_JSON_METADATA_KEY] = self.dtype.to_json(to_bytes=True)

        built = pa.field(
            name=self.name,
            type=self.arrow_type,
            nullable=self.nullable,
            metadata=metadata or None,
        )
        if not dump_json:
            object.__setattr__(self, "_arrow_field", built)
        return built

    def to_arrow_schema(self) -> pa.Schema:
        """Project this field as a top-level :class:`pa.Schema`.

        Struct-shaped fields (including :class:`~yggdrasil.data.Schema`)
        unfold their children into the schema's columns; non-struct
        fields produce a single-column schema with ``self`` as that
        column. The schema-level metadata mirrors ``self.metadata``,
        plus the field's name / nullable flag re-embedded as
        ``b"name"`` / ``b"nullable"`` so :meth:`Field.from_arrow_schema`
        can recover them (``pa.Schema`` has no native slot for either).
        """
        if self._arrow_schema is not None:
            return self._arrow_schema

        if self.type_id is DataTypeId.STRUCT:
            # ``self.fields`` excludes constraint-only children — those
            # are a yggdrasil concept with no Arrow equivalent.
            arrow_fields = [child.to_arrow_field() for child in self.fields]
        else:
            arrow_fields = [self.to_arrow_field()]

        meta = dict(self.metadata) if self.metadata else {}
        if self.name and self.name != DEFAULT_FIELD_NAME:
            meta.setdefault(b"name", self.name.encode("utf-8"))
        if self.nullable:
            meta.setdefault(b"nullable", b"true")
        built = pa.schema(arrow_fields, metadata=meta or None)
        object.__setattr__(self, "_arrow_schema", built)
        return built

    def to_polars_field(self) -> "polars.Field":
        if self._polars_field is not None:
            return self._polars_field

        pl = polars_module()
        built = pl.Field(self.name, self.dtype.to_polars())
        try:
            built.nullable = self.nullable
        except AttributeError:
            pass
        try:
            built.metadata = self.metadata
        except AttributeError:
            pass
        object.__setattr__(self, "_polars_field", built)
        return built

    def to_polars_schema(self) -> "polars.Schema":
        """Project this field as a :class:`polars.Schema`.

        Struct-shaped fields unfold into the schema's columns;
        non-struct fields produce a single-column schema.
        """
        if self._polars_schema is not None:
            return self._polars_schema

        pl = polars_module()
        if self.type_id is DataTypeId.STRUCT:
            entries = [(c.name, c.dtype.to_polars()) for c in self.fields]
        else:
            entries = [(self.name, self.dtype.to_polars())]
        built = pl.Schema(entries)
        object.__setattr__(self, "_polars_schema", built)
        return built

    def to_pyspark_field(self) -> "pst.StructField":
        """Project to a Spark :class:`StructField`.

        Spark's :class:`StructType` preserves struct children with
        their own metadata, so primitive and struct dtypes don't
        need a ``type_json`` round-trip blob. Spark's :class:`MapType`
        / :class:`ArrayType` only carry the element / key+value Spark
        types and lose any field-level metadata on the way through, so
        we dump the dtype JSON for those (and only those) to recover
        the original yggdrasil dtype on read.
        """
        if self._spark_field is not None:
            return self._spark_field

        import pyspark.sql as pyspark_sql

        metadata = (
            {
                (key.decode("utf-8") if isinstance(key, bytes) else str(key)): (
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                )
                for key, value in self.metadata.items()
            }
            if self.metadata
            else {}
        )
        if self._needs_spark_type_json():
            metadata[_TYPE_JSON_METADATA_KEY.decode("utf-8")] = self.dtype.to_json(
                to_bytes=False
            )
        built = pyspark_sql.types.StructField(
            self.name,
            self.dtype.to_spark(),
            self.nullable,
            metadata=metadata,
        )
        object.__setattr__(self, "_spark_field", built)
        return built

    def _needs_spark_type_json(self) -> bool:
        """True iff this field's dtype loses fidelity through Spark.

        Spark's :class:`MapType` / :class:`ArrayType` carry only the
        element Spark types — no per-field metadata — so a yggdrasil
        dtype that lives below them (key/value/item field) only
        round-trips if we dump the parent dtype as JSON. Struct,
        primitives, and scalars are preserved natively by
        :class:`StructField`'s recursive metadata, so the dump is
        skipped there.
        """
        return self.type_id in (DataTypeId.MAP, DataTypeId.ARRAY)

    def to_spark_schema(self) -> "pst.StructType":
        """Project this field as a top-level Spark :class:`StructType`.

        Struct-shaped fields unfold their children into the
        StructType's fields; non-struct fields produce a single-field
        StructType.
        """
        if self._spark_schema is not None:
            return self._spark_schema

        pyspark_sql = spark_sql_module()
        if self.type_id is DataTypeId.STRUCT:
            fields = [c.to_pyspark_field() for c in self.fields]
        else:
            fields = [self.to_pyspark_field()]
        built = pyspark_sql.types.StructType(fields)
        object.__setattr__(self, "_spark_schema", built)
        return built

    def as_spark(self) -> "Field":
        """Return a Field whose ``dtype`` is Spark-compatible.

        Stays on the yggdrasil side of the boundary — the result is
        still a :class:`Field`, just with :attr:`dtype` swapped for
        whatever ``self.dtype.as_spark()`` produced (an unsigned int
        widens to signed, a non-UTC timestamp drops to naive,
        ``TimeType`` becomes ``StringType``, …). When the dtype is
        already Spark-compatible the same instance is returned, so
        the call is cheap to make defensively.

        Use :meth:`to_pyspark_field` when you need an actual
        ``pyspark.sql.types.StructField``.
        """
        spark_dtype = self.dtype.as_spark()
        if spark_dtype is self.dtype:
            return self
        return Field(
            name=self.name,
            dtype=spark_dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )

    def as_polars(self) -> "Field":
        """Return a Field whose ``dtype`` is Polars-compatible.

        Mirrors :meth:`as_spark` for Polars — :attr:`dtype` is
        swapped for ``self.dtype.as_polars()`` (sub-32-bit floats
        widen to ``Float32Type``, second-precision timestamps /
        durations widen to milliseconds, nested types recurse).
        Already-Polars-compatible fields return ``self`` so the call
        is cheap to make defensively. Use :meth:`to_polars_field`
        when you need a real ``pl.Field``.
        """
        polars_dtype = self.dtype.as_polars()
        if polars_dtype is self.dtype:
            return self
        return Field(
            name=self.name,
            dtype=polars_dtype,
            nullable=self.nullable,
            metadata=self.metadata,
        )

    def to_schema(
        self,
        metadata: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
    ) -> "Schema":
        from .schema import Schema

        base = self.to_struct()
        final_metadata = base.metadata.copy() if base.metadata else {}
        new_metadata = _normalize_metadata(metadata, tags=tags)
        if new_metadata:
            final_metadata.update(new_metadata)

        return Schema(base.children,
            metadata=final_metadata or None,
            name=base.name if base.name != DEFAULT_FIELD_NAME else None,
            nullable=base.nullable,
        )

    def to_struct(self):
        dtype = self.dtype.to_struct(name=self.name)
        return Field(self.name, dtype, self.nullable, self.metadata)

    def to_spark_name(
        self,
        *,
        with_name: bool = True,
        with_nullable: bool = True,
        with_comment: bool = True,
    ) -> str:
        from yggdrasil.databricks.sql.sql_utils import escape_sql_string, quote_ident

        name_str = f"{quote_ident(self.name)} " if with_name else ""
        nullable_str = " NOT NULL" if with_nullable and not self.nullable else ""

        comment_str = ""
        if with_comment and self.metadata and b"comment" in self.metadata:
            comment = (self.metadata[b"comment"] or b"").decode("utf-8")
            comment_str = f" COMMENT '{escape_sql_string(comment)}'"

        if not pa.types.is_nested(self.arrow_type):
            dtype = DataType.from_arrow_type(self.arrow_type)
            if isinstance(dtype, type) and issubclass(dtype, DataType):
                dtype = dtype()
            sql_type = dtype.to_spark_name()
            return f"{name_str}{sql_type}{nullable_str}{comment_str}"

        if pa.types.is_struct(self.arrow_type):
            # Nested children never carry ``NOT NULL`` in the rendered DDL.
            # Spark/Delta refuse the implicit cast between two structs that
            # differ only in child nullability (``DATATYPE_MISMATCH.``
            # ``CAST_WITHOUT_SUGGESTION``), and the parquet reader used by
            # ``MERGE … USING parquet.`<path>``` always hands back nullable
            # children regardless of file metadata — keeping them nullable
            # at the schema-rendering layer matches that and lets the
            # MERGE go through.
            struct_body = ", ".join(
                Field.from_arrow(child).to_spark_name(
                    with_comment=False,
                    with_nullable=False,
                )
                for child in self.arrow_type
            )
            return f"{name_str}STRUCT<{struct_body}>{nullable_str}{comment_str}"

        if pa.types.is_map(self.arrow_type):
            map_type: pa.MapType = self.arrow_type
            key_type = Field.from_arrow(map_type.key_field).to_spark_name(
                with_name=False,
                with_comment=False,
                with_nullable=False,
            )
            val_type = Field.from_arrow(map_type.item_field).to_spark_name(
                with_name=False,
                with_comment=False,
                with_nullable=False,
            )
            return f"{name_str}MAP<{key_type}, {val_type}>{nullable_str}{comment_str}"

        if pa.types.is_list(self.arrow_type) or pa.types.is_large_list(self.arrow_type):
            list_type: pa.ListType = self.arrow_type
            elem_type = Field.from_arrow(list_type.value_field).to_spark_name(
                with_name=False,
                with_comment=False,
                with_nullable=False,
            )
            return f"{name_str}ARRAY<{elem_type}>{nullable_str}{comment_str}"

        raise TypeError(
            f"Cannot make Databricks DDL from nested type: {self.arrow_type}"
        )

    # ==================================================================
    # Cast — top-level dispatch (`cast` / engine-level `cast_*`)
    # ==================================================================
    #
    # Three granularities, from coarsest to finest:
    #
    # 1. :meth:`cast` — "cast whatever this is". Inspects the module of
    #    *obj* via :meth:`ObjectSerde.module_and_name`, routes to the
    #    engine dispatcher. Also recurses into plain iterators /
    #    iterables as a lazy generator.
    #
    # 2. :meth:`cast_arrow` / :meth:`cast_polars` / :meth:`cast_pandas`
    #    / :meth:`cast_spark` — "I know this is pyarrow/polars/pandas/
    #    spark; figure out the shape". Runs an isinstance walk within
    #    the engine's own types, routes to the narrow method.
    #
    # 3. :meth:`cast_arrow_tabular`, :meth:`cast_polars_series`, ... —
    #    the narrow methods below. They do the actual cast work, then
    #    delegate the post-cast null-fill + alias pass to the matching
    #    :meth:`finalize_*` method. ``self.dtype.type_id == OBJECT`` is
    #    the variant-column passthrough: a variant column must never
    #    be cast.

    def cast(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast *obj* to this field using its native engine.

        Routing is by module prefix via :meth:`ObjectSerde.module_and_name`:

        * ``pyarrow.*`` → :meth:`cast_arrow`
        * ``polars.*``  → :meth:`cast_polars`
        * ``pandas.*``  → :meth:`cast_pandas`
        * ``pyspark.*`` → :meth:`cast_spark`
        * iterator / iterable → recurse per element (lazy generator)
        * everything else → :class:`TypeError`

        ``self.dtype.type_id == OBJECT`` is handled by the narrow
        methods — they pass *obj* through unchanged because a variant
        column must never be cast. No redundant guard here.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.cast_arrow(obj, options=options, **more)
        if ns.startswith("polars"):
            return self.cast_polars(obj, options=options, **more)
        if ns.startswith("pandas"):
            return self.cast_pandas(obj, options=options, **more)
        if ns.startswith("pyspark"):
            return self.cast_spark(obj, options=options, **more)

        # Iterator / iterable fallback — preserve laziness. An iterator
        # that yields pa.RecordBatch items passes through as a
        # generator, each batch cast on demand. str/bytes excluded —
        # they're iterable but never tabular.
        if isinstance(obj, Iterator):
            return (self.cast(item, options=options, **more) for item in obj)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (self.cast(item, options=options, **more) for item in obj)

        raise TypeError(
            f"Field.cast: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r}). Expected pyarrow/polars/pandas/pyspark "
            "tabular / array / series / column, or an iterable of such."
        )

    def cast_arrow(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any pyarrow object — dispatch by shape.

        Table/RecordBatch → :meth:`cast_arrow_tabular`,
        Array/ChunkedArray → :meth:`cast_arrow_array`.
        """
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return self.cast_arrow_tabular(obj, options=options, **more)
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.cast_arrow_array(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_arrow: expected pa.Table / pa.RecordBatch / pa.Array / "
            f"pa.ChunkedArray, got {type(obj).__name__}"
        )

    def cast_polars(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any polars object — dispatch by shape.

        DataFrame/LazyFrame → :meth:`cast_polars_tabular`,
        Series → :meth:`cast_polars_series`,
        Expr → :meth:`cast_polars_expr`.
        """
        pl = polars_module()
        # Tabular first — a DataFrame is never a Series, and the pl
        # lazy import dominates dispatch latency. Doing the isinstance
        # walk here beats paying for it at every call site.
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return self.cast_polars_tabular(obj, options=options, **more)
        if isinstance(obj, pl.Series):
            return self.cast_polars_series(obj, options=options, **more)
        if isinstance(obj, pl.Expr):
            return self.cast_polars_expr(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_polars: expected pl.DataFrame / LazyFrame / Series / "
            f"Expr, got {type(obj).__name__}"
        )

    def cast_pandas(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any pandas object — dispatch by shape.

        DataFrame → :meth:`cast_pandas_tabular` + index check,
        Series → :meth:`cast_pandas_series`.
        """
        pd = pandas_module()
        if isinstance(obj, pd.DataFrame):
            casted = self.cast_pandas_tabular(obj, options=options, **more)
            return self.check_pandas_indexes(casted)
        if isinstance(obj, pd.Series):
            return self.cast_pandas_series(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_pandas: expected pd.DataFrame / pd.Series, "
            f"got {type(obj).__name__}"
        )

    def cast_spark(
        self,
        obj: Any,
        options: "CastOptions | None" = None,
        **more: Any,
    ) -> Any:
        """Cast any spark object — dispatch by shape.

        DataFrame → :meth:`cast_spark_tabular`,
        Column → :meth:`cast_spark_column`.
        """
        if isinstance(obj, spark_dataframe_classes()):
            return self.cast_spark_tabular(obj, options=options, **more)
        if isinstance(obj, spark_column_classes()):
            return self.cast_spark_column(obj, options=options, **more)
        raise TypeError(
            f"Field.cast_spark: expected pyspark.sql.DataFrame / Column, "
            f"got {type(obj).__name__}"
        )

    # ==================================================================
    # Cast — narrow methods (cast body + `finalize_*` tail)
    # ==================================================================
    #
    # Each narrow method follows the same three-step pattern:
    #
    #     1. OBJECT passthrough — variant columns never cast.
    #     2. Resolve CastOptions (merges kwargs, binds target).
    #     3. Delegate the cast body to self.dtype.cast_<engine>_<shape>.
    #     4. Hand the result to self.finalize_<engine>_<shape> for the
    #        null-fill + alias tail.
    #
    # Having the finalize call here — instead of inlining fill + alias
    # — means there's one source of truth for the post-cast cleanup
    # shape. Changing the null-fill semantics or the rename policy
    # touches finalize only; cast body stays put.

    def cast_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        options: "CastOptions | None" = None,
        default_scalar: pa.Scalar | None = None,
        **more,
    ):
        # Object target is a variant — never touch the values.
        if self.dtype.type_id == DataTypeId.OBJECT:
            return array
        options = get_cast_options_class().check(options=options, **more)
        scoped = options.with_target(self)
        # Wrap any inner failure in CastError carrying the *current*
        # field as the target.  Nested rebuilds (struct / list / map
        # children, tabular per-column) all funnel through this method
        # before they reach the dtype-level cast, so one wrap here gets
        # us per-leaf context for every shape — without an explicit
        # try/except at each nested call site.  An already-wrapped
        # CastError (from a deeper leaf) propagates unchanged so the
        # innermost failing field stays named.
        try:
            casted = self.dtype.cast_arrow_array(array, options=scoped)
        except CastError:
            raise
        except Exception as exc:
            # Bind source from the actual array when options didn't
            # carry one — otherwise the wrap renders ``cast ? -> <tgt>``
            # and the reader has to guess what came in. ``check_source``
            # swallows its own peek failures so this never masks the
            # original exception.
            sourced = scoped.check_source(array, copy=False)
            raise CastError(
                str(exc),
                source=sourced.source,
                target=self,
                original=exc,
            ) from exc
        return self.finalize_arrow_array(casted, default_scalar=default_scalar)

    def cast_arrow_tabular(
        self,
        table: pa.Table | pa.RecordBatch,
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return table
        options = get_cast_options_class().check(options=options, **more)
        scoped = options.with_target(self)
        # Per-column failures raise CastError from the inner
        # ``cast_arrow_array`` wrap — that's the atomic field-level
        # context the caller wants. Don't re-wrap here: a tabular-level
        # rewrap would discard the failing leaf's name in favour of the
        # outer struct's, and any non-CastError that escapes the column
        # walk (schema-mismatch, missing column, ``pa.Table.from_arrays``
        # type mismatch) already carries its own diagnostic.
        casted = self.to_struct().dtype.cast_arrow_tabular(
            table, options=scoped,
        )
        # Tabular finalize is identity — per-column finalize already
        # ran inside the struct walk. Kept for shape symmetry.
        return self.finalize_arrow(casted)

    def cast_arrow_batch_iterator(
        self,
        batches: "Iterable[pa.RecordBatch]",
        options: "CastOptions | None" = None,
        **more,
    ) -> "Iterator[pa.RecordBatch]":
        """Cast a stream of :class:`pa.RecordBatch` against this field.

        Object targets passthrough (variant). Otherwise the dtype's
        struct view owns the per-batch tabular cast and ``byte_size``
        rechunk — same shape contract as :meth:`cast_arrow_tabular`,
        just lazy.
        """
        if self.dtype.type_id == DataTypeId.OBJECT:
            return iter(batches)
        options = get_cast_options_class().check(options=options, **more)
        return self.to_struct().dtype.cast_arrow_batch_iterator(
            batches, options=options.with_target(self)
        )

    def cast_polars_series(
        self,
        series: "polars.Series",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_series(
            series, options=options.with_target(self)
        )
        return self.finalize_polars_series(casted, default_scalar=default_scalar)

    def cast_polars_expr(
        self,
        series: "polars.Expr",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_polars_expr(series, options=options.with_target(self))
        return self.finalize_polars_expr(casted, default_scalar=default_scalar)

    def cast_polars_tabular(
        self,
        data: "polars.DataFrame | polars.LazyFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.to_struct().dtype.cast_polars_tabular(
            data, options=options.with_target(self)
        )
        return self.finalize_polars(casted)

    def cast_pandas_series(
        self,
        series: "pd.Series",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return series
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_pandas_series(
            series, options=options.with_target(self)
        )
        return self.finalize_pandas_series(casted, default_scalar=default_scalar)

    def cast_pandas_tabular(
        self,
        data: "pd.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_pandas_tabular(data, options=options.with_target(self))
        return self.finalize_pandas(casted)

    def cast_spark_column(
        self,
        column: "ps.Column",
        options: "CastOptions | None" = None,
        default_scalar: Any = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return column
        options = get_cast_options_class().check(options=options, **more)
        options = options.with_target(self).check_source(column)
        casted = self.dtype.cast_spark_column(column, options=options)
        return self.finalize_spark_column(casted, default_scalar=default_scalar)

    def cast_spark_tabular(
        self,
        data: "ps.DataFrame",
        options: "CastOptions | None" = None,
        **more,
    ):
        if self.dtype.type_id == DataTypeId.OBJECT:
            return data
        options = get_cast_options_class().check(options=options, **more)
        casted = self.dtype.cast_spark_tabular(data, options=options.with_target(self))
        return self.finalize_spark(casted)

    # ==================================================================
    # Select — pull this field out of a frame and cast in one shot
    # ==================================================================
    #
    # ``select_in_*`` is the read-side companion to ``cast_*``. Where
    # ``cast_*`` takes a column / frame the caller already extracted,
    # ``select_in_*`` resolves the column from a frame whose schema
    # may differ — checking :attr:`name` first, falling back to
    # :attr:`alias` — and casts the result to ``self.dtype`` so the
    # output is shape-stable regardless of which name resolved.
    #
    # Miss policy is one knob: ``default: Any = ...``. Leaving it as
    # the ellipsis sentinel raises ``KeyError`` on miss (matching
    # candidates and available column names included). Passing any
    # other value (a literal, a typed pa.Array, ``None``, another
    # Field for the schema variant) returns that value as-is.

    # ==================================================================
    # Fill — null-only dispatch (no cast), mirrors the cast dispatcher
    # ==================================================================
    #
    # Same three granularities as the cast side. Useful when the caller
    # already has cast data and only needs the null-fill pass —
    # typically you don't call these directly because the ``cast_*``
    # methods already fill inline via ``finalize_*``, but they're the
    # right entry point when source and target dtypes already agree.

    def fill_nulls(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in *obj* using the native engine — engine + shape detection.

        Routes the same way :meth:`cast` does. See
        :meth:`fill_arrow` / :meth:`fill_polars` / :meth:`fill_pandas`
        / :meth:`fill_spark` for the per-engine behaviour.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.fill_arrow(obj, default_scalar=default_scalar)
        if ns.startswith("polars"):
            return self.fill_polars(obj, default_scalar=default_scalar)
        if ns.startswith("pandas"):
            return self.fill_pandas(obj, default_scalar=default_scalar)
        if ns.startswith("pyspark"):
            return self.fill_spark(obj, default_scalar=default_scalar)

        if isinstance(obj, Iterator):
            return (
                self.fill_nulls(item, default_scalar=default_scalar) for item in obj
            )
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (
                self.fill_nulls(item, default_scalar=default_scalar) for item in obj
            )

        raise TypeError(
            f"Field.fill_nulls: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r})"
        )

    def fill_arrow(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any pyarrow object.

        Arrays go through :meth:`fill_arrow_array_nulls` directly.
        Tables / RecordBatches re-use the tabular cast path with
        ``self`` as the target — a no-op cast that still runs the
        per-column null-fill via the struct walk.
        """
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.fill_arrow_array_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return obj
        raise TypeError(
            f"Field.fill_arrow: expected pa.Array/ChunkedArray/Table/"
            f"RecordBatch, got {type(obj).__name__}"
        )

    def fill_polars(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any polars object.

        Series / Expr go through :meth:`fill_polars_array_nulls` —
        which handles both shapes uniformly (Expr is the lazy
        counterpart of Series; the fill operator grafts onto each
        identically). DataFrame / LazyFrame route through
        :meth:`cast_polars_tabular` as a self-targeted cast.
        """
        pl = polars_module()
        if isinstance(obj, (pl.Series, pl.Expr)):
            return self.fill_polars_array_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return self.cast_polars_tabular(obj)
        raise TypeError(
            f"Field.fill_polars: expected pl.DataFrame/LazyFrame/Series/Expr, "
            f"got {type(obj).__name__}"
        )

    def fill_pandas(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any pandas object."""
        pd = pandas_module()
        if isinstance(obj, pd.Series):
            return self.fill_pandas_series_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, pd.DataFrame):
            return self.cast_pandas_tabular(obj)
        raise TypeError(
            f"Field.fill_pandas: expected pd.DataFrame/pd.Series, "
            f"got {type(obj).__name__}"
        )

    def fill_spark(self, obj: Any, *, default_scalar: Any = None) -> Any:
        """Fill nulls in any spark object."""
        if isinstance(obj, spark_column_classes()):
            return self.fill_spark_column_nulls(obj, default_scalar=default_scalar)
        if isinstance(obj, spark_dataframe_classes()):
            return self.cast_spark_tabular(obj)
        raise TypeError(
            f"Field.fill_spark: expected pyspark.sql.DataFrame/Column, "
            f"got {type(obj).__name__}"
        )

    # ------------------------------------------------------------------
    # Fill — narrow per-engine null-fill primitives
    # ------------------------------------------------------------------

    def fill_arrow_nulls(
        self,
        obj: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.fill_arrow_array_nulls(
                obj,
                default_scalar=default_scalar,
            )
        raise TypeError(f"Cannot fill nulls in {type(obj).__name__}")

    def fill_arrow_array_nulls(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.fill_arrow_array_nulls(
            array,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_polars_array_nulls(
        self,
        series: "polars.Series | polars.Expr",
        *,
        default_scalar: Any = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_value

        return self.dtype.fill_polars_array_nulls(
            series,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_pandas_series_nulls(
        self,
        series: "pd.Series",
        *,
        default_scalar: Any = None,
    ):
        if default_scalar is None and self.has_default:
            default_scalar = self.default_value

        return self.dtype.fill_pandas_series_nulls(
            series,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    def fill_spark_column_nulls(
        self,
        column: "ps.Column",
        *,
        default_scalar: pa.Scalar | None = None,
    ) -> "ps.Column":
        if default_scalar is None:
            default_scalar = self.default_value

        return self.dtype.fill_spark_column_nulls(
            column,
            nullable=self.nullable,
            default_scalar=default_scalar,
        )

    # ==================================================================
    # Default value factories — zero-row / size-N default arrays
    # ==================================================================

    def default_arrow_array(
        self,
        size: int = 0,
        memory_pool: Optional[pa.MemoryPool] = None,
        chunks: Optional[list[int]] = None,
        default_scalar: Optional[pa.Scalar] = None,
    ) -> pa.Array | pa.ChunkedArray:
        if default_scalar is None and self.has_default:
            default_scalar = self.default_arrow_scalar

        return self.dtype.default_arrow_array(
            nullable=self.nullable,
            size=size,
            memory_pool=memory_pool,
            chunks=chunks,
            default_scalar=default_scalar,
        )

    def default_polars_series(
        self,
        *,
        size: int = 0,
        name: str | None = None,
    ):
        return self.dtype.default_polars_series(
            value=self.default_value,
            nullable=self.nullable,
            size=size,
            name=self.name if name is None else name,
        )

    def default_polars_expr(self, alias: str | None = None):
        return self.dtype.default_polars_expr(
            value=self.default_value, nullable=self.nullable, alias=alias or self.name
        )

    def default_pandas_series(
        self,
        *,
        size: int = 0,
        index: Any = None,
        name: str | None = None,
    ):
        return self.dtype.default_pandas_series(
            value=self.default_value,
            nullable=self.nullable,
            size=size,
            index=index,
            name=self.name if name is None else name,
        )

    def default_spark_column(self, alias: str | None = None):
        s = self.dtype.default_spark_column(
            value=self.default_value, nullable=self.nullable
        )
        return s.alias(alias) if alias else s.alias(self.name)

    # ==================================================================
    # Rename / alias helpers
    # ==================================================================
    #
    # polars and spark expose an ``.alias(name)`` method on their
    # column-like types. These helpers centralize the "rename only if
    # the target name differs from the current name, and only if the
    # target has a non-default name" logic so callers that want a
    # rename-only pass (skipping the full cast) don't have to
    # reimplement the guard every time.

    def polars_alias(self, obj: Any) -> Any:
        """Rename a polars Series / Expr to match this field's name.

        No-op when the target name matches the current name, or when
        this field only has the sentinel name. Calling defensively
        is free — zero-cost on the no-rename path.
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        current = getattr(obj, "name", None)
        if current == self.name:
            return obj
        alias = getattr(obj, "alias", None)
        if alias is None:
            # Neither Series nor Expr — nothing to rename against.
            return obj
        return alias(self.name)

    def spark_alias(self, obj: Any) -> Any:
        """Rename a Spark Column to match this field's name.

        Spark DataFrames aren't handled — renaming a DataFrame
        requires a projection with named columns, which isn't a
        single-method operation. Column is the rename target here.
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        alias = getattr(obj, "alias", None)
        if alias is None:
            return obj
        return alias(self.name)

    def pandas_alias(self, obj: Any) -> Any:
        """Rename a pandas Series to match this field's name.

        Pandas has no ``.alias()`` — rename is ``series.name = ...``,
        which mutates. This helper returns the series so it chains
        like :meth:`polars_alias` / :meth:`spark_alias`. DataFrames
        aren't handled (column rename is a projection, not a
        single-method op).
        """
        if not self.name or self.name == DEFAULT_FIELD_NAME:
            return obj
        if not hasattr(obj, "name"):
            return obj
        if obj.name == self.name:
            return obj
        obj.name = self.name
        return obj

    # ==================================================================
    # Finalize — post-cast fill + alias; the tail of every `cast_*`
    # ==================================================================
    #
    # The ``cast_*`` methods delegate their post-cast cleanup here, so
    # finalize is both the tail of a cast pipeline AND a standalone
    # entry point when source and target dtypes already agree (no
    # cast needed) but the caller still wants the null-fill + rename
    # pass. :class:`CastOptions` delegates its ``finalize_*_cast``
    # methods here.
    #
    # Tabular finalize is identity: per-column fill + alias already
    # ran inside the struct walk during the tabular cast. The tabular
    # method is kept for shape symmetry so dispatchers can call
    # ``finalize_<engine>`` uniformly regardless of input shape.

    def finalize_arrow_array(
        self,
        array: pa.Array | pa.ChunkedArray,
        *,
        default_scalar: pa.Scalar | None = None,
    ):
        """Fill nulls on a pyarrow Array / ChunkedArray.

        No alias step: pa.Array / ChunkedArray don't carry a name.
        Tabular naming lives in the pa.Field that wraps the array in
        a Table/RecordBatch, which :meth:`cast_arrow_tabular` handles
        through the struct walk.
        """
        return self.fill_arrow_array_nulls(array, default_scalar=default_scalar)

    def finalize_arrow(
        self,
        obj: Any,
        *,
        default_scalar: pa.Scalar | None = None,
    ) -> Any:
        """Finalize any pyarrow object — dispatch by shape.

        Array/ChunkedArray → fill.
        Table/RecordBatch → identity.
        """
        if isinstance(obj, (pa.Array, pa.ChunkedArray)):
            return self.finalize_arrow_array(obj, default_scalar=default_scalar)
        if isinstance(obj, (pa.Table, pa.RecordBatch)):
            return obj
        raise TypeError(
            f"Field.finalize_arrow: expected pa.Array/ChunkedArray/Table/"
            f"RecordBatch, got {type(obj).__name__}"
        )

    def finalize_polars_series(
        self,
        series: "polars.Series",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, alias a polars Series to the target name."""
        filled = self.fill_polars_array_nulls(series, default_scalar=default_scalar)
        return self.polars_alias(filled)

    def finalize_polars_expr(
        self,
        expr: "polars.Expr",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, alias a polars Expr to the target name.

        Same as :meth:`finalize_polars_series` — polars Series and Expr
        share the fill + alias primitives, so the finalize shape is
        identical. Separate method for call-site clarity.
        """
        filled = self.fill_polars_array_nulls(expr, default_scalar=default_scalar)
        return self.polars_alias(filled)

    def finalize_polars(
        self,
        obj: "polars.Series | polars.Expr | polars.DataFrame | polars.LazyFrame",
        *,
        default_scalar: Any = None,
    ):
        """Finalize any polars object — dispatch by shape.

        Series/Expr → fill + alias.
        DataFrame/LazyFrame → identity (tabular cast already finalized
        per-column via the struct walk).
        """
        pl = polars_module()
        if isinstance(obj, (pl.Series, pl.Expr)):
            filled = self.fill_polars_array_nulls(obj, default_scalar=default_scalar)
            return self.polars_alias(filled)
        if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
            return obj
        raise TypeError(
            f"Field.finalize_polars: expected pl.Series/Expr/DataFrame/LazyFrame, "
            f"got {type(obj).__name__}"
        )

    def finalize_pandas_series(
        self,
        series: "pd.Series",
        *,
        default_scalar: Any = None,
    ):
        """Fill nulls, rename a pandas Series to the target name."""
        filled = self.fill_pandas_series_nulls(series, default_scalar=default_scalar)
        return self.pandas_alias(filled)

    def finalize_pandas(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize any pandas object — dispatch by shape.

        Series → fill + rename.
        DataFrame → identity.
        """
        pd = pandas_module()
        if isinstance(obj, pd.Series):
            return self.finalize_pandas_series(obj, default_scalar=default_scalar)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError(
            f"Field.finalize_pandas: expected pd.DataFrame/pd.Series, "
            f"got {type(obj).__name__}"
        )

    def check_pandas_indexes(self, obj: Any) -> Any:
        """Promote columns tagged ``index_key`` to the DataFrame index.

        Collects children with :attr:`index_key` set, sorted by
        :attr:`index_key_level`, and calls ``set_index`` on the
        DataFrame. ``__index_level_N__`` placeholder names are mapped
        back to ``None`` so the round-trip matches the source.

        For a Series whose field is itself tagged ``index_key``, the
        Series is returned as-is — the caller decides how to attach it
        as an index.

        Passthrough when no children carry the tag or when the object
        is not a DataFrame.
        """
        pd = pandas_module()
        if isinstance(obj, pd.DataFrame) and self.fields:
            levels: list[tuple[int, str]] = []
            for child in self.fields:
                if child.index_key:
                    level = child.index_key_level
                    levels.append((level if level is not None else 0, child.name))
            if levels:
                levels.sort()
                names = [name for _, name in levels]
                present = [n for n in names if n in obj.columns]
                if present:
                    obj = obj.set_index(present)
                    obj.index.names = [
                        None if isinstance(n, str) and n.startswith("__index_level_")
                        else n
                        for n in obj.index.names
                    ]
        return obj

    def finalize_spark_column(
        self,
        column: "ps.Column",
        *,
        default_scalar: Any = None,
    ) -> "ps.Column":
        """Fill nulls, alias a Spark Column to the target name."""
        filled = self.fill_spark_column_nulls(column, default_scalar=default_scalar)
        return self.spark_alias(filled)

    def finalize_spark(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize any spark object — dispatch by shape.

        Column → fill + alias.
        DataFrame → identity (tabular cast already finalized).
        """
        if isinstance(obj, spark_column_classes()):
            return self.finalize_spark_column(obj, default_scalar=default_scalar)
        if isinstance(obj, spark_dataframe_classes()):
            return obj
        raise TypeError(
            f"Field.finalize_spark: expected pyspark.sql.Column/DataFrame, "
            f"got {type(obj).__name__}"
        )

    def finalize(
        self,
        obj: Any,
        *,
        default_scalar: Any = None,
    ) -> Any:
        """Finalize *obj* using its native engine — module-prefix dispatch.

        Mirrors :meth:`cast` / :meth:`fill_nulls` routing.
        """
        ns, _name = ObjectSerde.module_and_name(obj)

        if ns.startswith("pyarrow"):
            return self.finalize_arrow(obj, default_scalar=default_scalar)
        if ns.startswith("polars"):
            return self.finalize_polars(obj, default_scalar=default_scalar)
        if ns.startswith("pandas"):
            return self.finalize_pandas(obj, default_scalar=default_scalar)
        if ns.startswith("pyspark"):
            return self.finalize_spark(obj, default_scalar=default_scalar)

        if isinstance(obj, Iterator):
            return (self.finalize(item, default_scalar=default_scalar) for item in obj)
        if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
            return (self.finalize(item, default_scalar=default_scalar) for item in obj)

        raise TypeError(
            f"Field.finalize: unsupported input {type(obj).__name__!r} "
            f"(module={ns!r})"
        )

    # ==================================================================
    # Mapping surface — meaningful for struct-shaped fields (schemas);
    # delegates to ``self.dtype.fields`` so the struct dtype stays the
    # single source of truth for children.
    # ==================================================================

    def __setitem__(self, key: str, value: "Field | pa.Field") -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"Field assignment key must be str; got {type(key).__name__}"
            )

        f = Field.from_any(value)
        if f.name != key:
            f = f.copy(name=key)

        new_fields: list[Field] = []
        replaced = False
        for existing in self.children:
            if existing.name == key:
                new_fields.append(f)
                replaced = True
            else:
                new_fields.append(existing)
        if not replaced:
            new_fields.append(f)
        self._set_dtype_fields(new_fields)

    def __delitem__(self, key: int | str) -> None:
        resolved = self.field(key)
        self._set_dtype_fields(
            f for f in self.children if f.name != resolved.name
        )

    def __iter__(self) -> Iterator[str]:
        return iter(f.name for f in self.children)

    def __len__(self) -> int:
        return len(self.children)

    def __contains__(self, key: object) -> bool:
        return self.field(key, raise_error=False) is not None

    def __bool__(self) -> bool:
        # Struct fields are truthy iff they have children — mirrors a
        # mapping's ``bool``. Non-struct fields fall back to the
        # default-true behaviour (a primitive Field is always truthy).
        if isinstance(self.dtype, StructType):
            return bool(self.dtype.fields)
        return True

    @overload
    def get(self, key: int, default: None = None) -> "Field | None": ...
    @overload
    def get(self, key: str, default: None = None) -> "Field | None": ...
    @overload
    def get(self, key: int | str, default: Any = None) -> Any: ...

    def get(self, key: int | str, default: Any = None) -> Any:
        resolved = self.field(key, raise_error=False)
        if resolved is None:
            return default
        return resolved

    @overload
    def pop(self, key: int) -> "Field": ...
    @overload
    def pop(self, key: str) -> "Field": ...
    @overload
    def pop(self, key: int | str, default: Any = None) -> "Field | Any": ...

    def pop(self, key: int | str, default: Any = ...):
        resolved = self.field(key, raise_error=False)
        if resolved is None:
            if default is ...:
                if isinstance(key, int):
                    raise IndexError(key)
                raise KeyError(key)
            return default
        self._set_dtype_fields(
            f for f in self.children if f.name != resolved.name
        )
        return resolved

    def setdefault(
        self, key: str, default: "Field | pa.Field | None" = None
    ) -> "Field":
        if not isinstance(key, str):
            raise TypeError(
                f"Field.setdefault key must be str; got {type(key).__name__}"
            )

        resolved = self.field(key, raise_error=False)
        if resolved is not None:
            return resolved

        if default is None:
            raise ValueError("Field.setdefault requires a Field default for new keys")

        f = Field.from_any(default)
        if f.name != key:
            f = f.copy(name=key)
        self._set_dtype_fields(list(self.children) + [f])
        return f

    def keys(self) -> list[str]:
        return [f.name for f in self.children]

    def values(self) -> list["Field"]:
        return list(self.children)

    def items(self) -> list[tuple[str, "Field"]]:
        return [(f.name, f) for f in self.children]

    def popitem(self, last: bool = True) -> tuple[str, "Field"]:
        fields = list(self.children)
        if not fields:
            raise KeyError("popitem(): field is empty")
        idx = -1 if last else 0
        f = fields.pop(idx)
        self._set_dtype_fields(fields)
        return f.name, f

    def clear(self) -> None:
        self._set_dtype_fields(())

    def append(self, *more_fields: "Field | pa.Field") -> "Field":
        out = self.copy()
        for value in more_fields:
            f = Field.from_any(value)
            out[f.name] = f
        return out

    def extend(self, fields: Iterable["Field | pa.Field"]) -> "Field":
        out = self.copy()
        for value in fields:
            f = Field.from_any(value)
            out[f.name] = f
        return out

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        # Mapping.update — accept another mapping/iterable of pairs or
        # kwargs. We don't inherit MutableMapping's mixin (Field is a
        # frozen dataclass) so spell it out.
        if args:
            if len(args) > 1:
                raise TypeError(
                    f"update expected at most 1 positional argument, got {len(args)}"
                )
            other = args[0]
            if isinstance(other, Mapping):
                for k in other:
                    self[k] = other[k]
            else:
                for k, v in other:
                    self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    # ==================================================================
    # Set operators — schema reconciliation reads like set algebra.
    # ==================================================================

    @staticmethod
    def _merge_metadata(
        left: dict[bytes, bytes] | None,
        right: dict[bytes, bytes] | None,
    ) -> dict[bytes, bytes] | None:
        if not left and not right:
            return None
        return {**(left or {}), **(right or {})} or None

    @classmethod
    def _coerce_other(cls, other: Any) -> "Field":
        if isinstance(other, cls):
            return other
        if isinstance(other, Field):
            return cls.from_field(other)
        return cls.from_any(other)

    @staticmethod
    def _rhs_wins_name(left: "Field", right: "Field") -> str:
        """RHS-wins conflict resolution mirroring metadata merge semantics."""
        if right.name and right.name != DEFAULT_FIELD_NAME:
            return right.name
        return left.name

    def __add__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        merged: "OrderedDict[str, Field]" = OrderedDict(
            (f.name, f.copy()) for f in self.children
        )
        for f in other.children:
            merged[f.name] = f.copy()
        return type(self)._make_struct(
            children=merged.values(),
            metadata=self._merge_metadata(self.metadata, other.metadata),
            name=self._rhs_wins_name(self, other),
            nullable=self.nullable or other.nullable,
        )

    def __radd__(self, other: Any) -> "Field":
        if other == 0:
            return self.copy()
        other = self._coerce_other(other)
        return other.__add__(self)

    def __iadd__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        merged = list(self.children)
        seen = {f.name: i for i, f in enumerate(merged)}
        for f in other.children:
            if f.name in seen:
                merged[seen[f.name]] = f.copy()
            else:
                seen[f.name] = len(merged)
                merged.append(f.copy())
        self._set_dtype_fields(merged)
        object.__setattr__(
            self, "metadata", self._merge_metadata(self.metadata, other.metadata)
        )
        if other.name and other.name != DEFAULT_FIELD_NAME:
            object.__setattr__(self, "name", other.name)
        return self

    def __sub__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        remove_names = {f.name for f in other.children}
        return type(self)._make_struct(
            children=[
                f.copy() for f in self.children if f.name not in remove_names
            ],
            metadata=dict(self.metadata) if self.metadata else None,
            name=self.name,
            nullable=self.nullable,
        )

    def __isub__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        remove_names = {f.name for f in other.children}
        self._set_dtype_fields(
            f for f in self.children if f.name not in remove_names
        )
        return self

    def __and__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        keep_names = {f.name for f in other.children}
        return type(self)._make_struct(
            children=[f.copy() for f in self.children if f.name in keep_names],
            metadata=self._merge_metadata(self.metadata, other.metadata),
            name=self._rhs_wins_name(self, other),
            nullable=self.nullable or other.nullable,
        )

    def __iand__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        keep_names = {f.name for f in other.children}
        self._set_dtype_fields(f for f in self.children if f.name in keep_names)
        object.__setattr__(
            self, "metadata", self._merge_metadata(self.metadata, other.metadata)
        )
        if other.name and other.name != DEFAULT_FIELD_NAME:
            object.__setattr__(self, "name", other.name)
        return self

    def __or__(self, other: Any) -> "Field":
        return self.__add__(other)

    def __ror__(self, other: Any) -> "Field":
        other = self._coerce_other(other)
        return other.__add__(self)

    def __ior__(self, other: Any) -> "Field":
        return self.__iadd__(other)


# ======================================================================
# Cast registry — `Any → Field` converter
# ======================================================================


@register_converter(Any, Field)
def any_to_field(obj: Any, _: Any):
    return Field.from_any(obj)
