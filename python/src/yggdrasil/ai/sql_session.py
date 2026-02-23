# sql_session.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pyarrow as pa

from .session import AISession, ChatResponse

__all__ = [
    "SQLFlavor",
    "SQLDialectConfig",
    "SchemaEntry",
    "SQLSession",
    "dialect",
    "extract_object_name",
    "extract_column_comment",
    "parse_qualified_name",
    "arrow_schema_to_fields",
    "schema_to_ddl",
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dialect configuration
# ---------------------------------------------------------------------------

class SQLFlavor(str, Enum):
    DATABRICKS  = "databricks"   # Spark SQL / Unity Catalog
    POSTGRESQL  = "postgresql"
    DUCKDB      = "duckdb"       # in-process DuckDB
    POLARS_SQL  = "polars_sql"   # Polars SQLContext — DuckDB-dialect SQL, ref by alias
    MONGODB     = "mongodb"      # aggregation pipeline JSON array


@dataclass(frozen=True)
class SQLDialectConfig:
    """
    All dialect-specific rendering knobs in one immutable record.
    Extend by adding a new entry to ``_DIALECT_REGISTRY`` — no if/elif needed.
    """
    flavor:           SQLFlavor
    display_name:     str

    # Name qualification
    qualify_names:    bool  = True    # emit catalog.schema.table
    quote_char:       str   = '"'     # identifier quoting character

    # Aggregate functions
    array_agg:        str   = "ARRAY_AGG"
    string_agg:       str   = "STRING_AGG"

    # Syntax snippets (format templates)
    limit_clause:     str   = "LIMIT {n}"
    cast_syntax:      str   = "CAST({expr} AS {type})"
    timestamp_now:    str   = "CURRENT_TIMESTAMP"

    # Feature flags
    supports_cte:     bool  = True
    supports_window:  bool  = True

    # Output shape
    is_pipeline:      bool  = False   # True → expect JSON array, not SQL text

    # ------------------------------------------------------------------ #
    # Arrow → SQL type mapping                                            #
    # ------------------------------------------------------------------ #

    def arrow_to_sql_type(self, dt: pa.DataType) -> str:
        """Map an Arrow ``DataType`` to this dialect's SQL type name."""
        if pa.types.is_int8(dt):   return "TINYINT"
        if pa.types.is_int16(dt):  return "SMALLINT"
        if pa.types.is_int32(dt):  return "INT"
        if pa.types.is_int64(dt):  return "BIGINT"
        if pa.types.is_uint8(dt):  return "TINYINT UNSIGNED"
        if pa.types.is_uint16(dt): return "SMALLINT UNSIGNED"
        if pa.types.is_uint32(dt): return "INT UNSIGNED"
        if pa.types.is_uint64(dt): return "BIGINT UNSIGNED"
        if pa.types.is_float16(dt) or pa.types.is_float32(dt): return "FLOAT"
        if pa.types.is_float64(dt): return "DOUBLE"
        if pa.types.is_decimal(dt): return f"DECIMAL({dt.precision},{dt.scale})"
        if pa.types.is_boolean(dt): return "BOOLEAN"
        if pa.types.is_string(dt) or pa.types.is_large_string(dt): return "STRING"
        if pa.types.is_binary(dt) or pa.types.is_large_binary(dt): return "BINARY"
        if pa.types.is_timestamp(dt):
            tz = " WITH TIME ZONE" if dt.tz else ""
            return f"TIMESTAMP{tz}"
        if pa.types.is_date32(dt) or pa.types.is_date64(dt): return "DATE"
        if pa.types.is_time32(dt) or pa.types.is_time64(dt): return "TIME"
        if pa.types.is_duration(dt): return "INTERVAL"
        if pa.types.is_list(dt) or pa.types.is_large_list(dt):
            return f"ARRAY<{self.arrow_to_sql_type(dt.value_type)}>"
        if pa.types.is_struct(dt):
            fields = ", ".join(
                f"{dt.field(i).name} {self.arrow_to_sql_type(dt.field(i).type)}"
                for i in range(dt.num_fields)
            )
            return f"STRUCT<{fields}>"
        if pa.types.is_map(dt):
            return f"MAP<{self.arrow_to_sql_type(dt.key_type)}, {self.arrow_to_sql_type(dt.item_type)}>"
        if pa.types.is_dictionary(dt):
            return self.arrow_to_sql_type(dt.value_type)
        return "STRING"  # safe fallback

    def quote(self, identifier: str) -> str:
        q = self.quote_char
        return f"{q}{identifier}{q}"

    def system_prompt(self) -> str:
        if self.is_pipeline:
            return (
                "Return ONLY a MongoDB aggregation pipeline as a strict JSON array. "
                "No prose, no markdown.\n"
                "Stage order: $match → $group/$project → $sort → $limit.\n"
                "Use only fields present in the schema context."
            )
        rules = [
            f"Return ONLY a valid {self.display_name} SQL query.",
            "No prose, no markdown fences, no explanations.",
            "Use only the columns listed in the schema context.",
            "Emit a single statement.",
        ]
        if self.qualify_names:
            rules.append("Use fully-qualified table names (catalog.schema.table) when provided.")
        if not self.supports_cte:
            rules.append("Do NOT use CTEs (WITH clauses); use subqueries instead.")
        return "\n".join(rules)


# ---------------------------------------------------------------------------
# Per-dialect type overrides via subclassing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PostgreSQLConfig(SQLDialectConfig):
    def arrow_to_sql_type(self, dt: pa.DataType) -> str:
        if pa.types.is_string(dt) or pa.types.is_large_string(dt): return "TEXT"
        if pa.types.is_float32(dt) or pa.types.is_float16(dt): return "REAL"
        if pa.types.is_float64(dt): return "DOUBLE PRECISION"
        if pa.types.is_binary(dt) or pa.types.is_large_binary(dt): return "BYTEA"
        if pa.types.is_list(dt) or pa.types.is_large_list(dt):
            return f"{self.arrow_to_sql_type(dt.value_type)}[]"
        return super().arrow_to_sql_type(dt)


@dataclass(frozen=True)
class _DuckDBConfig(SQLDialectConfig):
    def arrow_to_sql_type(self, dt: pa.DataType) -> str:
        if pa.types.is_float32(dt) or pa.types.is_float16(dt): return "FLOAT"
        if pa.types.is_float64(dt): return "DOUBLE"
        if pa.types.is_string(dt) or pa.types.is_large_string(dt): return "VARCHAR"
        if pa.types.is_list(dt) or pa.types.is_large_list(dt):
            return f"{self.arrow_to_sql_type(dt.value_type)}[]"
        if pa.types.is_map(dt):
            return (
                f"MAP({self.arrow_to_sql_type(dt.key_type)}, "
                f"{self.arrow_to_sql_type(dt.item_type)})"
            )
        return super().arrow_to_sql_type(dt)


# ---------------------------------------------------------------------------
# Dialect registry
# ---------------------------------------------------------------------------

_DIALECT_REGISTRY: Dict[SQLFlavor, SQLDialectConfig] = {
    SQLFlavor.DATABRICKS: SQLDialectConfig(
        SQLFlavor.DATABRICKS, "Databricks/Spark SQL",
        array_agg="COLLECT_LIST",
        string_agg="COLLECT_LIST",
        timestamp_now="CURRENT_TIMESTAMP()",
        quote_char="`",
    ),
    SQLFlavor.POSTGRESQL: _PostgreSQLConfig(
        SQLFlavor.POSTGRESQL, "PostgreSQL",
        cast_syntax="{expr}::{type}",
        string_agg="STRING_AGG",
    ),
    SQLFlavor.DUCKDB: _DuckDBConfig(
        SQLFlavor.DUCKDB, "DuckDB",
        qualify_names=False,           # in-process: no catalog prefix
        cast_syntax="{expr}::{type}",
        timestamp_now="NOW()",
    ),
    SQLFlavor.POLARS_SQL: _DuckDBConfig(
        # Polars SQLContext uses DuckDB SQL syntax; tables referenced by alias only
        SQLFlavor.POLARS_SQL, "Polars SQL",
        qualify_names=False,
        cast_syntax="{expr}::{type}",
        timestamp_now="NOW()",
    ),
    SQLFlavor.MONGODB: SQLDialectConfig(
        SQLFlavor.MONGODB, "MongoDB",
        qualify_names=False,
        supports_cte=False,
        supports_window=False,
        is_pipeline=True,
    ),
}


def dialect(flavor: SQLFlavor) -> SQLDialectConfig:
    """Retrieve the ``SQLDialectConfig`` for a given flavor."""
    return _DIALECT_REGISTRY[flavor]


# ---------------------------------------------------------------------------
# Arrow helpers
# ---------------------------------------------------------------------------

def arrow_schema_to_fields(schema: pa.Schema) -> List[pa.Field]:
    return [schema.field(i) for i in range(len(schema))]


def _arrow_type_compact(dt: pa.DataType) -> str:
    """Ultra-compact token-light type tag for LLM context lines."""
    if pa.types.is_int64(dt):   return "i64"
    if pa.types.is_int32(dt):   return "i32"
    if pa.types.is_int16(dt):   return "i16"
    if pa.types.is_int8(dt):    return "i8"
    if pa.types.is_uint64(dt):  return "u64"
    if pa.types.is_uint32(dt):  return "u32"
    if pa.types.is_uint16(dt):  return "u16"
    if pa.types.is_uint8(dt):   return "u8"
    if pa.types.is_float64(dt): return "f64"
    if pa.types.is_float32(dt) or pa.types.is_float16(dt): return "f32"
    if pa.types.is_boolean(dt): return "bool"
    if pa.types.is_string(dt) or pa.types.is_large_string(dt): return "str"
    if pa.types.is_binary(dt) or pa.types.is_large_binary(dt): return "bytes"
    if pa.types.is_timestamp(dt):
        return f"ts[{dt.tz}]" if dt.tz else "ts"
    if pa.types.is_date32(dt) or pa.types.is_date64(dt): return "date"
    if pa.types.is_time32(dt) or pa.types.is_time64(dt): return "time"
    if pa.types.is_duration(dt): return "dur"
    if pa.types.is_decimal(dt): return f"dec({dt.precision},{dt.scale})"
    if pa.types.is_list(dt) or pa.types.is_large_list(dt) or pa.types.is_fixed_size_list(dt):
        return f"arr<{_arrow_type_compact(dt.value_type)}>"
    if pa.types.is_map(dt):
        return f"map<{_arrow_type_compact(dt.key_type)},{_arrow_type_compact(dt.item_type)}>"
    if pa.types.is_struct(dt):
        inner = ",".join(
            f"{dt.field(i).name}:{_arrow_type_compact(dt.field(i).type)}"
            for i in range(dt.num_fields)
        )
        return f"struct<{inner}>"
    if pa.types.is_dictionary(dt):
        return f"dict<{_arrow_type_compact(dt.value_type)}>"
    return "any"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _b2s(x: object) -> str:
    if x is None:            return ""
    if isinstance(x, bytes): return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def _norm_metadata(raw: Optional[Dict]) -> Dict[str, str]:
    return {_b2s(k).lower(): _b2s(v) for k, v in (raw or {}).items() if _b2s(k)}


def _try_json_dict(s: str) -> Optional[Dict]:
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def parse_qualified_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """``'catalog.schema.table'`` → ``(catalog, schema, table)``."""
    n = (name or "").strip().strip('"').strip()
    if not n:
        return None, None, None
    parts = [p.strip().strip('"') for p in n.split(".") if p.strip()]
    if len(parts) == 1: return None,     None,     parts[0]
    if len(parts) == 2: return None,     parts[0], parts[1]
    return parts[-3], parts[-2], parts[-1]


_OBJECT_KEYS     = ("table_ref", "full_table_name", "qualified_name", "table", "table_name", "name", "object", "object_name")
_CATALOG_KEYS    = ("catalog", "unity_catalog", "uc_catalog", "db_catalog")
_SCHEMA_KEYS     = ("schema", "namespace", "database_schema")
_TABLE_KEYS      = ("table", "table_name", "relation", "object", "object_name")
_MONGO_COLL_KEYS = ("collection", "collection_name", "mongo.collection", "mongodb.collection")
_COMMENT_KEYS    = ("comment", "description", "doc", "column_comment", "spark.comment", "delta.comment")
_FALLBACK_QNAME  = ("table_ref", "full_table_name", "qualified_name", "name")


def extract_object_name(metadata: Optional[Dict], flavor: SQLFlavor) -> Optional[str]:
    """
    Best-effort qualified name from Arrow schema metadata.

    Returns ``catalog.schema.table`` for qualifying dialects, a bare table
    name for non-qualifying ones (DuckDB, Polars), and a collection name
    for MongoDB.
    """
    md = _norm_metadata(metadata)

    for k in list(_OBJECT_KEYS):
        obj = _try_json_dict(md.get(k, ""))
        if obj:
            for kk, vv in obj.items():
                md.setdefault(str(kk).lower(), _b2s(vv))

    if flavor == SQLFlavor.MONGODB:
        for ck in _MONGO_COLL_KEYS:
            if md.get(ck):
                return md[ck]
        for tk in ("table", "table_name", "name", "full_table_name", "table_ref"):
            if md.get(tk):
                _, _, coll = parse_qualified_name(md[tk])
                return coll or md[tk]
        return None

    catalog    = next((md[k] for k in _CATALOG_KEYS if md.get(k)), "")
    database   = md.get("database") or md.get("db") or ""
    catalog    = catalog or database
    schema_val = next((md[k] for k in _SCHEMA_KEYS if md.get(k)), "")
    table      = next((md[k] for k in _TABLE_KEYS  if md.get(k)), "")

    if not table:
        for fk in _FALLBACK_QNAME:
            if md.get(fk):
                c, s, t = parse_qualified_name(md[fk])
                catalog    = catalog    or (c or "")
                schema_val = schema_val or (s or "")
                table      = t or ""
                if table:
                    break

    if not table:
        return None

    cfg = _DIALECT_REGISTRY.get(flavor)
    if cfg and not cfg.qualify_names:
        return table

    return ".".join(p for p in (catalog, schema_val, table) if p)


def extract_column_comment(f: pa.Field) -> Optional[str]:
    """Pull the first non-empty comment / description from a field's metadata."""
    md = _norm_metadata(f.metadata)
    if not md:
        return None
    for key in _COMMENT_KEYS:
        val = md.get(key, "").strip()
        if val:
            return val
    for k in ("meta", "metadata", "attrs", "properties"):
        obj = _try_json_dict(md.get(k, ""))
        if obj:
            for kk in ("comment", "description", "doc"):
                vv = _b2s(obj.get(kk, "")).strip()
                if vv:
                    return vv
    return None


# ---------------------------------------------------------------------------
# DDL generation
# ---------------------------------------------------------------------------

def schema_to_ddl(
    schema: pa.Schema,
    table_name: str,
    flavor: SQLFlavor,
    *,
    if_not_exists: bool = True,
) -> str:
    """
    Generate a ``CREATE TABLE`` DDL string from an Arrow schema.

    Useful for seeding DuckDB / Polars SQLContext / Databricks with correct
    types before running generated queries.

    Parameters
    ----------
    schema:
        Source Arrow schema.
    table_name:
        Target table name (already qualified if needed).
    flavor:
        Target SQL dialect for type mapping.
    if_not_exists:
        Emit ``IF NOT EXISTS`` guard.
    """
    cfg  = dialect(flavor)
    cols = []
    for i in range(len(schema)):
        f       = schema.field(i)
        sql_t   = cfg.arrow_to_sql_type(f.type)
        comment = extract_column_comment(f)
        line    = f"  {cfg.quote(f.name)} {sql_t}"
        if comment:
            line += f"  -- {comment}"
        cols.append(line)

    guard = "IF NOT EXISTS " if if_not_exists else ""
    body  = ",\n".join(cols)
    return f"CREATE TABLE {guard}{table_name} (\n{body}\n)"


# ---------------------------------------------------------------------------
# Schema entry
# ---------------------------------------------------------------------------

@dataclass
class SchemaEntry:
    """
    Everything the session knows about one table / collection / DataFrame.

    Parameters
    ----------
    alias:
        Short reference name used in prompts and look-ups (e.g. ``"trades"``).
        For ``POLARS_SQL`` this is also the name passed to ``SQLContext.register()``.
    schema:
        The Arrow schema defining columns and types.
    object_name:
        Fully-qualified SQL name (``catalog.schema.table``). Auto-extracted
        from ``schema.metadata`` when omitted; falls back to ``alias``.
    tags:
        Free-form labels for context filtering (e.g. ``["prices", "eod"]``).
    """
    alias:       str
    schema:      pa.Schema
    object_name: Optional[str] = None
    tags:        List[str]     = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.object_name is None:
            self.object_name = (
                extract_object_name(dict(self.schema.metadata or {}), SQLFlavor.DATABRICKS)
                or self.alias
            )

    # ------------------------------------------------------------------ #

    @classmethod
    def from_fields(
        cls,
        alias: str,
        fields: List[pa.Field],
        *,
        metadata: Optional[Dict] = None,
        object_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "SchemaEntry":
        """Construct from a list of ``pa.Field`` objects."""
        return cls(
            alias=alias,
            schema=pa.schema(fields, metadata=metadata),
            object_name=object_name,
            tags=tags or [],
        )

    # ------------------------------------------------------------------ #

    @property
    def fields(self) -> List[pa.Field]:
        return arrow_schema_to_fields(self.schema)

    def ref(self, cfg: SQLDialectConfig) -> str:
        """
        The name to use when referencing this table in a generated query.

        Qualifying dialects use ``object_name``; non-qualifying use ``alias``.
        """
        if cfg.qualify_names and self.object_name and self.object_name != self.alias:
            return self.object_name
        return self.alias

    def compact_signature(self, *, include_comments: bool = True, max_comment: int = 60) -> str:
        """``col:compact_type[#comment],…`` — token-light schema fingerprint."""
        parts: List[str] = []
        for f in self.fields:
            t = _arrow_type_compact(f.type)
            if include_comments:
                c = extract_column_comment(f)
                if c:
                    c = c.replace("\n", " ").strip()
                    if len(c) > max_comment:
                        c = c[:max_comment - 3] + "..."
                    parts.append(f"{f.name}:{t}#{c}")
                    continue
            parts.append(f"{f.name}:{t}")
        return ",".join(parts)

    def context_line(self, cfg: SQLDialectConfig, *, include_comments: bool = True) -> str:
        """
        Single prompt context line::

            alias=>ref|col:type[#comment],…
        """
        sig = self.compact_signature(include_comments=include_comments)
        ref = self.ref(cfg)
        if ref != self.alias:
            return f"{self.alias}=>{ref}|{sig}"
        return f"{self.alias}|{sig}"

    def to_ddl(self, flavor: SQLFlavor, *, if_not_exists: bool = True) -> str:
        """Generate a ``CREATE TABLE`` DDL string for this entry."""
        return schema_to_ddl(
            self.schema,
            self.ref(dialect(flavor)),
            flavor,
            if_not_exists=if_not_exists,
        )

    def score_against(self, prompt: str, *, cap_column_hits: int = 8) -> int:
        """Relevance score: alias / object name / tag / column name hits in prompt."""
        p = prompt.lower()
        s = 0
        if self.alias.lower() in p:               s += 5
        if self.object_name and self.object_name.lower() in p: s += 5
        for tag in self.tags:
            if tag.lower() in p:                  s += 3
        hits = sum(1 for f in self.fields if f.name.lower() in p)
        s += min(hits, cap_column_hits)
        return s


# ---------------------------------------------------------------------------
# SQL session
# ---------------------------------------------------------------------------

@dataclass
class SQLSession(AISession):
    """
    Generic Arrow-schema-driven SQL generation session.

    Generates syntactically correct SQL (or a MongoDB aggregation pipeline)
    for any registered ``SQLFlavor``. The session holds a registry of
    ``SchemaEntry`` objects and selects the most relevant subset for each
    prompt automatically.

    Quick-start — Databricks
    ------------------------
    >>> sess = MySession(api_key="…", base_url="…", flavor=SQLFlavor.DATABRICKS)
    >>> sess.register(SchemaEntry("trades", trades_arrow_schema))
    >>> sql = sess.generate_text("total notional per instrument last 30 days")

    Quick-start — Polars SQLContext
    --------------------------------
    >>> ctx  = pl.SQLContext()
    >>> sess = MySession(api_key="…", base_url="…", flavor=SQLFlavor.POLARS_SQL)
    >>> sess.register_from_lazyframes({"trades": trades_lf, "positions": pos_lf})
    >>> query  = sess.generate_text("net position per book")
    >>> result = ctx.execute(query).collect()

    Quick-start — DuckDB
    --------------------
    >>> sess = MySession(api_key="…", base_url="…", flavor=SQLFlavor.DUCKDB)
    >>> sess.register(SchemaEntry("trades", trades_arrow_schema))
    >>> con.execute(sess.generate_text("vwap per symbol today"))
    """

    flavor:               SQLFlavor = SQLFlavor.DATABRICKS
    max_context_objects:  int       = 64    # LRU eviction threshold
    max_tables_in_prompt: int       = 12    # token budget per prompt
    include_comments:     bool      = True

    _registry: Dict[str, SchemaEntry] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    # AISession contract
    # ------------------------------------------------------------------

    def system_prompt(self) -> str:
        return dialect(self.flavor).system_prompt()

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def register(self, entry: SchemaEntry) -> None:
        """Add / replace a ``SchemaEntry`` (LRU eviction when at capacity)."""
        if len(self._registry) >= self.max_context_objects and entry.alias not in self._registry:
            oldest = next(iter(self._registry))
            del self._registry[oldest]
            log.debug("SQLSession evicted '%s'", oldest)
        self._registry[entry.alias] = entry
        log.debug("SQLSession registered '%s' → %s (%d cols)",
                  entry.alias, entry.object_name, len(entry.fields))

    def register_schema(
        self,
        alias: str,
        schema: pa.Schema,
        *,
        object_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Convenience: register directly from a ``pa.Schema``."""
        self.register(SchemaEntry(alias=alias, schema=schema, object_name=object_name, tags=tags or []))

    def register_fields(
        self,
        alias: str,
        fields: List[pa.Field],
        *,
        metadata: Optional[Dict] = None,
        object_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Convenience: register from a list of ``pa.Field`` objects."""
        self.register(SchemaEntry.from_fields(
            alias, fields,
            metadata=metadata,
            object_name=object_name,
            tags=tags,
        ))

    def register_from_lazyframes(
        self,
        lf_map: Dict[str, Any],              # alias → polars.LazyFrame
        *,
        tags: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Register Arrow schemas from a dict of Polars LazyFrames.

        Parameters
        ----------
        lf_map:
            ``{"alias": lazyframe, …}``
        tags:
            Optional per-alias tag lists: ``{"trades": ["market_data"]}``.

        Note
        ----
        Uses ``lf.collect_schema().to_arrow()`` — no data is materialised.
        """
        for alias, lf in lf_map.items():
            arrow_schema: pa.Schema = lf.collect_schema().to_arrow()
            self.register_schema(
                alias, arrow_schema,
                tags=(tags or {}).get(alias),
            )

    def unregister(self, alias: str) -> None:
        self._registry.pop(alias, None)

    def clear_registry(self) -> None:
        self._registry.clear()

    @property
    def registered_aliases(self) -> List[str]:
        return list(self._registry.keys())

    # ------------------------------------------------------------------
    # Context selection
    # ------------------------------------------------------------------

    def select_entries(
        self,
        prompt: str,
        *,
        pin: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[SchemaEntry]:
        """
        Score all entries against ``prompt``, return the top-N most relevant.

        Parameters
        ----------
        pin:
            Aliases forced into context regardless of score.
        limit:
            Override ``max_tables_in_prompt`` for this call.
        """
        cap    = limit if limit is not None else self.max_tables_in_prompt
        pinned = set(pin or [])

        pinned_entries = [self._registry[a] for a in pinned if a in self._registry]
        scored = sorted(
            [(e.score_against(prompt), e) for a, e in self._registry.items() if a not in pinned],
            key=lambda x: x[0], reverse=True,
        )
        budget   = max(0, cap - len(pinned_entries))
        selected = pinned_entries + [e for s, e in scored if s > 0][:budget]

        # Fallback: nothing matched → include the first registered entry
        if not selected and self._registry:
            selected = [next(iter(self._registry.values()))]

        return selected

    def build_context(
        self,
        entries: Iterable[SchemaEntry],
        *,
        include_comments: Optional[bool] = None,
    ) -> str:
        """Render selected entries as a compact schema context block."""
        ic  = include_comments if include_comments is not None else self.include_comments
        cfg = dialect(self.flavor)
        lines = ["schema:"]
        for e in entries:
            lines.append(e.context_line(cfg, include_comments=ic))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    def generate(
        self,
        user_prompt: str,
        *,
        temperature:         float               = 0.0,
        max_output_tokens:   int                 = 4096,
        extra_instructions:  Optional[str]       = None,
        pin:                 Optional[List[str]] = None,
        tables:              Optional[List[str]] = None,
        include_comments:    Optional[bool]      = None,
    ) -> ChatResponse:
        """
        Generate a SQL query (or MongoDB pipeline) for ``user_prompt``.

        Returns a full ``ChatResponse`` — use ``.text`` for the query string,
        ``.prompt_tokens`` + ``.completion_tokens`` for cost tracking.

        Parameters
        ----------
        pin:
            Aliases always included in schema context regardless of score.
        tables:
            Explicit alias list; bypasses relevance scoring entirely.
        include_comments:
            Override session-level ``include_comments`` for this call.
        """
        entries = (
            [self._registry[a] for a in tables if a in self._registry]
            if tables is not None
            else self.select_entries(user_prompt, pin=pin)
        )
        context = self.build_context(entries, include_comments=include_comments)
        return self.chat(
            user_prompt,
            context_system=context,
            extra_instructions=extra_instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate_text(self, user_prompt: str, **kwargs) -> str:
        """Return just the SQL / pipeline string."""
        return self.generate(user_prompt, **kwargs).text

    def generate_pipeline(self, user_prompt: str, **kwargs) -> List[Dict]:
        """
        MongoDB only — generate and parse the aggregation pipeline JSON array.

        Raises ``TypeError`` for non-MongoDB flavors.
        Raises ``ValueError`` if the output isn't a valid JSON array.
        """
        if self.flavor != SQLFlavor.MONGODB:
            raise TypeError(f"generate_pipeline() requires MONGODB flavor, got {self.flavor}")
        resp = self.generate(user_prompt, **kwargs)
        try:
            pipeline = json.loads(resp.text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON:\n{resp.text}") from exc
        if not isinstance(pipeline, list):
            raise ValueError(f"Expected JSON array, got {type(pipeline).__name__}")
        return pipeline

    # ------------------------------------------------------------------
    # DDL helpers
    # ------------------------------------------------------------------

    def ddl_for(self, alias: str, *, if_not_exists: bool = True) -> str:
        """``CREATE TABLE`` DDL for a single registered entry."""
        return self._registry[alias].to_ddl(self.flavor, if_not_exists=if_not_exists)

    def all_ddl(self, *, if_not_exists: bool = True) -> str:
        """``CREATE TABLE`` DDL for every registered entry, joined by blank lines."""
        return "\n\n".join(
            e.to_ddl(self.flavor, if_not_exists=if_not_exists)
            for e in self._registry.values()
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def preview_context(
        self,
        user_prompt: str,
        *,
        pin: Optional[List[str]] = None,
        tables: Optional[List[str]] = None,
        include_comments: Optional[bool] = None,
    ) -> str:
        """
        Dry-run: return the exact schema context block that would be sent to the
        LLM for ``user_prompt``, without making an API call.
        Useful for inspecting token usage and relevance scoring.
        """
        entries = (
            [self._registry[a] for a in tables if a in self._registry]
            if tables is not None
            else self.select_entries(user_prompt, pin=pin)
        )
        return self.build_context(entries, include_comments=include_comments)

    def registry_summary(self) -> List[Dict]:
        """List of dicts describing all registered entries."""
        return [
            {
                "alias":       e.alias,
                "object_name": e.object_name,
                "columns":     len(e.fields),
                "tags":        e.tags,
            }
            for e in self._registry.values()
        ]