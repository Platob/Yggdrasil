# sql_session.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa

from .session import AISession

__all__ = ["SQLFlavor", "SQLAISession"]


class SQLFlavor(str, Enum):
    DATABRICKS = "databricks"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"  # aggregation pipeline JSON array, not SQL


def _b2s(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore").strip()
    return str(x).strip()


def _try_parse_json_blob(s: str) -> Optional[dict]:
    s = (s or "").strip()
    if not s or not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_qualified_name(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    n = (name or "").strip().strip('"').strip()
    if not n:
        return None, None, None
    parts = [p.strip().strip('"') for p in n.split(".") if p.strip()]
    if len(parts) == 1:
        return None, None, parts[0]
    if len(parts) == 2:
        return None, parts[0], parts[1]
    return parts[-3], parts[-2], parts[-1]


def extract_object_name_from_metadata(metadata: Optional[Dict[object, object]], flavor: SQLFlavor) -> Optional[str]:
    md_raw = metadata or {}
    md: Dict[str, str] = {}
    for k, v in md_raw.items():
        ks = _b2s(k).lower()
        vs = _b2s(v)
        if ks:
            md[ks] = vs

    for k in ("table_ref", "full_table_name", "qualified_name", "table", "table_name", "name", "object", "object_name"):
        if k in md:
            obj = _try_parse_json_blob(md[k])
            if obj:
                for kk, vv in obj.items():
                    if kk is None:
                        continue
                    md[str(kk).lower()] = _b2s(vv)

    if flavor == SQLFlavor.MONGODB:
        for ck in ("collection", "collection_name", "mongo.collection", "mongodb.collection"):
            if md.get(ck):
                return md[ck]
        for tk in ("table", "table_name", "name", "full_table_name", "table_ref"):
            if md.get(tk):
                _, _, coll = parse_qualified_name(md[tk])
                return coll or md[tk]
        return None

    catalog = md.get("catalog") or md.get("unity_catalog") or md.get("uc_catalog") or md.get("db_catalog") or ""
    database = md.get("database") or md.get("db") or ""
    if not catalog and database:
        catalog = database

    schema_name = md.get("schema") or md.get("namespace") or md.get("database_schema") or ""
    table = md.get("table") or md.get("table_name") or md.get("relation") or md.get("object") or md.get("object_name") or ""

    if not table:
        for fk in ("table_ref", "full_table_name", "qualified_name", "name"):
            if md.get(fk):
                c, s, t = parse_qualified_name(md[fk])
                catalog = catalog or (c or "")
                schema_name = schema_name or (s or "")
                table = table or (t or "")
                if table:
                    break

    if not table:
        return None

    parts = [p for p in (catalog, schema_name, table) if p]
    return ".".join(parts) if parts else None


def extract_column_comment(field: pa.Field) -> Optional[str]:
    md = field.metadata or {}
    if not md:
        return None

    norm: Dict[str, str] = {}
    for k, v in md.items():
        ks = _b2s(k).lower()
        vs = _b2s(v)
        if ks:
            norm[ks] = vs

    for key in ("comment", "description", "doc", "column_comment", "spark.comment", "delta.comment"):
        val = norm.get(key, "").strip()
        if val:
            return val

    for k in ("meta", "metadata", "attrs", "properties"):
        obj = _try_parse_json_blob(norm.get(k, ""))
        if isinstance(obj, dict):
            for kk in ("comment", "description", "doc"):
                vv = _b2s(obj.get(kk, "")).strip()
                if vv:
                    return vv

    return None


# Token-light types for context (not for DDL)
def _arrow_type_compact(dt: pa.DataType) -> str:
    if pa.types.is_int64(dt):
        return "i64"
    if pa.types.is_int32(dt):
        return "i32"
    if pa.types.is_int16(dt):
        return "i16"
    if pa.types.is_int8(dt):
        return "i8"
    if pa.types.is_float64(dt):
        return "f64"
    if pa.types.is_float32(dt) or pa.types.is_float16(dt):
        return "f32"
    if pa.types.is_boolean(dt):
        return "bool"
    if pa.types.is_string(dt) or pa.types.is_large_string(dt):
        return "str"
    if pa.types.is_timestamp(dt):
        return "ts"
    if pa.types.is_date32(dt) or pa.types.is_date64(dt):
        return "date"
    if pa.types.is_decimal(dt):
        return f"dec({dt.precision},{dt.scale})"
    if pa.types.is_list(dt) or pa.types.is_large_list(dt) or pa.types.is_fixed_size_list(dt):
        return f"arr<{_arrow_type_compact(dt.value_type)}>"
    if pa.types.is_struct(dt):
        return "struct"
    if pa.types.is_map(dt):
        return "map"
    return "any"


def _fields_signature_compact(fields: Iterable[pa.Field], *, include_comments: bool) -> str:
    # Super compact: col:type[, ...] + optional short comment
    parts: List[str] = []
    for f in fields:
        t = _arrow_type_compact(f.type)
        if include_comments:
            c = extract_column_comment(f)
            if c:
                c = c.replace("\n", " ").strip()
                if len(c) > 60:
                    c = c[:57] + "..."
                parts.append(f"{f.name}:{t}#{c}")
                continue
        parts.append(f"{f.name}:{t}")
    return ",".join(parts)


@dataclass
class SQLAISession(AISession):
    flavor: SQLFlavor = SQLFlavor.DATABRICKS
    max_context_objects: int = 25

    # Registry
    _fields: Dict[str, List[pa.Field]] = field(default_factory=dict, init=False)
    _objects: Dict[str, str] = field(default_factory=dict, init=False)  # alias -> table/collection name
    _meta: Dict[str, Dict[object, object]] = field(default_factory=dict, init=False)  # alias -> schema-level metadata

    # Token controls
    include_comments_in_context: bool = True
    max_tables_in_context: int = 16  # big token saver

    def system_prompt(self) -> str:
        if self.flavor == SQLFlavor.MONGODB:
            return (
                "Return ONLY a MongoDB aggregation pipeline as a strict JSON array. No prose.\n"
                "Use only fields in context. Prefer $match then $group/$project then $sort then $limit."
            )

        # SQL
        dialect = "Databricks/Spark SQL" if self.flavor == SQLFlavor.DATABRICKS else "PostgreSQL"
        return (
            f"Return ONLY {dialect} query text. No prose, no markdown.\n"
            "Use only columns in context. Prefer fully-qualified table names when provided.\n"
            "Keep it short: avoid CTE unless necessary; return a single statement."
        )

    def set_flavor(self, flavor: SQLFlavor) -> None:
        self.flavor = flavor

    def register_fields(
        self,
        alias: str,
        fields: List[pa.Field],
        *,
        schema_metadata: Optional[Dict[object, object]] = None,
        object_name: Optional[str] = None,
        prefer_metadata_object_name: bool = True,
    ) -> None:
        if len(self._fields) >= self.max_context_objects and alias not in self._fields:
            oldest = next(iter(self._fields.keys()))
            self._fields.pop(oldest, None)
            self._objects.pop(oldest, None)
            self._meta.pop(oldest, None)

        self._fields[alias] = fields
        self._meta[alias] = dict(schema_metadata or {})

        resolved = object_name
        if resolved is None and prefer_metadata_object_name:
            resolved = extract_object_name_from_metadata(self._meta[alias], self.flavor)
        if resolved:
            self._objects[alias] = resolved

    def _pick_relevant_aliases(self, prompt: str) -> List[str]:
        """
        Cheap heuristic to keep context tiny:
        score aliases by presence of alias/object/column names in prompt.
        """
        p = prompt.lower()
        scores: List[Tuple[int, str]] = []

        for alias, cols in self._fields.items():
            s = 0
            if alias.lower() in p:
                s += 5
            obj = self._objects.get(alias, "")
            if obj and obj.lower() in p:
                s += 5
            # column hits (cap influence)
            hit = 0
            for f in cols:
                n = f.name.lower()
                if n in p:
                    hit += 1
                    if hit >= 6:
                        break
            s += min(hit, 6)
            scores.append((s, alias))

        scores.sort(reverse=True)
        picked = [a for s, a in scores if s > 0][: self.max_tables_in_context]

        # fallback: if nothing matched, include first table only (still minimal)
        if not picked and self._fields:
            picked = [next(iter(self._fields.keys()))]

        return picked

    def _build_schema_context(self, aliases: List[str]) -> str:
        lines = ["ctx:"]
        for a in aliases:
            obj = self._objects.get(a, "")
            sig = _fields_signature_compact(
                self._fields[a],
                include_comments=self.include_comments_in_context and self.flavor != SQLFlavor.MONGODB,
            )
            # tiny format: alias=>obj|cols
            if obj:
                lines.append(f"{a}=>{obj}|{sig}")
            else:
                lines.append(f"{a}|{sig}")
        return "\n".join(lines)

    def generate_query(
        self,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 4200,
        extra_instructions: Optional[str] = None,
        tables: Optional[List[str]] = None,
    ) -> str:
        # decide which tables to include (token saver)
        aliases = tables if tables else self._pick_relevant_aliases(user_prompt)
        context_system = self._build_schema_context(aliases)

        return self.chat(
            user_prompt,
            context_system=context_system,
            extra_instructions=extra_instructions,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            strip_code_fences=True,
        )
