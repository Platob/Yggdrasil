// Decode an Arrow IPC stream (the backend's fast tabular wire) into a plain
// {columns, rows} shape the grid can render. Int64 comes back as BigInt in
// arrow-js — coerced to Number when safe, else a string, so React/JSON stay
// happy.
import { tableFromIPC, DataType, type Field } from "apache-arrow";
import type { TabularCell } from "./api";

export interface DecodedTable {
  columns: { name: string; type: string }[];
  rows: TabularCell[][];
  numRows: number;
}

function normalizeCell(v: unknown): TabularCell {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") return Number.isSafeInteger(Number(v)) ? Number(v) : v.toString();
  if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") return v;
  if (v instanceof Date) return v.toISOString();
  return String(v);
}

// Small LRU+TTL cache of decoded tables keyed by URL — sheet-switches, zoom-
// back, and re-opening the same window hit the client instead of re-fetching
// and re-decoding. Bounded so it never grows heavy.
const _cache = new Map<string, { t: number; v: DecodedTable }>();
const ARROW_TTL = 30_000;
const ARROW_MAX = 8;

export function clearArrowCache(): void { _cache.clear(); }

export async function fetchArrowTable(url: string, useCache = true): Promise<DecodedTable> {
  if (useCache) {
    const hit = _cache.get(url);
    if (hit && Date.now() - hit.t < ARROW_TTL) return hit.v;
  }
  const res = await fetch(url);
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json())?.detail ?? ""; } catch { /* not json */ }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
  const columns = table.schema.fields.map((f) => ({ name: f.name, type: String(f.type) }));
  const colVectors = columns.map((_, i) => table.getChildAt(i));
  const rows: TabularCell[][] = [];
  for (let r = 0; r < table.numRows; r++) {
    const row: TabularCell[] = [];
    for (let c = 0; c < colVectors.length; c++) row.push(normalizeCell(colVectors[c]?.get(r)));
    rows.push(row);
  }
  const result = { columns, rows, numRows: table.numRows };
  if (useCache) {
    _cache.set(url, { t: Date.now(), v: result });
    if (_cache.size > ARROW_MAX) _cache.delete(_cache.keys().next().value as string);
  }
  return result;
}

// ── Rich decode: keeps nested list/struct/map values + a column "kind" so the
// grid can render them collapsibly instead of stringifying. ─────────────────

export type ColKind = "number" | "string" | "bool" | "date" | "list" | "struct" | "map" | "other";
export interface RichColumn { name: string; type: string; kind: ColKind }
export interface RichTable { columns: RichColumn[]; rows: unknown[][]; numRows: number }

function kindOf(f: Field): ColKind {
  const t = f.type;
  if (DataType.isList(t) || DataType.isFixedSizeList(t)) return "list";
  if (DataType.isStruct(t)) return "struct";
  if (DataType.isMap(t)) return "map";
  if (DataType.isDate(t) || DataType.isTimestamp(t) || DataType.isTime(t)) return "date";
  if (DataType.isBool(t)) return "bool";
  if (DataType.isInt(t) || DataType.isFloat(t) || DataType.isDecimal(t)) return "number";
  if (DataType.isUtf8(t) || DataType.isLargeUtf8(t)) return "string";
  return "other";
}

// Materialise Arrow's lazy row proxies into plain JS so React/JSON can handle
// nested values.
function plain(v: unknown): unknown {
  if (v == null) return null;
  if (typeof v === "bigint") return Number.isSafeInteger(Number(v)) ? Number(v) : v.toString();
  if (v instanceof Date) return v.toISOString();
  if (Array.isArray(v)) return v.map(plain);
  if (typeof v === "object") {
    const o = v as { toArray?: () => unknown[]; toJSON?: () => unknown };
    if (typeof o.toArray === "function") return o.toArray().map(plain);
    if (typeof o.toJSON === "function") return plain(o.toJSON());
    const out: Record<string, unknown> = {};
    for (const [k, val] of Object.entries(v as Record<string, unknown>)) out[k] = plain(val);
    return out;
  }
  return v;
}

export function decodeArrowRich(buf: Uint8Array): RichTable {
  const table = tableFromIPC(buf);
  const fields = table.schema.fields;
  const columns: RichColumn[] = fields.map((f) => ({ name: f.name, type: String(f.type), kind: kindOf(f) }));
  const cols = fields.map((_, i) => table.getChildAt(i));
  const rows: unknown[][] = [];
  for (let r = 0; r < table.numRows; r++) {
    const row: unknown[] = [];
    for (let c = 0; c < cols.length; c++) row.push(plain(cols[c]?.get(r)));
    rows.push(row);
  }
  return { columns, rows, numRows: table.numRows };
}

export async function fetchArrowRichPost(url: string, body: unknown): Promise<RichTable> {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let detail = ""; try { detail = (await res.json())?.detail ?? ""; } catch { /* not json */ }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  return decodeArrowRich(new Uint8Array(await res.arrayBuffer()));
}

// A staged-session window: rich Arrow body + the has-more / rows headers.
export async function fetchWindowRich(body: unknown): Promise<{ table: RichTable; hasMore: boolean; rows: number }> {
  const res = await fetch("/api/v2/saga/session/window", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!res.ok) {
    let detail = ""; try { detail = (await res.json())?.detail ?? ""; } catch { /* not json */ }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  const hasMore = res.headers.get("X-Has-More") === "1";
  const rows = Number(res.headers.get("X-Window-Rows") ?? 0);
  return { table: decodeArrowRich(new Uint8Array(await res.arrayBuffer())), hasMore, rows };
}
