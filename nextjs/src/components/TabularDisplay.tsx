"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { fetchWindowRich, type ColKind } from "@/lib/arrow";
import { createSession, closeSession, materializeSql, type PlanGraph, type SagaFilter, type WindowTransform } from "@/lib/api";
import TabularAnalyze from "@/components/TabularAnalyze";
import RegisterSagaModal from "@/components/RegisterSagaModal";

export interface TabularColumnDef { name: string; dtype?: string; kind?: ColKind }
export type Cell = unknown;

/** A LazyTabular-style source: explicit SQL, or a straight source path that
 *  becomes `SELECT * FROM '<source>'`. The result is staged server-side as
 *  Arrow IPC and scrolled in lazy windows — it is never decoded whole. */
export interface QuerySpec {
  sql?: string;
  source?: string;
  catalog?: string;
  schema?: string;
  node?: string;
  /** Optional cap on rows staged (a safety bound; the grid still windows). */
  limit?: number;
}

interface Props {
  /** In-memory mode (small, already-decoded rows). */
  data?: { columns: TabularColumnDef[]; rows: Cell[][] };
  /** Staged-session mode: stage → window lazily. */
  query?: QuerySpec;
  plan?: PlanGraph | null;
  caption?: string;
  maxHeight?: number | string;
  filterable?: boolean;
  onExpand?: () => void;
  /** Rows per window / per client page (default 200). */
  pageSize?: number;
}

const NUMERIC = /int|float|double|decimal|num/i;
const NESTED: ColKind[] = ["list", "struct", "map"];
const FILTER_OP = (s: string) => (s.trim() !== "" && !isNaN(Number(s)) ? "==" : "contains");

function kindFromDtype(d?: string): ColKind {
  const s = (d ?? "").toLowerCase();
  if (s.startsWith("list") || s.startsWith("large_list") || s.includes("[]")) return "list";
  if (s.startsWith("struct")) return "struct";
  if (s.startsWith("map")) return "map";
  if (/date|time/.test(s)) return "date";
  if (/bool/.test(s)) return "bool";
  if (NUMERIC.test(s)) return "number";
  return "string";
}

function preview(v: unknown): string {
  if (Array.isArray(v)) return `[${v.length}]`;
  if (v && typeof v === "object") return `{${Object.keys(v as object).length}}`;
  return "";
}
function fmt(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

interface XForm { op: "explode" | "unnest"; col: string }
type Grid = { columns: TabularColumnDef[]; rows: Cell[][] };

// Client-side nested transforms for in-memory mode (server-side in session mode).
function applyXforms(columns: TabularColumnDef[], rows: Cell[][], xs: XForm[]): Grid {
  let c = columns, r = rows;
  for (const x of xs) {
    const i = c.findIndex((col) => col.name === x.col);
    if (i < 0) continue;
    if (x.op === "explode") {
      const nr: Cell[][] = [];
      for (const row of r) {
        const v = row[i];
        if (Array.isArray(v)) { for (const el of v) { const nrow = [...row]; nrow[i] = el; nr.push(nrow); } }
        else nr.push(row);
      }
      r = nr;
      c = c.map((col, j) => (j === i ? { ...col, dtype: "item", kind: "other" } : col));
    } else {
      const fields: string[] = [];
      for (const row of r) { const v = row[i]; if (v && typeof v === "object" && !Array.isArray(v)) for (const k of Object.keys(v)) if (!fields.includes(k)) fields.push(k); }
      const cols: TabularColumnDef[] = fields.map((f) => ({ name: `${x.col}.${f}`, dtype: "", kind: "other" }));
      c = [...c.slice(0, i), ...cols, ...c.slice(i + 1)];
      r = r.map((row) => { const v = row[i] as Record<string, unknown> | null; const vals = fields.map((f) => (v && typeof v === "object" ? (v[f] ?? null) : null)); return [...row.slice(0, i), ...vals, ...row.slice(i + 1)]; });
    }
  }
  return { columns: c, rows: r };
}

/**
 * Embeddable typed tabular grid. In `query` mode the result is staged server-
 * side as Arrow IPC and scrolled in lazy windows — filter, sort and nested
 * explode/unnest run as polars **lazy** transforms over the mmap'd file, so the
 * browser only ever holds the visible window no matter how big the result. The
 * session is cleared on close/unmount/disconnect. `data` mode keeps small
 * already-decoded rows in memory.
 */
export default function TabularDisplay({
  data, query, plan, caption, maxHeight = 460, filterable = true, onExpand, pageSize = 200,
}: Props) {
  const isSession = !!query;
  const querySql = useMemo(() => {
    if (!query) return "";
    const cap = query.limit ? ` LIMIT ${query.limit}` : "";
    if (query.sql) return /\blimit\b/i.test(query.sql) ? query.sql : query.sql + cap;
    return `SELECT * FROM '${query.source}'${cap}`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query?.sql, query?.source, query?.limit]);

  // -- shared view state (column-name keyed so it survives unnest reshaping)
  const [colFilters, setColFilters] = useState<Record<string, string>>({});
  const [colSort, setColSort] = useState<{ name: string; desc: boolean } | null>(null);
  const [xforms, setXforms] = useState<XForm[]>([]);
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [copied, setCopied] = useState(false);

  // -- session mode state
  const [session, setSession] = useState<{ path: string; node?: string; cols: TabularColumnDef[]; total: number; ms: number } | null>(null);
  const [winCols, setWinCols] = useState<TabularColumnDef[]>([]);
  const [winRows, setWinRows] = useState<Cell[][]>([]);
  const [winMore, setWinMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const winOffset = useRef(0);
  const reqId = useRef(0);

  // -- memory mode load-more
  const [visible, setVisible] = useState(pageSize);

  // -- analyze / register
  const [pane, setPane] = useState<"grid" | "analyze">("grid");
  const [aPath, setAPath] = useState<string | null>(query?.source ?? null);
  const [aBusy, setABusy] = useState(false);
  const [aErr, setAErr] = useState("");
  const [regOpen, setRegOpen] = useState(false);
  const [err, setErr] = useState("");

  const serverParams = useCallback(() => {
    const filters: SagaFilter[] = Object.entries(colFilters)
      .filter(([, v]) => v.trim())
      .map(([column, v]) => {
        const op = FILTER_OP(v);
        return { column, op, value: op === "==" ? Number(v) : v };
      });
    const transforms: WindowTransform[] = xforms.map((x) => ({ op: x.op, column: x.col }));
    return { filters, transforms, sort: colSort?.name, descending: colSort?.desc ?? false };
  }, [colFilters, xforms, colSort]);

  // Stage a session whenever the query changes; clear the previous one.
  useEffect(() => {
    if (!isSession) return;
    let path: string | null = null, node: string | undefined;
    const id = ++reqId.current;
    setLoading(true); setErr(""); setSession(null); setWinRows([]); winOffset.current = 0;
    createSession({ sql: querySql, catalog: query?.catalog, schema: query?.schema, node: query?.node })
      .then((s) => { if (id !== reqId.current) return; path = s.path; node = s.node_id; setSession({ path: s.path, node: s.node_id, cols: s.columns.map((c) => ({ name: c.name, dtype: c.dtype, kind: kindFromDtype(c.dtype) })), total: s.row_count, ms: s.elapsed_ms }); })
      .catch((e) => { if (id === reqId.current) { setErr(String(e)); setLoading(false); } });
    return () => { if (path) closeSession(path, node); };
  }, [isSession, querySql, query?.catalog, query?.schema, query?.node]);

  // Fetch the first window (reset) whenever the session or view params change.
  const fetchWindow = useCallback(async (offset: number, append: boolean) => {
    if (!session) return;
    const id = reqId.current;
    setLoading(true); setErr("");
    try {
      const { table, hasMore } = await fetchWindowRich({
        path: session.path, node: session.node, offset, limit: pageSize, ...serverParams(),
      });
      if (id !== reqId.current) return;
      const cols = table.columns.map((c) => ({ name: c.name, dtype: c.type, kind: c.kind }));
      setWinCols(cols);
      setWinRows((prev) => (append ? [...prev, ...table.rows] : table.rows));
      setWinMore(hasMore);
      winOffset.current = offset;
    } catch (e) { if (id === reqId.current) setErr(String(e)); }
    finally { if (id === reqId.current) setLoading(false); }
  }, [session, pageSize, serverParams]);

  useEffect(() => {
    if (!session) return;
    const t = setTimeout(() => fetchWindow(0, false), 250);  // debounce filter typing
    return () => clearTimeout(t);
  }, [session, colFilters, colSort, xforms, fetchWindow]);

  // Clear the staged session if the tab/page goes away.
  useEffect(() => {
    if (!isSession) return;
    const bye = () => { if (session) closeSession(session.path, session.node); };
    window.addEventListener("beforeunload", bye);
    return () => window.removeEventListener("beforeunload", bye);
  }, [isSession, session]);

  // -- resolve the grid's columns + rows per mode
  const memGrid = useMemo<Grid>(() => {
    if (isSession) return { columns: [], rows: [] };
    const base = data ?? { columns: [], rows: [] };
    let g = xforms.length ? applyXforms(base.columns, base.rows, xforms) : base;
    const active = Object.entries(colFilters).filter(([, v]) => v.trim());
    let r = g.rows;
    if (active.length) r = r.filter((row) => active.every(([name, v]) => {
      const ci = g.columns.findIndex((c) => c.name === name); return ci < 0 || fmt(row[ci]).toLowerCase().includes(v.toLowerCase());
    }));
    if (colSort) {
      const ci = g.columns.findIndex((c) => c.name === colSort.name);
      if (ci >= 0) r = [...r].sort((a, b) => { const x = a[ci], y = b[ci]; if (x == null) return 1; if (y == null) return -1; const c = typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y)); return colSort.desc ? -c : c; });
    }
    g = { columns: g.columns, rows: r };
    return g;
  }, [isSession, data, xforms, colFilters, colSort]);

  const gcols = useMemo(
    () => (isSession ? (winCols.length ? winCols : session?.cols ?? []) : memGrid.columns),
    [isSession, winCols, session, memGrid.columns],
  );
  const grows = isSession ? winRows : memGrid.rows;
  const shown = isSession ? grows : grows.slice(0, visible);
  const hasMore = isSession ? winMore : grows.length > visible;
  const kinds = useMemo(() => gcols.map((c) => c.kind ?? kindFromDtype(c.dtype)), [gcols]);

  const onSort = (name: string) => setColSort((s) => (s?.name === name ? (s.desc ? null : { name, desc: true }) : { name, desc: false }));
  const onFilter = (name: string, v: string) => setColFilters((f) => ({ ...f, [name]: v }));
  const addXform = (op: "explode" | "unnest", col: string) => setXforms((x) => [...x, { op, col }]);
  const copy = (v: unknown) => navigator.clipboard?.writeText(fmt(v)).then(() => { setCopied(true); setTimeout(() => setCopied(false), 900); }).catch(() => {});
  const toggleCell = (key: string) => setOpen((o) => { const n = new Set(o); if (n.has(key)) n.delete(key); else n.add(key); return n; });
  const loadMore = () => { if (isSession) fetchWindow(winOffset.current + pageSize, true); else setVisible((v) => v + pageSize); };

  // -- analyze (materialised path) + register
  const canAnalyze = !!(query && (query.source || query.sql));
  const analyzeCols = useMemo(() => (session?.cols ?? data?.columns ?? []).map((c) => ({ name: c.name, type: c.dtype })), [session, data]);
  const openAnalyze = async () => {
    setPane("analyze");
    if (aPath) return;
    if (query?.source) { setAPath(query.source); return; }
    if (query?.sql) {
      setABusy(true); setAErr("");
      try { const r = await materializeSql({ sql: query.sql, catalog: query.catalog, schema: query.schema, node: query.node }); setAPath(r.path); }
      catch (e) { setAErr(String(e)); } finally { setABusy(false); }
    }
  };
  useEffect(() => { setAPath(query?.source ?? null); setPane("grid"); }, [query?.sql, query?.source]);

  const total = isSession ? session?.total : grows.length;
  const cap = caption ?? (isSession && session ? `${total?.toLocaleString()} rows · ${session.ms} ms · ${grows.length} loaded` : undefined);

  return (
    <div className="flex flex-col h-full min-h-0">
      {plan && plan.ops.length > 0 && (
        <div className="flex items-center gap-1 px-1 pb-1.5 overflow-x-auto text-[10px] font-mono">
          {plan.ops.map((o, i) => (
            <span key={o.id} className="flex items-center gap-1 shrink-0">
              {i > 0 && <span className="text-muted">▸</span>}
              <span className="px-1.5 py-0.5 rounded bg-white/[0.04] border border-white/[0.08] text-frost/80" title={o.detail}>
                {o.title}{o.rows != null ? ` ·${o.rows >= 1000 ? `${Math.round(o.rows / 1000)}k` : o.rows}r` : ""}{o.elapsed_ms != null ? ` ·${o.elapsed_ms}ms` : ""}
              </span>
            </span>
          ))}
        </div>
      )}
      {(cap || onExpand || canAnalyze) && (
        <div className="flex items-center gap-2 px-1 pb-1.5 text-[11px] text-muted">
          {canAnalyze && (
            <span className="flex items-center gap-1 font-mono">
              <button onClick={() => setPane("grid")} className={pane === "grid" ? "text-frost font-semibold" : "text-muted hover:text-foreground-dim"}>Grid</button>
              <span className="text-muted/40">|</span>
              <button onClick={openAnalyze} className={pane === "analyze" ? "text-frost font-semibold" : "text-muted hover:text-foreground-dim"}>Analyze</button>
              <span className="w-px h-3 bg-white/10 mx-1" />
            </span>
          )}
          {cap && <span className="font-mono">{cap}</span>}
          {xforms.map((x, i) => (
            <span key={i} className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-amber/10 text-amber border border-amber/20 font-mono">
              {x.op} {x.col}
              <button onClick={() => setXforms((xs) => xs.filter((_, j) => j !== i))} className="text-amber/60 hover:text-rose">✕</button>
            </span>
          ))}
          {copied && <span className="text-emerald/80">· copied</span>}
          {(loading || aBusy) && <span className="text-frost/80">· {aBusy ? "materialising…" : "loading…"}</span>}
          <div className="flex-1" />
          {query?.sql && (
            <button onClick={() => setRegOpen(true)} className="text-emerald/70 hover:text-emerald" title="register this query as a Saga view">⊕ view</button>
          )}
          {onExpand && <button onClick={onExpand} className="text-frost/70 hover:text-frost" title="expand">⤢</button>}
        </div>
      )}
      {err && <div className="text-[12px] text-rose/90 font-mono p-2 bg-rose/5 border border-rose/20 rounded-lg mb-1 break-words">{err}</div>}
      {aErr && <div className="text-[12px] text-rose/90 font-mono p-2 bg-rose/5 border border-rose/20 rounded-lg mb-1 break-words">{aErr}</div>}

      {pane === "analyze" && canAnalyze ? (
        aPath ? (
          <TabularAnalyze path={aPath} node={query?.node} columns={analyzeCols} />
        ) : (
          <div className="flex-1 p-4 text-[12px] text-muted">{aBusy ? "materialising result…" : "preparing analytics…"}</div>
        )
      ) : (
      <div className="flex-1 overflow-auto border border-white/[0.06] rounded-lg min-h-0" style={{ maxHeight }}>
        <table className="w-full text-[12px] font-mono border-collapse">
          <thead className="sticky top-0 z-10 bg-[#0a0a1a]">
            <tr>
              <th className="w-10 text-right px-2 py-1.5 border-b border-white/[0.08] text-muted font-normal select-none">#</th>
              {gcols.map((c, ci) => (
                <th key={ci} className="text-left px-2 py-1.5 border-b border-white/[0.08] text-frost/80 whitespace-nowrap select-none group">
                  <span onClick={() => onSort(c.name)} className="cursor-pointer hover:text-frost">
                    {c.name}
                    <span className="text-muted ml-1 font-normal">{NESTED.includes(kinds[ci]) ? kinds[ci] : (c.dtype ?? "")}</span>
                    {colSort?.name === c.name && <span className="ml-1 text-frost">{colSort.desc ? "▼" : "▲"}</span>}
                  </span>
                  {kinds[ci] === "list" && (
                    <button onClick={() => addXform("explode", c.name)} title="explode list → one row per element"
                      className="ml-1.5 opacity-0 group-hover:opacity-100 text-amber/70 hover:text-amber">⊞</button>
                  )}
                  {kinds[ci] === "struct" && (
                    <button onClick={() => addXform("unnest", c.name)} title="unnest struct → one column per field"
                      className="ml-1.5 opacity-0 group-hover:opacity-100 text-frost/70 hover:text-frost">⊟</button>
                  )}
                </th>
              ))}
            </tr>
            {filterable && gcols.length > 0 && (
              <tr>
                <th className="bg-[#0a0a1a] border-b border-white/[0.06]" />
                {gcols.map((c, ci) => (
                  <th key={ci} className="px-1 py-1 border-b border-white/[0.06] bg-[#0a0a1a]">
                    <input value={colFilters[c.name] ?? ""} placeholder="filter"
                      onChange={(e) => onFilter(c.name, e.target.value)}
                      className="w-full min-w-[60px] bg-white/[0.04] border border-white/[0.06] rounded px-1.5 py-0.5 text-[10px] font-normal outline-none focus:border-frost/30" />
                  </th>
                ))}
              </tr>
            )}
          </thead>
          <tbody>
            {shown.map((row, ri) => (
              <tr key={ri} className="hover:bg-white/[0.02] align-top">
                <td className="text-right px-2 py-1 border-b border-white/[0.04] text-muted/60 select-none">{ri + 1}</td>
                {row.map((cell, ci) => {
                  const nested = NESTED.includes(kinds[ci]) && cell != null && typeof cell === "object";
                  const key = `${ri}:${ci}`;
                  if (nested) {
                    const isOpen = open.has(key);
                    return (
                      <td key={ci} className="px-2 py-1 border-b border-white/[0.04] text-foreground-dim max-w-[360px]">
                        <button onClick={() => toggleCell(key)} className="text-frost/70 hover:text-frost mr-1">{isOpen ? "▾" : "▸"}</button>
                        <span className="text-amber/70">{preview(cell)}</span>
                        {isOpen && (
                          <pre className="mt-1 text-[11px] whitespace-pre-wrap break-words bg-[#06060f] border border-white/[0.06] rounded p-1.5 max-h-48 overflow-auto">{JSON.stringify(cell, null, 2)}</pre>
                        )}
                      </td>
                    );
                  }
                  return (
                    <td key={ci} onClick={() => copy(cell)} title="click to copy"
                      className={`px-2 py-1 border-b border-white/[0.04] whitespace-nowrap max-w-[320px] truncate cursor-copy ${kinds[ci] === "number" ? "text-right text-frost/80" : "text-foreground-dim"}`}>
                      {cell === null || cell === undefined ? <span className="text-muted italic">null</span> : fmt(cell)}
                    </td>
                  );
                })}
              </tr>
            ))}
            {shown.length === 0 && !loading && (
              <tr><td className="px-2 py-3 text-muted text-center" colSpan={gcols.length + 1}>no rows</td></tr>
            )}
          </tbody>
        </table>
        {hasMore && (
          <button onClick={loadMore} disabled={loading}
            className="w-full py-2 text-[11px] text-frost/80 hover:text-frost hover:bg-white/[0.03] border-t border-white/[0.06] disabled:opacity-40">
            {loading ? "loading…" : `load ${pageSize} more${isSession && total ? ` (${grows.length} / ${total.toLocaleString()})` : ""}`}
          </button>
        )}
      </div>
      )}

      {regOpen && query?.sql && (
        <RegisterSagaModal node={query.node} defaults={{ objectType: "VIEW", definition: query.sql }} onClose={() => setRegOpen(false)} />
      )}
    </div>
  );
}
