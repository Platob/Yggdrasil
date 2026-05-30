"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { fetchArrowRichPost, type ColKind } from "@/lib/arrow";
import type { PlanGraph } from "@/lib/api";

export interface TabularColumnDef { name: string; dtype?: string; kind?: ColKind }
export type Cell = unknown;

/** A LazyTabular-style source: explicit SQL, or a straight source path that
 *  becomes `SELECT * FROM '<source>'`. Either way it streams typed Arrow. */
export interface QuerySpec {
  sql?: string;
  source?: string;
  catalog?: string;
  schema?: string;
  node?: string;
  /** Row cap pushed into the plan (default 1000). */
  limit?: number;
}

interface Props {
  /** In-memory mode. */
  data?: { columns: TabularColumnDef[]; rows: Cell[][] };
  /** Arrow-fetch mode (typed, nested, windowed). */
  query?: QuerySpec;
  /** Optional execution-plan strip shown above the grid. */
  plan?: PlanGraph | null;
  caption?: string;
  maxHeight?: number | string;
  filterable?: boolean;
  onExpand?: () => void;
  /** DOM rows rendered before "load more" (default 200) — keeps it smooth. */
  pageSize?: number;
}

const NUMERIC = /int|float|double|decimal|num/i;
const NESTED: ColKind[] = ["list", "struct", "map"];

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

/**
 * Embeddable typed tabular grid. Two modes: in-memory `data`, or `query` which
 * streams Arrow IPC from `/sql.arrow` (a straight source becomes
 * `SELECT * FROM '<source>'`). Nested list/struct/map cells render collapsibly;
 * rows are windowed ("load more") so a wide/deep result never freezes the page.
 * An optional plan strip shows the execution plan that produced the rows.
 */
export default function TabularDisplay({
  data, query, plan, caption, maxHeight = 460, filterable = true, onExpand, pageSize = 200,
}: Props) {
  const [fetched, setFetched] = useState<{ columns: TabularColumnDef[]; rows: Cell[][] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [tookMs, setTookMs] = useState<number | null>(null);

  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  const [filters, setFilters] = useState<Record<number, string>>({});
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [visible, setVisible] = useState(pageSize);
  const [copied, setCopied] = useState(false);
  const reqId = useRef(0);

  useEffect(() => {
    if (!query) return;
    const limit = query.limit ?? 1000;
    const sql = query.sql ?? `SELECT * FROM '${query.source}' LIMIT ${limit}`;
    const id = ++reqId.current;
    setLoading(true); setErr(""); setVisible(pageSize);
    const t0 = performance.now();
    fetchArrowRichPost("/api/v2/saga/sql.arrow", {
      sql, catalog: query.catalog, schema: query.schema, node: query.node, limit,
    })
      .then((t) => {
        if (id !== reqId.current) return;
        setFetched({ columns: t.columns, rows: t.rows });
        setTookMs(Math.round(performance.now() - t0));
      })
      .catch((e) => { if (id === reqId.current) setErr(String(e)); })
      .finally(() => { if (id === reqId.current) setLoading(false); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query?.sql, query?.source, query?.catalog, query?.schema, query?.node, query?.limit, pageSize]);

  const src = data ?? fetched ?? { columns: [], rows: [] };
  const kinds = useMemo(
    () => src.columns.map((c) => c.kind ?? kindFromDtype(c.dtype)),
    [src.columns],
  );

  const view = useMemo(() => {
    let r = src.rows;
    const active = Object.entries(filters).filter(([, v]) => v.trim());
    if (active.length) {
      r = r.filter((row) => active.every(([ci, v]) => fmt(row[Number(ci)]).toLowerCase().includes(v.toLowerCase())));
    }
    if (sort) {
      const { col, dir } = sort;
      r = [...r].sort((a, b) => {
        const x = a[col], y = b[col];
        if (x == null) return 1;
        if (y == null) return -1;
        if (typeof x === "number" && typeof y === "number") return (x - y) * dir;
        return String(x).localeCompare(String(y)) * dir;
      });
    }
    return r;
  }, [src.rows, filters, sort]);

  const shown = view.slice(0, visible);
  const toggleSort = (c: number) =>
    setSort((s) => (s?.col === c ? (s.dir === 1 ? { col: c, dir: -1 } : null) : { col: c, dir: 1 }));
  const copy = (v: unknown) => navigator.clipboard?.writeText(fmt(v)).then(() => { setCopied(true); setTimeout(() => setCopied(false), 900); }).catch(() => {});
  const toggleCell = (key: string) => setOpen((o) => {
    const n = new Set(o);
    if (n.has(key)) n.delete(key); else n.add(key);
    return n;
  });

  const cap = caption ?? (tookMs != null ? `${src.rows.length} rows · ${tookMs} ms` : undefined);

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
      {(cap || onExpand) && (
        <div className="flex items-center gap-2 px-1 pb-1.5 text-[11px] text-muted">
          {cap && <span className="font-mono">{cap}</span>}
          {view.length !== src.rows.length && <span className="text-amber/80">· {view.length} shown</span>}
          {copied && <span className="text-emerald/80">· copied</span>}
          {loading && <span className="text-frost/80">· loading…</span>}
          <div className="flex-1" />
          {onExpand && <button onClick={onExpand} className="text-frost/70 hover:text-frost" title="expand">⤢</button>}
        </div>
      )}
      {err && <div className="text-[12px] text-rose/90 font-mono p-2 bg-rose/5 border border-rose/20 rounded-lg mb-1 break-words">{err}</div>}

      <div className="flex-1 overflow-auto border border-white/[0.06] rounded-lg min-h-0" style={{ maxHeight }}>
        <table className="w-full text-[12px] font-mono border-collapse">
          <thead className="sticky top-0 z-10 bg-[#0a0a1a]">
            <tr>
              <th className="w-10 text-right px-2 py-1.5 border-b border-white/[0.08] text-muted font-normal select-none">#</th>
              {src.columns.map((c, ci) => (
                <th key={ci} onClick={() => toggleSort(ci)}
                  className="text-left px-2 py-1.5 border-b border-white/[0.08] text-frost/80 whitespace-nowrap cursor-pointer hover:text-frost select-none">
                  {c.name}
                  <span className="text-muted ml-1 font-normal">{NESTED.includes(kinds[ci]) ? kinds[ci] : (c.dtype ?? "")}</span>
                  {sort?.col === ci && <span className="ml-1 text-frost">{sort.dir === 1 ? "▲" : "▼"}</span>}
                </th>
              ))}
            </tr>
            {filterable && src.columns.length > 0 && (
              <tr>
                <th className="bg-[#0a0a1a] border-b border-white/[0.06]" />
                {src.columns.map((c, ci) => (
                  <th key={ci} className="px-1 py-1 border-b border-white/[0.06] bg-[#0a0a1a]">
                    <input value={filters[ci] ?? ""} placeholder="filter"
                      onChange={(e) => setFilters((f) => ({ ...f, [ci]: e.target.value }))}
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
              <tr><td className="px-2 py-3 text-muted text-center" colSpan={src.columns.length + 1}>no rows</td></tr>
            )}
          </tbody>
        </table>
        {view.length > visible && (
          <button onClick={() => setVisible((v) => v + pageSize)}
            className="w-full py-2 text-[11px] text-frost/80 hover:text-frost hover:bg-white/[0.03] border-t border-white/[0.06]">
            load {Math.min(pageSize, view.length - visible)} more ({visible} / {view.length})
          </button>
        )}
      </div>
    </div>
  );
}
