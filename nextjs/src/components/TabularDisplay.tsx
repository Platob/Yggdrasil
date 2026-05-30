"use client";

import { useMemo, useState } from "react";

export interface TabularColumnDef { name: string; dtype?: string }
export type Cell = string | number | boolean | null;

interface Props {
  columns: TabularColumnDef[];
  rows: Cell[][];
  /** Optional caption shown above the grid (e.g. "5 rows · 12 ms"). */
  caption?: string;
  /** Max body height; the grid scrolls inside it. */
  maxHeight?: number | string;
  /** Show the per-column quick filter row. */
  filterable?: boolean;
  /** Called when the expand button is clicked (omit to hide it). */
  onExpand?: () => void;
}

function fmt(v: Cell): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(v);
  return String(v);
}

const NUMERIC = /int|float|double|decimal|num/i;

/**
 * Embeddable rich tabular grid over in-memory {columns, rows}. Sticky header
 * with dtype chips, row numbers, click-to-sort, a per-column quick filter,
 * right-aligned numerics and click-to-copy cells. Used inline (Saga results)
 * and inside a modal — it is *not* itself a modal.
 */
export default function TabularDisplay({ columns, rows, caption, maxHeight = 420, filterable = true, onExpand }: Props) {
  const [sort, setSort] = useState<{ col: number; dir: 1 | -1 } | null>(null);
  const [filters, setFilters] = useState<Record<number, string>>({});
  const [copied, setCopied] = useState<string | null>(null);

  const isNum = useMemo(() => columns.map((c) => NUMERIC.test(c.dtype ?? "")), [columns]);

  const view = useMemo(() => {
    let r = rows;
    const active = Object.entries(filters).filter(([, v]) => v.trim());
    if (active.length) {
      r = r.filter((row) => active.every(([ci, v]) => fmt(row[Number(ci)]).toLowerCase().includes(v.toLowerCase())));
    }
    if (sort) {
      const { col, dir } = sort;
      r = [...r].sort((a, b) => {
        const x = a[col], y = b[col];
        if (x === null) return 1;
        if (y === null) return -1;
        if (typeof x === "number" && typeof y === "number") return (x - y) * dir;
        return String(x).localeCompare(String(y)) * dir;
      });
    }
    return r;
  }, [rows, filters, sort]);

  const toggleSort = (col: number) =>
    setSort((s) => (s?.col === col ? (s.dir === 1 ? { col, dir: -1 } : null) : { col, dir: 1 }));

  const copy = (v: Cell) => {
    const s = fmt(v);
    navigator.clipboard?.writeText(s).then(() => { setCopied(s); setTimeout(() => setCopied(null), 900); }).catch(() => {});
  };

  return (
    <div className="flex flex-col h-full min-h-0">
      {(caption || onExpand) && (
        <div className="flex items-center gap-2 px-1 pb-1.5 text-[11px] text-muted">
          {caption && <span className="font-mono">{caption}</span>}
          {view.length !== rows.length && <span className="text-amber/80">· {view.length} shown</span>}
          {copied !== null && <span className="text-emerald/80">· copied</span>}
          <div className="flex-1" />
          {onExpand && <button onClick={onExpand} className="text-frost/70 hover:text-frost" title="expand">⤢</button>}
        </div>
      )}
      <div className="flex-1 overflow-auto border border-white/[0.06] rounded-lg min-h-0" style={{ maxHeight }}>
        <table className="w-full text-[12px] font-mono border-collapse">
          <thead className="sticky top-0 z-10 bg-[#0a0a1a]">
            <tr>
              <th className="w-10 text-right px-2 py-1.5 border-b border-white/[0.08] text-muted font-normal select-none">#</th>
              {columns.map((c, ci) => (
                <th key={ci} onClick={() => toggleSort(ci)}
                  className="text-left px-2 py-1.5 border-b border-white/[0.08] text-frost/80 whitespace-nowrap cursor-pointer hover:text-frost select-none">
                  {c.name}
                  {c.dtype && <span className="text-muted ml-1 font-normal">{c.dtype}</span>}
                  {sort?.col === ci && <span className="ml-1 text-frost">{sort.dir === 1 ? "▲" : "▼"}</span>}
                </th>
              ))}
            </tr>
            {filterable && columns.length > 0 && (
              <tr>
                <th className="bg-[#0a0a1a] border-b border-white/[0.06]" />
                {columns.map((c, ci) => (
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
            {view.map((row, ri) => (
              <tr key={ri} className="hover:bg-white/[0.02] group">
                <td className="text-right px-2 py-1 border-b border-white/[0.04] text-muted/60 select-none">{ri + 1}</td>
                {row.map((cell, ci) => (
                  <td key={ci} onClick={() => copy(cell)} title="click to copy"
                    className={`px-2 py-1 border-b border-white/[0.04] whitespace-nowrap max-w-[320px] truncate cursor-copy ${isNum[ci] ? "text-right text-frost/80" : "text-foreground-dim"}`}>
                    {cell === null ? <span className="text-muted italic">null</span> : fmt(cell)}
                  </td>
                ))}
              </tr>
            ))}
            {view.length === 0 && (
              <tr><td className="px-2 py-3 text-muted text-center" colSpan={columns.length + 1}>no rows</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
