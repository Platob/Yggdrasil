"use client";

// Standalone, reusable tabular / workbook editor. Renders any node+path table
// from the backend's Arrow-IPC wire (no JSON row materialization), with a
// metadata header and an editable grid:
//   - parquet/csv/json/…  -> /tabular (inspect + preview.arrow), save rewrites
//     the whole bounded file via /tabular/write.
//   - xlsx/xls            -> /workbook (sheet tabs + read.arrow per sheet),
//     edits saved surgically via /workbook/edit (formulas/other sheets kept).
//
// Decoupled from the files page so runs/DAGs/artifacts can open it, and built
// to grow into the full Excel grid (virtualization, formulas) without touching
// callers.

import { useCallback, useEffect, useState } from "react";
import {
  getTabularInspect,
  tabularPreviewArrowUrl,
  writeTabular,
  isWorkbookName,
  getWorkbookSheets,
  workbookReadArrowUrl,
  editWorkbook,
  fsDownloadUrl,
  type TabularInspect,
  type TabularCell,
  type WorkbookSheet,
} from "@/lib/api";
import { fetchArrowTable } from "@/lib/arrow";

const EDIT_CAP = 2000; // rows we'll load into an editable grid
const PAGE = 100;      // row page size for read-only viewports

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

// Excel-like cell coercion for save: blank -> null, =.. -> formula, numeric
// strings -> numbers, else the raw string.
function coerce(v: string): TabularCell {
  if (v === "") return null;
  if (v.startsWith("=")) return v;
  if (/^-?\d+$/.test(v)) return Number(v);
  if (/^-?\d*\.\d+$/.test(v)) return Number(v);
  return v;
}

interface Props {
  node?: string;
  nodeLabel?: string;
  path: string;
  name: string;
  onClose: () => void;
}

export default function TabularModal({ node, nodeLabel, path, name, onClose }: Props) {
  const isWorkbook = isWorkbookName(name);
  const [info, setInfo] = useState<TabularInspect | null>(null);
  const [sheets, setSheets] = useState<WorkbookSheet[]>([]);
  const [activeSheet, setActiveSheet] = useState<string>("");
  const [columns, setColumns] = useState<{ name: string; type: string }[]>([]);
  const [grid, setGrid] = useState<string[][]>([]);
  const [readOnly, setReadOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [edited, setEdited] = useState<Set<string>>(new Set());
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [page, setPage] = useState(0);   // read-only viewport page
  const [total, setTotal] = useState(0); // total data rows (for paging)

  const decodeInto = useCallback(async (url: string) => {
    const t = await fetchArrowTable(url);
    setColumns(t.columns);
    setGrid(t.rows.map((r) => r.map((v) => (v === null || v === undefined ? "" : String(v)))));
  }, []);

  // ``dims`` is passed explicitly (never read from `sheets` state) so this
  // callback stays stable — otherwise it would re-create whenever sheets load
  // and retrigger the load effect in a loop.
  const loadSheet = useCallback(async (sheet: string, dims: WorkbookSheet[]) => {
    setLoading(true);
    setError(null);
    setEdited(new Set());
    setSaved(false);
    try {
      const dim = dims.find((s) => s.name === sheet);
      const dataRows = dim ? Math.max(0, dim.rows - 1) : 0;
      const editable = dataRows <= EDIT_CAP;
      setReadOnly(!editable);
      setTotal(dataRows);
      setPage(0);
      // Editable sheets load whole (so a save covers every row); large sheets
      // stream the first page and page from there.
      await decodeInto(workbookReadArrowUrl(path, sheet, editable ? { node } : { n_rows: PAGE, skip_rows: 0, node }));
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to read sheet");
    } finally {
      setLoading(false);
    }
  }, [path, node, decodeInto]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setEdited(new Set());
    setSaved(false);
    try {
      if (isWorkbook) {
        const res = await getWorkbookSheets(path, node);
        setSheets(res.sheets);
        const first = res.sheets[0]?.name ?? "";
        setActiveSheet(first);
        if (first) await loadSheet(first, res.sheets);
        else setLoading(false);
      } else {
        const meta = await getTabularInspect(path, node);
        setInfo(meta);
        if (meta.schema_error) { setError(meta.schema_error); return; }
        setReadOnly(!meta.editable);
        setTotal(meta.row_count ?? 0);
        setPage(0);
        const limit = meta.editable && meta.row_count ? meta.row_count : PAGE;
        await decodeInto(tabularPreviewArrowUrl(path, limit, 0, node));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to read table");
    } finally {
      if (!isWorkbook) setLoading(false);
    }
  }, [isWorkbook, path, node, loadSheet, decodeInto]);

  useEffect(() => { load(); }, [load]);

  const switchSheet = (sheet: string) => { setActiveSheet(sheet); loadSheet(sheet, sheets); };

  // Stream a read-only viewport page (Arrow window) without reloading metadata.
  const goToPage = useCallback(async (p: number) => {
    if (p < 0 || p * PAGE >= total) return;
    setLoading(true);
    setError(null);
    try {
      const offset = p * PAGE;
      const url = isWorkbook
        ? workbookReadArrowUrl(path, activeSheet, { n_rows: PAGE, skip_rows: offset, node })
        : tabularPreviewArrowUrl(path, PAGE, offset, node);
      await decodeInto(url);
      setPage(p);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to load page");
    } finally {
      setLoading(false);
    }
  }, [isWorkbook, path, activeSheet, node, total, decodeInto]);

  const setCell = (r: number, c: number, value: string) => {
    setGrid((g) => { const next = g.map((row) => row.slice()); next[r][c] = value; return next; });
    setEdited((e) => new Set(e).add(`${r},${c}`));
    setSaved(false);
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      if (isWorkbook) {
        // Surgical: only the cells the user touched (xlsx is 1-based, row 1 is
        // the header, so grid (r,c) -> sheet (r+2, c+1)).
        const cells = [...edited].map((k) => {
          const [r, c] = k.split(",").map(Number);
          return [r + 2, c + 1, coerce(grid[r][c])] as [number, number, TabularCell];
        });
        if (cells.length) await editWorkbook(path, activeSheet, cells, node);
      } else {
        const rows: TabularCell[][] = grid.map((row) => row.map(coerce));
        await writeTabular(path, columns.map((c) => c.name), rows, node);
      }
      setEdited(new Set());
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "save failed");
    } finally {
      setSaving(false);
    }
  };

  const dirty = edited.size > 0;
  const meta: [string, string][] = isWorkbook
    ? [
        ["kind", "workbook"],
        ["sheets", String(sheets.length)],
        ["active rows", String(sheets.find((s) => s.name === activeSheet)?.rows ?? "--")],
        ["columns", String(columns.length)],
        ["node", nodeLabel ?? node ?? "local"],
        ["mode", readOnly ? "read-only" : "editable"],
      ]
    : info
      ? [
          ["format", info.media_type.split(/[/.]/).pop() ?? "--"],
          ["columns", String(info.column_count)],
          ["rows", info.row_count != null ? String(info.row_count) : "large"],
          ["size", formatSize(info.size_bytes)],
          ["schema", info.schema_hash || "--"],
          ["node", nodeLabel ?? node ?? "local"],
        ]
      : [];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative glass-card p-5 w-full max-w-5xl max-h-[88vh] z-10 flex flex-col gap-3">
        {/* Header */}
        <div className="flex items-start justify-between shrink-0">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="1.8">
                <rect x="3" y="3" width="18" height="18" rx="2" /><line x1="3" y1="9" x2="21" y2="9" /><line x1="9" y1="3" x2="9" y2="21" />
              </svg>
              <h3 className="text-sm font-mono font-semibold text-foreground truncate">{name}</h3>
              <span className={`text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded ${readOnly ? "bg-white/[0.06] text-muted" : "bg-emerald/15 text-emerald"}`}>
                {readOnly ? "read-only" : "editable"}
              </span>
              {isWorkbook && <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-frost/15 text-frost">arrow</span>}
            </div>
            <p className="text-[11px] text-muted font-mono truncate mt-1" title={info?.source_url}>
              <span className="text-frost/70">{nodeLabel ?? node ?? "local"}</span>{" : "}{info?.source_url ?? path}
            </p>
          </div>
          <button onClick={onClose} className="text-muted hover:text-foreground shrink-0 ml-4">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Metadata strip */}
        {meta.length > 0 && (
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2 shrink-0 text-[10px] font-mono">
            {meta.map(([k, v]) => (
              <div key={k} className="rounded bg-white/[0.03] border border-white/[0.06] px-2 py-1.5 min-w-0">
                <div className="text-muted/60 uppercase tracking-wider text-[9px]">{k}</div>
                <div className="text-foreground/80 truncate">{v}</div>
              </div>
            ))}
          </div>
        )}

        {/* Sheet tabs (workbook) */}
        {isWorkbook && sheets.length > 0 && (
          <div className="flex items-center gap-1 shrink-0 overflow-x-auto">
            {sheets.map((s) => (
              <button
                key={s.name}
                onClick={() => switchSheet(s.name)}
                className={`px-3 py-1 rounded-t text-[11px] font-mono whitespace-nowrap border-b-2 ${
                  s.name === activeSheet
                    ? "text-emerald border-emerald bg-emerald/[0.06]"
                    : "text-muted border-transparent hover:text-foreground"
                }`}
              >
                {s.name} <span className="text-muted/50">{s.rows}×{s.cols}</span>
              </button>
            ))}
          </div>
        )}

        {readOnly && !loading && !error && (
          <div className="shrink-0 rounded bg-amber/[0.06] border border-amber/15 px-3 py-1.5 text-[10px] font-mono text-amber/90">
            Read-only — over the {EDIT_CAP}-row editable cap. Showing the first {grid.length} rows; download for the full file.
          </div>
        )}

        {/* Grid */}
        <div className="flex-1 min-h-0 overflow-auto rounded-lg border border-white/[0.06] bg-black/30">
          {loading ? (
            <div className="flex items-center justify-center py-16">
              <div className="w-6 h-6 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
            </div>
          ) : error ? (
            <div className="p-6 text-center text-xs text-rose/90 font-mono">{error}</div>
          ) : grid.length === 0 ? (
            <div className="p-6 text-center text-xs text-muted font-mono">empty table</div>
          ) : (
            <table className="w-full text-[11px] font-mono border-collapse">
              <thead className="sticky top-0 bg-[#0d1117] z-10">
                <tr>
                  <th className="px-2 py-1.5 text-right text-muted/40 border-b border-white/[0.06] w-10">#</th>
                  {columns.map((col, ci) => (
                    <th key={ci} className="px-2 py-1.5 text-left text-frost/80 border-b border-white/[0.06] whitespace-nowrap">
                      {col.name}<span className="ml-1.5 text-muted/50 font-normal">{col.type}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid.map((row, ri) => (
                  <tr key={ri} className="hover:bg-white/[0.02]">
                    <td className="px-2 py-1 text-right text-muted/40 border-b border-white/[0.03] select-none">{readOnly ? page * PAGE + ri : ri}</td>
                    {row.map((cell, ci) => (
                      <td key={ci} className="border-b border-white/[0.03] p-0">
                        {readOnly ? (
                          <span className="block px-2 py-1 text-foreground/75 truncate max-w-[280px]">{cell}</span>
                        ) : (
                          <input
                            value={cell}
                            onChange={(e) => setCell(ri, ci, e.target.value)}
                            spellCheck={false}
                            className={`w-full bg-transparent px-2 py-1 text-foreground/85 outline-none focus:bg-frost/10 ${edited.has(`${ri},${ci}`) ? "bg-emerald/[0.07]" : ""}`}
                          />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center gap-3 pt-1 shrink-0">
          {!readOnly && (
            <button
              onClick={save}
              disabled={!dirty || saving}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40"
            >
              {saving ? "Saving…" : dirty ? `Save ${edited.size} cell${edited.size === 1 ? "" : "s"}` : saved ? "Saved" : "Save"}
            </button>
          )}
          {dirty && <span className="text-[10px] text-amber/80 font-mono">unsaved edits</span>}
          {readOnly && total > PAGE && (
            <div className="flex items-center gap-2 text-[10px] font-mono text-muted">
              <button
                onClick={() => goToPage(page - 1)}
                disabled={page === 0 || loading}
                className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground disabled:opacity-30"
              >‹ prev</button>
              <span className="text-foreground/70">
                rows {page * PAGE + 1}–{Math.min((page + 1) * PAGE, total)} of {total}
              </span>
              <button
                onClick={() => goToPage(page + 1)}
                disabled={(page + 1) * PAGE >= total || loading}
                className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground disabled:opacity-30"
              >next ›</button>
            </div>
          )}
          <a href={fsDownloadUrl(path, node)} className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20">
            Download
          </a>
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground ml-auto">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
