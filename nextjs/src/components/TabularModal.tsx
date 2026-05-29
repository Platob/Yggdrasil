"use client";

// Standalone, reusable tabular viewer/editor. Given any node + path it loads
// the schema + a bounded row window from /api/v2/tabular and renders a metadata
// header plus a grid. Small files (under the node's preview cap → `editable`)
// can be edited in place and saved back; larger files are read-only previews.
//
// It's deliberately decoupled from the files page so runs, DAGs, and node
// artifacts can open the same editor — and so it can grow into the full
// Excel-like editor (formulas, sorting, column ops) without touching callers.

import { useCallback, useEffect, useState } from "react";
import {
  getTabularInspect,
  getTabularPreview,
  writeTabular,
  fsDownloadUrl,
  type TabularInspect,
  type TabularCell,
} from "@/lib/api";

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

interface Props {
  node?: string;
  nodeLabel?: string;
  path: string;
  name: string;
  onClose: () => void;
}

export default function TabularModal({ node, nodeLabel, path, name, onClose }: Props) {
  const [info, setInfo] = useState<TabularInspect | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [grid, setGrid] = useState<string[][]>([]);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDirty(false);
    setSaved(false);
    try {
      const meta = await getTabularInspect(path, node);
      setInfo(meta);
      if (meta.schema_error) {
        setError(meta.schema_error);
        return;
      }
      // Editable files fit under the cap — pull every row so a save rewrites
      // the whole file; otherwise just a bounded read-only window.
      const limit = meta.editable && meta.row_count ? meta.row_count : 200;
      const prev = await getTabularPreview(path, limit, node);
      setColumns(prev.columns.map((c) => c.name));
      setGrid(prev.rows.map((r) => r.map((v) => (v === null || v === undefined ? "" : String(v)))));
      setTruncated(prev.truncated);
    } catch (e) {
      setError(e instanceof Error ? e.message : "failed to read table");
    } finally {
      setLoading(false);
    }
  }, [path, node]);

  useEffect(() => { load(); }, [load]);

  const editable = !!info?.editable && !truncated;

  const setCell = (r: number, c: number, value: string) => {
    setGrid((g) => {
      const next = g.map((row) => row.slice());
      next[r][c] = value;
      return next;
    });
    setDirty(true);
    setSaved(false);
  };

  const save = async () => {
    setSaving(true);
    setError(null);
    try {
      // Empty cells become null so numeric columns re-cast cleanly.
      const rows: TabularCell[][] = grid.map((row) => row.map((v) => (v === "" ? null : v)));
      await writeTabular(path, columns, rows, node);
      setDirty(false);
      setSaved(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "save failed");
    } finally {
      setSaving(false);
    }
  };

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
              {editable
                ? <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald/15 text-emerald">editable</span>
                : <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-white/[0.06] text-muted">read-only</span>}
            </div>
            {info && (
              <p className="text-[11px] text-muted font-mono truncate mt-1" title={info.source_url}>
                <span className="text-frost/70">{nodeLabel ?? node ?? "local"}</span>{" : "}{info.source_url}
              </p>
            )}
          </div>
          <button onClick={onClose} className="text-muted hover:text-foreground shrink-0 ml-4">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Metadata strip — everything about the source at a glance */}
        {info && (
          <div className="grid grid-cols-2 md:grid-cols-6 gap-2 shrink-0 text-[10px] font-mono">
            {[
              ["format", info.media_type.split(/[/.]/).pop() ?? "--"],
              ["columns", String(info.column_count)],
              ["rows", info.row_count != null ? String(info.row_count) : "large"],
              ["size", formatSize(info.size_bytes)],
              ["schema", info.schema_hash || "--"],
              ["node", nodeLabel ?? node ?? "local"],
            ].map(([k, v]) => (
              <div key={k} className="rounded bg-white/[0.03] border border-white/[0.06] px-2 py-1.5 min-w-0">
                <div className="text-muted/60 uppercase tracking-wider text-[9px]">{k}</div>
                <div className="text-foreground/80 truncate">{v}</div>
              </div>
            ))}
          </div>
        )}

        {truncated && (
          <div className="shrink-0 rounded bg-amber/[0.06] border border-amber/15 px-3 py-1.5 text-[10px] font-mono text-amber/90">
            Read-only — file exceeds the editable row cap. Showing the first {grid.length} rows. Download for the full file.
          </div>
        )}

        {/* Body */}
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
                  {columns.map((c, ci) => (
                    <th key={ci} className="px-2 py-1.5 text-left text-frost/80 border-b border-white/[0.06] whitespace-nowrap">
                      {c}
                      <span className="ml-1.5 text-muted/50 font-normal">{info?.columns[ci]?.type}</span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grid.map((row, ri) => (
                  <tr key={ri} className="hover:bg-white/[0.02]">
                    <td className="px-2 py-1 text-right text-muted/40 border-b border-white/[0.03] select-none">{ri}</td>
                    {row.map((cell, ci) => (
                      <td key={ci} className="border-b border-white/[0.03] p-0">
                        {editable ? (
                          <input
                            value={cell}
                            onChange={(e) => setCell(ri, ci, e.target.value)}
                            spellCheck={false}
                            className="w-full bg-transparent px-2 py-1 text-foreground/85 outline-none focus:bg-frost/10"
                          />
                        ) : (
                          <span className="block px-2 py-1 text-foreground/75 truncate max-w-[280px]">{cell}</span>
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Footer actions */}
        <div className="flex items-center gap-3 pt-1 shrink-0">
          {editable && (
            <button
              onClick={save}
              disabled={!dirty || saving}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-emerald/15 text-emerald border border-emerald/30 hover:bg-emerald/25 disabled:opacity-40"
            >
              {saving ? "Saving…" : dirty ? "Save changes" : saved ? "Saved" : "Save"}
            </button>
          )}
          {dirty && <span className="text-[10px] text-amber/80 font-mono">unsaved edits</span>}
          <a
            href={fsDownloadUrl(path, node)}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20"
          >
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
