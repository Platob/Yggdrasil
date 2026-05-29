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

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getTabularInspect,
  getFsStat,
  tabularPreviewArrowUrl,
  writeTabular,
  isWorkbookName,
  getWorkbookSheets,
  workbookReadArrowUrl,
  editWorkbook,
  fsDownloadUrl,
  aggregate,
  analysisSeries,
  analysisOhlc,
  downloadExport,
  type TabularInspect,
  type TabularCell,
  type WorkbookSheet,
  type AggFunc,
  type AggregateResult,
  type SeriesResult,
  type OhlcResult,
  type FilterSpec,
  type CastSpec,
} from "@/lib/api";
import { fetchArrowTable, clearArrowCache } from "@/lib/arrow";
import Chart, { type ChartType } from "@/components/Chart";

const AGGS: AggFunc[] = ["sum", "mean", "min", "max", "count", "median", "std", "var"];
const isNumericType = (t: string) => /int|float|double|decimal/i.test(t);

// Client-side column profile over the loaded grid rows — distinct values +
// counts and, for numeric columns, basic stats. Cheap (grid is bounded) and
// keeps the facet/filter pickers off the server.
function columnProfile(rows: string[][], ci: number) {
  const counts = new Map<string, number>();
  const nums: number[] = [];
  let nulls = 0;
  for (const r of rows) {
    const v = r[ci] ?? "";
    if (v === "") { nulls++; continue; }
    counts.set(v, (counts.get(v) ?? 0) + 1);
    const n = Number(v);
    if (!Number.isNaN(n)) nums.push(n);
  }
  const distinct = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const numeric = nums.length === rows.length - nulls && nums.length > 0
    ? { min: Math.min(...nums), max: Math.max(...nums), mean: nums.reduce((s, x) => s + x, 0) / nums.length }
    : null;
  return { distinct, numeric, nulls, total: rows.length };
}

// Client-side predicate eval — mirrors the backend filter ops so the grid
// reflects filters instantly without a round-trip.
function matchRow(row: string[], colIdx: Map<string, number>, filters: FilterSpec[]): boolean {
  for (const f of filters) {
    const i = colIdx.get(f.column);
    if (i === undefined) continue;
    const cell = row[i] ?? "";
    const op = f.op;
    if (op === "is_null") { if (cell !== "") return false; continue; }
    if (op === "not_null") { if (cell === "") return false; continue; }
    if (op === "contains") { if (!cell.includes(String(f.value ?? ""))) return false; continue; }
    if (op === "in") { const vs = (Array.isArray(f.value) ? f.value : [f.value]).map(String); if (!vs.includes(cell)) return false; continue; }
    const a = Number(cell), b = Number(f.value);
    const numeric = !Number.isNaN(a) && !Number.isNaN(b);
    const cmp = numeric
      ? { "==": a === b, "!=": a !== b, ">": a > b, ">=": a >= b, "<": a < b, "<=": a <= b }
      : { "==": cell === String(f.value), "!=": cell !== String(f.value), ">": cell > String(f.value), ">=": cell >= String(f.value), "<": cell < String(f.value), "<=": cell <= String(f.value) };
    if (op in cmp && !cmp[op as keyof typeof cmp]) return false;
  }
  return true;
}

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

  // Analyze mode (pivot + adaptive series + candlesticks)
  const [mode, setMode] = useState<"grid" | "analyze">("grid");
  const [analyzeKind, setAnalyzeKind] = useState<"pivot" | "series" | "candles">("pivot");
  const [groupBy, setGroupBy] = useState("");
  const [measureCol, setMeasureCol] = useState("");
  const [aggFn, setAggFn] = useState<AggFunc>("sum");
  const [chartType, setChartType] = useState<ChartType>("bar");
  const [pivot, setPivot] = useState<AggregateResult | null>(null);
  const [seriesCol, setSeriesCol] = useState("");   // value/price column
  const [xCol, setXCol] = useState("");              // optional x/order column
  const [volCol, setVolCol] = useState("");          // optional volume (candles)
  const [seriesType, setSeriesType] = useState<ChartType>("area");
  const [seriesData, setSeriesData] = useState<SeriesResult | null>(null);
  const [zoom, setZoom] = useState<{ min: number; max: number } | null>(null);
  const [candles, setCandles] = useState<OhlcResult | null>(null);
  const [maWindow, setMaWindow] = useState(20);          // client-side MA overlay
  const [analyzing, setAnalyzing] = useState(false);
  const [analyzeErr, setAnalyzeErr] = useState<string | null>(null);

  // Transform (filters + casts incl. tz) — collapsible so it doesn't clutter.
  const [transformOpen, setTransformOpen] = useState(false);
  const [filters, setFilters] = useState<FilterSpec[]>([]);
  const [casts, setCasts] = useState<CastSpec[]>([]);
  const [dlOpen, setDlOpen] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [infoOpen, setInfoOpen] = useState(false);
  const [stat, setStat] = useState<{ modified_at?: string; size?: number } | null>(null);
  const [facetCol, setFacetCol] = useState<number | null>(null);   // open column profile
  const [sort, setSort] = useState<{ col: number; dir: "asc" | "desc" } | null>(null);
  const [hidden, setHidden] = useState<Set<number>>(new Set());    // hidden columns

  const decodeInto = useCallback(async (url: string) => {
    const t = await fetchArrowTable(url);
    setColumns(t.columns);
    setGrid(t.rows.map((r) => r.map((v) => (v === null || v === undefined ? "" : String(v)))));
    // Schema-driven pre-fill so OHLC/series are ready without fiddling:
    //   x        -> a timestamp/date column, else the first integer
    //   price    -> prefer float/double (decimal), then any numeric
    //   volume   -> a numeric column named like "vol*"
    //   group/of -> first non-numeric dimension + first numeric measure
    const nums = t.columns.filter((c) => isNumericType(c.type)).map((c) => c.name);
    const dim = t.columns.find((c) => !isNumericType(c.type))?.name ?? "";
    const temporal = t.columns.find((c) => /date|time/i.test(c.type))?.name;
    const intCol = t.columns.find((c) => /\bint/i.test(c.type))?.name;
    const floatCol = t.columns.find((c) => /float|double|decimal/i.test(c.type))?.name;
    const volCol = t.columns.find((c) => /vol/i.test(c.name) && isNumericType(c.type))?.name;
    if (nums.length) { setMeasureCol((m) => m || nums[0]); setSeriesCol((m) => m || floatCol || nums[0]); }
    setGroupBy((g) => g || dim);
    setXCol((x) => x || temporal || intCol || "");
    setVolCol((v) => v || volCol || "");
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

  // Lazily fetch stat (mtime/size) the first time the info popover opens.
  useEffect(() => {
    if (infoOpen && !stat) {
      getFsStat(path, node)
        .then((s) => setStat({ modified_at: s.modified_at, size: s.size }))
        .catch(() => setStat({}));
    }
  }, [infoOpen, stat, path, node]);

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
      clearArrowCache();   // file changed — drop stale decoded windows
    } catch (e) {
      setError(e instanceof Error ? e.message : "save failed");
    } finally {
      setSaving(false);
    }
  };

  const numericCols = columns.filter((c) => isNumericType(c.type)).map((c) => c.name);
  const allCols = columns.map((c) => c.name);
  const colIdx = useMemo(() => new Map(columns.map((c, i) => [c.name, i])), [columns]);

  // Filters + sort applied client-side to the loaded rows — instant, no
  // round-trip. (Backend filters still drive the analyze charts + export.)
  const viewOrder = useMemo(() => {
    let idx = grid.map((_, i) => i);
    if (filters.length) idx = idx.filter((i) => matchRow(grid[i], colIdx, filters));
    if (sort) {
      const ci = sort.col, numeric = isNumericType(columns[ci]?.type ?? "");
      idx = idx.slice().sort((a, b) => {
        const va = grid[a][ci] ?? "", vb = grid[b][ci] ?? "";
        const c = numeric ? (Number(va) || 0) - (Number(vb) || 0) : va.localeCompare(vb);
        return sort.dir === "asc" ? c : -c;
      });
    }
    return idx;
  }, [grid, filters, sort, colIdx, columns]);
  const facet = useMemo(() => (facetCol != null ? columnProfile(grid, facetCol) : null), [facetCol, grid]);

  const runPivot = async () => {
    if (!measureCol) return;
    setAnalyzing(true); setAnalyzeErr(null);
    try {
      const res = await aggregate(path, groupBy ? [groupBy] : [], [{ column: measureCol, agg: aggFn }], node, 500, filters);
      setPivot(res);
    } catch (e) {
      setAnalyzeErr(e instanceof Error ? e.message : "aggregate failed");
    } finally {
      setAnalyzing(false);
    }
  };

  // Adaptive downsample. Re-fetches ~800 buckets over the current zoom window;
  // zooming in narrows the range so the backend reads fewer source rows and
  // returns finer detail (sampled=false once the range fits under the cap).
  const runSeries = async (z: { min: number; max: number } | null = zoom) => {
    if (!seriesCol) return;
    setAnalyzing(true); setAnalyzeErr(null);
    try {
      const res = await analysisSeries(path, seriesCol, {
        x: xCol || undefined, points: 800, x_min: z?.min, x_max: z?.max, filters, node,
      });
      setSeriesData(res);
    } catch (e) {
      setAnalyzeErr(e instanceof Error ? e.message : "series failed");
    } finally {
      setAnalyzing(false);
    }
  };

  const runCandles = async () => {
    if (!seriesCol) return;
    setAnalyzing(true); setAnalyzeErr(null);
    try {
      const res = await analysisOhlc(path, seriesCol, {
        x: xCol || undefined, volume: volCol || undefined, buckets: 120, filters, node,
      });
      setCandles(res);
    } catch (e) {
      setAnalyzeErr(e instanceof Error ? e.message : "ohlc failed");
    } finally {
      setAnalyzing(false);
    }
  };

  // Zoom adjusts the x window from the data's current extent and re-fetches.
  const xExtent = (): [number, number] | null => {
    if (!seriesData || !seriesData.x.length) return null;
    const xs = seriesData.x.map(Number);
    return [xs[0], xs[xs.length - 1]];
  };
  const zoomIn = () => { const e = xExtent(); if (!e) return; const mid = (e[0] + e[1]) / 2, half = (e[1] - e[0]) / 4; const z = { min: mid - half, max: mid + half }; setZoom(z); runSeries(z); };
  const zoomOut = () => { const e = xExtent(); if (!e) return; const span = e[1] - e[0] || 1; const z = { min: e[0] - span / 2, max: e[1] + span / 2 }; setZoom(z); runSeries(z); };
  const resetZoom = () => { setZoom(null); runSeries(null); };

  // Download-as: apply the current filters + casts and write any media type.
  const exportAs = async (fmt: string) => {
    setDlOpen(false); setExporting(true); setAnalyzeErr(null);
    try {
      await downloadExport(path, fmt, { filters, casts }, node);
    } catch (e) {
      setAnalyzeErr(e instanceof Error ? e.message : "export failed");
    } finally {
      setExporting(false);
    }
  };

  // Client-side moving average over the candle closes — computed on the client
  // so the node doesn't recompute it on every overlay toggle.
  const maLine = candles
    ? candles.close.map((_, i) => {
        if (i < maWindow - 1) return null;
        let s = 0, c = 0;
        for (let k = i - maWindow + 1; k <= i; k++) { const v = candles.close[k]; if (v != null) { s += v; c++; } }
        return c ? s / c : null;
      })
    : [];

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
      <div className="absolute inset-0 bg-[var(--modal-scrim)] backdrop-blur-sm" onClick={onClose} />
      <div className="relative modal-surface p-5 w-full max-w-5xl max-h-[88vh] z-10 flex flex-col gap-3">
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
              <button onClick={() => setInfoOpen((v) => !v)} title="Source info"
                className={`ml-0.5 ${infoOpen ? "text-frost" : "text-muted hover:text-foreground"}`}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><line x1="12" y1="16" x2="12" y2="12" /><line x1="12" y1="8" x2="12.01" y2="8" /></svg>
              </button>
            </div>
            <p className="text-[11px] text-muted font-mono truncate mt-1" title={info?.source_url}>
              <span className="text-frost/70">{nodeLabel ?? node ?? "local"}</span>{" : "}{info?.source_url ?? path}
            </p>
            {infoOpen && (
              <div className="absolute left-0 top-14 z-30 modal-surface p-3 w-[26rem] max-w-[80vw] text-[11px] font-mono space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-frost/80 uppercase tracking-wider text-[10px]">source</span>
                  <button onClick={() => setInfoOpen(false)} className="text-muted hover:text-foreground">✕</button>
                </div>
                {[
                  ["name", name],
                  ["node", nodeLabel ?? node ?? "local"],
                  ["url", info?.source_url ?? path],
                  ["media type", info?.media_type ?? "--"],
                  ["format", path.split(".").pop() ?? "--"],
                  ["size", stat?.size != null ? formatSize(stat.size) : (info ? formatSize(info.size_bytes) : "…")],
                  ["modified", stat?.modified_at ? new Date(stat.modified_at).toLocaleString() : "…"],
                  ["rows", isWorkbook ? String(total) : (info?.row_count != null ? String(info.row_count) : "large")],
                  ["columns", String(columns.length || info?.column_count || 0)],
                  ["schema hash", info?.schema_hash || "--"],
                  ["editable", String(!readOnly)],
                ].map(([k, v]) => (
                  <div key={k} className="flex gap-2">
                    <span className="text-muted/70 w-24 shrink-0">{k}</span>
                    <span className="text-foreground/85 break-all">{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="flex items-center gap-2 shrink-0 ml-4">
            <div className="flex rounded-lg overflow-hidden border border-white/[0.08] text-[10px] font-mono">
              {(["grid", "analyze"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`px-3 py-1 uppercase tracking-wider ${mode === m ? "bg-emerald/15 text-emerald" : "text-muted hover:text-foreground"}`}
                >{m}</button>
              ))}
            </div>
            <button onClick={onClose} className="text-muted hover:text-foreground">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
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

        {mode === "grid" && readOnly && !loading && !error && (
          <div className="shrink-0 rounded bg-amber/[0.06] border border-amber/15 px-3 py-1.5 text-[10px] font-mono text-amber/90">
            Read-only — over the {EDIT_CAP}-row editable cap. Showing the first {grid.length} rows; download for the full file.
          </div>
        )}

        {/* Analyze panel — pivot + adaptive series + candlesticks */}
        {mode === "analyze" && (
          <div className="flex-1 min-h-0 overflow-auto rounded-lg border border-white/[0.06] bg-black/30 p-3 space-y-3">
            <div className="flex items-center gap-2 text-[11px] font-mono flex-wrap">
              {(["pivot", "series", "candles"] as const).map((k) => (
                <button key={k} onClick={() => setAnalyzeKind(k)}
                  className={`px-2.5 py-1 rounded ${analyzeKind === k ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>{k}</button>
              ))}
              <span className="w-px h-4 bg-white/10 mx-1" />
              {analyzeKind === "pivot" && (
                <>
                  <label className="text-muted">group</label>
                  <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                    <option value="">(none)</option>
                    {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <label className="text-muted">of</label>
                  <select value={measureCol} onChange={(e) => setMeasureCol(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                    {(numericCols.length ? numericCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <select value={aggFn} onChange={(e) => setAggFn(e.target.value as AggFunc)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                    {AGGS.map((a) => <option key={a} value={a}>{a}</option>)}
                  </select>
                  <button onClick={runPivot} disabled={analyzing || !measureCol} className="px-2.5 py-1 rounded bg-emerald/15 text-emerald border border-emerald/30 disabled:opacity-40">run</button>
                  <span className="ml-auto flex items-center gap-1">
                    {(["bar", "line", "area"] as ChartType[]).map((t) => (
                      <button key={t} onClick={() => setChartType(t)} className={`px-2 py-1 rounded ${chartType === t ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>{t}</button>
                    ))}
                  </span>
                </>
              )}
              {(analyzeKind === "series" || analyzeKind === "candles") && (
                <>
                  <label className="text-muted">{analyzeKind === "candles" ? "price" : "series"}</label>
                  <select value={seriesCol} onChange={(e) => setSeriesCol(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                    {(numericCols.length ? numericCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <label className="text-muted">x</label>
                  <select value={xCol} onChange={(e) => setXCol(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                    <option value="">(index)</option>
                    {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  {analyzeKind === "candles" && (
                    <>
                      <label className="text-muted">vol</label>
                      <select value={volCol} onChange={(e) => setVolCol(e.target.value)} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none">
                        <option value="">(none)</option>
                        {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                    </>
                  )}
                  <button onClick={analyzeKind === "series" ? () => { setZoom(null); runSeries(null); } : runCandles} disabled={analyzing || !seriesCol} className="px-2.5 py-1 rounded bg-emerald/15 text-emerald border border-emerald/30 disabled:opacity-40">run</button>
                  {analyzeKind === "series" && (
                    <span className="ml-auto flex items-center gap-1">
                      {(["area", "line"] as ChartType[]).map((t) => (
                        <button key={t} onClick={() => setSeriesType(t)} className={`px-2 py-1 rounded ${seriesType === t ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>{t}</button>
                      ))}
                    </span>
                  )}
                </>
              )}
            </div>

            {/* Transform: filters + casts (collapsible — applies to charts + download) */}
            <div className="text-[11px] font-mono">
              <button onClick={() => setTransformOpen((v) => !v)} className="text-muted hover:text-foreground">
                {transformOpen ? "▾" : "▸"} filters &amp; casts {filters.length + casts.length > 0 ? `(${filters.length + casts.length})` : ""}
              </button>
              {transformOpen && (
                <div className="mt-2 space-y-2 rounded border border-white/[0.06] bg-white/[0.02] p-2">
                  <div className="text-[10px] text-muted/60 uppercase tracking-wider">filters (rows)</div>
                  {filters.map((f, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <select value={f.column} onChange={(e) => setFilters((fs) => fs.map((x, j) => j === i ? { ...x, column: e.target.value } : x))} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1">
                        {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                      </select>
                      <select value={f.op} onChange={(e) => setFilters((fs) => fs.map((x, j) => j === i ? { ...x, op: e.target.value } : x))} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1">
                        {["==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "not_null"].map((o) => <option key={o} value={o}>{o}</option>)}
                      </select>
                      {!["is_null", "not_null"].includes(f.op) && (
                        <>
                          <input list={`fv-${i}`} value={String(f.value ?? "")} onChange={(e) => { const s = e.target.value; const v = s === "" ? null : (s.trim() !== "" && !isNaN(Number(s)) ? Number(s) : s); setFilters((fs) => fs.map((x, j) => j === i ? { ...x, value: v } : x)); }} placeholder="value" className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 w-28 outline-none" />
                          <datalist id={`fv-${i}`}>
                            {colIdx.has(f.column) ? columnProfile(grid, colIdx.get(f.column)!).distinct.slice(0, 30).map(([v]) => <option key={v} value={v} />) : null}
                          </datalist>
                        </>
                      )}
                      <button onClick={() => setFilters((fs) => fs.filter((_, j) => j !== i))} className="text-rose/70 hover:text-rose px-1">✕</button>
                    </div>
                  ))}
                  <button onClick={() => setFilters((fs) => [...fs, { column: allCols[0] ?? "", op: ">", value: 0 }])} className="text-frost/70 hover:text-frost">+ filter</button>

                  <div className="text-[10px] text-muted/60 uppercase tracking-wider pt-1">casts (download)</div>
                  {casts.map((c, i) => (
                    <div key={i} className="flex items-center gap-1.5">
                      <select value={c.column} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, column: e.target.value } : x))} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1">
                        {allCols.map((col) => <option key={col} value={col}>{col}</option>)}
                      </select>
                      <span className="text-muted">→</span>
                      <select value={c.dtype} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, dtype: e.target.value } : x))} className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1">
                        {["int", "double", "float", "bool", "string", "date", "datetime"].map((d) => <option key={d} value={d}>{d}</option>)}
                      </select>
                      {c.dtype === "datetime" && (
                        <input value={c.tz ?? "UTC"} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, tz: e.target.value } : x))} placeholder="UTC" className="bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 w-32 outline-none" title="target timezone (UTC by default)" />
                      )}
                      <button onClick={() => setCasts((cs) => cs.filter((_, j) => j !== i))} className="text-rose/70 hover:text-rose px-1">✕</button>
                    </div>
                  ))}
                  <button onClick={() => setCasts((cs) => [...cs, { column: allCols[0] ?? "", dtype: "double" }])} className="text-frost/70 hover:text-frost">+ cast</button>
                </div>
              )}
            </div>

            {analyzeErr && <div className="text-[11px] text-rose/90 font-mono">{analyzeErr}</div>}
            {analyzing && <div className="text-[11px] text-muted font-mono">computing…</div>}

            {/* Pivot */}
            {analyzeKind === "pivot" && pivot && !analyzing && (
              <div className="space-y-3">
                <Chart type={chartType} labels={pivot.rows.map((r) => String(r[0]))}
                  values={pivot.rows.map((r) => Number(r[pivot.columns.length - 1]))}
                  color="var(--emerald)" yLabel={pivot.columns[pivot.columns.length - 1]} />
                <div className="text-[10px] text-muted font-mono">{pivot.group_count} groups · {pivot.source_rows.toLocaleString()} rows streamed</div>
                <table className="w-full text-[11px] font-mono border-collapse">
                  <thead><tr>{pivot.columns.map((c) => <th key={c} className="px-2 py-1 text-left text-frost/80 border-b border-white/[0.06]">{c}</th>)}</tr></thead>
                  <tbody>
                    {pivot.rows.slice(0, 50).map((r, i) => (
                      <tr key={i} className="hover:bg-white/[0.02]">
                        {r.map((v, j) => <td key={j} className="px-2 py-1 border-b border-white/[0.03] text-foreground/80">{v == null ? "" : typeof v === "number" ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(v)}</td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            {/* Adaptive series + zoom */}
            {analyzeKind === "series" && seriesData && !analyzing && (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-[10px] font-mono text-muted">
                  <button onClick={zoomIn} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">zoom in</button>
                  <button onClick={zoomOut} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">zoom out</button>
                  <button onClick={resetZoom} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">reset</button>
                  <span>
                    {seriesData.sampled
                      ? `${seriesData.x.length} buckets of ${seriesData.source_rows.toLocaleString()} rows (downsampled)`
                      : `${seriesData.x.length} points (raw)`}
                    {zoom ? ` · zoom [${zoom.min.toFixed(0)}, ${zoom.max.toFixed(0)}]` : ""}
                  </span>
                </div>
                <Chart type={seriesType} labels={seriesData.x} values={seriesData.y}
                  band={seriesData.sampled ? { min: seriesData.y_min, max: seriesData.y_max } : undefined}
                  color="var(--emerald)" yLabel={seriesData.column} height={300} />
              </div>
            )}

            {/* Candlesticks + client-side MA overlay + volume sub-panel */}
            {analyzeKind === "candles" && candles && !analyzing && (
              <div className="space-y-1">
                <div className="flex items-center gap-3 text-[10px] font-mono text-muted">
                  <span>{candles.bars} OHLC bars from {candles.source_rows.toLocaleString()} rows</span>
                  <label className="flex items-center gap-1 ml-auto">
                    <span className="text-amber/80">MA</span>
                    <input type="number" min={1} max={candles.bars} value={maWindow}
                      onChange={(e) => setMaWindow(Math.max(1, Number(e.target.value) || 1))}
                      className="w-12 bg-white/[0.04] border border-white/10 rounded px-1 py-0.5 outline-none" />
                  </label>
                </div>
                <Chart type="candle" labels={candles.x}
                  ohlc={{ open: candles.open, high: candles.high, low: candles.low, close: candles.close }}
                  overlay={maLine} yLabel={candles.column} height={300} />
                {candles.volume && (
                  <>
                    <div className="text-[10px] text-frost/70 font-mono">volume</div>
                    <Chart type="bar" labels={candles.x} values={candles.volume} color="var(--frost)" height={90} />
                  </>
                )}
              </div>
            )}

            {((analyzeKind === "pivot" && !pivot) || (analyzeKind === "series" && !seriesData) || (analyzeKind === "candles" && !candles)) && !analyzing && !analyzeErr && (
              <div className="text-[11px] text-muted font-mono py-8 text-center">
                Pick a {analyzeKind === "pivot" ? "group + measure" : analyzeKind === "candles" ? "price (+ x / volume)" : "series (+ x)"} and hit run.
              </div>
            )}
          </div>
        )}

        {/* Active view state — predicate chips, sort, hidden cols (grid mode) */}
        {mode === "grid" && (filters.length > 0 || hidden.size > 0 || sort) && (
          <div className="flex items-center gap-1.5 flex-wrap shrink-0 text-[10px] font-mono">
            {filters.length > 0 && <span className="text-muted/60">filters:</span>}
            {filters.map((f, i) => (
              <span key={i} className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-frost/10 text-frost border border-frost/20">
                {f.column} {f.op} {!["is_null", "not_null"].includes(f.op) ? String(f.value) : ""}
                <button onClick={() => setFilters((fs) => fs.filter((_, j) => j !== i))} className="text-frost/60 hover:text-rose">✕</button>
              </span>
            ))}
            {filters.length > 0 && <button onClick={() => setFilters([])} className="text-muted hover:text-foreground underline">clear</button>}
            {sort && <span className="px-1.5 py-0.5 rounded bg-emerald/10 text-emerald border border-emerald/20">sort {columns[sort.col]?.name} {sort.dir === "asc" ? "▲" : "▼"} <button onClick={() => setSort(null)} className="text-emerald/60 hover:text-rose">✕</button></span>}
            {hidden.size > 0 && <button onClick={() => setHidden(new Set())} className="px-1.5 py-0.5 rounded bg-amber/10 text-amber border border-amber/20">{hidden.size} hidden ⟲</button>}
            <span className="text-muted/50 ml-1">showing {viewOrder.length} of {grid.length}{readOnly ? " loaded" : ""}</span>
          </div>
        )}

        {/* Grid */}
        {mode === "grid" && (
        <div className="flex-1 min-h-0 overflow-auto rounded-lg border border-white/[0.06] bg-black/30" onClick={() => facetCol != null && setFacetCol(null)}>
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
                  {columns.map((col, ci) => hidden.has(ci) ? null : (
                    <th key={ci} className="px-2 py-1.5 text-left border-b border-white/[0.06] whitespace-nowrap relative">
                      <button
                        onClick={() => setFacetCol((c) => (c === ci ? null : ci))}
                        onContextMenu={(e) => { e.preventDefault(); setFacetCol(ci); }}
                        className="text-frost/80 hover:text-frost inline-flex items-center gap-1" title="Click or right-click for column tools">
                        {col.name}<span className="text-muted/50 font-normal">{col.type}</span>
                        {sort?.col === ci && <span className="text-emerald">{sort.dir === "asc" ? "▲" : "▼"}</span>}
                        <span className="text-muted/40">▾</span>
                      </button>
                      {facetCol === ci && facet && (
                        <div className="absolute left-0 top-full mt-1 z-30 modal-surface p-2.5 w-64 text-[11px] font-mono normal-case font-normal space-y-1.5" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-between">
                            <span className="text-frost/80 truncate">{col.name}</span>
                            <div className="flex gap-1.5 text-[10px]">
                              <button onClick={() => setSort({ col: ci, dir: "asc" })} className="text-muted hover:text-emerald">sort ▲</button>
                              <button onClick={() => setSort({ col: ci, dir: "desc" })} className="text-muted hover:text-emerald">sort ▼</button>
                              {sort?.col === ci && <button onClick={() => setSort(null)} className="text-muted hover:text-foreground">clr</button>}
                            </div>
                          </div>
                          {/* Column actions */}
                          <div className="flex flex-wrap gap-1 text-[10px]">
                            <button onClick={() => { setHidden((h) => new Set(h).add(ci)); setFacetCol(null); }} className="px-1.5 py-0.5 rounded bg-white/[0.05] text-muted hover:text-foreground">hide</button>
                            <button onClick={() => { navigator.clipboard?.writeText(grid.map((r) => r[ci] ?? "").join("\n")); setFacetCol(null); }} className="px-1.5 py-0.5 rounded bg-white/[0.05] text-muted hover:text-foreground">copy</button>
                            <button onClick={() => { setMode("analyze"); setAnalyzeKind("pivot"); setGroupBy(col.name); setFacetCol(null); }} className="px-1.5 py-0.5 rounded bg-frost/10 text-frost">pivot by</button>
                            {isNumericType(col.type) && <button onClick={() => { setMode("analyze"); setAnalyzeKind("series"); setSeriesCol(col.name); setFacetCol(null); }} className="px-1.5 py-0.5 rounded bg-emerald/10 text-emerald">plot</button>}
                            <button onClick={() => { setXCol(col.name); setFacetCol(null); }} className="px-1.5 py-0.5 rounded bg-white/[0.05] text-muted hover:text-foreground">as x</button>
                          </div>
                          <div className="text-[10px] text-muted/70">{facet.distinct.length} distinct · {facet.nulls} null{facet.numeric ? ` · min ${facet.numeric.min.toLocaleString(undefined, { maximumFractionDigits: 2 })} · max ${facet.numeric.max.toLocaleString(undefined, { maximumFractionDigits: 2 })} · μ ${facet.numeric.mean.toLocaleString(undefined, { maximumFractionDigits: 2 })}` : ""}</div>
                          <div className="max-h-40 overflow-auto space-y-0.5">
                            {facet.distinct.slice(0, 12).map(([val, cnt]) => (
                              <button key={val}
                                onClick={() => { setFilters((fs) => [...fs, { column: col.name, op: "==", value: Number.isNaN(Number(val)) || val === "" ? val : Number(val) }]); setFacetCol(null); }}
                                className="w-full flex items-center justify-between gap-2 px-1.5 py-0.5 rounded hover:bg-emerald/10 group">
                                <span className="truncate text-foreground/80 group-hover:text-emerald">{val || "∅"}</span>
                                <span className="text-muted/60 shrink-0">{cnt}</span>
                              </button>
                            ))}
                            {facet.distinct.length > 12 && <div className="text-[10px] text-muted/50 px-1.5">+{facet.distinct.length - 12} more</div>}
                          </div>
                          <div className="text-[9px] text-muted/40">click a value → filter == (loaded rows)</div>
                        </div>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {viewOrder.map((ri) => {
                  const row = grid[ri];
                  return (
                    <tr key={ri} className="hover:bg-white/[0.02]">
                      <td className="px-2 py-1 text-right text-muted/40 border-b border-white/[0.03] select-none">{readOnly ? page * PAGE + ri : ri}</td>
                      {row.map((cell, ci) => hidden.has(ci) ? null : (
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
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
        )}

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
          <div className="relative">
            <button onClick={() => setDlOpen((v) => !v)} disabled={exporting}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
              {exporting ? "exporting…" : `Download as ▾${filters.length + casts.length ? ` (${filters.length + casts.length})` : ""}`}
            </button>
            {dlOpen && (
              <div className="absolute bottom-full mb-1 left-0 z-20 glass-card p-1 min-w-36 text-[11px] font-mono">
                <a href={fsDownloadUrl(path, node)} onClick={() => setDlOpen(false)} className="block px-2 py-1.5 rounded hover:bg-white/[0.05] text-muted hover:text-foreground">raw file</a>
                <div className="h-px bg-white/10 my-1" />
                {["csv", "parquet", "json", "ndjson", "arrow", "xlsx"].map((fmt) => (
                  <button key={fmt} onClick={() => exportAs(fmt)} className="block w-full text-left px-2 py-1.5 rounded hover:bg-emerald/10 text-foreground/80 hover:text-emerald">
                    {fmt}{filters.length + casts.length ? " · transformed" : ""}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button onClick={onClose} className="px-4 py-2 rounded-lg text-xs font-medium text-muted hover:text-foreground ml-auto">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
