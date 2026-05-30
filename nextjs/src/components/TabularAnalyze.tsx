"use client";

import { useMemo, useState } from "react";
import {
  aggregate,
  analysisSeries,
  analysisOhlc,
  downloadExport,
  type AggFunc,
  type AggregateResult,
  type SeriesResult,
  type OhlcResult,
  type FilterSpec,
  type CastSpec,
} from "@/lib/api";
import Chart, { type ChartType } from "@/components/Chart";

const AGGS: AggFunc[] = ["sum", "mean", "min", "max", "count", "median", "std", "var"];
const EXPORT_FORMATS = ["csv", "parquet", "json", "ndjson", "arrow", "xlsx"];
const NUMERIC = /int|float|double|decimal|num/i;

export interface AnalyzeColumn { name: string; type?: string }

interface Props {
  /** Node-home-relative path to a tabular file (a materialised SQL result or a
   *  browsed file) — the analytics + download all run against this. */
  path: string;
  node?: string;
  columns: AnalyzeColumn[];
}

/**
 * Reusable analytics + download surface over a tabular *path*: pivot (aggregate
 * → table + chart), adaptive series with zoom, OHLC candles with MA + volume,
 * a filters+casts transform, and download-as in any media type. Shared by the
 * Saga result viewer and the Files browser so both analyse the same way.
 */
export default function TabularAnalyze({ path, node, columns }: Props) {
  const allCols = useMemo(() => columns.map((c) => c.name), [columns]);
  const numericCols = useMemo(() => columns.filter((c) => NUMERIC.test(c.type ?? "")).map((c) => c.name), [columns]);

  const [kind, setKind] = useState<"pivot" | "series" | "candles">("pivot");
  const [groupBy, setGroupBy] = useState("");
  const [measureCol, setMeasureCol] = useState(numericCols[0] ?? allCols[0] ?? "");
  const [aggFn, setAggFn] = useState<AggFunc>("sum");
  const [chartType, setChartType] = useState<ChartType>("bar");
  const [pivot, setPivot] = useState<AggregateResult | null>(null);
  const [seriesCol, setSeriesCol] = useState(numericCols[0] ?? allCols[0] ?? "");
  const [xCol, setXCol] = useState("");
  const [volCol, setVolCol] = useState("");
  const [seriesType, setSeriesType] = useState<ChartType>("area");
  const [seriesData, setSeriesData] = useState<SeriesResult | null>(null);
  const [zoom, setZoom] = useState<{ min: number; max: number } | null>(null);
  const [candles, setCandles] = useState<OhlcResult | null>(null);
  const [maWindow, setMaWindow] = useState(20);
  const [analyzing, setAnalyzing] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [transformOpen, setTransformOpen] = useState(false);
  const [filters, setFilters] = useState<FilterSpec[]>([]);
  const [casts, setCasts] = useState<CastSpec[]>([]);
  const [dlOpen, setDlOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  const runPivot = async () => {
    if (!measureCol) return;
    setAnalyzing(true); setErr(null);
    try { setPivot(await aggregate(path, groupBy ? [groupBy] : [], [{ column: measureCol, agg: aggFn }], node, 500, filters)); }
    catch (e) { setErr(e instanceof Error ? e.message : "aggregate failed"); }
    finally { setAnalyzing(false); }
  };
  const runSeries = async (z: { min: number; max: number } | null = zoom) => {
    if (!seriesCol) return;
    setAnalyzing(true); setErr(null);
    try { setSeriesData(await analysisSeries(path, seriesCol, { x: xCol || undefined, points: 800, x_min: z?.min, x_max: z?.max, filters, node })); }
    catch (e) { setErr(e instanceof Error ? e.message : "series failed"); }
    finally { setAnalyzing(false); }
  };
  const runCandles = async () => {
    if (!seriesCol) return;
    setAnalyzing(true); setErr(null);
    try { setCandles(await analysisOhlc(path, seriesCol, { x: xCol || undefined, volume: volCol || undefined, buckets: 120, filters, node })); }
    catch (e) { setErr(e instanceof Error ? e.message : "ohlc failed"); }
    finally { setAnalyzing(false); }
  };
  const exportAs = async (fmt: string) => {
    setDlOpen(false); setExporting(true); setErr(null);
    try { await downloadExport(path, fmt, { filters, casts }, node); }
    catch (e) { setErr(e instanceof Error ? e.message : "export failed"); }
    finally { setExporting(false); }
  };

  const xExtent = (): [number, number] | null => {
    if (!seriesData || !seriesData.x.length) return null;
    const xs = seriesData.x.map(Number);
    return [xs[0], xs[xs.length - 1]];
  };
  const zoomIn = () => { const e = xExtent(); if (!e) return; const mid = (e[0] + e[1]) / 2, half = (e[1] - e[0]) / 4; const z = { min: mid - half, max: mid + half }; setZoom(z); runSeries(z); };
  const zoomOut = () => { const e = xExtent(); if (!e) return; const span = e[1] - e[0] || 1; const z = { min: e[0] - span / 2, max: e[1] + span / 2 }; setZoom(z); runSeries(z); };
  const resetZoom = () => { setZoom(null); runSeries(null); };

  const maLine = candles ? candles.close.map((_, i) => {
    if (i < maWindow - 1) return null;
    let s = 0, c = 0;
    for (let k = i - maWindow + 1; k <= i; k++) { const v = candles.close[k]; if (v != null) { s += v; c++; } }
    return c ? s / c : null;
  }) : [];

  const sel = "bg-white/[0.04] border border-white/10 rounded px-1.5 py-1 outline-none";

  return (
    <div className="flex-1 min-h-0 overflow-auto rounded-lg border border-white/[0.06] bg-black/30 p-3 space-y-3">
      <div className="flex items-center gap-2 text-[11px] font-mono flex-wrap">
        {(["pivot", "series", "candles"] as const).map((k) => (
          <button key={k} onClick={() => setKind(k)}
            className={`px-2.5 py-1 rounded ${kind === k ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>{k}</button>
        ))}
        <span className="w-px h-4 bg-white/10 mx-1" />
        {kind === "pivot" && (
          <>
            <label className="text-muted">group</label>
            <select value={groupBy} onChange={(e) => setGroupBy(e.target.value)} className={sel}>
              <option value="">(none)</option>
              {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <label className="text-muted">of</label>
            <select value={measureCol} onChange={(e) => setMeasureCol(e.target.value)} className={sel}>
              {(numericCols.length ? numericCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <select value={aggFn} onChange={(e) => setAggFn(e.target.value as AggFunc)} className={sel}>
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
        {(kind === "series" || kind === "candles") && (
          <>
            <label className="text-muted">{kind === "candles" ? "price" : "series"}</label>
            <select value={seriesCol} onChange={(e) => setSeriesCol(e.target.value)} className={sel}>
              {(numericCols.length ? numericCols : allCols).map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <label className="text-muted">x</label>
            <select value={xCol} onChange={(e) => setXCol(e.target.value)} className={sel}>
              <option value="">(index)</option>
              {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            {kind === "candles" && (
              <>
                <label className="text-muted">vol</label>
                <select value={volCol} onChange={(e) => setVolCol(e.target.value)} className={sel}>
                  <option value="">(none)</option>
                  {numericCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </>
            )}
            <button onClick={kind === "series" ? () => { setZoom(null); runSeries(null); } : runCandles} disabled={analyzing || !seriesCol} className="px-2.5 py-1 rounded bg-emerald/15 text-emerald border border-emerald/30 disabled:opacity-40">run</button>
            {kind === "series" && (
              <span className="ml-auto flex items-center gap-1">
                {(["area", "line"] as ChartType[]).map((t) => (
                  <button key={t} onClick={() => setSeriesType(t)} className={`px-2 py-1 rounded ${seriesType === t ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground"}`}>{t}</button>
                ))}
              </span>
            )}
          </>
        )}
        {/* Download-as (applies filters + casts) */}
        <div className="relative ml-1">
          <button onClick={() => setDlOpen((v) => !v)} disabled={exporting}
            className="px-2.5 py-1 rounded bg-white/[0.04] border border-white/10 text-foreground-dim hover:bg-white/[0.08] disabled:opacity-40">
            {exporting ? "…" : "Download ▾"}
          </button>
          {dlOpen && (
            <div className="absolute right-0 mt-1 z-20 rounded-lg border border-white/[0.1] bg-[#0a0a1a] shadow-xl py-1 min-w-[110px]">
              {EXPORT_FORMATS.map((f) => (
                <button key={f} onClick={() => exportAs(f)} className="block w-full text-left px-3 py-1.5 text-xs font-mono text-foreground-dim hover:bg-white/[0.06] hover:text-frost">{f}</button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Transform: filters (rows) + casts (download) */}
      <div className="text-[11px] font-mono">
        <button onClick={() => setTransformOpen((v) => !v)} className="text-muted hover:text-foreground">
          {transformOpen ? "▾" : "▸"} filters &amp; casts {filters.length + casts.length > 0 ? `(${filters.length + casts.length})` : ""}
        </button>
        {transformOpen && (
          <div className="mt-2 space-y-2 rounded border border-white/[0.06] bg-white/[0.02] p-2">
            <div className="text-[10px] text-muted/60 uppercase tracking-wider">filters (rows)</div>
            {filters.map((f, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <select value={f.column} onChange={(e) => setFilters((fs) => fs.map((x, j) => j === i ? { ...x, column: e.target.value } : x))} className={sel}>
                  {allCols.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
                <select value={f.op} onChange={(e) => setFilters((fs) => fs.map((x, j) => j === i ? { ...x, op: e.target.value } : x))} className={sel}>
                  {["==", "!=", ">", ">=", "<", "<=", "contains", "in", "is_null", "not_null"].map((o) => <option key={o} value={o}>{o}</option>)}
                </select>
                {!["is_null", "not_null"].includes(f.op) && (
                  <input value={String(f.value ?? "")} onChange={(e) => { const sv = e.target.value; const v = sv === "" ? null : (sv.trim() !== "" && !isNaN(Number(sv)) ? Number(sv) : sv); setFilters((fs) => fs.map((x, j) => j === i ? { ...x, value: v } : x)); }} placeholder="value" className={`${sel} w-28`} />
                )}
                <button onClick={() => setFilters((fs) => fs.filter((_, j) => j !== i))} className="text-rose/70 hover:text-rose px-1">✕</button>
              </div>
            ))}
            <button onClick={() => setFilters((fs) => [...fs, { column: allCols[0] ?? "", op: ">", value: 0 }])} className="text-frost/70 hover:text-frost">+ filter</button>

            <div className="text-[10px] text-muted/60 uppercase tracking-wider pt-1">casts (download)</div>
            {casts.map((c, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <select value={c.column} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, column: e.target.value } : x))} className={sel}>
                  {allCols.map((col) => <option key={col} value={col}>{col}</option>)}
                </select>
                <span className="text-muted">→</span>
                <select value={c.dtype} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, dtype: e.target.value } : x))} className={sel}>
                  {["int", "double", "float", "bool", "string", "date", "datetime"].map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
                {c.dtype === "datetime" && (
                  <input value={c.tz ?? "UTC"} onChange={(e) => setCasts((cs) => cs.map((x, j) => j === i ? { ...x, tz: e.target.value } : x))} placeholder="UTC" className={`${sel} w-32`} title="target timezone (UTC by default)" />
                )}
                <button onClick={() => setCasts((cs) => cs.filter((_, j) => j !== i))} className="text-rose/70 hover:text-rose px-1">✕</button>
              </div>
            ))}
            <button onClick={() => setCasts((cs) => [...cs, { column: allCols[0] ?? "", dtype: "double" }])} className="text-frost/70 hover:text-frost">+ cast</button>
          </div>
        )}
      </div>

      {err && <div className="text-[11px] text-rose/90 font-mono">{err}</div>}
      {analyzing && <div className="text-[11px] text-muted font-mono">computing…</div>}

      {kind === "pivot" && pivot && !analyzing && (
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

      {kind === "series" && seriesData && !analyzing && (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-[10px] font-mono text-muted">
            <button onClick={zoomIn} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">zoom in</button>
            <button onClick={zoomOut} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">zoom out</button>
            <button onClick={resetZoom} className="px-2 py-1 rounded border border-white/[0.08] hover:text-foreground">reset</button>
            <span>{seriesData.sampled ? `${seriesData.x.length} buckets of ${seriesData.source_rows.toLocaleString()} rows (downsampled)` : `${seriesData.x.length} points (raw)`}{zoom ? ` · zoom [${zoom.min.toFixed(0)}, ${zoom.max.toFixed(0)}]` : ""}</span>
          </div>
          <Chart type={seriesType} labels={seriesData.x} values={seriesData.y}
            band={seriesData.sampled ? { min: seriesData.y_min, max: seriesData.y_max } : undefined}
            color="var(--emerald)" yLabel={seriesData.column} height={300} />
        </div>
      )}

      {kind === "candles" && candles && !analyzing && (
        <div className="space-y-1">
          <div className="flex items-center gap-3 text-[10px] font-mono text-muted">
            <span>{candles.bars} OHLC bars from {candles.source_rows.toLocaleString()} rows</span>
            <label className="flex items-center gap-1 ml-auto">
              <span className="text-amber/80">MA</span>
              <input type="number" min={1} max={candles.bars} value={maWindow} onChange={(e) => setMaWindow(Math.max(1, Number(e.target.value) || 1))} className="w-12 bg-white/[0.04] border border-white/10 rounded px-1 py-0.5 outline-none" />
            </label>
          </div>
          <Chart type="candle" labels={candles.x} ohlc={{ open: candles.open, high: candles.high, low: candles.low, close: candles.close }} overlay={maLine} yLabel={candles.column} height={300} />
          {candles.volume && (
            <>
              <div className="text-[10px] text-frost/70 font-mono">volume</div>
              <Chart type="bar" labels={candles.x} values={candles.volume} color="var(--frost)" height={90} />
            </>
          )}
        </div>
      )}

      {((kind === "pivot" && !pivot) || (kind === "series" && !seriesData) || (kind === "candles" && !candles)) && !analyzing && !err && (
        <div className="text-[11px] text-muted font-mono py-8 text-center">
          Pick a {kind === "pivot" ? "group + measure" : kind === "candles" ? "price (+ x / volume)" : "series (+ x)"} and hit run.
        </div>
      )}
    </div>
  );
}
