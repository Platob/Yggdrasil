"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  getWatchlist,
  addToWatchlist,
  removeFromWatchlist,
  getMarketQuote,
  getIndicators,
  aiGenerateInsight,
  aiAnalyzeFile,
  type MarketQuote,
  type IndicatorSpec,
  type IndicatorResult,
  type IndicatorSeries,
} from "@/lib/api";

const inputCls =
  "bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors";
const frostBtn =
  "bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-30 disabled:cursor-not-allowed";

function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-muted">
      <span className="w-6 h-6 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
      {label && <span className="text-sm font-mono">{label}</span>}
    </div>
  );
}

function isNotAvailable(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /HTTP 404|HTTP 501|Failed to fetch|NetworkError/i.test(msg);
}

function fmtPrice(v: number | null): string {
  if (v == null) return "—";
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ── SVG chart ────────────────────────────────────────────────────────────────
// Plots one or more series onto a fixed 800×160 viewBox. y-values are normalized
// across all series so multi-line indicators (MACD, BB) share a vertical scale.

interface ChartLine {
  values: (number | null)[];
  color: string;
  dash?: boolean;
  fillTo?: number; // index of another line in `lines` to fill the band toward
}

function LineChart({
  lines,
  x,
  histogram,
  refLines,
}: {
  lines: ChartLine[];
  x: (string | number)[];
  histogram?: { values: (number | null)[]; posColor: string; negColor: string };
  refLines?: { value: number; color: string; label: string }[];
}) {
  const W = 800;
  const H = 160;
  const TOP = 10;
  const BOT = 150;

  const all: number[] = [];
  for (const l of lines) for (const v of l.values) if (v != null && Number.isFinite(v)) all.push(v);
  if (histogram) for (const v of histogram.values) if (v != null && Number.isFinite(v)) all.push(v);
  if (refLines) for (const r of refLines) all.push(r.value);

  if (all.length === 0) {
    return <p className="text-xs text-muted/60 italic py-8 text-center">no data</p>;
  }

  let lo = Math.min(...all);
  let hi = Math.max(...all);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const norm = (v: number) => BOT - ((v - lo) / (hi - lo)) * (BOT - TOP);

  const n = Math.max(lines[0]?.values.length ?? 0, histogram?.values.length ?? 0);
  const xPos = (i: number) => (n <= 1 ? W / 2 : (i / (n - 1)) * W);

  const labelEvery = Math.max(1, Math.ceil(n / 8));
  const fmtX = (val: string | number): string => {
    if (typeof val === "number") return String(val);
    const d = new Date(val);
    if (!Number.isNaN(d.getTime())) {
      return `${d.getMonth() + 1}/${d.getDate()}`;
    }
    return String(val).slice(0, 10);
  };

  const polyline = (values: (number | null)[]): string =>
    values
      .map((v, i) => (v == null || !Number.isFinite(v) ? null : `${xPos(i)},${norm(v)}`))
      .filter((p): p is string => p !== null)
      .join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[160px]" preserveAspectRatio="none">
      {refLines?.map((r, i) => (
        <g key={`ref-${i}`}>
          <line
            x1={0}
            x2={W}
            y1={norm(r.value)}
            y2={norm(r.value)}
            stroke={r.color}
            strokeWidth={1}
            strokeDasharray="4 4"
            opacity={0.5}
          />
          <text x={4} y={norm(r.value) - 3} fill={r.color} fontSize={10} opacity={0.7}>
            {r.label}
          </text>
        </g>
      ))}

      {histogram &&
        histogram.values.map((v, i) => {
          if (v == null || !Number.isFinite(v)) return null;
          const zero = norm(0);
          const y = norm(v);
          const barW = Math.max(1, (W / Math.max(1, n)) * 0.6);
          return (
            <rect
              key={`h-${i}`}
              x={xPos(i) - barW / 2}
              y={Math.min(zero, y)}
              width={barW}
              height={Math.abs(zero - y)}
              fill={v >= 0 ? histogram.posColor : histogram.negColor}
              opacity={0.45}
            />
          );
        })}

      {lines.map((l, li) =>
        l.fillTo != null && lines[l.fillTo] ? (
          <polygon
            key={`fill-${li}`}
            points={`${polyline(l.values)} ${lines[l.fillTo].values
              .map((v, i) => (v == null || !Number.isFinite(v) ? null : `${xPos(i)},${norm(v)}`))
              .filter((p): p is string => p !== null)
              .reverse()
              .join(" ")}`}
            fill={l.color}
            opacity={0.06}
          />
        ) : null,
      )}

      {lines.map((l, li) => (
        <polyline
          key={`line-${li}`}
          points={polyline(l.values)}
          fill="none"
          stroke={l.color}
          strokeWidth={1.5}
          strokeDasharray={l.dash ? "4 3" : undefined}
          strokeLinejoin="round"
          strokeLinecap="round"
        >
          <title>{`${l.values.filter((v) => v != null).length} points`}</title>
        </polyline>
      ))}

      {x.map((val, i) =>
        i % labelEvery === 0 ? (
          <text
            key={`xl-${i}`}
            x={xPos(i)}
            y={H - 1}
            fill="var(--muted)"
            fontSize={9}
            textAnchor="middle"
            opacity={0.6}
          >
            {fmtX(val)}
          </text>
        ) : null,
      )}
    </svg>
  );
}

function IndicatorChart({
  series,
  x,
}: {
  series: IndicatorSeries;
  x: (string | number)[];
}) {
  const v = series.values;
  let chart: React.ReactNode = null;

  if (series.type === "rsi") {
    const key = Object.keys(v)[0];
    chart = (
      <LineChart
        x={x}
        lines={[{ values: v[key] ?? [], color: "var(--frost)" }]}
        refLines={[
          { value: 70, color: "var(--rose)", label: "70 overbought" },
          { value: 30, color: "var(--emerald)", label: "30 oversold" },
        ]}
      />
    );
  } else if (series.type === "ema") {
    const key = Object.keys(v)[0];
    chart = <LineChart x={x} lines={[{ values: v[key] ?? [], color: "var(--frost)" }]} />;
  } else if (series.type === "macd") {
    chart = (
      <LineChart
        x={x}
        lines={[
          { values: v.macd ?? [], color: "var(--frost)" },
          { values: v.signal ?? [], color: "var(--amber)" },
        ]}
        histogram={
          v.histogram
            ? { values: v.histogram, posColor: "var(--emerald)", negColor: "var(--rose)" }
            : undefined
        }
      />
    );
  } else if (series.type === "bb") {
    chart = (
      <LineChart
        x={x}
        lines={[
          { values: v.upper ?? [], color: "var(--rose)", fillTo: 2 },
          { values: v.middle ?? [], color: "var(--frost)" },
          { values: v.lower ?? [], color: "var(--rose)" },
        ]}
      />
    );
  } else {
    chart = (
      <LineChart
        x={x}
        lines={Object.entries(v).map(([, vals], i) => ({
          values: vals,
          color: ["var(--frost)", "var(--amber)", "var(--emerald)", "var(--rose)"][i % 4],
        }))}
      />
    );
  }

  const legend = Object.keys(v);
  return (
    <div className="runic-card p-4 space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-foreground">{series.name || series.type.toUpperCase()}</h3>
        <div className="flex items-center gap-3">
          {legend.map((k) => (
            <span key={k} className="text-[10px] text-muted font-mono uppercase tracking-wider">
              {k}
            </span>
          ))}
        </div>
      </div>
      {chart}
    </div>
  );
}

// ── Watchlist row ────────────────────────────────────────────────────────────

function WatchRow({ ticker, onRemove }: { ticker: string; onRemove: (t: string) => void }) {
  const [quote, setQuote] = useState<MarketQuote | null>(null);
  const [unavailable, setUnavailable] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const q = await getMarketQuote(ticker);
      setQuote(q);
      setUnavailable(!q.available);
    } catch (err) {
      if (isNotAvailable(err)) setUnavailable(true);
    }
  }, [ticker]);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30000);
    return () => clearInterval(id);
  }, [refresh]);

  const change = quote?.change_pct ?? null;
  const changeColor =
    change == null ? "var(--muted)" : change >= 0 ? "var(--emerald)" : "var(--rose)";

  return (
    <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
      <span className="font-mono font-semibold text-sm text-foreground w-20 truncate">{ticker}</span>
      {unavailable ? (
        <span className="text-xs text-muted/50 italic flex-1">market data unavailable</span>
      ) : (
        <>
          <span className="font-mono text-sm flex-1" style={{ color: "var(--frost)" }}>
            {fmtPrice(quote?.price ?? null)}
          </span>
          <span className="font-mono text-xs w-24 text-right" style={{ color: changeColor }}>
            {change == null ? "—" : `${change >= 0 ? "+" : ""}${change.toFixed(2)}%`}
          </span>
        </>
      )}
      <button
        onClick={() => onRemove(ticker)}
        title={`Remove ${ticker}`}
        className="text-muted hover:text-rose transition-colors shrink-0 w-6 text-center"
      >
        ×
      </button>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

const DEFAULT_PARAMS: Record<string, Record<string, number>> = {
  rsi: { period: 14 },
  macd: { fast: 12, slow: 26, signal: 9 },
  ema: { period: 20 },
  bb: { period: 20, num_std: 2 },
};

const INDICATOR_LABELS: Record<string, string> = {
  rsi: "RSI",
  macd: "MACD",
  ema: "EMA",
  bb: "Bollinger Bands",
};

export default function TradingPage() {
  // Section A — watchlist
  const [tickers, setTickers] = useState<string[]>([]);
  const [watchUnavailable, setWatchUnavailable] = useState(false);
  const [newTicker, setNewTicker] = useState("");
  const [watchErr, setWatchErr] = useState("");

  const loadWatchlist = useCallback(async () => {
    try {
      const w = await getWatchlist();
      setTickers(w.tickers);
      setWatchUnavailable(false);
    } catch (err) {
      if (isNotAvailable(err)) setWatchUnavailable(true);
    }
  }, []);

  useEffect(() => {
    loadWatchlist();
  }, [loadWatchlist]);

  const handleAdd = async () => {
    const t = newTicker.trim().toUpperCase();
    if (!t) return;
    setWatchErr("");
    try {
      const w = await addToWatchlist(t);
      setTickers(w.tickers);
      setNewTicker("");
    } catch (err) {
      setWatchErr(isNotAvailable(err) ? "Feature not yet available" : err instanceof Error ? err.message : "Add failed");
    }
  };

  const handleRemove = async (t: string) => {
    try {
      const w = await removeFromWatchlist(t);
      setTickers(w.tickers);
    } catch {
      setTickers((prev) => prev.filter((x) => x !== t));
    }
  };

  // Section B — indicators
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [xCol, setXCol] = useState("");
  const [selected, setSelected] = useState<Record<string, boolean>>({ rsi: true });
  const [params, setParams] = useState<Record<string, Record<string, number>>>(
    JSON.parse(JSON.stringify(DEFAULT_PARAMS)),
  );
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [indResult, setIndResult] = useState<IndicatorResult | null>(null);
  const [indLoading, setIndLoading] = useState(false);
  const [indErr, setIndErr] = useState("");

  const toggleIndicator = (k: string) =>
    setSelected((prev) => ({ ...prev, [k]: !prev[k] }));

  const setParam = (ind: string, key: string, value: number) =>
    setParams((prev) => ({ ...prev, [ind]: { ...prev[ind], [key]: value } }));

  const handleAnalyze = async () => {
    if (!path.trim()) {
      setIndErr("Enter a file path first");
      return;
    }
    const specs: IndicatorSpec[] = (Object.keys(INDICATOR_LABELS) as IndicatorSpec["type"][])
      .filter((k) => selected[k])
      .map((k) => ({ type: k, params: params[k] }));
    if (specs.length === 0) {
      setIndErr("Select at least one indicator");
      return;
    }
    setIndLoading(true);
    setIndErr("");
    setIndResult(null);
    try {
      const r = await getIndicators(path.trim(), column.trim() || "close", specs, {
        x: xCol.trim() || undefined,
      });
      setIndResult(r);
    } catch (err) {
      setIndErr(
        isNotAvailable(err)
          ? "Feature not yet available"
          : err instanceof Error
            ? err.message
            : "Analysis failed",
      );
    } finally {
      setIndLoading(false);
    }
  };

  // Section C — AI
  const [context, setContext] = useState("Analyze the current market conditions");
  const [aiOutput, setAiOutput] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiErr, setAiErr] = useState("");
  const aiRef = useRef<HTMLDivElement | null>(null);

  const runAi = async (fn: () => Promise<string>) => {
    setAiLoading(true);
    setAiErr("");
    setAiOutput("");
    try {
      setAiOutput(await fn());
      aiRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    } catch (err) {
      setAiErr(
        isNotAvailable(err)
          ? "Feature not yet available"
          : err instanceof Error
            ? err.message
            : "Generation failed",
      );
    } finally {
      setAiLoading(false);
    }
  };

  const handleInsight = () =>
    runAi(async () => (await aiGenerateInsight(context.trim())).insight);

  const handleAnalyzeFile = () =>
    runAi(async () => (await aiAnalyzeFile(path.trim(), context.trim())).analysis);

  return (
    <div className="relative p-6 space-y-6 overflow-y-auto h-screen animate-in">
      <div className="aurora-bg" />

      <div className="relative">
        <h1 className="text-2xl font-bold text-foreground glow-frost">Trading</h1>
        <p className="text-sm text-muted mt-1">Market watchlist, technical indicators, and AI market briefs</p>
      </div>

      {/* Section A — Watchlist */}
      <section className="relative runic-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">Market Watchlist</h2>

        <div className="flex gap-2">
          <input
            value={newTicker}
            onChange={(e) => setNewTicker(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleAdd()}
            placeholder="AAPL"
            className={`${inputCls} flex-1 uppercase`}
          />
          <button onClick={handleAdd} className={frostBtn}>
            Add
          </button>
        </div>
        {watchErr && <p className="text-xs text-rose font-mono">{watchErr}</p>}

        {watchUnavailable ? (
          <p className="text-xs text-muted/50 italic py-3">market data unavailable — yfinance not installed</p>
        ) : tickers.length === 0 ? (
          <p className="text-xs text-muted/60 italic py-3">No tickers yet — add one above</p>
        ) : (
          <div className="space-y-1">
            {tickers.map((t) => (
              <WatchRow key={t} ticker={t} onRemove={handleRemove} />
            ))}
          </div>
        )}
      </section>

      {/* Section B — Technical Indicators */}
      <section className="relative glass-card p-5 space-y-4">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">Technical Indicators</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <div className="md:col-span-3">
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium">File path</label>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="path/to/prices.parquet"
              className={`${inputCls} w-full mt-1`}
            />
          </div>
          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Price column</label>
            <input
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              placeholder="close"
              className={`${inputCls} w-full mt-1`}
            />
          </div>
          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium">X / time column (optional)</label>
            <input
              value={xCol}
              onChange={(e) => setXCol(e.target.value)}
              placeholder="date"
              className={`${inputCls} w-full mt-1`}
            />
          </div>
        </div>

        <div className="space-y-2">
          {(Object.keys(INDICATOR_LABELS) as (keyof typeof INDICATOR_LABELS)[]).map((k) => (
            <div key={k} className="rounded-lg bg-white/[0.02] border border-white/[0.06]">
              <div className="flex items-center gap-3 px-3 py-2">
                <label className="flex items-center gap-2 cursor-pointer select-none flex-1">
                  <input
                    type="checkbox"
                    checked={!!selected[k]}
                    onChange={() => toggleIndicator(k)}
                    className="accent-[var(--frost)]"
                  />
                  <span className="text-sm font-medium text-foreground">{INDICATOR_LABELS[k]}</span>
                </label>
                <button
                  onClick={() => setExpanded((p) => ({ ...p, [k]: !p[k] }))}
                  className="text-[10px] text-muted hover:text-frost font-mono uppercase tracking-wider"
                >
                  {expanded[k] ? "hide params" : "params"}
                </button>
              </div>
              {expanded[k] && (
                <div className="flex flex-wrap gap-3 px-3 pb-3">
                  {Object.entries(params[k]).map(([pk, pv]) => (
                    <div key={pk} className="flex items-center gap-1.5">
                      <span className="text-[10px] text-muted font-mono">{pk}</span>
                      <input
                        type="number"
                        value={pv}
                        onChange={(e) => setParam(k, pk, Number(e.target.value))}
                        className={`${inputCls} w-20 py-1`}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="flex items-center gap-3">
          <button onClick={handleAnalyze} disabled={indLoading} className={frostBtn}>
            {indLoading ? "Computing..." : "Analyze"}
          </button>
          {indLoading && <Spinner />}
        </div>
        {indErr && <p className="text-xs text-rose font-mono">{indErr}</p>}

        {indResult && (
          <div className="space-y-3 pt-2">
            <p className="text-[10px] text-muted/60 font-mono">
              {indResult.source_rows.toLocaleString()} rows · {indResult.column}
            </p>
            {indResult.indicators.map((s, i) => (
              <IndicatorChart key={`${s.type}-${i}`} series={s} x={indResult.x} />
            ))}
          </div>
        )}
      </section>

      {/* Section C — AI Market Brief */}
      <section className="relative runic-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">AI Market Brief</h2>

        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          rows={3}
          placeholder="Analyze the current market conditions"
          className={`${inputCls} w-full resize-y`}
        />

        <div className="flex flex-wrap items-center gap-3">
          <button onClick={handleInsight} disabled={aiLoading} className={frostBtn}>
            {aiLoading ? "Generating..." : "Generate Insight"}
          </button>
          <button
            onClick={handleAnalyzeFile}
            disabled={aiLoading || !path.trim()}
            title={path.trim() ? "" : "Enter a file path in Technical Indicators first"}
            className={frostBtn}
          >
            Analyze File
          </button>
          {aiLoading && <Spinner label="thinking" />}
        </div>
        {aiErr && <p className="text-xs text-rose font-mono">{aiErr}</p>}

        {aiOutput && (
          <div
            ref={aiRef}
            className="rounded-lg border border-frost/20 p-4 text-sm text-foreground-dim whitespace-pre-wrap leading-relaxed"
            style={{
              background: "linear-gradient(135deg, rgba(103,232,249,0.06), rgba(103,232,249,0.01))",
            }}
          >
            {aiOutput}
          </div>
        )}
      </section>
    </div>
  );
}
