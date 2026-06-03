"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getCatalogs, getSchemas, getTables,
  getIndicators, getVaR, getSignals, getCorrelation, getPortfolio,
  getFsNodes,
  type TableEntry,
  type PortfolioAsset,
} from "@/lib/api";
import type {
  IndicatorResult, VaRResult, SignalResult,
  CorrelationResult, PortfolioResult, TradeSignal,
} from "@/lib/types";
import Chart from "@/components/Chart";
import { NUMERIC } from "@/lib/format";

// ── Helpers ──────────────────────────────────────────────────────────────────

function pct(v: number | null | undefined, d = 2) {
  return v == null ? "--" : `${(v * 100).toFixed(d)}%`;
}
function num(v: number | null | undefined, d = 2) {
  return v == null ? "--" : v.toFixed(d);
}
function tone(v: number | null | undefined, goodAbove = 0): "good" | "bad" | "neutral" {
  if (v == null) return "neutral";
  return v >= goodAbove ? "good" : "bad";
}

const PRICEY = /close|price|adj|nav|value|last|open|high|low/i;

// ── Small components ─────────────────────────────────────────────────────────

function Chip({ children, color = "text-muted" }: { children: React.ReactNode; color?: string }) {
  return (
    <span className={`px-2 py-0.5 rounded text-[10px] font-mono border border-white/[0.08] bg-white/[0.03] ${color}`}>
      {children}
    </span>
  );
}

function KpiCard({
  label, value, tone: t = "neutral", hint,
}: { label: string; value: string; tone?: "good" | "bad" | "neutral"; hint?: string }) {
  const color = t === "good" ? "text-emerald" : t === "bad" ? "text-rose/90" : "text-foreground";
  return (
    <div className="glass-card p-3 flex flex-col gap-0.5 min-w-0" title={hint}>
      <span className="text-[10px] uppercase tracking-wider text-muted truncate">{label}</span>
      <span className={`font-mono text-base font-semibold ${color}`}>{value}</span>
    </div>
  );
}

function SectionHead({ children }: { children: React.ReactNode }) {
  return <h2 className="text-[11px] uppercase tracking-widest text-muted font-semibold pt-1">{children}</h2>;
}

// Signal badge: BUY=emerald SELL=rose HOLD=muted
function SignalBadge({ action, strength }: { action: string; strength: number }) {
  const cls =
    action === "BUY" ? "text-emerald border-emerald/30 bg-emerald/10" :
    action === "SELL" ? "text-rose border-rose/30 bg-rose/10" :
    "text-muted border-white/[0.08] bg-white/[0.03]";
  return (
    <span className={`px-2 py-0.5 rounded text-[11px] font-bold font-mono border ${cls}`}>
      {action} {action !== "HOLD" && <span className="opacity-60 font-normal">({(strength * 100).toFixed(0)}%)</span>}
    </span>
  );
}

// Correlation heatmap rendered as an SVG grid
function CorrelationHeatmap({ result }: { result: CorrelationResult }) {
  const { labels, matrix } = result;
  const n = labels.length;
  const CELL = 48;
  const LABEL_W = 80;
  const W = LABEL_W + n * CELL;
  const H = LABEL_W + n * CELL;

  function cellColor(v: number | null): string {
    if (v == null) return "rgba(255,255,255,0.03)";
    const abs = Math.abs(v);
    if (v > 0) return `rgba(103,232,249,${0.1 + abs * 0.7})`; // frost
    return `rgba(244,63,94,${0.1 + abs * 0.7})`; // rose
  }

  return (
    <div className="overflow-x-auto">
      <svg width={W} height={H} className="block">
        {/* Row labels */}
        {labels.map((l, i) => (
          <text key={i} x={LABEL_W - 4} y={LABEL_W + i * CELL + CELL / 2 + 4}
            textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="monospace"
            className="select-none">{l.slice(0, 12)}</text>
        ))}
        {/* Col labels */}
        {labels.map((l, j) => (
          <text key={j} x={LABEL_W + j * CELL + CELL / 2} y={LABEL_W - 6}
            textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="monospace"
            transform={`rotate(-40, ${LABEL_W + j * CELL + CELL / 2}, ${LABEL_W - 6})`}
            className="select-none">{l.slice(0, 10)}</text>
        ))}
        {/* Cells */}
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const cx = LABEL_W + j * CELL;
            const cy = LABEL_W + i * CELL;
            return (
              <g key={`${i}-${j}`}>
                <rect x={cx} y={cy} width={CELL - 1} height={CELL - 1} fill={cellColor(v)} rx="3" />
                <text x={cx + CELL / 2} y={cy + CELL / 2 + 4}
                  textAnchor="middle" fontSize="9" fill="var(--foreground)" fontFamily="monospace" opacity="0.85">
                  {v != null ? v.toFixed(2) : "—"}
                </text>
              </g>
            );
          })
        )}
      </svg>
    </div>
  );
}

// ── RSI sub-chart (small, below main) ────────────────────────────────────────
function RsiPanel({ index, rsi }: { index: (string | number)[]; rsi: (number | null)[] }) {
  const valid = rsi.filter((v) => v != null);
  if (!valid.length) return null;
  const labels = index as (string | number)[];
  return (
    <div className="glass-card p-3">
      <div className="flex items-center gap-2 mb-1">
        <span className="text-[10px] uppercase tracking-wider text-muted">RSI (14)</span>
        <Chip color="text-amber">30</Chip>
        <Chip color="text-amber">70</Chip>
      </div>
      <Chart type="line" labels={labels} values={rsi} color="var(--amber)" height={90} />
    </div>
  );
}

// MACD sub-chart
function MacdPanel({ index, hist, macd, signal }: {
  index: (string | number)[];
  hist: (number | null)[];
  macd: (number | null)[];
  signal: (number | null)[];
}) {
  if (!hist.some((v) => v != null)) return null;
  return (
    <div className="glass-card p-3">
      <span className="text-[10px] uppercase tracking-wider text-muted">MACD histogram</span>
      <Chart type="bar" labels={index} values={hist} color="var(--emerald)" height={80} />
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

interface SeriesInfo { full_name: string; source_url: string; numeric: string[]; high?: string; low?: string; }

type Tab = "indicators" | "signals" | "portfolio" | "var" | "correlation";

export default function TradingPage() {
  const [nodes, setNodes] = useState<{ node_id: string; self: boolean }[]>([]);
  const [node, setNode] = useState<string | undefined>();
  const [series, setSeries] = useState<SeriesInfo[]>([]);
  const [tab, setTab] = useState<Tab>("indicators");

  // Indicators state
  const [sel, setSel] = useState("");
  const [col, setCol] = useState("");
  const [xCol, setXCol] = useState("");
  const [inds, setInds] = useState({ rsi: true, macd: true, bb: true });
  const [rsiPeriod, setRsiPeriod] = useState(14);
  const [bbPeriod, setBbPeriod] = useState(20);
  const [indResult, setIndResult] = useState<IndicatorResult | null>(null);
  const [indBusy, setIndBusy] = useState(false);
  const [indErr, setIndErr] = useState("");

  // Signals state
  const [sigSel, setSigSel] = useState("");
  const [sigCol, setSigCol] = useState("");
  const [sigResult, setSigResult] = useState<SignalResult | null>(null);
  const [sigBusy, setSigBusy] = useState(false);
  const [sigErr, setSigErr] = useState("");

  // VaR state
  const [varSel, setVarSel] = useState("");
  const [varCol, setVarCol] = useState("");
  const [varMethod, setVarMethod] = useState("historical");
  const [varConf, setVarConf] = useState(0.95);
  const [varHorizon, setVarHorizon] = useState(1);
  const [varResult, setVarResult] = useState<VaRResult | null>(null);
  const [varBusy, setVarBusy] = useState(false);
  const [varErr, setVarErr] = useState("");

  // Correlation state
  const [corrCols, setCorrCols] = useState<string[]>([]);  // selected series (full_name)
  const [corrColName, setCorrColName] = useState("");
  const [corrResult, setCorrResult] = useState<CorrelationResult | null>(null);
  const [corrBusy, setCorrBusy] = useState(false);
  const [corrErr, setCorrErr] = useState("");

  // Portfolio state
  const [portAssets, setPortAssets] = useState<(PortfolioAsset & { full_name: string })[]>([]);
  const [portResult, setPortResult] = useState<PortfolioResult | null>(null);
  const [portBusy, setPortBusy] = useState(false);
  const [portErr, setPortErr] = useState("");

  // Load nodes + catalog tree once
  useEffect(() => {
    getFsNodes().then((ns) => {
      setNodes(ns);
      const self = ns.find((n) => n.self);
      if (self) setNode(self.node_id);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cats = (await getCatalogs(node, true)).catalogs;
        const out: SeriesInfo[] = [];
        for (const c of cats) {
          const schs = (await getSchemas(c.name, node, true)).schemas;
          for (const s of schs) {
            const tbls: TableEntry[] = (await getTables(c.name, s.name, node, true)).tables;
            for (const t of tbls) {
              if (!t.source_url) continue;
              const numeric = (t.columns ?? []).filter((cc) => NUMERIC.test(cc.dtype)).map((cc) => cc.name);
              if (!numeric.length) continue;
              const allCols = (t.columns ?? []).map((cc) => cc.name);
              const high = allCols.find((c) => /high/i.test(c));
              const low = allCols.find((c) => /low/i.test(c));
              out.push({ full_name: t.full_name, source_url: t.source_url, numeric, high, low });
            }
          }
        }
        if (cancelled) return;
        setSeries(out);
        const pick = out.find((s) => PRICEY.test(s.full_name) || s.numeric.some((n) => PRICEY.test(n))) ?? out[0];
        if (pick) {
          const c = pick.numeric.find((n) => PRICEY.test(n)) ?? pick.numeric[0];
          setSel(pick.full_name); setCol(c);
          setSigSel(pick.full_name); setSigCol(c);
          setVarSel(pick.full_name); setVarCol(c);
          if (pick.numeric[0]) setCorrColName(pick.numeric[0]);
        }
      } catch { /* silent */ }
    })();
    return () => { cancelled = true; };
  }, [node]);

  const currentSeries = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);
  const currentSig = useMemo(() => series.find((s) => s.full_name === sigSel), [series, sigSel]);
  const currentVar = useMemo(() => series.find((s) => s.full_name === varSel), [series, varSel]);

  // Run indicators
  const runIndicators = useCallback(async () => {
    if (!currentSeries || !col) return;
    setIndBusy(true); setIndErr("");
    try {
      const activeInds = Object.entries(inds).filter(([, v]) => v).map(([k]) => k);
      setIndResult(await getIndicators({
        path: currentSeries.source_url, column: col,
        x: xCol || undefined, indicators: activeInds,
        rsi_period: rsiPeriod, bb_period: bbPeriod,
        high: currentSeries.high, low: currentSeries.low,
      }, node));
    } catch (e) { setIndErr(String(e)); } finally { setIndBusy(false); }
  }, [currentSeries, col, xCol, inds, rsiPeriod, bbPeriod, node]);

  useEffect(() => { if (currentSeries && col) runIndicators(); }, [currentSeries, col, runIndicators]);

  // Run signals
  const runSignals = useCallback(async () => {
    if (!currentSig || !sigCol) return;
    setSigBusy(true); setSigErr("");
    try {
      setSigResult(await getSignals(currentSig.source_url, sigCol, { node }));
    } catch (e) { setSigErr(String(e)); } finally { setSigBusy(false); }
  }, [currentSig, sigCol, node]);

  useEffect(() => { if (tab === "signals" && currentSig && sigCol) runSignals(); }, [tab, currentSig, sigCol, runSignals]);

  // Run VaR
  const runVar = useCallback(async () => {
    if (!currentVar || !varCol) return;
    setVarBusy(true); setVarErr("");
    try {
      setVarResult(await getVaR(currentVar.source_url, varCol, {
        method: varMethod, confidence: varConf, horizon: varHorizon, node,
      }));
    } catch (e) { setVarErr(String(e)); } finally { setVarBusy(false); }
  }, [currentVar, varCol, varMethod, varConf, varHorizon, node]);

  useEffect(() => { if (tab === "var" && currentVar && varCol) runVar(); }, [tab, currentVar, varCol, runVar]);

  // Run correlation
  const runCorrelation = useCallback(async () => {
    if (corrCols.length < 2 || !corrColName) return;
    setCorrBusy(true); setCorrErr("");
    try {
      const paths = corrCols.map((fn) => series.find((s) => s.full_name === fn)?.source_url).filter(Boolean) as string[];
      const labels = corrCols.map((fn) => fn.split(".").pop() ?? fn);
      setCorrResult(await getCorrelation(paths, corrColName, { labels, node }));
    } catch (e) { setCorrErr(String(e)); } finally { setCorrBusy(false); }
  }, [corrCols, corrColName, series, node]);

  useEffect(() => { if (tab === "correlation" && corrCols.length >= 2) runCorrelation(); }, [tab, corrCols, corrColName, runCorrelation]);

  // Run portfolio
  const runPortfolio = useCallback(async () => {
    if (portAssets.length < 2) return;
    setPortBusy(true); setPortErr("");
    try {
      setPortResult(await getPortfolio(portAssets.map((a) => ({ path: a.path, column: a.column, label: a.label, weight: a.weight })), { node }));
    } catch (e) { setPortErr(String(e)); } finally { setPortBusy(false); }
  }, [portAssets, node]);

  useEffect(() => { if (tab === "portfolio" && portAssets.length >= 2) runPortfolio(); }, [tab, portAssets, runPortfolio]);

  // Select box helper
  function SeriesSelect({ value, onChange, label }: { value: string; onChange: (v: string) => void; label?: string }) {
    return (
      <div className="flex flex-col gap-1 min-w-[160px]">
        {label && <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>}
        <select value={value} onChange={(e) => onChange(e.target.value)}
          className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
          {series.length === 0 && <option value="">no tables</option>}
          {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
        </select>
      </div>
    );
  }

  function ColSelect({ cols, value, onChange, label }: { cols: string[]; value: string; onChange: (v: string) => void; label?: string }) {
    return (
      <div className="flex flex-col gap-1 min-w-[120px]">
        {label && <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>}
        <select value={value} onChange={(e) => onChange(e.target.value)}
          className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
          {cols.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>
    );
  }

  const tabClass = (t: Tab) =>
    `px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${tab === t ? "bg-frost/10 text-frost border border-frost/20" : "text-muted hover:text-foreground border border-transparent"}`;

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-auto">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-lg font-bold tracking-tight">Trading Analytics</h1>
          <p className="text-[11px] text-muted mt-0.5">Technical indicators · Risk metrics · Portfolio · Signals</p>
        </div>
        {nodes.length > 1 && (
          <select value={node ?? ""} onChange={(e) => setNode(e.target.value || undefined)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1 text-xs font-mono outline-none focus:border-frost/30">
            {nodes.map((n) => <option key={n.node_id} value={n.node_id}>{n.node_id}{n.self ? " (self)" : ""}</option>)}
          </select>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-2 flex-wrap">
        <button className={tabClass("indicators")} onClick={() => setTab("indicators")}>Indicators</button>
        <button className={tabClass("signals")} onClick={() => setTab("signals")}>Signals</button>
        <button className={tabClass("var")} onClick={() => setTab("var")}>Value-at-Risk</button>
        <button className={tabClass("correlation")} onClick={() => setTab("correlation")}>Correlation</button>
        <button className={tabClass("portfolio")} onClick={() => setTab("portfolio")}>Portfolio</button>
      </div>

      {/* ── Indicators Tab ── */}
      {tab === "indicators" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-wrap items-end gap-3">
            <SeriesSelect value={sel} onChange={(v) => { setSel(v); const s = series.find((x) => x.full_name === v); if (s) setCol(s.numeric[0]); }} label="Series" />
            <ColSelect cols={currentSeries?.numeric ?? []} value={col} onChange={setCol} label="Column" />
            {/* X axis */}
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">X axis (opt)</span>
              <input type="text" value={xCol} onChange={(e) => setXCol(e.target.value)} placeholder="time col"
                className="w-24 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            </div>
            {/* Indicator toggles */}
            <div className="flex items-center gap-3 self-end pb-1">
              {(["rsi", "macd", "bb"] as const).map((k) => (
                <label key={k} className="flex items-center gap-1 text-xs cursor-pointer select-none">
                  <input type="checkbox" checked={inds[k]} onChange={(e) => setInds((p) => ({ ...p, [k]: e.target.checked }))}
                    className="accent-frost w-3 h-3" />
                  <span className="text-muted uppercase">{k}</span>
                </label>
              ))}
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">RSI period</span>
              <input type="number" value={rsiPeriod} min={2} max={100} onChange={(e) => setRsiPeriod(+e.target.value || 14)}
                className="w-16 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">BB period</span>
              <input type="number" value={bbPeriod} min={2} max={200} onChange={(e) => setBbPeriod(+e.target.value || 20)}
                className="w-16 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            </div>
            <button onClick={runIndicators} disabled={indBusy}
              className="self-end px-3 py-1.5 rounded-lg text-xs font-medium bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
              {indBusy ? "Computing…" : "Run"}
            </button>
          </div>

          {indErr && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{indErr}</div>}

          {indResult && (
            <div className="flex flex-col gap-2">
              {/* Price + Bollinger Bands */}
              <div className="glass-card p-3">
                <span className="text-[11px] uppercase tracking-wide text-muted">{col} · price + Bollinger Bands</span>
                <Chart
                  type="line"
                  labels={indResult.index}
                  values={indResult.value}
                  overlay={indResult.bb_mid ?? undefined}
                  color="var(--frost)"
                  height={220}
                />
                {indResult.bb_upper && (
                  <div className="flex gap-3 mt-1 text-[10px] text-muted font-mono">
                    <span>Upper: {indResult.bb_upper.find((v) => v != null)?.toFixed(2) ?? "--"}</span>
                    <span>Lower: {indResult.bb_lower?.find((v) => v != null)?.toFixed(2) ?? "--"}</span>
                  </div>
                )}
              </div>

              {/* RSI */}
              {inds.rsi && indResult.rsi && (
                <RsiPanel index={indResult.index} rsi={indResult.rsi} />
              )}

              {/* MACD */}
              {inds.macd && indResult.macd_hist && (
                <MacdPanel
                  index={indResult.index}
                  hist={indResult.macd_hist}
                  macd={indResult.macd ?? []}
                  signal={indResult.macd_signal ?? []}
                />
              )}

              {/* ATR */}
              {indResult.atr && indResult.atr.some((v) => v != null) && (
                <div className="glass-card p-3">
                  <span className="text-[10px] uppercase tracking-wider text-muted">ATR (14)</span>
                  <Chart type="line" labels={indResult.index} values={indResult.atr} color="var(--amber)" height={80} />
                </div>
              )}

              <div className="text-[10px] text-muted font-mono">
                {indResult.source_rows.toLocaleString()} rows{indResult.truncated ? " (truncated)" : ""}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Signals Tab ── */}
      {tab === "signals" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-wrap items-end gap-3">
            <SeriesSelect value={sigSel} onChange={(v) => { setSigSel(v); const s = series.find((x) => x.full_name === v); if (s) setSigCol(s.numeric[0]); }} label="Series" />
            <ColSelect cols={currentSig?.numeric ?? []} value={sigCol} onChange={setSigCol} label="Column" />
            <button onClick={runSignals} disabled={sigBusy}
              className="self-end px-3 py-1.5 rounded-lg text-xs font-medium bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
              {sigBusy ? "Scanning…" : "Generate Signals"}
            </button>
          </div>

          {sigErr && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{sigErr}</div>}

          {sigResult && (
            <div className="flex flex-col gap-3">
              {/* Summary */}
              <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
                <KpiCard label="Last Signal" value={sigResult.last_action}
                  tone={sigResult.last_action === "BUY" ? "good" : sigResult.last_action === "SELL" ? "bad" : "neutral"} />
                <KpiCard label="BUY signals" value={String(sigResult.buy_count)} tone="good" />
                <KpiCard label="SELL signals" value={String(sigResult.sell_count)} tone="bad" />
                <KpiCard label="Source rows" value={sigResult.source_rows.toLocaleString()} />
              </div>

              {/* Signal log */}
              <div className="glass-card overflow-hidden">
                <div className="px-3 py-2 border-b border-white/[0.06] text-[10px] uppercase tracking-wider text-muted">
                  {sigResult.signals.length} non-hold signals
                </div>
                <div className="overflow-auto max-h-72">
                  <table className="w-full text-xs font-mono">
                    <thead>
                      <tr className="border-b border-white/[0.06]">
                        <th className="text-left px-3 py-1.5 text-muted font-normal">Bar</th>
                        <th className="text-left px-3 py-1.5 text-muted font-normal">Action</th>
                        <th className="text-left px-3 py-1.5 text-muted font-normal">RSI</th>
                        <th className="text-left px-3 py-1.5 text-muted font-normal">MACD Hist</th>
                        <th className="text-left px-3 py-1.5 text-muted font-normal">Reasons</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sigResult.signals.slice(-50).reverse().map((s, i) => (
                        <tr key={i} className="border-b border-white/[0.04] hover:bg-white/[0.02]">
                          <td className="px-3 py-1 text-muted">{String(s.index)}</td>
                          <td className="px-3 py-1"><SignalBadge action={s.action} strength={s.strength} /></td>
                          <td className="px-3 py-1">{s.rsi != null ? s.rsi.toFixed(1) : "—"}</td>
                          <td className="px-3 py-1 tabular-nums">{s.macd_hist != null ? s.macd_hist.toFixed(4) : "—"}</td>
                          <td className="px-3 py-1 text-muted/70 text-[10px]">{s.reasons.join(", ")}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── VaR Tab ── */}
      {tab === "var" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-wrap items-end gap-3">
            <SeriesSelect value={varSel} onChange={(v) => { setVarSel(v); const s = series.find((x) => x.full_name === v); if (s) setVarCol(s.numeric[0]); }} label="Series" />
            <ColSelect cols={currentVar?.numeric ?? []} value={varCol} onChange={setVarCol} label="Column" />
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">Method</span>
              <select value={varMethod} onChange={(e) => setVarMethod(e.target.value)}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
                <option value="historical">Historical</option>
                <option value="parametric">Parametric</option>
                <option value="cornish_fisher">Cornish-Fisher</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">Confidence</span>
              <select value={varConf} onChange={(e) => setVarConf(+e.target.value)}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
                <option value={0.90}>90%</option>
                <option value={0.95}>95%</option>
                <option value={0.99}>99%</option>
              </select>
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">Horizon (days)</span>
              <input type="number" value={varHorizon} min={1} max={30}
                onChange={(e) => setVarHorizon(Math.max(1, +e.target.value || 1))}
                className="w-16 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
            </div>
            <button onClick={runVar} disabled={varBusy}
              className="self-end px-3 py-1.5 rounded-lg text-xs font-medium bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
              {varBusy ? "Computing…" : "Compute VaR"}
            </button>
          </div>

          {varErr && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{varErr}</div>}

          {varResult && (
            <div className="flex flex-col gap-3">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
                <KpiCard label={`VaR ${(varResult.confidence * 100).toFixed(0)}%`}
                  value={varResult.var_pct != null ? `${varResult.var_pct.toFixed(2)}%` : "--"}
                  tone="bad" hint="Daily loss not exceeded at this confidence level" />
                <KpiCard label="CVaR (ES)"
                  value={varResult.cvar_pct != null ? `${varResult.cvar_pct.toFixed(2)}%` : "--"}
                  tone="bad" hint="Expected Shortfall — average loss beyond VaR" />
                <KpiCard label="Ann. Vol"
                  value={varResult.ann_volatility != null ? pct(varResult.ann_volatility) : "--"}
                  hint="Annualized volatility of returns" />
                <KpiCard label="Method" value={varResult.method} />
                <KpiCard label="Horizon" value={`${varResult.horizon}d`} />
              </div>

              <div className="glass-card p-4">
                <p className="text-xs text-muted leading-relaxed">
                  At <span className="text-frost font-semibold">{(varResult.confidence * 100).toFixed(0)}% confidence</span>,
                  the {varResult.horizon}-day VaR is{" "}
                  <span className="text-rose font-semibold">
                    {varResult.var_pct != null ? `${Math.abs(varResult.var_pct).toFixed(2)}%` : "--"}
                  </span>{" "}
                  — meaning the portfolio will not lose more than this amount on{" "}
                  {(varResult.confidence * 100).toFixed(0)}% of trading days.
                  The Expected Shortfall (CVaR) — average loss on the worst{" "}
                  {(100 - varResult.confidence * 100).toFixed(0)}% of days — is{" "}
                  <span className="text-rose font-semibold">
                    {varResult.cvar_pct != null ? `${Math.abs(varResult.cvar_pct).toFixed(2)}%` : "--"}
                  </span>.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Correlation Tab ── */}
      {tab === "correlation" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-col gap-3">
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wider text-muted">Add series</span>
                <select onChange={(e) => {
                  const fn = e.target.value;
                  if (fn && !corrCols.includes(fn)) setCorrCols((p) => [...p, fn]);
                  e.target.value = "";
                }}
                  className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30 min-w-[180px]">
                  <option value="">Select series…</option>
                  {series.filter((s) => !corrCols.includes(s.full_name)).map((s) => (
                    <option key={s.full_name} value={s.full_name}>{s.full_name}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wider text-muted">Column (all)</span>
                <input type="text" value={corrColName} onChange={(e) => setCorrColName(e.target.value)} placeholder="price"
                  className="w-28 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
              </div>
              <button onClick={runCorrelation} disabled={corrBusy || corrCols.length < 2}
                className="self-end px-3 py-1.5 rounded-lg text-xs font-medium bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
                {corrBusy ? "Computing…" : "Correlate"}
              </button>
            </div>
            {corrCols.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {corrCols.map((fn) => (
                  <span key={fn} className="flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono border border-frost/20 bg-frost/[0.06] text-frost">
                    {fn.split(".").pop()}
                    <button onClick={() => setCorrCols((p) => p.filter((x) => x !== fn))}
                      className="text-muted hover:text-rose ml-1">×</button>
                  </span>
                ))}
              </div>
            )}
          </div>

          {corrErr && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{corrErr}</div>}
          {corrCols.length < 2 && !corrResult && (
            <div className="glass-card p-4 text-muted text-xs">Add at least 2 series to compute correlation.</div>
          )}

          {corrResult && (
            <div className="flex flex-col gap-3">
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
                  {corrResult.method} correlation matrix
                </div>
                <CorrelationHeatmap result={corrResult} />
              </div>
              <div className="text-[10px] text-muted font-mono">
                {corrResult.source_rows.map((r, i) => `${corrResult.labels[i]}: ${r.toLocaleString()} rows`).join(" · ")}
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Portfolio Tab ── */}
      {tab === "portfolio" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-col gap-3">
            <div className="flex flex-wrap items-end gap-3">
              <div className="flex flex-col gap-1">
                <span className="text-[10px] uppercase tracking-wider text-muted">Add asset</span>
                <select onChange={(e) => {
                  const fn = e.target.value;
                  const s = series.find((x) => x.full_name === fn);
                  if (!s || portAssets.some((a) => a.full_name === fn)) return;
                  const c = s.numeric.find((n) => PRICEY.test(n)) ?? s.numeric[0];
                  setPortAssets((p) => [...p, { full_name: fn, path: s.source_url, column: c, label: fn.split(".").pop() ?? fn, weight: 1.0 }]);
                  e.target.value = "";
                }}
                  className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30 min-w-[200px]">
                  <option value="">Add series…</option>
                  {series.filter((s) => !portAssets.some((a) => a.full_name === s.full_name)).map((s) => (
                    <option key={s.full_name} value={s.full_name}>{s.full_name}</option>
                  ))}
                </select>
              </div>
              <button onClick={runPortfolio} disabled={portBusy || portAssets.length < 2}
                className="self-end px-3 py-1.5 rounded-lg text-xs font-medium bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-50">
                {portBusy ? "Computing…" : "Analyze Portfolio"}
              </button>
            </div>

            {portAssets.length > 0 && (
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    <th className="text-left py-1 text-muted font-normal">Asset</th>
                    <th className="text-left py-1 text-muted font-normal">Column</th>
                    <th className="text-left py-1 text-muted font-normal">Weight</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {portAssets.map((a, i) => (
                    <tr key={a.full_name} className="border-b border-white/[0.04]">
                      <td className="py-1 text-frost">{a.label}</td>
                      <td className="py-1">
                        <select value={a.column} onChange={(e) => setPortAssets((p) => p.map((x, j) => j === i ? { ...x, column: e.target.value } : x))}
                          className="bg-white/[0.04] border border-white/[0.08] rounded px-1 py-0.5 text-xs font-mono outline-none">
                          {(series.find((s) => s.full_name === a.full_name)?.numeric ?? []).map((c) => (
                            <option key={c} value={c}>{c}</option>
                          ))}
                        </select>
                      </td>
                      <td className="py-1">
                        <input type="number" value={a.weight} min={0.01} step={0.1}
                          onChange={(e) => setPortAssets((p) => p.map((x, j) => j === i ? { ...x, weight: +e.target.value || 1 } : x))}
                          className="w-16 bg-white/[0.04] border border-white/[0.08] rounded px-1 py-0.5 text-xs font-mono outline-none" />
                      </td>
                      <td className="py-1 text-right">
                        <button onClick={() => setPortAssets((p) => p.filter((_, j) => j !== i))}
                          className="text-muted hover:text-rose">×</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {portErr && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{portErr}</div>}
          {portAssets.length < 2 && !portResult && (
            <div className="glass-card p-4 text-muted text-xs">Add at least 2 assets to analyze a portfolio.</div>
          )}

          {portResult && (() => {
            const m = portResult.metrics;
            return (
              <div className="flex flex-col gap-3">
                {/* Risk metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-2">
                  <KpiCard label="Total return" value={pct(m.total_return)} tone={tone(m.total_return)} />
                  <KpiCard label="CAGR" value={pct(m.cagr)} tone={tone(m.cagr)} />
                  <KpiCard label="Ann. return" value={pct(m.ann_return)} />
                  <KpiCard label="Ann. vol" value={pct(m.ann_volatility)} />
                  <KpiCard label="Sharpe" value={num(m.sharpe)} tone={tone(m.sharpe, 1)} hint="(ann_return - rf) / ann_vol" />
                  <KpiCard label="Sortino" value={num(m.sortino)} tone={tone(m.sortino, 1)} />
                  <KpiCard label="Max DD" value={pct(m.max_drawdown)} tone="bad" />
                  <KpiCard label="Calmar" value={num(m.calmar)} tone={tone(m.calmar, 1)} />
                </div>

                {/* Portfolio value chart */}
                <div className="glass-card p-3">
                  <span className="text-[11px] uppercase tracking-wide text-muted">Portfolio cumulative return</span>
                  <Chart type="line" labels={portResult.index} values={portResult.portfolio_value}
                    color="var(--frost)" height={200} />
                </div>

                {/* Drawdown */}
                <div className="glass-card p-3">
                  <span className="text-[11px] uppercase tracking-wide text-muted">Drawdown</span>
                  <Chart type="area" labels={portResult.index} values={portResult.drawdown}
                    color="var(--rose)" height={120} />
                </div>

                {/* Individual asset returns */}
                {portResult.individual_returns.length > 0 && (
                  <div className="glass-card p-3">
                    <span className="text-[11px] uppercase tracking-wide text-muted mb-2 block">Individual asset returns</span>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {portResult.individual_returns.map((ret, i) => (
                        <div key={i}>
                          <span className="text-[10px] text-muted font-mono">{portResult.labels[i]}</span>
                          <Chart type="line" labels={portResult.index} values={ret}
                            color={["var(--emerald)", "var(--amber)", "var(--frost)", "var(--rose)"][i % 4]}
                            height={100} />
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Correlation matrix */}
                <div className="glass-card p-3">
                  <span className="text-[11px] uppercase tracking-wide text-muted mb-2 block">Asset correlation</span>
                  <CorrelationHeatmap result={{
                    node_id: portResult.node_id,
                    labels: portResult.labels,
                    method: "pearson",
                    matrix: portResult.correlation_matrix,
                    source_rows: portResult.labels.map(() => portResult.source_rows),
                  }} />
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
