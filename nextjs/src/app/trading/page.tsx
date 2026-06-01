"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalogs, getSchemas, getTables, finance, technical, correlation,
  getPeers,
} from "@/lib/api";
import type {
  TableEntry, FinanceResult, TechnicalResult, CorrelationResult,
} from "@/lib/api";
import Chart from "@/components/Chart";

const NUMERIC = /int|float|double|decimal|number|real/i;
const PRICEY = /close|price|adj|nav|value|amount|last|mid/i;

interface Series { full_name: string; source_url: string; numeric: string[]; high?: string; low?: string; volume?: string; }

function pct(v: number | null | undefined) { return v == null ? "--" : `${(v * 100).toFixed(2)}%`; }
function num(v: number | null | undefined, d = 2) { return v == null ? "--" : v.toFixed(d); }

function Metric({ label, value, tone = "neutral", hint }: { label: string; value: string; tone?: "good" | "bad" | "neutral"; hint?: string }) {
  const color = tone === "good" ? "text-emerald" : tone === "bad" ? "text-rose/90" : "text-foreground";
  return (
    <div className="glass-card p-3 flex flex-col gap-0.5 min-w-[100px]" title={hint}>
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className={`font-mono text-lg font-semibold ${color}`}>{value}</span>
    </div>
  );
}

// Correlation heatmap — pure SVG, no deps
function CorrelationHeatmap({ result }: { result: CorrelationResult }) {
  const { columns, matrix } = result;
  const n = columns.length;
  if (n === 0) return null;
  const cell = Math.max(28, Math.min(56, Math.floor(480 / n)));
  const labelW = 80;
  const w = labelW + n * cell;
  const h = labelW + n * cell;

  const color = (v: number | null) => {
    if (v == null) return "var(--foreground)";
    const t = (v + 1) / 2; // 0..1
    if (t > 0.5) {
      const s = (t - 0.5) * 2;
      return `rgba(52,211,153,${0.15 + s * 0.7})`; // emerald
    } else {
      const s = (0.5 - t) * 2;
      return `rgba(239,68,68,${0.15 + s * 0.7})`; // rose
    }
  };

  return (
    <div className="overflow-auto">
      <svg width={w} height={h} style={{ fontFamily: "monospace", fontSize: 9 }}>
        {/* column labels */}
        {columns.map((c, i) => (
          <text key={i}
            x={labelW + i * cell + cell / 2}
            y={labelW - 4}
            textAnchor="middle"
            fill="var(--muted)"
            transform={`rotate(-45, ${labelW + i * cell + cell / 2}, ${labelW - 4})`}
          >{c.length > 10 ? c.slice(0, 9) + "…" : c}</text>
        ))}
        {/* row labels */}
        {columns.map((c, i) => (
          <text key={i}
            x={labelW - 4}
            y={labelW + i * cell + cell / 2 + 3}
            textAnchor="end"
            fill="var(--muted)"
          >{c.length > 10 ? c.slice(0, 9) + "…" : c}</text>
        ))}
        {/* cells */}
        {matrix.map((row, ri) =>
          row.map((v, ci) => (
            <g key={`${ri}-${ci}`}>
              <rect
                x={labelW + ci * cell}
                y={labelW + ri * cell}
                width={cell}
                height={cell}
                fill={color(v)}
                stroke="var(--foreground)"
                strokeOpacity="0.06"
              />
              {v != null && (
                <text
                  x={labelW + ci * cell + cell / 2}
                  y={labelW + ri * cell + cell / 2 + 3}
                  textAnchor="middle"
                  fill="var(--foreground)"
                  fillOpacity="0.85"
                  fontSize={cell > 40 ? 9 : 7}
                >{v.toFixed(2)}</text>
              )}
            </g>
          )),
        )}
      </svg>
    </div>
  );
}

const SIGNAL_META: Record<string, { label: string; color: string; tone: string }> = {
  rsi_oversold:      { label: "RSI Oversold",       color: "text-emerald", tone: "bullish" },
  rsi_overbought:    { label: "RSI Overbought",      color: "text-rose/90", tone: "bearish" },
  macd_cross_up:     { label: "MACD Cross ↑",        color: "text-emerald", tone: "bullish" },
  macd_cross_down:   { label: "MACD Cross ↓",        color: "text-rose/90", tone: "bearish" },
  bb_breakout_up:    { label: "BB Breakout ↑",       color: "text-amber",   tone: "watch" },
  bb_breakout_down:  { label: "BB Breakdown ↓",      color: "text-frost",   tone: "watch" },
};

export default function TradingPage() {
  const [node, setNode] = useState<string | undefined>();
  const [peers, setPeers] = useState<{ node_id: string }[]>([]);
  const [series, setSeries] = useState<Series[]>([]);
  const [sel, setSel] = useState("");
  const [col, setCol] = useState("");
  const [tab, setTab] = useState<"overview" | "technical" | "signals" | "correlation">("overview");
  const [finRes, setFinRes] = useState<FinanceResult | null>(null);
  const [techRes, setTechRes] = useState<TechnicalResult | null>(null);
  const [corrRes, setCorrRes] = useState<CorrelationResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  // technical controls
  const [rsiPeriod, setRsiPeriod] = useState(14);
  const [bbPeriod, setBbPeriod] = useState(20);
  const [bbStd, setBbStd] = useState(2.0);
  const [win, setWin] = useState(20);

  useEffect(() => {
    getPeers().then((r) => setPeers(r.peers)).catch(() => {});
  }, []);

  // Discover finance-ready tables from the catalog
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cats = (await getCatalogs(node, true)).catalogs;
        const out: Series[] = [];
        for (const c of cats) {
          const schs = (await getSchemas(c.name, node, true)).schemas;
          for (const s of schs) {
            const tbls: TableEntry[] = (await getTables(c.name, s.name, node, true)).tables;
            for (const t of tbls) {
              if (t.object_type !== "TABLE" || !t.source_url) continue;
              const numeric = (t.columns ?? []).filter((cc) => NUMERIC.test(cc.dtype)).map((cc) => cc.name);
              if (!numeric.length) continue;
              const high = numeric.find((nm) => /high/i.test(nm));
              const low = numeric.find((nm) => /low/i.test(nm));
              const volume = numeric.find((nm) => /vol/i.test(nm));
              out.push({ full_name: t.full_name, source_url: t.source_url, numeric, high, low, volume });
            }
          }
        }
        if (cancelled) return;
        setSeries(out);
        const pick = out.find((s) => PRICEY.test(s.full_name) || s.numeric.some((nm) => PRICEY.test(nm))) ?? out[0];
        if (pick) {
          setSel(pick.full_name);
          setCol(pick.numeric.find((nm) => PRICEY.test(nm)) ?? pick.numeric[0]);
        }
      } catch (e) { if (!cancelled) setErr(String(e)); }
    })();
    return () => { cancelled = true; };
  }, [node]);

  const current = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);

  const runAll = useCallback(async () => {
    if (!current || !col) return;
    setBusy(true); setErr("");
    try {
      const [fr, tr] = await Promise.all([
        finance(current.source_url, col, { window: win, node }),
        technical(current.source_url, col, {
          high: current.high, low: current.low, volume: current.volume,
          rsi_period: rsiPeriod, bb_period: bbPeriod, bb_std: bbStd, node,
        }),
      ]);
      setFinRes(fr); setTechRes(tr);
      // Correlation: only when multiple numeric columns exist
      if (current.numeric.length > 1) {
        const cr = await correlation(current.source_url, { columns: current.numeric.slice(0, 8), node });
        setCorrRes(cr);
      }
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }, [current, col, win, rsiPeriod, bbPeriod, bbStd, node]);

  useEffect(() => { if (current && col) runAll(); }, [current, col, win, rsiPeriod, bbPeriod, bbStd, runAll]);

  const m = finRes?.metrics;
  const idx = finRes?.index ?? [];
  const techIdx = techRes?.index ?? [];

  const getIndicator = (name: string) => techRes?.indicators.find((i) => i.name === name)?.values ?? [];

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 overflow-auto animate-in">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">Trading Analytics</h1>
          <p className="text-xs text-muted mt-0.5">Technical indicators · Risk metrics · Signal detection</p>
        </div>
        <div className="flex items-center gap-2">
          {peers.length > 0 && (
            <select value={node ?? ""} onChange={(e) => setNode(e.target.value || undefined)}
              className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none">
              <option value="">local</option>
              {peers.map((p) => <option key={p.node_id} value={p.node_id}>{p.node_id.slice(0, 12)}</option>)}
            </select>
          )}
          <button onClick={() => runAll()} disabled={busy || !current}
            className="px-3 py-1.5 rounded-lg text-xs font-mono border border-white/[0.08] bg-white/[0.04] hover:bg-white/[0.08] disabled:opacity-40 transition-colors">
            {busy ? "…" : "↻ Refresh"}
          </button>
        </div>
      </div>

      {err && <div className="glass-card p-3 text-rose/90 text-xs font-mono">{err}</div>}

      {/* Series picker */}
      <div className="glass-card p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Series</span>
          <select value={sel} onChange={(e) => {
              setSel(e.target.value);
              const s = series.find((x) => x.full_name === e.target.value);
              if (s) setCol(s.numeric.find((nm) => PRICEY.test(nm)) ?? s.numeric[0]);
            }}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30 min-w-[200px]">
            {series.length === 0 && <option value="">no tables found</option>}
            {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Column</span>
          <select value={col} onChange={(e) => setCol(e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none">
            {current?.numeric.map((nm) => <option key={nm} value={nm}>{nm}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Window</span>
          <input type="number" value={win} min={5} max={200} step={5}
            onChange={(e) => setWin(Number(e.target.value))}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none w-20" />
        </div>
        {finRes?.truncated && <span className="text-[10px] text-amber font-mono ml-auto">series truncated</span>}
      </div>

      {/* Tab bar */}
      <div className="flex gap-1">
        {(["overview", "technical", "signals", "correlation"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-3 py-1.5 rounded-lg text-xs font-mono capitalize transition-colors border ${
              tab === t
                ? "bg-frost/10 border-frost/30 text-frost"
                : "border-white/[0.06] bg-white/[0.02] text-muted hover:text-foreground"
            }`}>
            {t}
            {t === "signals" && techRes && techRes.signals.length > 0 && (
              <span className="ml-1.5 px-1 py-0.5 rounded text-[9px] bg-amber/20 text-amber font-bold">
                {techRes.signals.length}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === "overview" && finRes && (
        <div className="flex flex-col gap-3 animate-in">
          <div className="flex flex-wrap gap-2">
            <Metric label="Total Return" value={pct(m?.total_return)}
              tone={(m?.total_return ?? 0) >= 0 ? "good" : "bad"} />
            <Metric label="CAGR" value={pct(m?.cagr)} tone={(m?.cagr ?? 0) >= 0 ? "good" : "bad"} />
            <Metric label="Sharpe" value={num(m?.sharpe)}
              tone={!m?.sharpe ? "neutral" : m.sharpe >= 1 ? "good" : m.sharpe < 0 ? "bad" : "neutral"} />
            <Metric label="Sortino" value={num(m?.sortino)}
              tone={!m?.sortino ? "neutral" : m.sortino >= 1 ? "good" : m.sortino < 0 ? "bad" : "neutral"} />
            <Metric label="Max Drawdown" value={pct(m?.max_drawdown)} tone="bad" />
            <Metric label="Calmar" value={num(m?.calmar)}
              tone={!m?.calmar ? "neutral" : m.calmar >= 0.5 ? "good" : "bad"} />
            <Metric label="Ann. Vol" value={pct(m?.ann_volatility)} tone="neutral" />
            <Metric label="Ann. Return" value={pct(m?.ann_return)}
              tone={(m?.ann_return ?? 0) >= 0 ? "good" : "bad"} />
          </div>
          <div className="glass-card p-3">
            <div className="text-[10px] uppercase tracking-wider text-muted mb-2">Cumulative Return</div>
            <Chart type="area" labels={idx as (string | number)[]} values={finRes.cum_return} color="var(--emerald)" height={200} />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="glass-card p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted mb-2">Price + EMA({win})</div>
              <Chart type="line" labels={idx as (string | number)[]} values={finRes.value} overlay={finRes.ema} color="var(--frost)" height={180} />
            </div>
            <div className="glass-card p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted mb-2">Drawdown</div>
              <Chart type="area" labels={idx as (string | number)[]} values={finRes.drawdown} color="var(--rose)" height={180} />
            </div>
          </div>
        </div>
      )}

      {/* Technical Tab */}
      {tab === "technical" && (
        <div className="flex flex-col gap-3 animate-in">
          {/* Controls row */}
          <div className="glass-card p-3 flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">RSI Period</span>
              <input type="number" value={rsiPeriod} min={2} max={50} step={1}
                onChange={(e) => setRsiPeriod(Number(e.target.value))}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none w-20" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">BB Period</span>
              <input type="number" value={bbPeriod} min={5} max={100} step={1}
                onChange={(e) => setBbPeriod(Number(e.target.value))}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none w-20" />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-[10px] uppercase tracking-wider text-muted">BB Std Dev</span>
              <input type="number" value={bbStd} min={0.5} max={4} step={0.5}
                onChange={(e) => setBbStd(Number(e.target.value))}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none w-20" />
            </div>
          </div>

          {techRes ? (
            <>
              {/* Price + Bollinger Bands */}
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
                  Price + Bollinger Bands (±{bbStd}σ, {bbPeriod}-period)
                </div>
                <Chart type="area"
                  labels={techIdx as (string | number)[]}
                  values={techRes.price}
                  band={{ min: getIndicator("bb_lower"), max: getIndicator("bb_upper") }}
                  overlay={getIndicator("bb_middle")}
                  color="var(--frost)"
                  height={220}
                />
              </div>
              {/* RSI */}
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
                  RSI ({rsiPeriod}) — oversold &lt;30 · overbought &gt;70
                </div>
                <Chart type="line"
                  labels={techIdx as (string | number)[]}
                  values={getIndicator("rsi")}
                  color="var(--amber)"
                  height={120}
                  yLabel="RSI"
                />
              </div>
              {/* MACD histogram */}
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-2">
                  MACD ({rsiPeriod > 12 ? 12 : rsiPeriod},{26},{9}) — histogram + signal
                </div>
                <Chart type="bar"
                  labels={techIdx as (string | number)[]}
                  values={getIndicator("macd_hist")}
                  overlay={getIndicator("macd_signal")}
                  color="var(--emerald)"
                  height={120}
                  yLabel="MACD"
                />
              </div>
            </>
          ) : (
            <div className="glass-card p-6 text-center text-muted text-sm">
              {busy ? "Loading…" : "Select a series to compute indicators"}
            </div>
          )}
        </div>
      )}

      {/* Signals Tab */}
      {tab === "signals" && (
        <div className="flex flex-col gap-3 animate-in">
          {techRes && techRes.signals.length > 0 ? (
            <>
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-3">
                  {techRes.signals.length} signal{techRes.signals.length !== 1 ? "s" : ""} detected
                </div>
                <div className="overflow-auto max-h-[500px]">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="text-muted text-[10px] uppercase tracking-wider">
                        <th className="text-left pb-2 pr-4">Time / Index</th>
                        <th className="text-left pb-2 pr-4">Signal</th>
                        <th className="text-right pb-2">Value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {techRes.signals.slice().reverse().map((sig, i) => {
                        const meta = SIGNAL_META[sig.kind] ?? { label: sig.kind, color: "text-muted", tone: "neutral" };
                        return (
                          <tr key={i} className="border-t border-white/[0.04] hover:bg-white/[0.02]">
                            <td className="py-1.5 pr-4 text-muted">{String(sig.x_val)}</td>
                            <td className={`py-1.5 pr-4 ${meta.color}`}>{meta.label}</td>
                            <td className="py-1.5 text-right">{sig.value != null ? sig.value.toFixed(4) : "--"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              {/* Legend */}
              <div className="glass-card p-3">
                <div className="text-[10px] uppercase tracking-wider text-muted mb-2">Legend</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(SIGNAL_META).map(([k, v]) => (
                    <span key={k} className={`text-[10px] font-mono px-2 py-0.5 rounded border border-white/[0.08] ${v.color}`}>
                      {v.label}
                    </span>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="glass-card p-6 text-center text-muted text-sm">
              {busy ? "Scanning for signals…" : techRes ? "No signals detected in this series" : "Select a series to scan for signals"}
            </div>
          )}
        </div>
      )}

      {/* Correlation Tab */}
      {tab === "correlation" && (
        <div className="flex flex-col gap-3 animate-in">
          {corrRes ? (
            <div className="glass-card p-3">
              <div className="text-[10px] uppercase tracking-wider text-muted mb-3">
                Pearson correlation — {corrRes.columns.length} columns × {corrRes.source_rows.toLocaleString()} rows
              </div>
              <CorrelationHeatmap result={corrRes} />
            </div>
          ) : (
            <div className="glass-card p-6 text-center text-muted text-sm">
              {busy ? "Computing correlation…" : "Select a series with multiple numeric columns"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
