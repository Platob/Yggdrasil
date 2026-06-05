"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalogs, getSchemas, getTables, finance, analysisIndicators, analysisPortfolio,
  type FinanceResult, type IndicatorsResult, type PortfolioResult,
} from "@/lib/api";
import Chart from "@/components/Chart";
import { NUMERIC } from "@/lib/format";

interface Series { full_name: string; source_url: string; numeric: string[]; }

const PRICEY = /close|price|adj|nav|value|amount|mrr|rate|last/i;
const ALL_INDICATORS = ["rsi", "macd", "bb"] as const;

function pct(v: number | null | undefined): string {
  return v == null ? "--" : `${(v * 100).toFixed(2)}%`;
}
function num(v: number | null | undefined, d = 2): string {
  return v == null ? "--" : v.toFixed(d);
}
function ind(res: IndicatorsResult | null, name: string): (number | null)[] {
  return res?.indicators.find((i) => i.name === name)?.values ?? [];
}

// Discover finance-ready tables by fanning out the catalog tree per level.
async function discoverSeries(): Promise<Series[]> {
  const cats = (await getCatalogs(undefined, true)).catalogs;
  const schemaResults = await Promise.all(cats.map((c) => getSchemas(c.name, undefined, true)));
  const pairs: { cat: string; sch: string }[] = [];
  schemaResults.forEach((r, i) => r.schemas.forEach((s) => pairs.push({ cat: cats[i].name, sch: s.name })));
  const tableResults = await Promise.all(pairs.map((p) => getTables(p.cat, p.sch, undefined, true)));
  const out: Series[] = [];
  tableResults.forEach((r) => {
    r.tables.forEach((t) => {
      if (t.object_type !== "TABLE" || !t.source_url) return;
      const numeric = (t.columns ?? []).filter((cc) => NUMERIC.test(cc.dtype)).map((cc) => cc.name);
      if (numeric.length) out.push({ full_name: t.full_name, source_url: t.source_url, numeric });
    });
  });
  return out;
}

function Metric({ label, value, tone = "neutral", hint }: { label: string; value: string; tone?: "good" | "bad" | "neutral"; hint?: string }) {
  const color = tone === "good" ? "text-emerald" : tone === "bad" ? "text-rose/90" : "text-foreground";
  return (
    <div className="glass-card p-3 flex flex-col gap-0.5" title={hint}>
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className={`font-mono text-lg font-semibold ${color}`}>{value}</span>
    </div>
  );
}

const selectCls = "bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30";

// ── Analysis tab ─────────────────────────────────────────────────────────────

function AnalysisTab({ series }: { series: Series[] }) {
  const [sel, setSel] = useState("");
  const [col, setCol] = useState("");
  const [win, setWin] = useState(14);
  const [ppy, setPpy] = useState(252);
  const [indicators, setIndicators] = useState<Set<string>>(new Set(ALL_INDICATORS));
  const [finRes, setFinRes] = useState<FinanceResult | null>(null);
  const [indRes, setIndRes] = useState<IndicatorsResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (sel || !series.length) return;
    const pick = series.find((s) => PRICEY.test(s.full_name) || s.numeric.some((n) => PRICEY.test(n))) ?? series[0];
    setSel(pick.full_name);
    setCol(pick.numeric.find((n) => PRICEY.test(n)) ?? pick.numeric[0]);
  }, [series, sel]);

  const current = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);

  const run = useCallback(async () => {
    if (!current || !col) return;
    setBusy(true); setErr("");
    try {
      const [fr, ir] = await Promise.all([
        finance(current.source_url, col, { window: win, periods_per_year: ppy, node: undefined }),
        analysisIndicators(current.source_url, col, { indicators: [...indicators], window: win, node: undefined }),
      ]);
      setFinRes(fr); setIndRes(ir);
    } catch (e) { setErr(String(e)); setFinRes(null); setIndRes(null); } finally { setBusy(false); }
  }, [current, col, win, ppy, indicators]);

  useEffect(() => { if (current && col) run(); }, [current, col, win, ppy, indicators, run]);

  const m = finRes?.metrics;
  const labels = finRes?.index ?? [];
  const indLabels = indRes?.index ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="glass-card p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Asset</span>
          <select value={sel} onChange={(e) => { setSel(e.target.value); const s = series.find((x) => x.full_name === e.target.value); if (s) setCol(s.numeric[0]); }}
            className={`${selectCls} min-w-[180px]`}>
            {series.length === 0 && <option>no numeric tables</option>}
            {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Column</span>
          <select value={col} onChange={(e) => setCol(e.target.value)} className={selectCls}>
            {(current?.numeric ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Window</span>
          <input type="number" value={win} min={2} onChange={(e) => setWin(Math.max(2, +e.target.value || 14))}
            className={`w-20 ${selectCls}`} />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Periods/yr</span>
          <select value={ppy} onChange={(e) => setPpy(+e.target.value)} className={selectCls}>
            <option value={252}>252 (daily)</option>
            <option value={52}>52 (weekly)</option>
            <option value={12}>12 (monthly)</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Indicators</span>
          <div className="flex gap-2">
            {ALL_INDICATORS.map((key) => {
              const on = indicators.has(key);
              return (
                <button key={key} onClick={() => setIndicators((prev) => { const next = new Set(prev); if (on) next.delete(key); else next.add(key); return next; })}
                  className={`px-2 py-1 rounded text-[11px] font-mono border uppercase ${on ? "bg-frost/15 text-frost border-frost/30" : "bg-white/[0.03] text-muted border-white/[0.08]"}`}>
                  {key}
                </button>
              );
            })}
          </div>
        </div>
        {busy && <span className="text-[11px] text-muted">computing…</span>}
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      {m && (
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-8 gap-2">
          <Metric label="Total return" value={pct(m.total_return)} tone={(m.total_return ?? 0) >= 0 ? "good" : "bad"} />
          <Metric label="CAGR" value={pct(m.cagr)} tone={(m.cagr ?? 0) >= 0 ? "good" : "bad"} />
          <Metric label="Ann. return" value={pct(m.ann_return)} hint="mean periodic return × periods/yr" />
          <Metric label="Ann. vol" value={pct(m.ann_volatility)} hint="stdev of returns, annualized" />
          <Metric label="Sharpe" value={num(m.sharpe)} tone={(m.sharpe ?? 0) >= 1 ? "good" : "neutral"} hint="(ann. return − rf) / ann. vol" />
          <Metric label="Sortino" value={num(m.sortino)} tone={(m.sortino ?? 0) >= 1 ? "good" : "neutral"} hint="downside-deviation Sharpe" />
          <Metric label="Max drawdown" value={pct(m.max_drawdown)} tone="bad" hint="worst peak-to-trough" />
          <Metric label="Calmar" value={num(m.calmar)} tone={(m.calmar ?? 0) >= 1 ? "good" : "neutral"} hint="CAGR / |max drawdown|" />
        </div>
      )}

      {finRes && (
        <div className="glass-card p-3">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-[11px] uppercase tracking-wide text-muted">{col} · price</span>
            <span className="text-[10px] text-amber/80">— EMA({finRes.window})</span>
          </div>
          <Chart type="line" labels={labels} values={finRes.value} overlay={finRes.ema} color="var(--frost)" yLabel={col} height={240} />
        </div>
      )}

      {indicators.has("bb") && indRes && (
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">Bollinger Bands (±2σ)</span>
          <Chart type="area" labels={indLabels} values={indRes.price}
            band={{ min: ind(indRes, "bb_lower"), max: ind(indRes, "bb_upper") }}
            color="var(--frost)" yLabel="bb" height={200} />
        </div>
      )}

      {indicators.has("rsi") && indRes && (
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">RSI · OB:70 OS:30</span>
          <Chart type="line" labels={indLabels} values={ind(indRes, "rsi")} color="var(--amber)" yLabel="rsi" height={120} />
        </div>
      )}

      {indicators.has("macd") && indRes && (
        <div className="glass-card p-3">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-[11px] uppercase tracking-wide text-muted">MACD</span>
            <span className="text-[10px] text-frost/80">— line</span>
            <span className="text-[10px] text-amber/80">— signal</span>
          </div>
          <Chart type="line" labels={indLabels} values={ind(indRes, "macd_line")} overlay={ind(indRes, "macd_signal")} color="var(--frost)" yLabel="macd" height={140} />
        </div>
      )}

      {finRes && (
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">Drawdown (peak-to-trough)</span>
          <Chart type="area" labels={labels} values={finRes.drawdown} color="var(--rose)" yLabel="dd" height={140} />
        </div>
      )}

      {!finRes && !err && <div className="glass-card p-4 text-muted text-xs">Pick an asset to run technical analysis.</div>}
    </div>
  );
}

// ── Portfolio tab ────────────────────────────────────────────────────────────

interface Pick { full_name: string; source_url: string; col: string; }

function corrColor(v: number | null): string {
  if (v == null) return "transparent";
  const a = Math.min(1, Math.abs(v)).toFixed(2);
  return v >= 0 ? `color-mix(in srgb, var(--emerald) ${+a * 100}%, transparent)` : `color-mix(in srgb, var(--rose) ${+a * 100}%, transparent)`;
}

function PortfolioTab({ series }: { series: Series[] }) {
  const [picks, setPicks] = useState<Pick[]>([]);
  const [win, setWin] = useState(20);
  const [ppy, setPpy] = useState(252);
  const [portResult, setPortResult] = useState<PortfolioResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (picks.length || series.length < 1) return;
    const seed = series.slice(0, 2).map((s) => ({ full_name: s.full_name, source_url: s.source_url, col: s.numeric.find((n) => PRICEY.test(n)) ?? s.numeric[0] }));
    setPicks(seed);
  }, [series, picks.length]);

  const setPick = (i: number, patch: Partial<Pick>) =>
    setPicks((prev) => prev.map((p, k) => (k === i ? { ...p, ...patch } : p)));

  const analyze = useCallback(async () => {
    const valid = picks.filter((p) => p.source_url && p.col);
    if (valid.length < 2) { setErr("Pick at least two assets to compare."); return; }
    setBusy(true); setErr("");
    try {
      setPortResult(await analysisPortfolio(
        valid.map((p) => p.source_url),
        valid.map((p) => p.col),
        { labels: valid.map((p) => `${p.full_name}:${p.col}`), window: win, periods_per_year: ppy, node: undefined },
      ));
    } catch (e) { setErr(String(e)); setPortResult(null); } finally { setBusy(false); }
  }, [picks, win, ppy]);

  const labels = portResult?.index ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="glass-card p-3 flex flex-col gap-2">
        {picks.map((p, i) => {
          const s = series.find((x) => x.full_name === p.full_name);
          return (
            <div key={i} className="flex items-center gap-2">
              <select value={p.full_name}
                onChange={(e) => { const ns = series.find((x) => x.full_name === e.target.value); setPick(i, { full_name: e.target.value, source_url: ns?.source_url ?? "", col: ns?.numeric[0] ?? "" }); }}
                className={`${selectCls} min-w-[200px]`}>
                {series.map((x) => <option key={x.full_name} value={x.full_name}>{x.full_name}</option>)}
              </select>
              <select value={p.col} onChange={(e) => setPick(i, { col: e.target.value })} className={selectCls}>
                {(s?.numeric ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
              <button onClick={() => setPicks((prev) => prev.filter((_, k) => k !== i))}
                className="px-2 py-1 rounded text-[11px] font-mono border border-rose/30 text-rose/80 hover:bg-rose/10">remove</button>
            </div>
          );
        })}
        <div className="flex items-center gap-3">
          {picks.length < 6 && series.length > 0 && (
            <button onClick={() => { const s = series[0]; setPicks((prev) => [...prev, { full_name: s.full_name, source_url: s.source_url, col: s.numeric[0] }]); }}
              className="px-2 py-1 rounded text-[11px] font-mono border border-frost/30 text-frost hover:bg-frost/10">+ Add Asset</button>
          )}
          <div className="flex items-center gap-1">
            <span className="text-[10px] uppercase tracking-wider text-muted">Window</span>
            <input type="number" value={win} min={2} onChange={(e) => setWin(Math.max(2, +e.target.value || 20))} className={`w-20 ${selectCls}`} />
          </div>
          <select value={ppy} onChange={(e) => setPpy(+e.target.value)} className={selectCls}>
            <option value={252}>252 (daily)</option>
            <option value={52}>52 (weekly)</option>
            <option value={12}>12 (monthly)</option>
          </select>
          <button onClick={analyze} className="px-3 py-1.5 rounded text-[11px] font-mono bg-frost/20 text-frost border border-frost/40 font-semibold hover:bg-frost/30">Analyze</button>
          {busy && <span className="text-[11px] text-muted">computing…</span>}
        </div>
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      {portResult && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            <Metric label="VaR (95%)" value={pct(portResult.var_95)} tone="bad" hint="1-period value-at-risk, equal-weight" />
            <Metric label="CVaR (95%)" value={pct(portResult.cvar_95)} tone="bad" hint="expected shortfall beyond VaR" />
          </div>

          <div className="glass-card p-3 overflow-x-auto">
            <span className="text-[11px] uppercase tracking-wide text-muted">Per-asset risk</span>
            <table className="w-full mt-2 text-xs font-mono">
              <thead>
                <tr className="text-muted text-[10px] uppercase tracking-wider">
                  <th className="text-left py-1 pr-3">Asset</th>
                  <th className="text-right px-2">Total ret</th>
                  <th className="text-right px-2">Ann ret</th>
                  <th className="text-right px-2">Ann vol</th>
                  <th className="text-right px-2">Sharpe</th>
                  <th className="text-right px-2">Max DD</th>
                  <th className="text-right px-2">Beta</th>
                </tr>
              </thead>
              <tbody>
                {portResult.assets.map((a) => (
                  <tr key={a.label} className="border-t border-white/[0.05]">
                    <td className="text-left py-1.5 pr-3 text-foreground">{a.label}</td>
                    <td className={`text-right px-2 ${(a.total_return ?? 0) >= 0 ? "text-emerald" : "text-rose/90"}`}>{pct(a.total_return)}</td>
                    <td className="text-right px-2">{pct(a.ann_return)}</td>
                    <td className="text-right px-2">{pct(a.ann_volatility)}</td>
                    <td className={`text-right px-2 ${(a.sharpe ?? 0) >= 1 ? "text-emerald" : ""}`}>{num(a.sharpe)}</td>
                    <td className="text-right px-2 text-rose/90">{pct(a.max_drawdown)}</td>
                    <td className="text-right px-2">{num(a.beta)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="glass-card p-3 overflow-x-auto">
            <span className="text-[11px] uppercase tracking-wide text-muted">Correlation matrix</span>
            <table className="mt-2 text-[11px] font-mono border-collapse">
              <thead>
                <tr>
                  <th className="p-1"></th>
                  {portResult.labels.map((l) => <th key={l} className="p-1 text-muted text-[9px] font-normal max-w-[80px] truncate" title={l}>{l.split(":")[0].split(".").pop()}</th>)}
                </tr>
              </thead>
              <tbody>
                {portResult.correlation.map((row, i) => (
                  <tr key={i}>
                    <td className="p-1 text-muted text-[9px] max-w-[100px] truncate" title={portResult.labels[i]}>{portResult.labels[i]?.split(":")[0].split(".").pop()}</td>
                    {row.map((v, j) => (
                      <td key={j} className="p-1 text-center text-foreground" style={{ background: corrColor(v) }}>{v == null ? "--" : v.toFixed(2)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
            {portResult.labels.map((l, i) => (
              <div key={l} className="glass-card p-3">
                <span className="text-[11px] uppercase tracking-wide text-muted truncate block" title={l}>{l}</span>
                <Chart type="line" labels={labels} values={portResult.prices[i]} color="var(--frost)" yLabel="px" height={140} />
              </div>
            ))}
          </div>
        </>
      )}

      {!portResult && !err && <div className="glass-card p-4 text-muted text-xs">Pick two or more assets and hit Analyze.</div>}
    </div>
  );
}

// ── Signals tab ──────────────────────────────────────────────────────────────

interface Signal { name: string; value: string; tone: "good" | "neutral" | "bad"; why: string; }

function lastFinite(arr: (number | null)[]): number | null {
  for (let i = arr.length - 1; i >= 0; i--) { const v = arr[i]; if (v != null && isFinite(v)) return v; }
  return null;
}
function mean(arr: (number | null)[]): number | null {
  let s = 0, c = 0;
  for (const v of arr) if (v != null && isFinite(v)) { s += v; c++; }
  return c ? s / c : null;
}

function SignalsTab({ series }: { series: Series[] }) {
  const [sel, setSel] = useState("");
  const [col, setCol] = useState("");
  const [res, setRes] = useState<FinanceResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (sel || !series.length) return;
    const pick = series.find((s) => PRICEY.test(s.full_name) || s.numeric.some((n) => PRICEY.test(n))) ?? series[0];
    setSel(pick.full_name);
    setCol(pick.numeric.find((n) => PRICEY.test(n)) ?? pick.numeric[0]);
  }, [series, sel]);

  const current = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);

  const run = useCallback(async () => {
    if (!current || !col) return;
    setBusy(true); setErr("");
    try { setRes(await finance(current.source_url, col, { node: undefined })); }
    catch (e) { setErr(String(e)); setRes(null); } finally { setBusy(false); }
  }, [current, col]);

  useEffect(() => { if (current && col) run(); }, [current, col, run]);

  const signals: Signal[] = useMemo(() => {
    if (!res) return [];
    const price = lastFinite(res.value);
    const ema = lastFinite(res.ema);
    const rm = res.roll_mean.filter((v) => v != null && isFinite(v)) as number[];
    const slope = rm.length >= 5 ? rm[rm.length - 1] - rm[rm.length - 5] : null;
    const curVol = lastFinite(res.roll_vol);
    const meanVol = mean(res.roll_vol);
    const dd = lastFinite(res.drawdown);
    const sharpe = res.metrics.sharpe;

    const out: Signal[] = [];
    if (price != null && ema != null)
      out.push({ name: "Momentum", value: price >= ema ? "Bullish" : "Bearish", tone: price >= ema ? "good" : "bad", why: "Last price vs EMA — above the average signals upward momentum." });
    if (slope != null)
      out.push({ name: "Trend", value: slope > 0 ? "Uptrend" : slope < 0 ? "Downtrend" : "Flat", tone: slope > 0 ? "good" : slope < 0 ? "bad" : "neutral", why: "Slope of the rolling mean over the last 5 points." });
    if (curVol != null && meanVol != null)
      out.push({ name: "Volatility", value: curVol > 1.5 * meanVol ? "High Vol" : "Normal", tone: curVol > 1.5 * meanVol ? "bad" : "good", why: "Current rolling vol vs its own average — >1.5× flags a turbulent regime." });
    if (dd != null)
      out.push({ name: "Drawdown", value: dd < -0.2 ? "Severe" : dd < -0.1 ? "Elevated" : "Contained", tone: dd < -0.2 ? "bad" : dd < -0.1 ? "neutral" : "good", why: "Distance below the running peak; deeper than -20% is severe." });
    if (sharpe != null)
      out.push({ name: "Risk-adjusted", value: sharpe > 1 ? "Good" : sharpe < 0 ? "Negative" : "Modest", tone: sharpe > 1 ? "good" : sharpe < 0 ? "bad" : "neutral", why: "Sharpe ratio — return per unit of volatility." });
    return out;
  }, [res]);

  const snapshot = useMemo(() => {
    if (!res) return "";
    const m = res.metrics;
    const lines = [
      `SNAPSHOT — ${sel} :: ${col}`,
      `total_return  ${pct(m.total_return)}`,
      `cagr          ${pct(m.cagr)}`,
      `ann_return    ${pct(m.ann_return)}`,
      `ann_vol       ${pct(m.ann_volatility)}`,
      `sharpe        ${num(m.sharpe)}`,
      `sortino       ${num(m.sortino)}`,
      `max_drawdown  ${pct(m.max_drawdown)}`,
      `calmar        ${num(m.calmar)}`,
      "",
      ...signals.map((s) => `${s.name.padEnd(14)}${s.value}`),
    ];
    return lines.join("\n");
  }, [res, signals, sel, col]);

  const [showSnap, setShowSnap] = useState(false);
  const toneCls = (t: Signal["tone"]) => (t === "good" ? "text-emerald" : t === "bad" ? "text-rose/90" : "text-amber");

  return (
    <div className="flex flex-col gap-3">
      <div className="glass-card p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Asset</span>
          <select value={sel} onChange={(e) => { setSel(e.target.value); const s = series.find((x) => x.full_name === e.target.value); if (s) setCol(s.numeric[0]); }}
            className={`${selectCls} min-w-[180px]`}>
            {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Column</span>
          <select value={col} onChange={(e) => setCol(e.target.value)} className={selectCls}>
            {(current?.numeric ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <button onClick={() => setShowSnap((v) => !v)} className="px-3 py-1.5 rounded text-[11px] font-mono border border-frost/30 text-frost hover:bg-frost/10">
          {showSnap ? "Hide snapshot" : "Export snapshot"}
        </button>
        {busy && <span className="text-[11px] text-muted">computing…</span>}
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      {signals.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {signals.map((s) => (
            <div key={s.name} className="glass-card p-4 flex flex-col gap-1">
              <span className="text-sm font-semibold text-foreground">{s.name}</span>
              <span className={`font-mono text-lg ${toneCls(s.tone)}`}>{s.value}</span>
              <span className="text-[10px] text-muted leading-snug">{s.why}</span>
            </div>
          ))}
        </div>
      )}

      {showSnap && res && (
        <pre className="glass-card p-4 text-[11px] font-mono text-foreground-dim whitespace-pre overflow-x-auto">{snapshot}</pre>
      )}

      {!res && !err && <div className="glass-card p-4 text-muted text-xs">Pick an asset to derive signals.</div>}
    </div>
  );
}

// ── Page shell ───────────────────────────────────────────────────────────────

const TABS = ["Analysis", "Portfolio", "Signals"] as const;
type Tab = (typeof TABS)[number];

export default function TradingPage() {
  const [series, setSeries] = useState<Series[]>([]);
  const [tab, setTab] = useState<Tab>("Analysis");
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;
    discoverSeries().then((s) => { if (!cancelled) setSeries(s); }).catch((e) => { if (!cancelled) setErr(String(e)); });
    return () => { cancelled = true; };
  }, []);

  return (
    <div className="relative p-6 space-y-5 overflow-y-auto h-screen animate-in">
      <div className="aurora-bg" />

      <div className="relative flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground glow-frost">Trading</h1>
          <p className="text-sm text-muted mt-1">Technical analysis, portfolio risk &amp; signals over Saga price series</p>
        </div>
      </div>

      <div className="relative flex gap-2">
        {TABS.map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-1.5 rounded-lg text-xs font-mono font-medium border transition-colors ${tab === t ? "bg-frost/15 text-frost border-frost/30" : "bg-white/[0.03] text-muted border-white/[0.08] hover:text-foreground"}`}>
            {t}
          </button>
        ))}
      </div>

      {err && <div className="relative glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      <div className="relative">
        {tab === "Analysis" && <AnalysisTab series={series} />}
        {tab === "Portfolio" && <PortfolioTab series={series} />}
        {tab === "Signals" && <SignalsTab series={series} />}
      </div>
    </div>
  );
}
