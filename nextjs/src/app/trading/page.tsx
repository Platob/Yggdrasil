"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalogs, getSchemas, getTables, finance, getIndicators, aiSummary,
  type TableEntry, type FinanceResult,
} from "@/lib/api";
import type { IndicatorsResult, AiSummaryResult } from "@/lib/types";
import Chart from "@/components/Chart";
import { NUMERIC } from "@/lib/format";

const PRICEY = /close|price|adj|nav|value|amount|last/i;

function pct(v: number | null | undefined) {
  return v == null ? "--" : `${(v * 100).toFixed(2)}%`;
}
function num(v: number | null | undefined, d = 2) {
  return v == null ? "--" : v.toFixed(d);
}

interface Series { full_name: string; source_url: string; numeric: string[]; }

function MetricBadge({ label, value, tone = "neutral" }: { label: string; value: string; tone?: "good" | "bad" | "neutral" }) {
  const c = tone === "good" ? "text-emerald" : tone === "bad" ? "text-rose/80" : "text-foreground/80";
  return (
    <div className="flex flex-col gap-0.5 min-w-[80px]">
      <span className="text-[9px] uppercase tracking-wider text-muted">{label}</span>
      <span className={`font-mono text-sm font-semibold ${c}`}>{value}</span>
    </div>
  );
}

export default function TradingPage() {
  const [series, setSeries] = useState<Series[]>([]);
  const [sel, setSel] = useState<string>("");
  const [col, setCol] = useState<string>("");
  const [win, setWin] = useState(20);
  const [res, setRes] = useState<FinanceResult | null>(null);
  const [ind, setInd] = useState<IndicatorsResult | null>(null);
  const [aiRes, setAiRes] = useState<AiSummaryResult | null>(null);
  const [aiQ, setAiQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [err, setErr] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const cats = (await getCatalogs(undefined, true)).catalogs;
        const out: Series[] = [];
        for (const c of cats) {
          const schs = (await getSchemas(c.name, undefined, true)).schemas;
          for (const s of schs) {
            const tbls: TableEntry[] = (await getTables(c.name, s.name, undefined, true)).tables;
            for (const t of tbls) {
              if (t.object_type !== "TABLE" || !t.source_url) continue;
              const numeric = (t.columns ?? []).filter((cc) => NUMERIC.test(cc.dtype)).map((cc) => cc.name);
              if (numeric.length) out.push({ full_name: t.full_name, source_url: t.source_url, numeric });
            }
          }
        }
        if (cancelled) return;
        setSeries(out);
        const pick = out.find((s) => PRICEY.test(s.full_name) || s.numeric.some((n) => PRICEY.test(n))) ?? out[0];
        if (pick) { setSel(pick.full_name); setCol(pick.numeric.find((n) => PRICEY.test(n)) ?? pick.numeric[0]); }
      } catch (e) { if (!cancelled) setErr(String(e)); }
    })();
    return () => { cancelled = true; };
  }, []);

  const current = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);

  const run = useCallback(async () => {
    if (!current || !col) return;
    setBusy(true); setErr("");
    try {
      const [f, i] = await Promise.all([
        finance(current.source_url, col, { window: win, periods_per_year: 252 }),
        getIndicators(current.source_url, col, { bb_period: win }),
      ]);
      setRes(f); setInd(i);
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }, [current, col, win]);

  useEffect(() => { if (current && col) run(); }, [current, col, win, run]);

  const analyze = async () => {
    if (!current || !col) return;
    setAiBusy(true); setAiRes(null);
    try { setAiRes(await aiSummary(current.source_url, col, aiQ || null)); }
    catch (e) { setErr(String(e)); } finally { setAiBusy(false); }
  };

  const m = res?.metrics;
  const idx = res?.index ?? [];
  const iidx = ind?.index ?? [];

  return (
    <div className="flex flex-col gap-4 p-6 animate-in min-h-screen">
      {/* Header */}
      <div className="flex items-center gap-4 mb-2">
        <h1 className="text-xl font-semibold text-foreground">Trading Terminal</h1>
        {busy && <span className="text-xs text-muted animate-pulse">computing…</span>}
        <div className="ml-auto flex items-center gap-2">
          <select value={sel} onChange={(e) => { setSel(e.target.value); const s = series.find((x) => x.full_name === e.target.value); if (s) setCol(s.numeric[0]); }}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-sm font-mono outline-none focus:border-frost/30 min-w-[200px]">
            {series.length === 0 && <option>No series found</option>}
            {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
          </select>
          <select value={col} onChange={(e) => setCol(e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-sm font-mono outline-none focus:border-frost/30">
            {(current?.numeric ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
          <select value={win} onChange={(e) => setWin(+e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-1.5 text-sm font-mono outline-none focus:border-frost/30">
            <option value={10}>Win 10</option>
            <option value={20}>Win 20</option>
            <option value={50}>Win 50</option>
            <option value={200}>Win 200</option>
          </select>
        </div>
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs">{err}</div>}

      {/* Metrics strip */}
      {m && (
        <div className="glass-card p-3 flex flex-wrap gap-6">
          <MetricBadge label="Total Return" value={pct(m.total_return)} tone={(m.total_return ?? 0) >= 0 ? "good" : "bad"} />
          <MetricBadge label="CAGR" value={pct(m.cagr)} tone={(m.cagr ?? 0) >= 0 ? "good" : "bad"} />
          <MetricBadge label="Sharpe" value={num(m.sharpe)} tone={(m.sharpe ?? 0) >= 1 ? "good" : "neutral"} />
          <MetricBadge label="Sortino" value={num(m.sortino)} tone={(m.sortino ?? 0) >= 1 ? "good" : "neutral"} />
          <MetricBadge label="Ann. Vol" value={pct(m.ann_volatility)} tone="neutral" />
          <MetricBadge label="Max DD" value={pct(m.max_drawdown)} tone="bad" />
          <MetricBadge label="Calmar" value={num(m.calmar)} tone={(m.calmar ?? 0) >= 1 ? "good" : "neutral"} />
        </div>
      )}

      {/* Price + EMA chart */}
      {res && (
        <div className="glass-card p-3">
          <div className="flex items-center gap-3 mb-2">
            <span className="text-[11px] uppercase tracking-wide text-muted font-semibold">{col}</span>
            <span className="text-[10px] text-frost/60">— price</span>
            <span className="text-[10px] text-amber/60">— EMA({win})</span>
            {ind && <><span className="text-[10px] text-emerald/60 ml-2">· BB upper</span><span className="text-[10px] text-emerald/40">· BB lower</span></>}
          </div>
          <Chart type="line" labels={idx} values={res.value} overlay={res.ema} color="var(--frost)" yLabel={col} height={280} />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* RSI */}
        {ind && (
          <div className="glass-card p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[11px] uppercase tracking-wide text-muted font-semibold">RSI (14)</span>
              <span className="text-[9px] text-rose/60">70 overbought</span>
              <span className="text-[9px] text-emerald/60">30 oversold</span>
            </div>
            <Chart type="line" labels={iidx} values={ind.rsi} color="var(--amber)" yLabel="RSI" height={160} />
          </div>
        )}

        {/* MACD */}
        {ind && (
          <div className="glass-card p-3">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[11px] uppercase tracking-wide text-muted font-semibold">MACD (12,26,9)</span>
              <span className="text-[9px] text-frost/60">histogram</span>
            </div>
            <Chart type="bar" labels={iidx} values={ind.macd_hist} overlay={ind.macd_line} color="var(--emerald)" yLabel="MACD" height={160} />
          </div>
        )}
      </div>

      {/* Drawdown */}
      {res && (
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">Drawdown</span>
          <Chart type="area" labels={idx} values={res.drawdown} color="var(--rose)" yLabel="dd" height={120} />
        </div>
      )}

      {/* AI Analysis */}
      <div className="glass-card p-4 flex flex-col gap-3">
        <div className="flex items-center gap-3">
          <span className="text-[11px] uppercase tracking-wide text-muted font-semibold">AI Analysis</span>
          {aiRes && <span className="text-[9px] font-mono text-muted/50 ml-auto">{aiRes.model}</span>}
        </div>
        <div className="flex gap-2">
          <input
            type="text" value={aiQ} onChange={(e) => setAiQ(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") analyze(); }}
            placeholder="Ask about this series… (optional)"
            className="flex-1 bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-xs font-mono outline-none focus:border-frost/30 text-foreground placeholder-muted/50"
          />
          <button onClick={analyze} disabled={!current || !col || aiBusy}
            className="px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all whitespace-nowrap">
            {aiBusy ? "Analyzing…" : "Ask AI"}
          </button>
        </div>
        {aiRes && (
          <>
            <p className="text-sm text-foreground/90 leading-relaxed">{aiRes.summary}</p>
            {aiRes.key_points.length > 0 && (
              <ul className="space-y-1">
                {aiRes.key_points.map((pt, i) => (
                  <li key={i} className="flex items-start gap-2 text-xs text-foreground/80">
                    <span className="text-frost mt-0.5 shrink-0">▸</span><span>{pt}</span>
                  </li>
                ))}
              </ul>
            )}
            {aiRes.chart_hint && <p className="text-[10px] text-amber/70 font-mono">Suggested: {aiRes.chart_hint}</p>}
          </>
        )}
        {!aiRes && !aiBusy && (
          <p className="text-xs text-muted">Get a Claude-powered quantitative analysis of the selected series.</p>
        )}
      </div>
    </div>
  );
}
