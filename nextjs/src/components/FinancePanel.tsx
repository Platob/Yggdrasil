"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalogs, getSchemas, getTables, finance, getIndicators, aiSummary,
  type TableEntry, type FinanceResult,
} from "@/lib/api";
import type { IndicatorsResult, AiSummaryResult } from "@/lib/types";
import Chart from "@/components/Chart";
import { NUMERIC } from "@/lib/format";

interface Series { full_name: string; source_url: string; numeric: string[]; }

const PRICEY = /close|price|adj|nav|value|amount|mrr|rate|last/i;

function pct(v: number | null | undefined): string {
  return v == null ? "--" : `${(v * 100).toFixed(2)}%`;
}
function num(v: number | null | undefined, d = 2): string {
  return v == null ? "--" : v.toFixed(d);
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

type Tab = "overview" | "indicators" | "ai";

export default function FinancePanel({ node }: { node?: string }) {
  const [series, setSeries] = useState<Series[]>([]);
  const [sel, setSel] = useState<string>("");
  const [col, setCol] = useState<string>("");
  const [win, setWin] = useState<number>(20);
  const [ppy, setPpy] = useState<number>(252);
  const [res, setRes] = useState<FinanceResult | null>(null);
  const [ind, setInd] = useState<IndicatorsResult | null>(null);
  const [aiRes, setAiRes] = useState<AiSummaryResult | null>(null);
  const [aiQuestion, setAiQuestion] = useState<string>("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [aiBusy, setAiBusy] = useState(false);
  const [tab, setTab] = useState<Tab>("overview");

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
              if (numeric.length) out.push({ full_name: t.full_name, source_url: t.source_url, numeric });
            }
          }
        }
        if (cancelled) return;
        setSeries(out);
        const pick = out.find((s) => PRICEY.test(s.full_name) || s.numeric.some((n) => PRICEY.test(n))) ?? out[0];
        if (pick) {
          setSel(pick.full_name);
          setCol(pick.numeric.find((n) => PRICEY.test(n)) ?? pick.numeric[0]);
        }
      } catch (e) { if (!cancelled) setErr(String(e)); }
    })();
    return () => { cancelled = true; };
  }, [node]);

  const current = useMemo(() => series.find((s) => s.full_name === sel), [series, sel]);

  const run = useCallback(async () => {
    if (!current || !col) return;
    setBusy(true); setErr("");
    try {
      const [finRes, indRes] = await Promise.all([
        finance(current.source_url, col, { window: win, periods_per_year: ppy, node }),
        getIndicators(current.source_url, col, { bb_period: win, node }),
      ]);
      setRes(finRes);
      setInd(indRes);
    } catch (e) { setErr(String(e)); setRes(null); setInd(null); } finally { setBusy(false); }
  }, [current, col, win, ppy, node]);

  useEffect(() => { if (current && col) run(); }, [current, col, win, ppy, run]);

  const handleAiAnalyze = async () => {
    if (!current || !col) return;
    setAiBusy(true); setAiRes(null);
    try {
      setAiRes(await aiSummary(current.source_url, col, aiQuestion || null));
    } catch (e) { setErr(String(e)); } finally { setAiBusy(false); }
  };

  const m = res?.metrics;
  const labels = res?.index ?? [];
  const indLabels = ind?.index ?? [];

  const TABS: { key: Tab; label: string }[] = [
    { key: "overview", label: "Overview" },
    { key: "indicators", label: "Indicators" },
    { key: "ai", label: "AI Analysis" },
  ];

  return (
    <div className="flex flex-col gap-3 overflow-auto min-h-0 pr-1 animate-in">
      {/* Controls */}
      <div className="glass-card p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Series</span>
          <select value={sel} onChange={(e) => { setSel(e.target.value); const s = series.find((x) => x.full_name === e.target.value); if (s) setCol(s.numeric[0]); }}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30 min-w-[180px]">
            {series.length === 0 && <option>no numeric tables</option>}
            {series.map((s) => <option key={s.full_name} value={s.full_name}>{s.full_name}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Column</span>
          <select value={col} onChange={(e) => setCol(e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
            {(current?.numeric ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Window</span>
          <input type="number" value={win} min={2} onChange={(e) => setWin(Math.max(2, +e.target.value || 20))}
            className="w-20 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Periods/yr</span>
          <select value={ppy} onChange={(e) => setPpy(+e.target.value)}
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30">
            <option value={252}>252 (daily)</option>
            <option value={52}>52 (weekly)</option>
            <option value={12}>12 (monthly)</option>
          </select>
        </div>
        {busy && <span className="text-[11px] text-muted">computing…</span>}
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-border pb-0">
        {TABS.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)}
            className={`px-4 py-2 text-xs font-semibold rounded-t-lg transition-all duration-150 border-b-2 -mb-px ${
              tab === t.key
                ? "text-frost border-frost bg-frost/5"
                : "text-muted border-transparent hover:text-foreground"
            }`}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Overview tab ─────────────────────────────────── */}
      {tab === "overview" && (
        <>
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
          {res && (
            <div className="glass-card p-3">
              <div className="flex items-center gap-3 mb-1">
                <span className="text-[11px] uppercase tracking-wide text-muted">{col} · price</span>
                <span className="text-[10px] text-amber/80">— EMA({res.window})</span>
              </div>
              <Chart type="line" labels={labels} values={res.value} overlay={res.ema} color="var(--frost)" yLabel={col} height={220} />
            </div>
          )}
          {res && (
            <div className="glass-card p-3">
              <span className="text-[11px] uppercase tracking-wide text-muted">Drawdown (peak-to-trough)</span>
              <Chart type="area" labels={labels} values={res.drawdown} color="var(--rose)" yLabel="dd" height={150} />
            </div>
          )}
          {!res && !err && <div className="glass-card p-4 text-muted text-xs">Pick a numeric series to compute risk metrics.</div>}
        </>
      )}

      {/* ── Indicators tab ───────────────────────────────── */}
      {tab === "indicators" && (
        <>
          {!ind && !err && <div className="glass-card p-4 text-muted text-xs">Select a series to compute indicators.</div>}
          {ind && (
            <>
              {/* Bollinger Bands */}
              <div className="glass-card p-3">
                <div className="flex items-center gap-3 mb-1">
                  <span className="text-[11px] uppercase tracking-wide text-muted">Bollinger Bands (BB{win})</span>
                  <span className="text-[10px] text-muted/60">upper · mid · lower</span>
                </div>
                <Chart type="line" labels={indLabels} values={ind.bb_mid} overlay={ind.bb_upper} color="var(--frost)" yLabel="price" height={200} />
              </div>

              {/* RSI */}
              <div className="glass-card p-3">
                <div className="flex items-center gap-3 mb-1">
                  <span className="text-[11px] uppercase tracking-wide text-muted">RSI (14)</span>
                  <span className="text-[10px] text-rose/60">— overbought 70</span>
                  <span className="text-[10px] text-emerald/60">— oversold 30</span>
                </div>
                <Chart type="line" labels={indLabels} values={ind.rsi} color="var(--amber)" yLabel="RSI" height={150} />
              </div>

              {/* MACD */}
              <div className="glass-card p-3">
                <div className="flex items-center gap-3 mb-1">
                  <span className="text-[11px] uppercase tracking-wide text-muted">MACD (12,26,9)</span>
                  <span className="text-[10px] text-frost/60">— line</span>
                  <span className="text-[10px] text-amber/60">— signal</span>
                </div>
                <Chart type="bar" labels={indLabels} values={ind.macd_hist} overlay={ind.macd_line} color="var(--emerald)" yLabel="MACD" height={150} />
              </div>
            </>
          )}
        </>
      )}

      {/* ── AI Analysis tab ──────────────────────────────── */}
      {tab === "ai" && (
        <div className="flex flex-col gap-3">
          <div className="glass-card p-3 flex flex-col gap-2">
            <span className="text-[10px] uppercase tracking-wider text-muted">Question (optional)</span>
            <input
              type="text"
              value={aiQuestion}
              onChange={(e) => setAiQuestion(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") handleAiAnalyze(); }}
              placeholder="e.g. Is this a good investment? What are the key risks?"
              className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-xs font-mono outline-none focus:border-frost/30 text-foreground placeholder-muted/50"
            />
            <button
              onClick={handleAiAnalyze}
              disabled={!current || !col || aiBusy}
              className="self-start px-4 py-2 rounded-lg text-xs font-semibold bg-frost/10 text-frost border border-frost/20 hover:bg-frost/20 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
            >
              {aiBusy ? "Analyzing…" : "Analyze with AI"}
            </button>
          </div>
          {aiRes && (
            <div className="glass-card p-4 flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <span className="text-[10px] uppercase tracking-wider text-muted">AI Summary</span>
                <span className="text-[9px] font-mono text-muted/50 ml-auto">{aiRes.model}</span>
              </div>
              <p className="text-sm text-foreground/90 leading-relaxed">{aiRes.summary}</p>
              {aiRes.key_points.length > 0 && (
                <ul className="space-y-1.5">
                  {aiRes.key_points.map((pt, i) => (
                    <li key={i} className="flex items-start gap-2 text-xs text-foreground/80">
                      <span className="text-frost mt-0.5 shrink-0">▸</span>
                      <span>{pt}</span>
                    </li>
                  ))}
                </ul>
              )}
              {aiRes.chart_hint && (
                <div className="flex items-center gap-2 text-xs text-amber/80 mt-1">
                  <span className="text-muted">Suggested chart:</span>
                  <span className="font-mono">{aiRes.chart_hint}</span>
                </div>
              )}
              {aiRes.error && <p className="text-xs text-rose/80 font-mono">{aiRes.error}</p>}
            </div>
          )}
          {!aiRes && !aiBusy && (
            <div className="glass-card p-4 text-muted text-xs">
              Click &quot;Analyze with AI&quot; to get a Claude-powered analysis of the selected series.
              {!current && " Select a series first."}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
