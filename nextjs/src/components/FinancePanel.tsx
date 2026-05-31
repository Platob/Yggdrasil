"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getCatalogs, getSchemas, getTables, finance,
  type TableEntry, type FinanceResult,
} from "@/lib/api";
import Chart from "@/components/Chart";

// A flat, finance-ready view of a registered table: its file/mount path plus the
// numeric columns worth treating as a price/return series.
interface Series { full_name: string; source_url: string; numeric: string[]; }

const NUMERIC = /int|float|double|decimal|number|real/i;
const PRICEY = /close|price|adj|nav|value|amount|mrr|rate|last/i;

function pct(v: number | null | undefined): string {
  return v == null ? "--" : `${(v * 100).toFixed(2)}%`;
}
function num(v: number | null | undefined, d = 2): string {
  return v == null ? "--" : v.toFixed(d);
}

// A risk metric tile — green when "good", rose when adverse, so the strip reads
// at a glance the way a desk would expect.
function Metric({ label, value, tone = "neutral", hint }: { label: string; value: string; tone?: "good" | "bad" | "neutral"; hint?: string }) {
  const color = tone === "good" ? "text-emerald" : tone === "bad" ? "text-rose/90" : "text-foreground";
  return (
    <div className="glass-card p-3 flex flex-col gap-0.5" title={hint}>
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className={`font-mono text-lg font-semibold ${color}`}>{value}</span>
    </div>
  );
}

export default function FinancePanel({ node }: { node?: string }) {
  const [series, setSeries] = useState<Series[]>([]);
  const [sel, setSel] = useState<string>("");          // full_name
  const [col, setCol] = useState<string>("");
  const [win, setWin] = useState<number>(20);
  const [ppy, setPpy] = useState<number>(252);
  const [res, setRes] = useState<FinanceResult | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  // Discover finance-ready tables by walking the (small) catalog tree once.
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
        // Auto-pick a price-like table + column for an instant, meaningful view.
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
      setRes(await finance(current.source_url, col, { window: win, periods_per_year: ppy, node }));
    } catch (e) { setErr(String(e)); setRes(null); } finally { setBusy(false); }
  }, [current, col, win, ppy, node]);

  // Auto-run whenever the selection settles.
  useEffect(() => { if (current && col) run(); }, [current, col, win, ppy, run]);

  const m = res?.metrics;
  const labels = res?.index ?? [];

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

      {/* Risk metrics strip */}
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

      {/* Price + EMA */}
      {res && (
        <div className="glass-card p-3">
          <div className="flex items-center gap-3 mb-1">
            <span className="text-[11px] uppercase tracking-wide text-muted">{col} · price</span>
            <span className="text-[10px] text-amber/80">— EMA({res.window})</span>
          </div>
          <Chart type="line" labels={labels} values={res.value} overlay={res.ema} color="var(--frost)" yLabel={col} height={220} />
        </div>
      )}

      {/* Drawdown */}
      {res && (
        <div className="glass-card p-3">
          <span className="text-[11px] uppercase tracking-wide text-muted">Drawdown (peak-to-trough)</span>
          <Chart type="area" labels={labels} values={res.drawdown} color="var(--rose)" yLabel="dd" height={150} />
        </div>
      )}

      {!res && !err && <div className="glass-card p-4 text-muted text-xs">Pick a numeric series to compute risk metrics.</div>}
    </div>
  );
}
