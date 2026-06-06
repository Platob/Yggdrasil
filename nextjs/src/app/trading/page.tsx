"use client";

import { useCallback, useState } from "react";
import Chart from "@/components/Chart";
import FinancePanel from "@/components/FinancePanel";
import {
  analysisIndicators, analysisTradingSignals, streamAiAnalyze,
  type IndicatorResult, type SignalResult, type TradingSignal,
} from "@/lib/api";

// Signal chips read like a desk would expect: emerald = bullish, rose = bearish.
function dirColor(d: string): string {
  return d === "bullish" ? "text-emerald" : d === "bearish" ? "text-rose/90" : "text-muted";
}
function dirBg(d: string): string {
  return d === "bullish" ? "rgba(52,211,153,0.1)" : d === "bearish" ? "rgba(244,63,94,0.1)" : "rgba(255,255,255,0.05)";
}

function BiasBadge({ bias }: { bias: string }) {
  return (
    <span className={`text-xs font-bold uppercase tracking-wide px-2 py-1 rounded ${dirColor(bias)}`} style={{ background: dirBg(bias) }}>
      {bias}
    </span>
  );
}

function StatTile({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="glass-card p-2.5 flex flex-col gap-0.5" title={hint}>
      <span className="text-[10px] uppercase tracking-wider text-muted">{label}</span>
      <span className="font-mono text-base font-semibold text-foreground">{value}</span>
    </div>
  );
}

export default function TradingPage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [x, setX] = useState("");
  const [node, setNode] = useState("");

  const [ind, setInd] = useState<IndicatorResult | null>(null);
  const [sig, setSig] = useState<SignalResult | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const [question, setQuestion] = useState("Analyze this trading data and identify key signals and risks.");
  const [aiText, setAiText] = useState("");
  const [aiBusy, setAiBusy] = useState(false);

  const analyze = useCallback(async () => {
    if (!path || !column) { setErr("Enter a file path and a price column."); return; }
    setBusy(true); setErr(""); setAiText("");
    const opts = { x: x || undefined, node: node || undefined };
    try {
      const [i, s] = await Promise.all([
        analysisIndicators(path, column, { ...opts, indicators: ["rsi", "macd", "bb", "sma", "ema"] }),
        analysisTradingSignals(path, column, { ...opts, last_n: 1000 }),
      ]);
      setInd(i); setSig(s);
    } catch (e) {
      setErr(String(e)); setInd(null); setSig(null);
    } finally { setBusy(false); }
  }, [path, column, x, node]);

  const askAi = useCallback(async () => {
    if (!sig || !ind) return;
    const summary = JSON.stringify({
      column: ind.column, rows: ind.source_rows, bias: sig.bias,
      current_rsi: sig.current_rsi, current_macd: sig.current_macd,
      current_bb_pct: sig.current_bb_pct,
      recent_signals: sig.signals.slice(-12).map((g) => ({ signal: g.signal, direction: g.direction, x: g.x, value: g.value })),
    });
    setAiBusy(true); setAiText("");
    try {
      for await (const chunk of streamAiAnalyze(summary, question)) {
        setAiText((prev) => prev + chunk);
      }
    } catch (e) {
      setAiText(`AI analysis failed: ${String(e)}`);
    } finally { setAiBusy(false); }
  }, [sig, ind, question]);

  const labels = ind?.x ?? [];

  return (
    <div className="p-6 max-w-[1500px] mx-auto flex flex-col gap-5 animate-in">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Trading Analysis</h1>
          <p className="text-xs text-muted mt-0.5">Technical indicators, signal detection and AI commentary over a price series.</p>
        </div>
      </header>

      {/* File selector row */}
      <div className="glass-card p-3 flex flex-wrap items-end gap-3">
        <div className="flex flex-col gap-1 flex-1 min-w-[220px]">
          <span className="text-[10px] uppercase tracking-wider text-muted">File path</span>
          <input value={path} onChange={(e) => setPath(e.target.value)} placeholder="data/prices.parquet"
            className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Price column</span>
          <input value={column} onChange={(e) => setColumn(e.target.value)} placeholder="close"
            className="w-32 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Time column</span>
          <input value={x} onChange={(e) => setX(e.target.value)} placeholder="(optional)"
            className="w-32 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-muted">Node</span>
          <input value={node} onChange={(e) => setNode(e.target.value)} placeholder="(local)"
            className="w-32 bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs font-mono outline-none focus:border-frost/30" />
        </div>
        <button onClick={analyze} disabled={busy}
          className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-frost/15 text-frost border border-frost/30 hover:bg-frost/25 disabled:opacity-50 transition-colors">
          {busy ? "Analyzing…" : "Analyze"}
        </button>
      </div>

      {err && <div className="glass-card p-3 text-rose/80 font-mono text-xs break-words">{err}</div>}

      {/* Main section: charts (2/3) + signals/AI (1/3) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: charts */}
        <div className="lg:col-span-2 flex flex-col gap-5">
          {ind && (
            <div className="glass-card p-3">
              <div className="flex items-center gap-3 mb-1 flex-wrap">
                <span className="text-[11px] uppercase tracking-wide text-muted">{ind.column} · price</span>
                <span className="text-[10px] text-amber/80">— SMA / EMA</span>
                <span className="text-[10px] text-frost/70">— Bollinger band</span>
              </div>
              <Chart type="line" labels={labels} values={ind.close}
                band={ind.bb_upper && ind.bb_lower ? { min: ind.bb_lower, max: ind.bb_upper } : undefined}
                overlay={ind.ema ?? undefined} color="var(--frost)" yLabel={ind.column} height={260} />
            </div>
          )}

          {ind?.rsi && (
            <div className="glass-card p-3">
              <div className="flex items-center gap-3 mb-1">
                <span className="text-[11px] uppercase tracking-wide text-muted">RSI ({"14"})</span>
                <span className="text-[10px] text-rose/70">70 overbought</span>
                <span className="text-[10px] text-emerald/70">30 oversold</span>
              </div>
              <Chart type="line" labels={labels} values={ind.rsi}
                band={{ min: ind.rsi.map(() => 30), max: ind.rsi.map(() => 70) }}
                color="var(--amber)" yLabel="RSI" height={150} />
            </div>
          )}

          {ind?.macd && (
            <div className="glass-card p-3">
              <div className="flex items-center gap-3 mb-1">
                <span className="text-[11px] uppercase tracking-wide text-muted">MACD</span>
                <span className="text-[10px] text-muted">histogram + signal line</span>
              </div>
              <Chart type="bar" labels={labels} values={ind.macd_hist ?? []} overlay={ind.macd_signal ?? undefined} color="var(--frost)" yLabel="MACD" height={150} />
              <Chart type="line" labels={labels} values={ind.macd} overlay={ind.macd_signal ?? undefined} color="var(--emerald)" yLabel="" height={120} />
            </div>
          )}

          {!ind && !busy && (
            <div className="glass-card p-6 text-muted text-xs">Enter a file path and price column, then Analyze to see indicators.</div>
          )}
        </div>

        {/* Right: signals + AI */}
        <div className="flex flex-col gap-5">
          {sig && (
            <div className="glass-card p-3 flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] uppercase tracking-wide text-muted">Signal bias</span>
                <BiasBadge bias={sig.bias} />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <StatTile label="RSI" value={sig.current_rsi == null ? "--" : sig.current_rsi.toFixed(1)} />
                <StatTile label="MACD" value={sig.current_macd == null ? "--" : sig.current_macd.toFixed(3)} />
                <StatTile label="BB %" value={sig.current_bb_pct == null ? "--" : `${(sig.current_bb_pct * 100).toFixed(0)}%`} hint="position within the Bollinger band" />
              </div>
            </div>
          )}

          {sig && (
            <div className="glass-card p-3 flex flex-col gap-2 max-h-[360px] overflow-auto">
              <span className="text-[11px] uppercase tracking-wide text-muted">Detected signals ({sig.signals.length})</span>
              {sig.signals.length === 0 && <span className="text-xs text-muted">No signals in the recent window.</span>}
              {sig.signals.slice().reverse().slice(0, 40).map((g: TradingSignal, k) => (
                <div key={k} className="flex items-center gap-2 text-xs rounded px-2 py-1.5" style={{ background: dirBg(g.direction) }}>
                  <span className={`font-semibold ${dirColor(g.direction)}`}>{g.signal}</span>
                  <span className="text-muted truncate">{g.label}</span>
                  <span className="ml-auto font-mono text-[10px] text-muted">{String(g.x).slice(0, 12)}</span>
                </div>
              ))}
            </div>
          )}

          {sig && (
            <div className="glass-card p-3 flex flex-col gap-2">
              <span className="text-[11px] uppercase tracking-wide text-muted">AI analysis</span>
              <textarea value={question} onChange={(e) => setQuestion(e.target.value)} rows={3}
                className="bg-white/[0.04] border border-white/[0.08] rounded-lg px-2 py-1.5 text-xs outline-none focus:border-frost/30 resize-none" />
              <button onClick={askAi} disabled={aiBusy}
                className="self-start px-3 py-1.5 rounded-lg text-xs font-semibold bg-frost/15 text-frost border border-frost/30 hover:bg-frost/25 disabled:opacity-50 transition-colors">
                {aiBusy ? "Thinking…" : "Ask AI"}
              </button>
              {aiText && <div className="text-xs text-foreground-dim whitespace-pre-wrap leading-relaxed border-t border-white/[0.06] pt-2">{aiText}</div>}
            </div>
          )}
        </div>
      </div>

      {/* Risk metrics over registered series */}
      <div className="flex flex-col gap-2">
        <h2 className="text-sm font-semibold text-muted uppercase tracking-wide">Risk metrics</h2>
        <FinancePanel node={node || undefined} />
      </div>
    </div>
  );
}
