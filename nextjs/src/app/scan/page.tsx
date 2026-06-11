"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import type { ScanEntry } from "@/lib/types";

function signalBar(sig: number) {
  const pct = Math.round(((sig + 1) / 2) * 100);
  const color = sig > 0.2 ? "bg-emerald-500" : sig < -0.2 ? "bg-red-500" : "bg-gray-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-800 rounded-full h-1.5 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs w-10 text-right ${sig > 0.2 ? "text-emerald-400" : sig < -0.2 ? "text-red-400" : "text-gray-400"}`}>
        {sig > 0 ? "+" : ""}{sig.toFixed(2)}
      </span>
    </div>
  );
}

function rsiColor(rsi: number | null | undefined) {
  if (rsi == null) return "text-gray-500";
  if (rsi < 30) return "text-emerald-400";
  if (rsi > 70) return "text-red-400";
  return "text-gray-300";
}

export default function ScanPage() {
  const [input, setInput] = useState("");
  const [column, setColumn] = useState("close");
  const [results, setResults] = useState<ScanEntry[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    const paths = input.split(/[\n,]+/).map(p => p.trim()).filter(Boolean);
    if (!paths.length) { setError("Enter at least one path"); return; }
    setLoading(true); setError("");
    try {
      const res = await api.scan(paths, column);
      setResults(res.results);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const sorted = results
    ? [...results].sort((a, b) => (b.signal ?? -2) - (a.signal ?? -2))
    : null;

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Signal Scan</h1>
      <p className="text-gray-400 text-sm">
        Scan multiple parquet/CSV files and rank them by composite signal (EMA cross + RSI + MACD).
      </p>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={"prices/aapl.parquet\nprices/msft.parquet\nprices/goog.parquet"}
          rows={4}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono resize-y"
        />
        <div className="flex gap-2 items-center">
          <label className="text-gray-400 text-sm shrink-0">Price column</label>
          <input value={column} onChange={e => setColumn(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-28" />
          <button onClick={run} disabled={loading}
            className="ml-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-1.5 rounded-lg text-sm font-medium">
            {loading ? "Scanning…" : "Scan"}
          </button>
        </div>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-lg px-3 py-2">{error}</div>}

      {sorted && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
          <div className="grid grid-cols-[1fr_80px_60px_60px_60px_140px] gap-x-3 px-4 py-2 bg-gray-800 text-xs text-gray-500 uppercase tracking-wider">
            <span>Path</span>
            <span className="text-right">Price</span>
            <span className="text-right">EMA</span>
            <span className="text-right">RSI</span>
            <span className="text-right">MACD</span>
            <span>Signal</span>
          </div>
          {sorted.map((r, i) => (
            <div key={i}
              className="grid grid-cols-[1fr_80px_60px_60px_60px_140px] gap-x-3 px-4 py-2.5 border-t border-gray-800 items-center hover:bg-gray-800/50 text-sm">
              <span className="font-mono text-xs text-gray-300 truncate" title={r.path}>
                {r.path.split("/").pop() ?? r.path}
              </span>
              {r.error ? (
                <span className="col-span-5 text-red-400 text-xs">{r.error}</span>
              ) : (
                <>
                  <span className="text-right text-gray-200">{r.price?.toFixed(2) ?? "–"}</span>
                  <span className={`text-right text-xs ${r.ema9 && r.ema21 ? (r.ema9 > r.ema21 ? "text-emerald-400" : "text-red-400") : "text-gray-500"}`}>
                    {r.ema9 && r.ema21 ? (r.ema9 > r.ema21 ? "↑" : "↓") : "–"}
                  </span>
                  <span className={`text-right text-xs ${rsiColor(r.rsi)}`}>
                    {r.rsi?.toFixed(0) ?? "–"}
                  </span>
                  <span className={`text-right text-xs ${r.macd_hist ? (r.macd_hist > 0 ? "text-emerald-400" : "text-red-400") : "text-gray-500"}`}>
                    {r.macd_hist ? (r.macd_hist > 0 ? "+" : "") + r.macd_hist.toFixed(3) : "–"}
                  </span>
                  <div>{r.signal != null ? signalBar(r.signal) : <span className="text-gray-500 text-xs">–</span>}</div>
                </>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
