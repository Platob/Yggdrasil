"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { LineChart } from "@/components/Chart";
import type { FinanceResult } from "@/lib/types";

function KPI({
  label, value, good, sub,
}: { label: string; value: string; good?: boolean | null; sub?: string }) {
  const color = good === true ? "text-emerald-400" : good === false ? "text-red-400" : "text-gray-100";
  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <div className="text-gray-500 text-xs uppercase tracking-wide mb-1">{label}</div>
      <div className={`text-xl font-bold ${color}`}>{value}</div>
      {sub && <div className="text-gray-600 text-xs mt-0.5">{sub}</div>}
    </div>
  );
}

export default function AnalysisPage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [result, setResult] = useState<FinanceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyze = async () => {
    if (!path) { setError("Enter a file path"); return; }
    setLoading(true); setError("");
    try {
      setResult(await api.finance(path, column));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false); }
  };

  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Finance Analysis</h1>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div className="flex gap-2 flex-wrap">
          <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path to price data…"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
          <input value={column} onChange={e => setColumn(e.target.value)} placeholder="Column"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-28" />
          <button onClick={analyze} disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-2 rounded-lg text-sm font-medium">
            {loading ? "Analyzing…" : "Analyze"}
          </button>
        </div>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-lg px-3 py-2">{error}</div>}

      {result && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPI label="Total Return" value={pct(result.total_return)} good={result.total_return > 0} />
            <KPI label="CAGR" value={pct(result.cagr)} good={result.cagr > 0} />
            <KPI label="Sharpe Ratio" value={result.sharpe.toFixed(2)} good={result.sharpe > 1}
              sub="≥1 = acceptable" />
            <KPI label="Sortino Ratio" value={result.sortino.toFixed(2)} good={result.sortino > 1}
              sub="downside-only vol" />
            <KPI label="Ann. Volatility" value={pct(result.ann_volatility)} />
            <KPI label="Max Drawdown" value={pct(result.max_drawdown)} good={false} />
            <KPI label="Calmar Ratio" value={result.calmar.toFixed(2)} good={result.calmar > 1}
              sub="CAGR / max DD" />
            <KPI label="Ann. Return" value={pct(result.ann_return)} good={result.ann_return > 0} />
          </div>

          {result.ema.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Price · EMA</div>
              <div className="relative">
                <LineChart data={result.ema} color="#6366f1" fill height={180} />
              </div>
            </div>
          )}

          {result.drawdown.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Drawdown</div>
              <LineChart data={result.drawdown} color="#f43f5e" fill height={100} />
              <div className="text-xs text-gray-600 mt-1">
                Max {pct(result.max_drawdown)} · Calmar {result.calmar.toFixed(2)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
