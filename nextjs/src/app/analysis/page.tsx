"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import type { FinanceResult } from "@/lib/types";

function KPI({ label, value, good }: { label: string; value: string; good?: boolean | null }) {
  return (
    <div className="bg-gray-800 rounded-lg p-3">
      <div className="text-gray-500 text-xs uppercase tracking-wide">{label}</div>
      <div className={`text-xl font-bold mt-1 ${good === true ? "text-emerald-400" : good === false ? "text-red-400" : "text-gray-100"}`}>
        {value}
      </div>
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
    } catch(e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Finance Analysis</h1>
      <div className="flex gap-3 flex-wrap">
        <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path to price data..."
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
        <input value={column} onChange={e => setColumn(e.target.value)} placeholder="Column"
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm w-32" />
        <button onClick={analyze} disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium">
          {loading ? "…" : "Analyze"}
        </button>
      </div>
      {error && <div className="text-red-400 text-sm">{error}</div>}
      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <KPI label="Total Return" value={pct(result.total_return)} good={result.total_return > 0} />
            <KPI label="CAGR" value={pct(result.cagr)} good={result.cagr > 0} />
            <KPI label="Sharpe" value={result.sharpe.toFixed(2)} good={result.sharpe > 1} />
            <KPI label="Sortino" value={result.sortino.toFixed(2)} good={result.sortino > 1} />
            <KPI label="Ann. Volatility" value={pct(result.ann_volatility)} />
            <KPI label="Max Drawdown" value={pct(result.max_drawdown)} good={false} />
            <KPI label="Calmar" value={result.calmar.toFixed(2)} good={result.calmar > 1} />
            <KPI label="Ann. Return" value={pct(result.ann_return)} good={result.ann_return > 0} />
          </div>
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-sm text-gray-400 mb-2">Drawdown curve</div>
            <svg viewBox="0 0 600 80" className="w-full h-20">
              {(() => {
                const dd = result.drawdown;
                const min = Math.min(...dd);
                const pts = dd.map((v, i) => `${(i/(dd.length-1))*600},${80-(v/min)*80}`).join(" L ");
                return <path d={`M ${pts}`} fill="none" stroke="#f43f5e" strokeWidth="1.5" />;
              })()}
            </svg>
          </div>
        </>
      )}
    </div>
  );
}
