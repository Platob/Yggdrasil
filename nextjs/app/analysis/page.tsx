"use client";
import { useState } from "react";
import FinancePanel from "@/components/FinancePanel";
import DrawdownChart from "@/components/DrawdownChart";
import CumReturnChart from "@/components/CumReturnChart";
import { financeAnalysis } from "@/lib/api";
import type { FinanceResult } from "@/lib/types";

export default function AnalysisPage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("price");
  const [rfr, setRfr] = useState("0.0");
  const [result, setResult] = useState<FinanceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze(e: React.FormEvent) {
    e.preventDefault();
    if (!path.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const r = await financeAnalysis({
        path: path.trim(),
        column: column.trim() || "price",
        risk_free_rate: parseFloat(rfr) || 0,
      });
      setResult(r);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <h1 className="text-zinc-100 text-xl font-semibold">Finance Analysis</h1>

      {/* input form */}
      <form onSubmit={analyze} className="bg-zinc-900 rounded-lg border border-zinc-800 p-5 space-y-4">
        <p className="text-zinc-400 text-sm">
          Point at a parquet or CSV file on the node to compute risk/return metrics.
        </p>
        <div className="grid grid-cols-3 gap-4">
          <div className="col-span-2">
            <label className="block text-zinc-500 text-xs mb-1">File path (relative to node home)</label>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              placeholder="data/prices.parquet"
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-600"
            />
          </div>
          <div>
            <label className="block text-zinc-500 text-xs mb-1">Price column</label>
            <input
              value={column}
              onChange={(e) => setColumn(e.target.value)}
              placeholder="price"
              className="w-full bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-600"
            />
          </div>
        </div>
        <div className="flex items-end gap-4">
          <div>
            <label className="block text-zinc-500 text-xs mb-1">Risk-free rate (annualised)</label>
            <input
              value={rfr}
              onChange={(e) => setRfr(e.target.value)}
              placeholder="0.0"
              className="w-28 bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:border-emerald-600"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !path.trim()}
            className="px-6 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm rounded transition-colors"
          >
            {loading ? "Analysing…" : "Analyse"}
          </button>
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
      </form>

      {result && (
        <div className="space-y-5">
          <FinancePanel metrics={result.metrics} />
          <div className="grid grid-cols-2 gap-4">
            <CumReturnChart data={result.cum_return} />
            <DrawdownChart data={result.drawdown} />
          </div>
          {result.ema.length > 0 && (
            <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
              <h3 className="text-zinc-400 text-xs font-medium mb-1">EMA(20) — last value</h3>
              <p className="text-emerald-400 text-2xl font-semibold tabular-nums">
                {result.ema[result.ema.length - 1].toFixed(4)}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
