"use client";
import { useState } from "react";
import { api } from "@/lib/api";

interface PortfolioResult {
  assets: string[];
  weights: number[];
  total_return: number;
  ann_return: number;
  ann_volatility: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  weighted_return: number;
  diversification_ratio: number;
  component_sharpes: number[];
}

function fmt2(v: number) { return v.toFixed(2); }
function pct(v: number) { return `${(v * 100).toFixed(2)}%`; }
function returnColor(v: number) { return v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : ""; }

export default function PortfolioPage() {
  const [input, setInput] = useState("");
  const [weightsRaw, setWeightsRaw] = useState("");
  const [column, setColumn] = useState("close");
  const [result, setResult] = useState<PortfolioResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    const paths = input.split(/[\n,]+/).map(p => p.trim()).filter(Boolean);
    if (paths.length < 1) { setError("Enter at least one path"); return; }
    const weights = weightsRaw
      ? weightsRaw.split(/[\s,]+/).map(Number).filter(x => !isNaN(x))
      : undefined;
    if (weights && weights.length !== paths.length) {
      setError(`Expected ${paths.length} weights, got ${weights.length}`);
      return;
    }
    setLoading(true); setError("");
    try {
      const res = await fetch("/api/v2/trading/portfolio", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paths, column, weights: weights ?? null }),
      });
      if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
      setResult(await res.json());
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Portfolio</h1>
      <p className="text-gray-400 text-sm">
        Combine multiple price series into a blended portfolio. Weights default to equal allocation.
      </p>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={"prices/aapl.parquet\nprices/msft.parquet\nprices/goog.parquet"}
          rows={4}
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm font-mono resize-y"
        />
        <div className="flex gap-2 flex-wrap items-center text-sm">
          <label className="text-gray-400 shrink-0">Column</label>
          <input value={column} onChange={e => setColumn(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-28" />
          <label className="text-gray-400 shrink-0 ml-2">Weights (optional)</label>
          <input value={weightsRaw} onChange={e => setWeightsRaw(e.target.value)}
            placeholder="0.5, 0.3, 0.2"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-40" />
          <button onClick={run} disabled={loading}
            className="ml-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-1.5 rounded-lg text-sm font-medium">
            {loading ? "Computing…" : "Compute"}
          </button>
        </div>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-lg px-3 py-2">{error}</div>}

      {result && (
        <div className="space-y-4">
          {/* Portfolio KPIs */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {[
              ["Total Return", pct(result.total_return), returnColor(result.total_return)],
              ["Ann. Return", pct(result.ann_return), returnColor(result.ann_return)],
              ["Ann. Volatility", pct(result.ann_volatility), ""],
              ["Max Drawdown", pct(result.max_drawdown), "text-red-400"],
              ["Sharpe", fmt2(result.sharpe), result.sharpe > 1 ? "text-emerald-400" : ""],
              ["Sortino", fmt2(result.sortino), result.sortino > 1 ? "text-emerald-400" : ""],
              ["Div. Ratio", result.diversification_ratio.toFixed(3), ""],
              ["Weighted Return", pct(result.weighted_return), returnColor(result.weighted_return)],
            ].map(([label, value, color]) => (
              <div key={label as string} className="bg-gray-900 border border-gray-800 rounded-lg p-2.5">
                <div className="text-gray-500 text-xs mb-0.5">{label}</div>
                <div className={`font-semibold text-sm ${color as string}`}>{value}</div>
              </div>
            ))}
          </div>

          {/* Asset breakdown */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
            <div className="px-4 py-2 bg-gray-800 text-xs text-gray-500 uppercase tracking-wider">
              Component assets
            </div>
            <div className="divide-y divide-gray-800">
              {result.assets.map((asset, i) => (
                <div key={asset} className="flex items-center px-4 py-3 gap-4 text-sm">
                  <span className="font-mono text-gray-300 truncate flex-1">{asset}</span>
                  <div className="flex items-center gap-1 w-32">
                    <div className="flex-1 bg-gray-800 rounded-full h-1.5 overflow-hidden">
                      <div className="h-full bg-indigo-500 rounded-full"
                        style={{ width: `${(result.weights[i] * 100).toFixed(0)}%` }} />
                    </div>
                    <span className="text-gray-400 text-xs w-10 text-right">
                      {(result.weights[i] * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span className={`text-xs w-14 text-right ${result.component_sharpes[i] > 1 ? "text-emerald-400" : "text-gray-400"}`}>
                    SR {result.component_sharpes[i].toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {result.diversification_ratio > 1 && (
            <div className="text-xs text-gray-500 bg-emerald-950/20 border border-emerald-900/50 rounded-lg px-3 py-2">
              Diversification ratio {result.diversification_ratio.toFixed(3)} &gt; 1 — the portfolio has lower
              volatility than its weighted-average component volatility.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
