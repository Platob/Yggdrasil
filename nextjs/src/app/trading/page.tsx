"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { LineChart, overlayPath } from "@/components/Chart";
import type { IndicatorsResult, BacktestResult } from "@/lib/types";

const STRATEGIES = ["ema_cross", "rsi_mean_reversion", "macd", "buy_and_hold"] as const;

export default function TradingPage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [strategy, setStrategy] = useState<typeof STRATEGIES[number]>("ema_cross");
  const [indicators, setIndicators] = useState<IndicatorsResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!path) { setError("Enter a file path"); return; }
    setLoading(true); setError("");
    try {
      const [ind, bt] = await Promise.all([
        api.indicators(path, column),
        api.backtest(path, column, strategy),
      ]);
      setIndicators(ind);
      setBacktest(bt);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trading</h1>

      {/* Controls */}
      <div className="flex gap-3 flex-wrap">
        <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path to parquet/CSV..."
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
        <input value={column} onChange={e => setColumn(e.target.value)} placeholder="Price column"
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm w-36" />
        <select value={strategy} onChange={e => setStrategy(e.target.value as any)}
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm">
          {STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <button onClick={run} disabled={loading}
          className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-2 rounded-lg text-sm font-medium">
          {loading ? "Running…" : "Analyze"}
        </button>
      </div>

      {error && <div className="text-red-400 text-sm">{error}</div>}

      {indicators && (
        <div className="space-y-4">
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-sm text-gray-400 mb-2">Price + EMA 9/21</div>
            <div className="relative">
              <LineChart data={indicators.price} color="#6366f1" />
              <div className="absolute inset-0 pointer-events-none">
                <svg viewBox="0 0 600 200" className="w-full h-full">
                  <path d={overlayPath(indicators.ema_9, indicators.price)}
                    fill="none" stroke="#f59e0b" strokeWidth="1" />
                  <path d={overlayPath(indicators.ema_21, indicators.price)}
                    fill="none" stroke="#ec4899" strokeWidth="1" />
                </svg>
              </div>
            </div>
          </div>
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-sm text-gray-400 mb-2">RSI (14) — dashed at 30/70</div>
            <LineChart data={indicators.rsi_14.map(v => v ?? 0)} color="#22d3ee" height={100} />
          </div>
        </div>
      )}

      {backtest && (
        <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
          <div className="text-sm text-gray-400 mb-3">Backtest: {backtest.strategy}</div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4 text-sm">
            {[
              ["Return", `${(backtest.total_return * 100).toFixed(2)}%`],
              ["Sharpe", backtest.sharpe.toFixed(2)],
              ["Max DD", `${(backtest.max_drawdown * 100).toFixed(2)}%`],
              ["Trades", backtest.n_trades],
              ["Win Rate", `${(backtest.win_rate * 100).toFixed(1)}%`],
              ["vs BnH", `${((backtest.total_return - backtest.benchmark_return) * 100).toFixed(2)}%`],
            ].map(([k, v]) => (
              <div key={k as string} className="bg-gray-800 rounded-lg p-2">
                <div className="text-gray-500 text-xs">{k}</div>
                <div className="font-semibold">{v}</div>
              </div>
            ))}
          </div>
          <div className="text-sm text-gray-400 mb-1">Equity curve vs benchmark</div>
          <div className="relative h-40">
            <LineChart data={backtest.equity_curve} color="#10b981" height={160} />
          </div>
        </div>
      )}
    </div>
  );
}
