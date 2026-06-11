"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { DualLineChart, LineChart, MacdChart, RsiChart, overlayPath } from "@/components/Chart";
import type { BacktestResult, IndicatorsResult } from "@/lib/types";

const STRATEGIES = [
  { id: "ema_cross", label: "EMA Cross" },
  { id: "rsi_mean_reversion", label: "RSI Mean Reversion" },
  { id: "macd", label: "MACD" },
  { id: "buy_and_hold", label: "Buy & Hold" },
];

const SIZING = ["full", "half", "quarter"] as const;

function fmt(v: number, decimals = 2) { return v.toFixed(decimals); }
function pct(v: number) { return `${(v * 100).toFixed(2)}%`; }

function KpiGrid({ items }: { items: [string, string | number, string?][] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
      {items.map(([label, value, color]) => (
        <div key={label} className="bg-gray-800 rounded-lg p-2.5">
          <div className="text-gray-500 text-xs mb-0.5">{label}</div>
          <div className={`font-semibold text-sm ${color ?? ""}`}>{value}</div>
        </div>
      ))}
    </div>
  );
}

export default function TradingPage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [strategy, setStrategy] = useState("ema_cross");
  const [initialCash, setInitialCash] = useState(10000);
  const [stopLoss, setStopLoss] = useState("");
  const [takeProfit, setTakeProfit] = useState("");
  const [sizing, setSizing] = useState<typeof SIZING[number]>("full");
  const [indicators, setIndicators] = useState<IndicatorsResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    if (!path) { setError("Enter a file path"); return; }
    setLoading(true); setError("");
    try {
      const sl = stopLoss ? parseFloat(stopLoss) / 100 : undefined;
      const tp = takeProfit ? parseFloat(takeProfit) / 100 : undefined;
      const [ind, bt] = await Promise.all([
        api.indicators(path, column),
        api.backtest(path, column, strategy, initialCash, sl, tp, sizing),
      ]);
      setIndicators(ind);
      setBacktest(bt);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const returnColor = (v: number) => v > 0 ? "text-emerald-400" : v < 0 ? "text-red-400" : "";

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Trading</h1>

      {/* Controls */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <div className="flex gap-2 flex-wrap">
          <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path to parquet/CSV…"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
          <input value={column} onChange={e => setColumn(e.target.value)} placeholder="Column"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-28" />
          <select value={strategy} onChange={e => setStrategy(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm">
            {STRATEGIES.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
        </div>
        <div className="flex gap-2 flex-wrap items-center text-sm">
          <label className="text-gray-400 shrink-0">Cash $</label>
          <input type="number" value={initialCash} onChange={e => setInitialCash(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-28" />
          <label className="text-gray-400 shrink-0 ml-2">Stop loss %</label>
          <input value={stopLoss} onChange={e => setStopLoss(e.target.value)} placeholder="e.g. 5"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-20" />
          <label className="text-gray-400 shrink-0">Take profit %</label>
          <input value={takeProfit} onChange={e => setTakeProfit(e.target.value)} placeholder="e.g. 10"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-20" />
          <label className="text-gray-400 shrink-0 ml-2">Size</label>
          <select value={sizing} onChange={e => setSizing(e.target.value as typeof SIZING[number])}
            className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm">
            {SIZING.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <button onClick={run} disabled={loading}
            className="ml-auto bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-1.5 rounded-lg text-sm font-medium">
            {loading ? "Running…" : "Analyze"}
          </button>
        </div>
      </div>

      {error && <div className="text-red-400 text-sm bg-red-950/30 border border-red-900 rounded-lg px-3 py-2">{error}</div>}

      {indicators && (
        <div className="space-y-3">
          {/* Price + EMA + BB */}
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">Price · EMA 9/21 · Bollinger Bands</div>
            <div className="relative">
              <LineChart data={indicators.price} color="#6366f1" fill />
              <div className="absolute inset-0 pointer-events-none">
                <svg viewBox="0 0 600 200" className="w-full h-full">
                  <path d={overlayPath(indicators.bb_upper, indicators.price)}
                    fill="none" stroke="#374151" strokeWidth="0.8" strokeDasharray="3 3" />
                  <path d={overlayPath(indicators.bb_lower, indicators.price)}
                    fill="none" stroke="#374151" strokeWidth="0.8" strokeDasharray="3 3" />
                  <path d={overlayPath(indicators.ema_9, indicators.price)}
                    fill="none" stroke="#f59e0b" strokeWidth="1.2" />
                  <path d={overlayPath(indicators.ema_21, indicators.price)}
                    fill="none" stroke="#ec4899" strokeWidth="1.2" />
                </svg>
              </div>
            </div>
            <div className="flex gap-4 mt-1.5 text-xs text-gray-500">
              <span><span className="inline-block w-3 h-0.5 bg-amber-400 mr-1" />EMA 9</span>
              <span><span className="inline-block w-3 h-0.5 bg-pink-400 mr-1" />EMA 21</span>
              <span><span className="inline-block w-3 h-0.5 bg-gray-600 mr-1" style={{ borderTop: "1px dashed" }} />BB ±2σ</span>
            </div>
          </div>

          {/* RSI */}
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">RSI (14) — oversold &lt;30 / overbought &gt;70</div>
            <RsiChart data={indicators.rsi_14} height={90} />
          </div>

          {/* MACD */}
          <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
            <div className="text-xs text-gray-500 mb-2 uppercase tracking-wider">MACD (12/26/9) — histogram · signal · line</div>
            <MacdChart hist={indicators.macd_hist} line={indicators.macd_line} signal={indicators.macd_signal} height={80} />
          </div>
        </div>
      )}

      {backtest && (
        <div className="bg-gray-900 rounded-xl p-4 border border-gray-800 space-y-4">
          <div className="text-xs text-gray-500 uppercase tracking-wider">
            Backtest — {STRATEGIES.find(s => s.id === backtest.strategy)?.label ?? backtest.strategy}
          </div>
          <KpiGrid items={[
            ["Total Return", pct(backtest.total_return), returnColor(backtest.total_return)],
            ["vs Buy & Hold", pct(backtest.total_return - backtest.benchmark_return), returnColor(backtest.total_return - backtest.benchmark_return)],
            ["Ann. Return", pct(backtest.ann_return), returnColor(backtest.ann_return)],
            ["Max Drawdown", pct(backtest.max_drawdown), "text-red-400"],
            ["Sharpe", fmt(backtest.sharpe)],
            ["Sortino", fmt(backtest.sortino)],
            ["Trades", backtest.n_trades],
            ["Win Rate", pct(backtest.win_rate)],
            ["Profit Factor", backtest.profit_factor != null ? fmt(backtest.profit_factor) : "∞"],
            ["Avg Win", pct(backtest.avg_win_pct), "text-emerald-400"],
            ["Avg Loss", pct(backtest.avg_loss_pct), "text-red-400"],
            ["Max Consec. Losses", backtest.max_consecutive_losses],
          ]} />

          <div>
            <div className="text-xs text-gray-500 mb-1 uppercase tracking-wider">Equity vs Buy & Hold</div>
            <DualLineChart a={backtest.equity_curve} b={backtest.benchmark_equity}
              colorA="#10b981" colorB="#6b7280" height={160} />
            <div className="flex gap-4 mt-1 text-xs text-gray-500">
              <span><span className="inline-block w-3 h-0.5 bg-emerald-400 mr-1" />Strategy</span>
              <span><span className="inline-block w-3 h-0.5 bg-gray-500 mr-1 opacity-70" />Benchmark</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
