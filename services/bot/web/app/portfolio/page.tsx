"use client";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "@/lib/api";
import { fmt, fmtPct, fmtCompact, signalBadge } from "@/lib/utils";
import type { Portfolio, PnL, Trade } from "@/lib/types";
import { PlusCircle, TrendingUp, TrendingDown } from "lucide-react";

export default function PortfolioPage() {
  const qc = useQueryClient();
  const { data: portfolio } = useQuery({ queryKey: ["portfolio"], queryFn: () => api.portfolio() });
  const { data: pnl } = useQuery({ queryKey: ["pnl"], queryFn: () => api.pnl() });
  const { data: trades = [] } = useQuery({ queryKey: ["trades"], queryFn: () => api.trades() });

  const [form, setForm] = useState({ symbol: "", side: "buy" as "buy" | "sell", quantity: "" });
  const [error, setError] = useState("");

  const trade = useMutation({
    mutationFn: () =>
      api.trade(1, { symbol: form.symbol, side: form.side, quantity: Number(form.quantity) }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["portfolio"] });
      qc.invalidateQueries({ queryKey: ["pnl"] });
      qc.invalidateQueries({ queryKey: ["trades"] });
      setForm({ symbol: "", side: "buy", quantity: "" });
      setError("");
    },
    onError: (e: Error) => setError(e.message),
  });

  const positions = Object.values(portfolio?.positions ?? {});

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 space-y-6">
      <h1 className="text-xl font-bold text-slate-100">Portfolio</h1>

      {/* PnL summary */}
      {pnl && <PnLSummary pnl={pnl} />}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Positions */}
        <div className="lg:col-span-2 rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-400 uppercase tracking-wider">Open Positions</h2>
          {positions.length === 0 ? (
            <p className="text-sm text-slate-500">No open positions. Place a trade to get started.</p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-800/60 text-xs text-slate-500">
                  <th className="pb-2 text-left">Symbol</th>
                  <th className="pb-2 text-right">Qty</th>
                  <th className="pb-2 text-right">Avg Cost</th>
                  <th className="pb-2 text-right">Price</th>
                  <th className="pb-2 text-right">Market Val</th>
                  <th className="pb-2 text-right">P&amp;L</th>
                  <th className="pb-2 text-right">Wt%</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => (
                  <tr key={p.symbol} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                    <td className="py-2 font-medium text-slate-200">{p.symbol}</td>
                    <td className="py-2 text-right text-slate-300">{p.quantity.toFixed(4)}</td>
                    <td className="py-2 text-right text-slate-300">${fmt(p.avg_cost)}</td>
                    <td className="py-2 text-right text-slate-300">${fmt(p.current_price)}</td>
                    <td className="py-2 text-right text-slate-300">${fmtCompact(p.market_value)}</td>
                    <td className={`py-2 text-right ${p.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                      {fmtPct(p.unrealized_pnl_pct)}
                    </td>
                    <td className="py-2 text-right text-slate-400">{p.weight.toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Trade form */}
        <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
          <h2 className="mb-4 flex items-center gap-2 text-sm font-semibold text-slate-400 uppercase tracking-wider">
            <PlusCircle className="h-4 w-4" /> New Trade
          </h2>
          <div className="space-y-3">
            <input
              value={form.symbol}
              onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })}
              placeholder="Symbol (e.g. AAPL)"
              className="w-full rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
            />
            <div className="grid grid-cols-2 gap-2">
              {(["buy", "sell"] as const).map((side) => (
                <button
                  key={side}
                  onClick={() => setForm({ ...form, side })}
                  className={`rounded-lg py-2 text-sm font-medium transition-colors ${
                    form.side === side
                      ? side === "buy"
                        ? "bg-green-500/30 text-green-300 border border-green-500/40"
                        : "bg-red-500/30 text-red-300 border border-red-500/40"
                      : "bg-slate-800/60 text-slate-400 border border-slate-700 hover:border-slate-600"
                  }`}
                >
                  {side.toUpperCase()}
                </button>
              ))}
            </div>
            <input
              type="number"
              value={form.quantity}
              onChange={(e) => setForm({ ...form, quantity: e.target.value })}
              placeholder="Quantity"
              min="0"
              step="0.0001"
              className="w-full rounded-lg border border-slate-700 bg-slate-800/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-indigo-500 focus:outline-none"
            />
            {error && <p className="text-xs text-red-400">{error}</p>}
            <button
              onClick={() => trade.mutate()}
              disabled={!form.symbol || !form.quantity || trade.isPending}
              className="w-full rounded-lg bg-indigo-600 py-2 text-sm font-medium text-white transition-colors hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {trade.isPending ? "Placing…" : "Place Trade (Market Price)"}
            </button>
            {portfolio && (
              <p className="text-xs text-slate-500">Cash available: ${fmtCompact(portfolio.cash)}</p>
            )}
          </div>
        </div>
      </div>

      {/* Trade history */}
      <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
        <h2 className="mb-4 text-sm font-semibold text-slate-400 uppercase tracking-wider">Recent Trades</h2>
        {trades.length === 0 ? (
          <p className="text-sm text-slate-500">No trades yet.</p>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800/60 text-xs text-slate-500">
                <th className="pb-2 text-left">Time</th>
                <th className="pb-2 text-left">Symbol</th>
                <th className="pb-2 text-left">Side</th>
                <th className="pb-2 text-right">Qty</th>
                <th className="pb-2 text-right">Price</th>
                <th className="pb-2 text-right">Notional</th>
              </tr>
            </thead>
            <tbody>
              {[...trades].reverse().map((t) => (
                <tr key={t.id} className="border-b border-slate-800/30 hover:bg-slate-800/20">
                  <td className="py-2 text-slate-500">{new Date(t.timestamp).toLocaleTimeString()}</td>
                  <td className="py-2 font-medium text-slate-200">{t.symbol}</td>
                  <td className={`py-2 font-medium ${t.side === "buy" ? "text-green-400" : "text-red-400"}`}>
                    {t.side.toUpperCase()}
                  </td>
                  <td className="py-2 text-right text-slate-300">{t.quantity.toFixed(4)}</td>
                  <td className="py-2 text-right text-slate-300">${fmt(t.price)}</td>
                  <td className="py-2 text-right text-slate-300">${fmtCompact(t.quantity * t.price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function PnLSummary({ pnl }: { pnl: PnL }) {
  const items = [
    { label: "Total Value",    value: `$${fmtCompact(pnl.total_value)}`,   color: "text-slate-200" },
    { label: "Cash",           value: `$${fmtCompact(pnl.cash)}`,          color: "text-slate-300" },
    { label: "Invested",       value: `$${fmtCompact(pnl.invested)}`,      color: "text-slate-300" },
    { label: "Unrealized P&L", value: fmtPct(pnl.unrealized_pnl_pct),     color: pnl.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400" },
    { label: "Total Return",   value: fmtPct(pnl.total_pnl_pct),          color: pnl.total_pnl >= 0 ? "text-green-400" : "text-red-400" },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {items.map(({ label, value, color }) => (
        <div key={label} className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-4">
          <p className="text-xs text-slate-500">{label}</p>
          <p className={`mt-1 text-lg font-bold ${color}`}>{value}</p>
        </div>
      ))}
    </div>
  );
}
