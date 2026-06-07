"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Wallet } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cancelOrder, getPortfolio, getPortfolioSummary, getTrades } from "@/lib/api";
import type { Portfolio, PortfolioSummary, Trade } from "@/lib/types";
import { fmtDate, fmtNum, fmtPct, fmtPnl, fmtPrice, pnlColor } from "@/lib/format";
import { ErrorBanner, Kpi, Panel, SideBadge, Spinner } from "@/components/ui";

const PORTFOLIO_ID = 1;
const PAGE_SIZE = 50;

export default function PortfolioPage() {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [page, setPage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [cancelling, setCancelling] = useState<number | null>(null);

  const load = useCallback(async () => {
    try {
      const [pf, sum, trd] = await Promise.all([
        getPortfolio(PORTFOLIO_ID),
        getPortfolioSummary(PORTFOLIO_ID),
        getTrades(PORTFOLIO_ID, 500),
      ]);
      setPortfolio(pf);
      setSummary(sum);
      setTrades(trd);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  const onCancel = async (orderId: number) => {
    setCancelling(orderId);
    try {
      await cancelOrder(PORTFOLIO_ID, orderId);
      await load();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setCancelling(null);
    }
  };

  // Build the equity curve from chronologically-ordered trade P&L, anchored to
  // the starting cash (equity minus cumulative realized P&L).
  const equityCurve = useMemo(() => {
    if (!trades.length) return [];
    const sorted = [...trades].sort((a, b) => a.ts - b.ts);
    const totalPnl = sorted.reduce((acc, t) => acc + t.pnl, 0);
    const base = (summary?.equity ?? 0) - totalPnl;
    let running = base;
    return sorted.map((t) => {
      running += t.pnl;
      return { ts: t.ts, equity: running };
    });
  }, [trades, summary]);

  const pagedTrades = useMemo(() => {
    const sorted = [...trades].sort((a, b) => b.ts - a.ts);
    return sorted.slice(page * PAGE_SIZE, page * PAGE_SIZE + PAGE_SIZE);
  }, [trades, page]);

  const totalPages = Math.max(1, Math.ceil(trades.length / PAGE_SIZE));

  return (
    <div className="p-4 md:p-6 pt-20 md:pt-6 space-y-6">
      <header className="flex items-center gap-2">
        <Wallet className="text-green-400" size={22} />
        <h1 className="text-xl font-semibold text-gray-100">Portfolio</h1>
      </header>

      {error && <ErrorBanner message={error} />}

      {loading && !portfolio ? (
        <Spinner />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <Kpi label="Equity" value={fmtPrice(summary?.equity ?? 0)} />
            <Kpi label="Cash" value={fmtPrice(summary?.cash ?? 0)} />
            <Kpi
              label="Total P&L"
              value={fmtPnl(summary?.total_pnl ?? 0)}
              colorBySign
              signValue={summary?.total_pnl ?? 0}
            />
            <Kpi
              label="Daily P&L"
              value={fmtPnl(summary?.daily_pnl ?? 0)}
              colorBySign
              signValue={summary?.daily_pnl ?? 0}
            />
            <Kpi label="Win Rate" value={fmtPct((summary?.win_rate ?? 0) * 100)} />
          </div>

          <Panel title="Equity Curve">
            <div className="p-4" style={{ height: 280 }}>
              {equityCurve.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={equityCurve}
                    margin={{ top: 8, right: 8, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid stroke="#1f2937" vertical={false} />
                    <XAxis
                      dataKey="ts"
                      tickFormatter={(t) => fmtDate(t).split(",")[0]}
                      tick={{ fill: "#6b7280", fontSize: 10 }}
                      axisLine={{ stroke: "#1f2937" }}
                      tickLine={false}
                      minTickGap={50}
                    />
                    <YAxis
                      orientation="right"
                      domain={["auto", "auto"]}
                      tick={{ fill: "#6b7280", fontSize: 10 }}
                      axisLine={{ stroke: "#1f2937" }}
                      tickLine={false}
                      tickFormatter={(v) => fmtNum(v, 0)}
                      width={64}
                    />
                    <Tooltip
                      cursor={{ stroke: "#374151" }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        const d = payload[0].payload as { ts: number; equity: number };
                        return (
                          <div className="rounded-md border border-gray-700 bg-gray-900/95 px-3 py-2 text-xs font-mono">
                            <div className="text-gray-400">{fmtDate(d.ts)}</div>
                            <div className="text-gray-100">{fmtPrice(d.equity)}</div>
                          </div>
                        );
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-gray-600 text-sm">
                  No trade history
                </div>
              )}
            </div>
          </Panel>

          <Panel title="Positions">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                    <th className="px-4 py-2 font-medium">Symbol</th>
                    <th className="px-4 py-2 font-medium">Side</th>
                    <th className="px-4 py-2 font-medium text-right">Qty</th>
                    <th className="px-4 py-2 font-medium text-right">Avg Entry</th>
                    <th className="px-4 py-2 font-medium text-right">Price</th>
                    <th className="px-4 py-2 font-medium text-right">uP&L</th>
                    <th className="px-4 py-2 font-medium text-right">%</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio?.positions.map((p) => {
                    const pct =
                      p.avg_entry !== 0
                        ? ((p.current_price - p.avg_entry) / p.avg_entry) *
                          100 *
                          (p.side?.toLowerCase() === "short" ? -1 : 1)
                        : 0;
                    return (
                      <tr
                        key={p.id}
                        className="border-b border-gray-800/60 hover:bg-gray-800/30"
                      >
                        <td className="px-4 py-2 font-medium text-gray-200">{p.symbol}</td>
                        <td className="px-4 py-2">
                          <SideBadge side={p.side} />
                        </td>
                        <td className="px-4 py-2 text-right font-mono">{p.qty}</td>
                        <td className="px-4 py-2 text-right font-mono">
                          {fmtPrice(p.avg_entry)}
                        </td>
                        <td className="px-4 py-2 text-right font-mono">
                          {fmtPrice(p.current_price)}
                        </td>
                        <td
                          className={`px-4 py-2 text-right font-mono ${pnlColor(
                            p.unrealized_pnl,
                          )}`}
                        >
                          {fmtPnl(p.unrealized_pnl)}
                        </td>
                        <td className={`px-4 py-2 text-right font-mono ${pnlColor(pct)}`}>
                          {fmtPct(pct)}
                        </td>
                      </tr>
                    );
                  })}
                  {!portfolio?.positions.length && (
                    <tr>
                      <td colSpan={7} className="px-4 py-6 text-center text-gray-600">
                        No open positions
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel title="Open Orders">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                    <th className="px-4 py-2 font-medium">Symbol</th>
                    <th className="px-4 py-2 font-medium">Side</th>
                    <th className="px-4 py-2 font-medium">Type</th>
                    <th className="px-4 py-2 font-medium text-right">Qty</th>
                    <th className="px-4 py-2 font-medium text-right">Price</th>
                    <th className="px-4 py-2 font-medium">Status</th>
                    <th className="px-4 py-2 font-medium text-right">Action</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolio?.open_orders.map((o) => (
                    <tr
                      key={o.id}
                      className="border-b border-gray-800/60 hover:bg-gray-800/30"
                    >
                      <td className="px-4 py-2 font-medium text-gray-200">{o.symbol}</td>
                      <td className="px-4 py-2">
                        <SideBadge side={o.side} />
                      </td>
                      <td className="px-4 py-2 text-gray-400 uppercase text-xs">
                        {o.type}
                      </td>
                      <td className="px-4 py-2 text-right font-mono">{o.qty}</td>
                      <td className="px-4 py-2 text-right font-mono">
                        {o.price != null ? fmtPrice(o.price) : "MKT"}
                      </td>
                      <td className="px-4 py-2 text-gray-400 text-xs">{o.status}</td>
                      <td className="px-4 py-2 text-right">
                        <button
                          onClick={() => onCancel(o.id)}
                          disabled={cancelling === o.id}
                          className="px-2 py-1 rounded text-xs bg-red-500/15 text-red-400 hover:bg-red-500/25 disabled:opacity-50"
                        >
                          {cancelling === o.id ? "…" : "Cancel"}
                        </button>
                      </td>
                    </tr>
                  ))}
                  {!portfolio?.open_orders.length && (
                    <tr>
                      <td colSpan={7} className="px-4 py-6 text-center text-gray-600">
                        No open orders
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>

          <Panel
            title="Trade History"
            action={
              <div className="flex items-center gap-2 text-xs text-gray-400">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="px-2 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700"
                >
                  Prev
                </button>
                <span className="font-mono">
                  {page + 1} / {totalPages}
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="px-2 py-1 rounded bg-gray-800 disabled:opacity-40 hover:bg-gray-700"
                >
                  Next
                </button>
              </div>
            }
          >
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                    <th className="px-4 py-2 font-medium">Time</th>
                    <th className="px-4 py-2 font-medium">Symbol</th>
                    <th className="px-4 py-2 font-medium">Side</th>
                    <th className="px-4 py-2 font-medium text-right">Qty</th>
                    <th className="px-4 py-2 font-medium text-right">Price</th>
                    <th className="px-4 py-2 font-medium text-right">Fee</th>
                    <th className="px-4 py-2 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {pagedTrades.map((t) => (
                    <tr
                      key={t.id}
                      className="border-b border-gray-800/60 hover:bg-gray-800/30"
                    >
                      <td className="px-4 py-2 font-mono text-gray-400 text-xs">
                        {fmtDate(t.ts)}
                      </td>
                      <td className="px-4 py-2 font-medium text-gray-200">{t.symbol}</td>
                      <td className="px-4 py-2">
                        <SideBadge side={t.side} />
                      </td>
                      <td className="px-4 py-2 text-right font-mono">{t.qty}</td>
                      <td className="px-4 py-2 text-right font-mono">{fmtPrice(t.price)}</td>
                      <td className="px-4 py-2 text-right font-mono text-gray-400">
                        {fmtPrice(t.fee)}
                      </td>
                      <td className={`px-4 py-2 text-right font-mono ${pnlColor(t.pnl)}`}>
                        {fmtPnl(t.pnl)}
                      </td>
                    </tr>
                  ))}
                  {!pagedTrades.length && (
                    <tr>
                      <td colSpan={7} className="px-4 py-6 text-center text-gray-600">
                        No trades
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
