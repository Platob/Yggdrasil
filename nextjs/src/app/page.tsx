"use client";

import { useCallback, useEffect, useState } from "react";
import { Activity } from "lucide-react";
import {
  getAssets,
  getPortfolio,
  getPortfolioSummary,
  getTick,
  getTrades,
} from "@/lib/api";
import type { AssetInfo, Portfolio, PortfolioSummary, Tick, Trade } from "@/lib/types";
import { fmtDate, fmtPct, fmtPnl, fmtPrice, fmtTime, pnlColor } from "@/lib/format";
import { ErrorBanner, Kpi, Panel, SideBadge, Spinner } from "@/components/ui";

const PORTFOLIO_ID = 1;

export default function DashboardPage() {
  const [now, setNow] = useState(() => Date.now());
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [ticks, setTicks] = useState<Record<string, Tick>>({});
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const load = useCallback(async () => {
    try {
      const [assetList, pf, sum, trd] = await Promise.all([
        getAssets(),
        getPortfolio(PORTFOLIO_ID),
        getPortfolioSummary(PORTFOLIO_ID),
        getTrades(PORTFOLIO_ID, 10),
      ]);
      setAssets(assetList);
      setPortfolio(pf);
      setSummary(sum);
      setTrades(trd);
      setError(null);

      const tickResults = await Promise.allSettled(
        assetList.map((a) => getTick(a.symbol)),
      );
      const map: Record<string, Tick> = {};
      tickResults.forEach((r, i) => {
        if (r.status === "fulfilled") map[assetList[i].symbol] = r.value;
      });
      setTicks(map);
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

  return (
    <div className="p-4 md:p-6 pt-20 md:pt-6 space-y-6">
      <header className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="text-green-400" size={22} />
          <h1 className="text-xl font-semibold text-gray-100">Dashboard</h1>
        </div>
        <div className="text-sm font-mono text-gray-400">{fmtTime(now)}</div>
      </header>

      {error && <ErrorBanner message={error} />}

      {loading && !portfolio ? (
        <Spinner />
      ) : (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Kpi label="Equity" value={fmtPrice(summary?.equity ?? 0)} />
            <Kpi
              label="Daily P&L"
              value={fmtPnl(summary?.daily_pnl ?? 0)}
              colorBySign
              signValue={summary?.daily_pnl ?? 0}
            />
            <Kpi label="Positions" value={String(summary?.position_count ?? 0)} />
            <Kpi
              label="Win Rate"
              value={fmtPct((summary?.win_rate ?? 0) * 100)}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Panel title="Market Overview">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                      <th className="px-4 py-2 font-medium">Symbol</th>
                      <th className="px-4 py-2 font-medium text-right">Price</th>
                      <th className="px-4 py-2 font-medium text-right">Side</th>
                    </tr>
                  </thead>
                  <tbody>
                    {assets.map((a) => {
                      const t = ticks[a.symbol];
                      return (
                        <tr
                          key={a.symbol}
                          className="border-b border-gray-800/60 hover:bg-gray-800/30"
                        >
                          <td className="px-4 py-2">
                            <div className="font-medium text-gray-200">{a.symbol}</div>
                            <div className="text-[11px] text-gray-500">{a.name}</div>
                          </td>
                          <td className="px-4 py-2 text-right font-mono">
                            {t ? fmtPrice(t.price) : "—"}
                          </td>
                          <td className="px-4 py-2 text-right">
                            {t ? <SideBadge side={t.side} /> : "—"}
                          </td>
                        </tr>
                      );
                    })}
                    {!assets.length && (
                      <tr>
                        <td colSpan={3} className="px-4 py-6 text-center text-gray-600">
                          No assets
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Panel>

            <Panel title="Active Positions">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                      <th className="px-4 py-2 font-medium">Symbol</th>
                      <th className="px-4 py-2 font-medium text-right">Qty</th>
                      <th className="px-4 py-2 font-medium text-right">Price</th>
                      <th className="px-4 py-2 font-medium text-right">uP&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {portfolio?.positions.map((p) => (
                      <tr
                        key={p.id}
                        className="border-b border-gray-800/60 hover:bg-gray-800/30"
                      >
                        <td className="px-4 py-2">
                          <span className="font-medium text-gray-200">{p.symbol}</span>{" "}
                          <SideBadge side={p.side} />
                        </td>
                        <td className="px-4 py-2 text-right font-mono">{p.qty}</td>
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
                      </tr>
                    ))}
                    {!portfolio?.positions.length && (
                      <tr>
                        <td colSpan={4} className="px-4 py-6 text-center text-gray-600">
                          No open positions
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </Panel>
          </div>

          <Panel title="Recent Trades">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 border-b border-gray-800">
                    <th className="px-4 py-2 font-medium">Time</th>
                    <th className="px-4 py-2 font-medium">Symbol</th>
                    <th className="px-4 py-2 font-medium">Side</th>
                    <th className="px-4 py-2 font-medium text-right">Qty</th>
                    <th className="px-4 py-2 font-medium text-right">Price</th>
                    <th className="px-4 py-2 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t) => (
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
                      <td
                        className={`px-4 py-2 text-right font-mono ${pnlColor(t.pnl)}`}
                      >
                        {fmtPnl(t.pnl)}
                      </td>
                    </tr>
                  ))}
                  {!trades.length && (
                    <tr>
                      <td colSpan={6} className="px-4 py-6 text-center text-gray-600">
                        No trades yet
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
