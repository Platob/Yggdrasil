"use client";
import useSWR from "swr";
import PriceChart from "@/components/PriceChart";
import { getMarketSymbols, getMonitor } from "@/lib/api";
import type { MarketSymbol, MonitorSnapshot } from "@/lib/types";

const DEMO_SYMBOLS: MarketSymbol[] = [
  { symbol: "BTC/USD", price: 67_420.15, change_pct: 2.34,  volume: 28_400 },
  { symbol: "ETH/USD", price:  3_812.50, change_pct: -1.12, volume: 14_200 },
  { symbol: "SPY",     price:    535.22, change_pct: 0.41,  volume: 89_200 },
  { symbol: "QQQ",     price:    456.88, change_pct: -0.23, volume: 52_100 },
  { symbol: "GLD",     price:    225.40, change_pct: 0.89,  volume: 11_300 },
];

export default function DashboardPage() {
  const { data: symbols } = useSWR(
    "symbols",
    () => getMarketSymbols().then((r) => r.symbols).catch(() => DEMO_SYMBOLS),
    { refreshInterval: 10_000, fallbackData: DEMO_SYMBOLS },
  );
  const { data: monitor } = useSWR<MonitorSnapshot>(
    "monitor",
    () => getMonitor().catch(() => null as unknown as MonitorSnapshot),
    { refreshInterval: 5000 },
  );

  return (
    <div className="space-y-6 max-w-6xl mx-auto">
      <h1 className="text-zinc-100 text-xl font-semibold">Dashboard</h1>

      {/* KPI row */}
      <div className="grid grid-cols-4 gap-4">
        <KpiCard label="Portfolio Value" value="$142,381" sub="+2.1% today" positive />
        <KpiCard label="Day P&L" value="+$2,941" sub="realised + unrealised" positive />
        <KpiCard label="Open Positions" value="7" sub="across 5 instruments" />
        <KpiCard label="CPU / MEM" value={monitor ? `${monitor.cpu_pct.toFixed(0)}% / ${monitor.mem_pct.toFixed(0)}%` : "—"} sub="node health" />
      </div>

      {/* live chart */}
      <PriceChart />

      {/* market table */}
      <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
        <div className="px-4 py-3 border-b border-zinc-800">
          <h2 className="text-zinc-200 text-sm font-medium">Market Overview</h2>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-zinc-800">
              {["Symbol", "Price", "Change", "Volume"].map((h) => (
                <th key={h} className="text-left px-4 py-2 text-zinc-500 text-xs font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(symbols ?? DEMO_SYMBOLS).map((s) => (
              <tr key={s.symbol} className="border-b border-zinc-900 hover:bg-zinc-800/40 transition-colors">
                <td className="px-4 py-2.5 font-medium text-zinc-200">{s.symbol}</td>
                <td className="px-4 py-2.5 font-mono text-zinc-100">{s.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                <td className={`px-4 py-2.5 font-mono ${s.change_pct >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                  {s.change_pct >= 0 ? "▲" : "▼"} {Math.abs(s.change_pct).toFixed(2)}%
                </td>
                <td className="px-4 py-2.5 text-zinc-500 text-xs">{s.volume.toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function KpiCard({ label, value, sub, positive }: { label: string; value: string; sub: string; positive?: boolean }) {
  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <p className="text-zinc-500 text-xs mb-1">{label}</p>
      <p className={`text-2xl font-semibold ${positive ? "text-emerald-400" : "text-zinc-100"}`}>{value}</p>
      <p className="text-zinc-600 text-xs mt-1">{sub}</p>
    </div>
  );
}
