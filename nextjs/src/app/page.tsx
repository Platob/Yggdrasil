"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

function uptime(s: number) {
  if (s < 60) return `${Math.floor(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <div className="text-gray-500 text-xs uppercase tracking-wider mb-1">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
      {sub && <div className="text-gray-500 text-xs mt-0.5">{sub}</div>}
    </div>
  );
}

function BackendBadge({ name, status }: { name: string; status: string }) {
  const online = status === "online";
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm ${
      online ? "border-emerald-800 bg-emerald-950/30" : "border-gray-800 bg-gray-900"
    }`}>
      <span className={`w-1.5 h-1.5 rounded-full ${online ? "bg-emerald-400" : "bg-gray-600"}`} />
      <span className={online ? "text-emerald-300" : "text-gray-500"}>{name}</span>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<{ uptime_s: number; requests: number } | null>(null);
  const [backends, setBackends] = useState<{ name: string; status: string }[] | null>(null);
  const [nodeStatus, setNodeStatus] = useState<"online" | "offline" | "loading">("loading");

  useEffect(() => {
    api.ping()
      .then(() => setNodeStatus("online"))
      .catch(() => setNodeStatus("offline"));
    api.stats().then(setStats).catch(() => {});
    api.backend().then(setBackends).catch(() => {});
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        <StatusBadge status={nodeStatus} />
      </div>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Uptime" value={uptime(stats.uptime_s)} />
          <StatCard label="Requests" value={stats.requests.toLocaleString()} />
        </div>
      )}

      {backends && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Backends</div>
          <div className="flex flex-wrap gap-2">
            {backends.map(b => <BackendBadge key={b.name} {...b} />)}
          </div>
        </div>
      )}

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Quick start</div>
        <div className="space-y-2 text-sm text-gray-400">
          <div>1. Upload a parquet or CSV price file via <a href="/files" className="text-emerald-400 hover:underline">Files</a></div>
          <div>2. Run <a href="/trading" className="text-emerald-400 hover:underline">Trading</a> to see indicators, EMA/MACD/RSI + backtest</div>
          <div>3. Use <a href="/scan" className="text-emerald-400 hover:underline">Signal Scan</a> to rank multiple datasets by composite signal</div>
          <div>4. Run <a href="/analysis" className="text-emerald-400 hover:underline">Analysis</a> for Sharpe, CAGR, and drawdown metrics</div>
        </div>
      </div>
    </div>
  );
}
