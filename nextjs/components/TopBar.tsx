"use client";
import useSWR from "swr";
import { getStats } from "@/lib/api";
import type { Stats } from "@/lib/types";

function pct(n: number) { return `${n.toFixed(1)}%`; }

export default function TopBar() {
  const { data } = useSWR<Stats>("stats", getStats, { refreshInterval: 5000 });
  return (
    <header className="h-12 shrink-0 flex items-center justify-between px-5 bg-zinc-950 border-b border-zinc-800 text-sm">
      <span className="text-zinc-400 text-xs font-mono">
        {data?.node_id ?? "—"}
      </span>
      {data ? (
        <div className="flex items-center gap-4">
          <Badge label="CPU" value={pct(data.cpu_pct)} warn={data.cpu_pct > 80} />
          <Badge label="MEM" value={pct(data.mem_pct)} warn={data.mem_pct > 85} />
          <span className="text-zinc-600 text-xs">{fmtUptime(data.uptime_s)}</span>
        </div>
      ) : (
        <span className="text-zinc-600 text-xs">connecting…</span>
      )}
    </header>
  );
}

function Badge({ label, value, warn }: { label: string; value: string; warn: boolean }) {
  return (
    <span className="flex items-center gap-1 text-xs">
      <span className="text-zinc-500">{label}</span>
      <span className={warn ? "text-amber-400" : "text-zinc-300"}>{value}</span>
    </span>
  );
}

function fmtUptime(s: number) {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}
