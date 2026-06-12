"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmt, fmtPct, fmtCompact } from "@/lib/utils";
import { TrendingUp, TrendingDown } from "lucide-react";

export function PortfolioCard() {
  const { data: pnl, isLoading } = useQuery({
    queryKey: ["pnl"],
    queryFn: () => api.pnl(1),
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  if (isLoading || !pnl) {
    return <div className="h-32 animate-pulse rounded-xl bg-slate-800/50" />;
  }

  const up = pnl.total_pnl >= 0;
  const Icon = up ? TrendingUp : TrendingDown;

  return (
    <div className="card-hover rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Portfolio</h2>
        <Icon className={`h-4 w-4 ${up ? "text-green-400" : "text-red-400"}`} />
      </div>

      <div className="mb-4">
        <p className="text-2xl font-bold text-slate-100">${fmtCompact(pnl.total_value)}</p>
        <p className={`text-sm ${up ? "text-green-400" : "text-red-400"}`}>
          {fmtPct(pnl.total_pnl_pct)} total return
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3 border-t border-slate-800/60 pt-4">
        <Stat label="Cash" value={`$${fmtCompact(pnl.cash)}`} />
        <Stat label="Invested" value={`$${fmtCompact(pnl.invested)}`} />
        <Stat
          label="Unrealized"
          value={`${pnl.unrealized_pnl >= 0 ? "+" : ""}$${fmtCompact(pnl.unrealized_pnl)}`}
          color={pnl.unrealized_pnl >= 0 ? "text-green-400" : "text-red-400"}
        />
      </div>
    </div>
  );
}

function Stat({ label, value, color = "text-slate-200" }: { label: string; value: string; color?: string }) {
  return (
    <div>
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`text-sm font-semibold ${color}`}>{value}</p>
    </div>
  );
}
