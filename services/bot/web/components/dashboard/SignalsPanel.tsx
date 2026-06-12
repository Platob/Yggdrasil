"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { signalBadge, signalLabel, cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import type { Signal } from "@/lib/types";

const SCAN_SYMBOLS = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "BTC-USD", "ETH-USD", "SPY"];

function DirectionIcon({ d }: { d: string }) {
  if (d === "strong_buy" || d === "buy") return <TrendingUp className="h-3.5 w-3.5 text-green-400" />;
  if (d === "strong_sell" || d === "sell") return <TrendingDown className="h-3.5 w-3.5 text-red-400" />;
  return <Minus className="h-3.5 w-3.5 text-slate-400" />;
}

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="h-1 w-20 rounded-full bg-slate-700/60">
      <div
        className="h-full rounded-full bg-indigo-400/70"
        style={{ width: `${value * 100}%` }}
      />
    </div>
  );
}

export function SignalsPanel() {
  const { data: signals = [], isLoading } = useQuery({
    queryKey: ["signals-scan"],
    queryFn: () => api.scanSignals(SCAN_SYMBOLS),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });

  const sorted = [...signals].sort((a, b) => b.confidence - a.confidence);

  return (
    <div className="card-hover rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
      <h2 className="mb-4 text-sm font-semibold text-slate-300 uppercase tracking-wider">
        Technical Signals
      </h2>
      {isLoading ? (
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-10 animate-pulse rounded bg-slate-800/50" />
          ))}
        </div>
      ) : (
        <div className="space-y-2">
          {sorted.map((s) => (
            <SignalRow key={s.symbol} signal={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function SignalRow({ signal: s }: { signal: Signal }) {
  return (
    <div className="flex items-center justify-between rounded-lg px-3 py-2 hover:bg-slate-800/40 transition-colors">
      <div className="flex items-center gap-2">
        <DirectionIcon d={s.direction} />
        <span className="text-sm font-medium text-slate-200">{s.symbol}</span>
      </div>
      <div className="flex items-center gap-3">
        <ConfidenceBar value={s.confidence} />
        <span className={cn("rounded border px-2 py-0.5 text-xs font-medium", signalBadge(s.direction))}>
          {signalLabel(s.direction)}
        </span>
      </div>
    </div>
  );
}
