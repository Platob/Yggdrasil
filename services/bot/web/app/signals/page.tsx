"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { signalBadge, signalLabel, cn, fmt } from "@/lib/utils";
import type { Signal } from "@/lib/types";
import { PriceChart } from "@/components/dashboard/PriceChart";

const DEFAULT_SYMBOLS = ["AAPL", "MSFT", "NVDA", "META", "TSLA", "GOOGL", "AMZN", "BTC-USD", "ETH-USD", "SPY", "QQQ"];

export default function SignalsPage() {
  const [selected, setSelected] = useState<string>("AAPL");

  const { data: signals = [], isLoading } = useQuery({
    queryKey: ["signals-scan-full"],
    queryFn: () => api.scanSignals(DEFAULT_SYMBOLS),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });

  const { data: detail } = useQuery({
    queryKey: ["signal-detail", selected],
    queryFn: () => api.signal(selected, "3mo"),
    enabled: !!selected,
    staleTime: 60_000,
  });

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 space-y-6">
      <h1 className="text-xl font-bold text-slate-100">Technical Signals</h1>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Signal list */}
        <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
          <h2 className="mb-4 text-sm font-semibold text-slate-400 uppercase tracking-wider">Signal Scan</h2>
          {isLoading ? (
            <div className="space-y-2">
              {[...Array(8)].map((_, i) => <div key={i} className="h-12 animate-pulse rounded bg-slate-800/50" />)}
            </div>
          ) : (
            <div className="space-y-1">
              {signals.map((s) => (
                <button
                  key={s.symbol}
                  onClick={() => setSelected(s.symbol)}
                  className={cn(
                    "w-full flex items-center justify-between rounded-lg px-3 py-2.5 text-left transition-colors",
                    selected === s.symbol ? "bg-indigo-500/15 border border-indigo-500/30" : "hover:bg-slate-800/40 border border-transparent"
                  )}
                >
                  <span className="text-sm font-medium text-slate-200">{s.symbol}</span>
                  <div className="flex items-center gap-2">
                    <div className="h-1 w-12 rounded-full bg-slate-700">
                      <div className="h-full rounded-full bg-indigo-400/70" style={{ width: `${s.confidence * 100}%` }} />
                    </div>
                    <span className={cn("rounded border px-1.5 py-0.5 text-xs", signalBadge(s.direction))}>
                      {signalLabel(s.direction)}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Signal detail */}
        <div className="lg:col-span-2 space-y-4">
          {detail && <SignalDetail signal={detail} />}
        </div>
      </div>
    </div>
  );
}

function SignalDetail({ signal: s }: { signal: Signal }) {
  return (
    <div className="rounded-xl border border-slate-800/60 bg-slate-900/50 p-5 space-y-5">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-slate-100">{s.symbol}</h2>
        <span className={cn("rounded-lg border px-3 py-1.5 text-sm font-semibold", signalBadge(s.direction))}>
          {signalLabel(s.direction)}
        </span>
      </div>

      <PriceChart symbol={s.symbol} period="3mo" />

      {/* Targets */}
      {(s.price_target || s.stop_loss) && (
        <div className="grid grid-cols-2 gap-3">
          {s.price_target && (
            <div className="rounded-lg bg-green-500/10 border border-green-500/20 p-3">
              <p className="text-xs text-green-500/70">Price Target</p>
              <p className="text-lg font-bold text-green-400">${fmt(s.price_target)}</p>
            </div>
          )}
          {s.stop_loss && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3">
              <p className="text-xs text-red-500/70">Stop Loss</p>
              <p className="text-lg font-bold text-red-400">${fmt(s.stop_loss)}</p>
            </div>
          )}
        </div>
      )}

      {/* Indicators */}
      <div>
        <h3 className="mb-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">Indicators</h3>
        <div className="space-y-2">
          {s.indicators.map((ind) => (
            <div key={ind.name} className="flex items-center justify-between rounded-lg bg-slate-800/40 px-3 py-2">
              <div>
                <span className="text-sm font-medium text-slate-200">{ind.name}</span>
                {ind.description && <p className="text-xs text-slate-500">{ind.description}</p>}
              </div>
              <span className={cn("rounded border px-2 py-0.5 text-xs font-medium", signalBadge(ind.signal))}>
                {signalLabel(ind.signal)}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Confidence */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-500">Confidence</span>
        <div className="flex-1 h-1.5 rounded-full bg-slate-700">
          <div className="h-full rounded-full bg-indigo-400" style={{ width: `${s.confidence * 100}%` }} />
        </div>
        <span className="text-xs font-medium text-slate-300">{(s.confidence * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
}
