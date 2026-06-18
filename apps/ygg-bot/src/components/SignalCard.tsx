"use client";

import type { Signal } from "@/lib/api";

interface SignalCardProps {
  signal: Signal;
}

const KIND_STYLE: Record<string, string> = {
  BUY: "text-green-400 border border-green-800",
  SELL: "text-red-400 border border-red-800",
  HOLD: "text-zinc-400 bg-zinc-900 border border-zinc-700",
};

const KIND_BG: Record<string, string> = {
  BUY: "#052e16",
  SELL: "#450a0a",
  HOLD: "",
};

export function SignalCard({ signal }: SignalCardProps) {
  const style = KIND_STYLE[signal.kind] ?? KIND_STYLE.HOLD;
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="text-text font-semibold">{signal.zone}</span>
        <span
          className={`text-xs font-semibold px-2 py-0.5 rounded ${style}`}
          style={KIND_BG[signal.kind] ? { backgroundColor: KIND_BG[signal.kind] } : undefined}
        >
          {signal.kind}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-xs mb-3">
        <div>
          <div className="text-muted">price</div>
          <div className="text-text">{signal.price.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-muted">7d mean</div>
          <div className="text-text">{signal.mean.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-muted">z-score</div>
          <div className={signal.zscore > 0 ? "text-red-400" : signal.zscore < 0 ? "text-green-400" : "text-muted"}>
            {signal.zscore > 0 ? "+" : ""}{signal.zscore.toFixed(2)}σ
          </div>
        </div>
      </div>
      <p className="text-muted text-xs leading-relaxed">{signal.reason}</p>
    </div>
  );
}

export function SignalPlaceholder() {
  return (
    <div className="bg-surface border border-border rounded-lg p-4 flex items-center justify-center h-32 text-muted text-xs">
      no signals — ENTSOE_API_TOKEN not configured
    </div>
  );
}
