"use client";

import type { FxRate } from "@/lib/api";

interface FxTickerProps {
  rates: FxRate[];
  base: string;
}

export function FxTicker({ rates, base }: FxTickerProps) {
  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="text-muted text-xs mb-3">FX rates ({base})</div>
      {rates.length === 0 ? (
        <p className="text-muted text-xs">unavailable (network)</p>
      ) : (
        <div className="flex flex-wrap gap-3">
          {rates.map((r) => (
            <div key={r.pair} className="text-xs">
              <span className="text-muted">{r.pair.split("/")[1]}</span>
              <span className="text-text ml-1.5 font-semibold tabular-nums">
                {r.rate.toFixed(4)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
