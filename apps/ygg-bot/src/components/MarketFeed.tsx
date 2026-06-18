"use client";

import { useEffect, useRef, useState } from "react";
import type { WsTick } from "@/lib/ws";

const KIND_COLOR: Record<string, string> = {
  BUY: "text-green-400",
  SELL: "text-red-400",
  HOLD: "text-zinc-400",
};

export function MarketFeed() {
  const [ticks, setTicks] = useState<(WsTick & { id: number })[]>([]);
  const counter = useRef(0);

  useEffect(() => {
    let cleanup: (() => void) | undefined;
    // Lazy import to avoid SSR issues
    import("@/lib/ws").then(({ createMarketFeed }) => {
      cleanup = createMarketFeed((tick) => {
        setTicks((prev) => [
          { ...tick, id: ++counter.current },
          ...prev.slice(0, 19),
        ]);
      });
    });
    return () => cleanup?.();
  }, []);

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="text-muted text-xs mb-3">live feed (ws)</div>
      {ticks.length === 0 ? (
        <p className="text-muted text-xs">waiting for ticks…</p>
      ) : (
        <div className="space-y-1 font-mono text-xs max-h-48 overflow-y-auto">
          {ticks.map((t) => (
            <div key={t.id} className="flex items-center gap-3 text-muted">
              <span className="w-[90px] shrink-0">
                {new Date(t.ts * 1000).toLocaleTimeString("en-GB")}
              </span>
              <span className="text-text w-10">{t.zone}</span>
              <span className="tabular-nums text-text w-14 text-right">
                {t.price != null ? t.price.toFixed(2) : "—"}
              </span>
              {t.signal && (
                <span className={`w-8 font-semibold ${KIND_COLOR[t.signal] ?? ""}`}>
                  {t.signal}
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
