"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmt, fmtPct } from "@/lib/utils";

const SYMBOLS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "SPY", "QQQ", "BTC-USD", "ETH-USD"];

export function MarketTicker() {
  const { data: quotes = [] } = useQuery({
    queryKey: ["ticker-quotes"],
    queryFn: () => api.quotes(SYMBOLS),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });

  const items = [...quotes, ...quotes]; // duplicate for seamless loop

  return (
    <div className="overflow-hidden border-b border-slate-800/50 bg-slate-900/40">
      <div className="ticker-scroll flex whitespace-nowrap py-2">
        {items.map((q, i) => (
          <span key={`${q.symbol}-${i}`} className="mx-6 inline-flex items-center gap-2 text-sm">
            <span className="font-semibold text-slate-200">{q.symbol}</span>
            <span className="text-slate-300">${fmt(q.price)}</span>
            <span className={q.change >= 0 ? "text-green-400" : "text-red-400"}>
              {fmtPct(q.change_pct)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}
