"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { fmt, fmtPct, fmtCompact } from "@/lib/utils";
import { PriceChart } from "./PriceChart";

interface Props {
  symbol: string;
}

export function QuoteCard({ symbol }: Props) {
  const { data: q, isLoading } = useQuery({
    queryKey: ["quote", symbol],
    queryFn: () => api.quote(symbol),
    staleTime: 15_000,
    refetchInterval: 15_000,
  });

  if (isLoading || !q) {
    return <div className="h-72 animate-pulse rounded-xl bg-slate-800/50" />;
  }

  const up = q.change >= 0;

  return (
    <div className="card-hover rounded-xl border border-slate-800/60 bg-slate-900/50 p-5">
      <div className="mb-3 flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider">{symbol}</p>
          <p className="text-2xl font-bold text-slate-100">${fmt(q.price)}</p>
        </div>
        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${up ? "bg-green-500/20 text-green-400" : "bg-red-500/20 text-red-400"}`}>
          {fmtPct(q.change_pct)}
        </span>
      </div>
      <PriceChart symbol={symbol} period="3mo" />
      <div className="mt-3 grid grid-cols-3 gap-2 border-t border-slate-800/60 pt-3 text-xs">
        {q.volume > 0 && <Kv k="Volume" v={fmtCompact(q.volume)} />}
        {q.high_52w && <Kv k="52W High" v={`$${fmt(q.high_52w)}`} />}
        {q.low_52w && <Kv k="52W Low" v={`$${fmt(q.low_52w)}`} />}
      </div>
    </div>
  );
}

function Kv({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <p className="text-slate-500">{k}</p>
      <p className="font-medium text-slate-300">{v}</p>
    </div>
  );
}
