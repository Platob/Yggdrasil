import type { FinanceMetrics } from "@/lib/types";

interface KPI {
  label: string;
  key: keyof FinanceMetrics;
  fmt: "pct" | "ratio";
  invert?: boolean; // true = lower is better (drawdown, vol)
}

const KPIS: KPI[] = [
  { label: "Total Return",    key: "total_return",    fmt: "pct" },
  { label: "CAGR",            key: "cagr",            fmt: "pct" },
  { label: "Ann. Return",     key: "ann_return",      fmt: "pct" },
  { label: "Ann. Volatility", key: "ann_volatility",  fmt: "pct", invert: true },
  { label: "Sharpe",          key: "sharpe",          fmt: "ratio" },
  { label: "Sortino",         key: "sortino",         fmt: "ratio" },
  { label: "Max Drawdown",    key: "max_drawdown",    fmt: "pct",  invert: true },
  { label: "Calmar",          key: "calmar",          fmt: "ratio" },
];

function fmtVal(v: number, fmt: "pct" | "ratio") {
  if (fmt === "pct") return `${(v * 100).toFixed(2)}%`;
  return v.toFixed(3);
}

function color(v: number, invert?: boolean) {
  const good = invert ? v < 0 : v > 0;
  if (v === 0) return "text-zinc-400";
  return good ? "text-emerald-400" : "text-red-400";
}

export default function FinancePanel({ metrics }: { metrics: FinanceMetrics }) {
  return (
    <div className="grid grid-cols-4 gap-3">
      {KPIS.map(({ label, key, fmt, invert }) => {
        const v = metrics[key];
        return (
          <div key={key} className="bg-zinc-900 rounded-lg border border-zinc-800 p-3">
            <p className="text-zinc-500 text-xs mb-1">{label}</p>
            <p className={`text-xl font-semibold tabular-nums ${color(v, invert)}`}>
              {fmtVal(v, fmt)}
            </p>
          </div>
        );
      })}
    </div>
  );
}
