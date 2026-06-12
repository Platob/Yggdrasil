"use client";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { api } from "@/lib/api";
import { format } from "date-fns";
import { fmt } from "@/lib/utils";

interface Props {
  symbol: string;
  period?: string;
}

export function PriceChart({ symbol, period = "3mo" }: Props) {
  const { data: bars = [], isLoading } = useQuery({
    queryKey: ["ohlcv", symbol, period],
    queryFn: () => api.ohlcv(symbol, period),
    staleTime: 60_000,
  });

  const chartData = bars.map((b) => ({
    date: format(new Date(b.timestamp), "MMM d"),
    close: b.close,
    volume: b.volume,
  }));

  const first = chartData[0]?.close ?? 0;
  const last = chartData[chartData.length - 1]?.close ?? 0;
  const up = last >= first;
  const color = up ? "#34d399" : "#f87171";
  const gradId = `grad-${symbol}`;

  if (isLoading) {
    return <div className="h-56 animate-pulse rounded-lg bg-slate-800/50" />;
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={chartData} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(100,116,139,0.12)" />
        <XAxis
          dataKey="date"
          tick={{ fontSize: 11, fill: "#64748b" }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={["auto", "auto"]}
          tick={{ fontSize: 11, fill: "#64748b" }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `$${fmt(v, 0)}`}
          width={60}
        />
        <Tooltip
          contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(v: number) => [`$${fmt(v)}`, "Close"]}
        />
        <Area
          type="monotone"
          dataKey="close"
          stroke={color}
          strokeWidth={1.5}
          fill={`url(#${gradId})`}
          dot={false}
          activeDot={{ r: 3, fill: color }}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
