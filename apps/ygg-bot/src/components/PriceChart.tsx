"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { PricePoint } from "@/lib/api";

interface PriceChartProps {
  prices: PricePoint[];
  zone: string;
  isLoading?: boolean;
}

function fmt(ts: string) {
  const d = new Date(ts);
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours().toString().padStart(2, "0")}h`;
}

const CustomTooltip = ({ active, payload, label }: {
  active?: boolean;
  payload?: { value: number; payload: PricePoint }[];
  label?: string;
}) => {
  if (!active || !payload?.length) return null;
  const pt = payload[0];
  return (
    <div className="bg-surface border border-border rounded px-3 py-2 text-xs">
      <div className="text-muted">{fmt(pt.payload.timestamp)}</div>
      <div className="text-text font-semibold">{pt.value.toFixed(2)} {pt.payload.currency}/{pt.payload.unit}</div>
    </div>
  );
};

export function PriceChart({ prices, zone, isLoading }: PriceChartProps) {
  const data = prices.map((p) => ({ ...p, ts: fmt(p.timestamp) }));
  const values = prices.map((p) => p.value);
  const min = Math.min(...values) * 0.97;
  const max = Math.max(...values) * 1.03;
  const latest = prices[prices.length - 1];

  return (
    <div className="bg-surface border border-border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <span className="text-text font-semibold">{zone}</span>
          <span className="text-muted ml-2 text-xs">day-ahead prices</span>
        </div>
        {latest && (
          <span className="text-accent font-semibold">
            {latest.value.toFixed(2)} {latest.currency}/{latest.unit}
          </span>
        )}
      </div>
      {isLoading ? (
        <div className="h-48 flex items-center justify-center text-muted text-xs">
          loading…
        </div>
      ) : prices.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-muted text-xs">
          no data — ENTSOE_API_TOKEN not configured
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={192}>
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis
              dataKey="ts"
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[min, max]}
              tick={{ fill: "#64748b", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={48}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#6366f1"
              strokeWidth={1.5}
              fill="url(#priceGrad)"
              dot={false}
              activeDot={{ r: 3, fill: "#6366f1" }}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
