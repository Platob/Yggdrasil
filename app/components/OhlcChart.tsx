"use client";

import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { OhlcCandle } from "@/lib/types";

interface OhlcChartProps {
  candles: OhlcCandle[];
}

// Transform candles for recharts: each bar needs [bottom, top] for the body and wick
interface CandleRow {
  time: string;
  // high-low wick: [low, high]
  wick: [number, number];
  // open-close body: [min(o,c), max(o,c)]
  body: [number, number];
  bullish: boolean;
  open: number;
  high: number;
  low: number;
  close: number;
}

function prepareCandleData(candles: OhlcCandle[]): CandleRow[] {
  return candles.map((c) => ({
    time: String(c.time),
    wick: [c.low, c.high],
    body: [Math.min(c.open, c.close), Math.max(c.open, c.close)],
    bullish: c.close >= c.open,
    open: c.open,
    high: c.high,
    low: c.low,
    close: c.close,
  }));
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function OhlcTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as CandleRow;
  if (!d) return null;
  return (
    <div className="bg-[#13131a] border border-[#1e1e2e] rounded-lg p-3 text-xs font-mono">
      <div className="text-gray-400 mb-1">{d.time}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-gray-500">O</span>
        <span className="text-gray-200">{d.open?.toFixed(4)}</span>
        <span className="text-gray-500">H</span>
        <span className="text-green-400">{d.high?.toFixed(4)}</span>
        <span className="text-gray-500">L</span>
        <span className="text-red-400">{d.low?.toFixed(4)}</span>
        <span className="text-gray-500">C</span>
        <span className={d.bullish ? "text-green-400" : "text-red-400"}>{d.close?.toFixed(4)}</span>
      </div>
    </div>
  );
}

export default function OhlcChart({ candles }: OhlcChartProps) {
  if (!candles.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-600 text-sm font-mono">
        No candle data to display
      </div>
    );
  }

  const data = prepareCandleData(candles);

  return (
    <ResponsiveContainer width="100%" height={360}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
        <XAxis
          dataKey="time"
          tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }}
          tickLine={false}
          axisLine={{ stroke: "#1e1e2e" }}
        />
        <YAxis
          tick={{ fill: "#6b7280", fontSize: 11, fontFamily: "monospace" }}
          tickLine={false}
          axisLine={false}
          width={70}
          domain={["auto", "auto"]}
        />
        <Tooltip content={<OhlcTooltip />} />
        {/* Wick: high-low range, narrow bar */}
        <Bar dataKey="wick" barSize={2} fill="#6b7280" isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.bullish ? "#22c55e" : "#ef4444"} />
          ))}
        </Bar>
        {/* Body: open-close range */}
        <Bar dataKey="body" barSize={8} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.bullish ? "#22c55e" : "#ef4444"} />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  );
}
