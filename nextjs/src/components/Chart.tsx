"use client";

import { useMemo } from "react";
import {
  Bar,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Candle } from "@/lib/types";
import { fmtNum, fmtPrice, fmtTime } from "@/lib/format";

const GREEN = "#22c55e";
const RED = "#ef4444";

interface Row {
  ts: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  up: boolean;
  // [low, high] domain for the candle bar so recharts allocates the wick range
  range: [number, number];
  // [bodyBottom, bodyTop] for the candle body
  body: [number, number];
  sma: number | null;
}

interface CandleShapeProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  payload?: Row;
  yAxis?: { scale: (v: number) => number };
}

function CandleShape(props: CandleShapeProps) {
  const { x = 0, width = 0, payload, yAxis } = props;
  if (!payload || !yAxis) return null;

  const color = payload.up ? GREEN : RED;
  const cx = x + width / 2;
  const bodyWidth = Math.max(width * 0.6, 1);
  const bodyX = cx - bodyWidth / 2;

  const yHigh = yAxis.scale(payload.high);
  const yLow = yAxis.scale(payload.low);
  const yOpen = yAxis.scale(payload.open);
  const yClose = yAxis.scale(payload.close);
  const bodyTop = Math.min(yOpen, yClose);
  const bodyH = Math.max(Math.abs(yClose - yOpen), 1);

  return (
    <g>
      {/* Wick */}
      <line x1={cx} x2={cx} y1={yHigh} y2={yLow} stroke={color} strokeWidth={1} />
      {/* Body */}
      <rect x={bodyX} y={bodyTop} width={bodyWidth} height={bodyH} fill={color} />
    </g>
  );
}

interface TooltipPayload {
  payload: Row;
}

function CandleTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const color = d.up ? "text-green-400" : "text-red-400";
  return (
    <div className="rounded-md border border-gray-700 bg-gray-900/95 px-3 py-2 text-xs font-mono shadow-lg">
      <div className="text-gray-400 mb-1">{fmtTime(d.ts)}</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
        <span className="text-gray-500">O</span>
        <span className="text-right">{fmtNum(d.open)}</span>
        <span className="text-gray-500">H</span>
        <span className="text-right">{fmtNum(d.high)}</span>
        <span className="text-gray-500">L</span>
        <span className="text-right">{fmtNum(d.low)}</span>
        <span className="text-gray-500">C</span>
        <span className={`text-right ${color}`}>{fmtNum(d.close)}</span>
        <span className="text-gray-500">V</span>
        <span className="text-right">{fmtNum(d.volume, 0)}</span>
      </div>
    </div>
  );
}

export default function Chart({
  candles,
  height = 420,
}: {
  candles: Candle[];
  height?: number;
}) {
  const rows = useMemo<Row[]>(() => {
    const period = 20;
    return candles.map((c, i) => {
      let sma: number | null = null;
      if (i >= period - 1) {
        let sum = 0;
        for (let j = i - period + 1; j <= i; j++) sum += candles[j].close;
        sma = sum / period;
      }
      return {
        ts: c.ts,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
        up: c.close >= c.open,
        range: [c.low, c.high],
        body: [Math.min(c.open, c.close), Math.max(c.open, c.close)],
        sma,
      };
    });
  }, [candles]);

  const priceDomain = useMemo<[number, number]>(() => {
    if (!rows.length) return [0, 1];
    let lo = Infinity;
    let hi = -Infinity;
    for (const r of rows) {
      if (r.low < lo) lo = r.low;
      if (r.high > hi) hi = r.high;
    }
    const pad = (hi - lo) * 0.05 || 1;
    return [lo - pad, hi + pad];
  }, [rows]);

  const volHeight = Math.round(height * 0.25);
  const priceHeight = height - volHeight;

  if (!candles.length) {
    return (
      <div
        className="flex items-center justify-center text-gray-600 text-sm"
        style={{ height }}
      >
        No candle data
      </div>
    );
  }

  return (
    <div style={{ width: "100%" }}>
      <ResponsiveContainer width="100%" height={priceHeight}>
        <ComposedChart data={rows} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <XAxis
            dataKey="ts"
            tickFormatter={(t) => fmtTime(t)}
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={{ stroke: "#1f2937" }}
            tickLine={false}
            minTickGap={40}
          />
          <YAxis
            domain={priceDomain}
            orientation="right"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={{ stroke: "#1f2937" }}
            tickLine={false}
            tickFormatter={(v) => fmtNum(v, 0)}
            width={56}
          />
          <Tooltip content={<CandleTooltip />} cursor={{ stroke: "#374151" }} />
          {/* Invisible bar to anchor the custom candle shapes across the price range */}
          <Bar dataKey="range" shape={<CandleShape />} isAnimationActive={false} />
          <Line
            type="monotone"
            dataKey="sma"
            stroke="#3b82f6"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
            connectNulls
          />
        </ComposedChart>
      </ResponsiveContainer>

      <ResponsiveContainer width="100%" height={volHeight}>
        <ComposedChart data={rows} margin={{ top: 0, right: 8, left: 0, bottom: 0 }}>
          <XAxis dataKey="ts" hide />
          <YAxis
            orientation="right"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={{ stroke: "#1f2937" }}
            tickLine={false}
            tickFormatter={(v) => fmtNum(v, 0)}
            width={56}
          />
          <Tooltip
            cursor={{ fill: "#1f2937", opacity: 0.4 }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0].payload as Row;
              return (
                <div className="rounded-md border border-gray-700 bg-gray-900/95 px-2 py-1 text-xs font-mono">
                  Vol {fmtNum(d.volume, 0)}
                </div>
              );
            }}
          />
          <Bar dataKey="volume" isAnimationActive={false}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.up ? GREEN : RED} fillOpacity={0.5} />
            ))}
          </Bar>
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

export { fmtPrice };
