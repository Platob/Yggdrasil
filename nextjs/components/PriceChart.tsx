"use client";
import { useEffect, useRef, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";
import { createPriceStream } from "@/lib/api";
import type { PriceTick } from "@/lib/types";

const MAX_TICKS = 120;

function fmt(ts: number) {
  const d = new Date(ts * 1000);
  return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`;
}

export default function PriceChart() {
  const [ticks, setTicks] = useState<{ t: string; price: number }[]>([]);
  const [connected, setConnected] = useState(false);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let active = true;
    function connect() {
      if (!active) return;
      try {
        const cleanup = createPriceStream((tick: PriceTick) => {
          setConnected(true);
          setTicks((prev) => {
            const next = [...prev, { t: fmt(tick.ts), price: tick.price }];
            return next.length > MAX_TICKS ? next.slice(-MAX_TICKS) : next;
          });
        });
        cleanupRef.current = cleanup;
      } catch {
        // WebSocket not available — show demo data
        setConnected(false);
      }
    }
    connect();
    return () => {
      active = false;
      cleanupRef.current?.();
    };
  }, []);

  const yDomain = ticks.length
    ? [
        Math.min(...ticks.map((t) => t.price)) * 0.999,
        Math.max(...ticks.map((t) => t.price)) * 1.001,
      ]
    : ["auto", "auto"];

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-zinc-200 text-sm font-medium">Live Price Stream</h2>
        <span className={`text-xs px-2 py-0.5 rounded-full ${connected ? "bg-emerald-900/50 text-emerald-400" : "bg-zinc-800 text-zinc-500"}`}>
          {connected ? "● live" : "○ demo"}
        </span>
      </div>
      {ticks.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-zinc-600 text-sm">
          Waiting for price data…
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={ticks} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="t"
              tick={{ fill: "#71717a", fontSize: 10 }}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={yDomain as [number, number]}
              tick={{ fill: "#71717a", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              width={55}
              tickFormatter={(v) => v.toFixed(2)}
            />
            <Tooltip
              contentStyle={{ background: "#18181b", border: "1px solid #27272a", borderRadius: 6 }}
              labelStyle={{ color: "#71717a", fontSize: 11 }}
              itemStyle={{ color: "#10b981", fontSize: 12 }}
            />
            <Line
              type="monotone"
              dataKey="price"
              stroke="#10b981"
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
