"use client";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function DrawdownChart({ data }: { data: number[] }) {
  const pts = data.map((v, i) => ({ i, dd: +(v * 100).toFixed(3) }));
  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-zinc-400 text-xs font-medium mb-3">Drawdown</h3>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={pts} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="i" hide />
          <YAxis
            tick={{ fill: "#71717a", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={45}
            tickFormatter={(v) => `${v.toFixed(1)}%`}
          />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #27272a", borderRadius: 6 }}
            itemStyle={{ color: "#f87171", fontSize: 12 }}
            formatter={(v) => [`${Number(v).toFixed(2)}%`, "Drawdown"]}
          />
          <Area
            type="monotone"
            dataKey="dd"
            stroke="#f87171"
            fill="#f87171"
            fillOpacity={0.15}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
