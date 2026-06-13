"use client";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

export default function CumReturnChart({ data }: { data: number[] }) {
  const pts = data.map((v, i) => ({ i, ret: +(v * 100).toFixed(3) }));
  const maxAbs = Math.max(...data.map(Math.abs)) * 100;
  const domain: [number, number] = [-maxAbs * 1.05, maxAbs * 1.05];

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 p-4">
      <h3 className="text-zinc-400 text-xs font-medium mb-3">Cumulative Return</h3>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={pts} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis dataKey="i" hide />
          <YAxis
            domain={domain}
            tick={{ fill: "#71717a", fontSize: 10 }}
            tickLine={false}
            axisLine={false}
            width={45}
            tickFormatter={(v) => `${v.toFixed(1)}%`}
          />
          <ReferenceLine y={0} stroke="#3f3f46" strokeDasharray="4 4" />
          <Tooltip
            contentStyle={{ background: "#18181b", border: "1px solid #27272a", borderRadius: 6 }}
            itemStyle={{ color: "#10b981", fontSize: 12 }}
            formatter={(v) => [`${Number(v).toFixed(2)}%`, "Cum. Return"]}
          />
          <Line
            type="monotone"
            dataKey="ret"
            stroke="#10b981"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
