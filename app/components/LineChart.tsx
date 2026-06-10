"use client";

import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { SeriesPoint } from "@/lib/types";

interface LineChartProps {
  data: SeriesPoint[];
  column?: string;
}

export default function LineChart({ data, column = "value" }: LineChartProps) {
  if (!data.length) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-600 text-sm font-mono">
        No data to display
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ReLineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
        <XAxis
          dataKey="x"
          tick={{ fill: "#6b7280", fontSize: 11, fontFamily: "monospace" }}
          tickLine={false}
          axisLine={{ stroke: "#1e1e2e" }}
        />
        <YAxis
          tick={{ fill: "#6b7280", fontSize: 11, fontFamily: "monospace" }}
          tickLine={false}
          axisLine={false}
          width={60}
        />
        <Tooltip
          contentStyle={{
            background: "#13131a",
            border: "1px solid #1e1e2e",
            borderRadius: "8px",
            fontFamily: "monospace",
            fontSize: "12px",
            color: "#e5e7eb",
          }}
        />
        <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "11px", color: "#6b7280" }} />
        <Line
          type="monotone"
          dataKey="y"
          name={column}
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4, fill: "#60a5fa" }}
        />
      </ReLineChart>
    </ResponsiveContainer>
  );
}
