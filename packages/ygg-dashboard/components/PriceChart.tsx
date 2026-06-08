'use client';

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from 'recharts';
import type { OhlcvBar } from '@/lib/types';

interface Props {
  data: OhlcvBar[];
  sma20?: (number | null)[];
  sma50?: (number | null)[];
  bbUpper?: (number | null)[];
  bbLower?: (number | null)[];
  height?: number;
}

export default function PriceChart({ data, sma20, sma50, bbUpper, bbLower, height = 300 }: Props) {
  const chartData = data.map((bar, i) => ({
    date: new Date(bar.ts).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    close: bar.close,
    open: bar.open,
    high: bar.high,
    low: bar.low,
    volume: bar.volume,
    sma20: sma20?.[i] ?? undefined,
    sma50: sma50?.[i] ?? undefined,
    bbUpper: bbUpper?.[i] ?? undefined,
    bbLower: bbLower?.[i] ?? undefined,
    positive: (bar.close ?? 0) >= (bar.open ?? 0),
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ top: 4, right: 4, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" vertical={false} />
        <XAxis
          dataKey="date"
          tick={{ fill: '#6b7280', fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          domain={['auto', 'auto']}
          tick={{ fill: '#6b7280', fontSize: 10 }}
          tickLine={false}
          axisLine={false}
          width={60}
          tickFormatter={(v) => v.toFixed(0)}
        />
        <Tooltip
          contentStyle={{ background: '#111827', border: '1px solid #1f2937', borderRadius: 4, fontSize: 11 }}
          labelStyle={{ color: '#9ca3af' }}
          itemStyle={{ color: '#f9fafb' }}
        />
        {/* Bollinger Bands */}
        <Line type="monotone" dataKey="bbUpper" stroke="#374151" strokeWidth={1} dot={false} name="BB Upper" strokeDasharray="4 2" />
        <Line type="monotone" dataKey="bbLower" stroke="#374151" strokeWidth={1} dot={false} name="BB Lower" strokeDasharray="4 2" />
        {/* SMAs */}
        <Line type="monotone" dataKey="sma20" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="SMA 20" />
        <Line type="monotone" dataKey="sma50" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="SMA 50" />
        {/* Price */}
        <Line type="monotone" dataKey="close" stroke="#e5e7eb" strokeWidth={2} dot={false} name="Close" />
        {/* Volume bars at bottom */}
        <Bar dataKey="volume" yAxisId={1} fill="#1f2937" opacity={0.5} name="Volume" />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
