"use client";

interface Bar {
  open: number;
  high: number;
  low: number;
  close: number;
}

interface Props {
  bars: Bar[];
  height?: number;
}

export default function OhlcChart({ bars, height = 200 }: Props) {
  if (!bars.length) return <div style={{ height, display: "flex", alignItems: "center", justifyContent: "center", color: "var(--muted)", fontSize: 12 }}>No data</div>;

  const allPrices = bars.flatMap(b => [b.high, b.low]);
  const minP = Math.min(...allPrices);
  const maxP = Math.max(...allPrices);
  const range = maxP - minP || 1;

  const W = 600;
  const H = height;
  const PAD = { top: 12, bottom: 20, left: 48, right: 12 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;
  const barW = Math.max(2, Math.floor(chartW / bars.length) - 1);

  const py = (v: number) => PAD.top + chartH - ((v - minP) / range) * chartH;
  const px = (i: number) => PAD.left + (i / bars.length) * chartW + barW / 2;

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => minP + (i / tickCount) * range);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height, display: "block" }}>
      {/* grid */}
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={PAD.left} y1={py(t)} x2={W - PAD.right} y2={py(t)} stroke="var(--border)" strokeWidth={0.5} />
          <text x={PAD.left - 4} y={py(t) + 4} textAnchor="end" fill="var(--muted)" fontSize={9}>
            {t.toFixed(4)}
          </text>
        </g>
      ))}
      {/* candles */}
      {bars.map((b, i) => {
        const up = b.close >= b.open;
        const color = up ? "var(--green)" : "var(--red)";
        const bodyTop = py(Math.max(b.open, b.close));
        const bodyH = Math.max(1, Math.abs(py(b.open) - py(b.close)));
        const x = px(i);
        return (
          <g key={i}>
            <line x1={x} y1={py(b.high)} x2={x} y2={py(b.low)} stroke={color} strokeWidth={1} />
            <rect x={x - barW / 2} y={bodyTop} width={barW} height={bodyH} fill={color} opacity={0.85} rx={0.5} />
          </g>
        );
      })}
    </svg>
  );
}
