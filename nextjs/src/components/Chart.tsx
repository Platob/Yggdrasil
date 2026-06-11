"use client";

const W = 600;

function scale(data: (number | null)[], h: number): { pts: string; d: string; min: number; max: number } {
  const valid = data.filter((x): x is number => x != null);
  const min = valid.length ? Math.min(...valid) : 0;
  const max = valid.length ? Math.max(...valid) : 1;
  const range = max - min || 1;
  const coords = data.map((v, i) =>
    v == null ? null : `${(i / Math.max(data.length - 1, 1)) * W},${h - ((v - min) / range) * h}`
  ).filter(Boolean) as string[];
  const d = coords.length ? `M ${coords.join(" L ")}` : "";
  return { pts: coords.join(" "), d, min, max };
}

/** Minimal SVG line chart. Null points are skipped so partially-warmed series render cleanly. */
export function LineChart({
  data,
  height = 200,
  color = "#10b981",
  fill = false,
}: {
  data: (number | null)[];
  height?: number;
  color?: string;
  fill?: boolean;
}) {
  if (!data.length) return null;
  const { d, min, max } = scale(data, height);
  if (!d) return null;
  const range = max - min || 1;
  const zeroY = height - ((0 - min) / range) * height;
  const fillPath = fill
    ? `${d} L ${W},${Math.min(height, Math.max(0, zeroY))} L 0,${Math.min(height, Math.max(0, zeroY))} Z`
    : "";

  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }}>
      {fill && <path d={fillPath} fill={color} fillOpacity={0.15} />}
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

/** Overlay a second series scaled to the same [min,max] domain as a reference series. */
export function overlayPath(
  series: (number | null)[],
  domainOf: (number | null)[],
  h = 200,
): string {
  const valid = domainOf.filter((x): x is number => x != null);
  const min = valid.length ? Math.min(...valid) : 0;
  const max = valid.length ? Math.max(...valid) : 1;
  const range = max - min || 1;
  const pts = series.map((v, i) =>
    v == null ? null : `${(i / Math.max(series.length - 1, 1)) * W},${h - ((v - min) / range) * h}`
  ).filter(Boolean) as string[];
  return pts.length ? `M ${pts.join(" L ")}` : "";
}

/** Chart with two series (e.g. equity vs. benchmark). Second series uses domainOf the combined range. */
export function DualLineChart({
  a, b, colorA = "#10b981", colorB = "#6b7280", height = 180,
}: {
  a: number[]; b: number[]; colorA?: string; colorB?: string; height?: number;
}) {
  const combined = [...a, ...b].filter((x): x is number => x != null);
  if (!combined.length) return null;
  const min = Math.min(...combined);
  const max = Math.max(...combined);
  const range = max - min || 1;

  const toPath = (data: (number | null)[]) => {
    const pts = data.map((v, i) =>
      v == null ? null : `${(i / Math.max(data.length - 1, 1)) * W},${height - ((v - min) / range) * height}`
    ).filter(Boolean) as string[];
    return pts.length ? `M ${pts.join(" L ")}` : "";
  };

  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }}>
      <path d={toPath(b)} fill="none" stroke={colorB} strokeWidth="1" strokeDasharray="4 3" />
      <path d={toPath(a)} fill="none" stroke={colorA} strokeWidth="1.5" />
    </svg>
  );
}

/** RSI chart with 30/70 reference bands shaded. */
export function RsiChart({ data, height = 100 }: { data: (number | null)[]; height?: number }) {
  const { d } = scale(data, height);
  const y30 = height - (30 / 100) * height;
  const y70 = height - (70 / 100) * height;
  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }}>
      <rect x={0} y={y70} width={W} height={y30 - y70} fill="#fbbf24" fillOpacity={0.07} />
      <line x1={0} y1={y70} x2={W} y2={y70} stroke="#fbbf24" strokeWidth="0.5" strokeDasharray="3 3" />
      <line x1={0} y1={y30} x2={W} y2={y30} stroke="#fbbf24" strokeWidth="0.5" strokeDasharray="3 3" />
      <path d={d} fill="none" stroke="#22d3ee" strokeWidth="1.5" />
    </svg>
  );
}

/** MACD histogram (bars) + signal line. */
export function MacdChart({
  hist, line, signal: sig, height = 80,
}: {
  hist: (number | null)[]; line: (number | null)[];
  signal: (number | null)[]; height?: number;
}) {
  const allVals = [...(hist ?? []), ...(line ?? []), ...(sig ?? [])].filter((x): x is number => x != null);
  if (!allVals.length) return null;
  const min = Math.min(...allVals);
  const max = Math.max(...allVals);
  const range = max - min || 1;
  const yOf = (v: number) => height - ((v - min) / range) * height;
  const zeroY = yOf(0);
  const n = hist.length;
  const barW = Math.max(1, W / n - 0.5);

  const toPath = (data: (number | null)[]) => {
    const pts = data.map((v, i) =>
      v == null ? null : `${(i / Math.max(n - 1, 1)) * W},${yOf(v)}`
    ).filter(Boolean) as string[];
    return pts.length ? `M ${pts.join(" L ")}` : "";
  };

  return (
    <svg viewBox={`0 0 ${W} ${height}`} className="w-full" style={{ height }}>
      <line x1={0} y1={zeroY} x2={W} y2={zeroY} stroke="#374151" strokeWidth="0.5" />
      {hist.map((v, i) => {
        if (v == null) return null;
        const x = (i / Math.max(n - 1, 1)) * W;
        const y = yOf(v);
        const isPos = v >= 0;
        return (
          <rect key={i} x={x - barW / 2} y={isPos ? y : zeroY}
            width={barW} height={Math.abs(y - zeroY)}
            fill={isPos ? "#10b981" : "#ef4444"} fillOpacity={0.7} />
        );
      })}
      <path d={toPath(line)} fill="none" stroke="#6366f1" strokeWidth="1" />
      <path d={toPath(sig)} fill="none" stroke="#f59e0b" strokeWidth="1" />
    </svg>
  );
}
