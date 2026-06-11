"use client";

/** Minimal SVG line chart. Skips null points so partially-warmed indicator
 * series (EMA/RSI/MACD) render cleanly from their first valid value. */
export function LineChart({
  data,
  height = 200,
  color = "#10b981",
}: {
  data: (number | null)[];
  height?: number;
  color?: string;
}) {
  if (!data.length) return null;
  const valid = data.filter((x): x is number => x != null);
  if (!valid.length) return null;
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const range = max - min || 1;
  const w = 600;
  const h = height;
  const pts = data
    .map((v, i) =>
      v == null ? null : `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`
    )
    .filter(Boolean) as string[];
  const d = pts.length ? `M ${pts.join(" L ")}` : "";
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full" style={{ height }}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" />
    </svg>
  );
}

/** Overlay a second series scaled to the same [min,max] domain as a base
 * series — used to draw EMA lines on top of the price chart. */
export function overlayPath(
  series: (number | null)[],
  domainOf: number[],
  w = 600,
  h = 200,
): string {
  const min = Math.min(...domainOf);
  const max = Math.max(...domainOf);
  const range = max - min || 1;
  const pts = series
    .map((v, i) =>
      v == null ? null : `${(i / (series.length - 1)) * w},${h - ((v - min) / range) * h}`
    )
    .filter(Boolean) as string[];
  return pts.length ? `M ${pts.join(" L ")}` : "";
}
