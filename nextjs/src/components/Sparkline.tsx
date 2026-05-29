"use client";

import { useId } from "react";

interface SparklineProps {
  /** Time-ordered samples, oldest → newest. Rendered left → right. */
  data: number[];
  /** Line/area color (CSS color or var). */
  color: string;
  /** Fixed upper bound for the y-axis (e.g. 100 for a percent). When
   *  omitted, the series auto-scales to its own max with a little headroom. */
  max?: number;
  height?: number;
  /** Optional big number + unit drawn at the top-left of the chart. */
  label?: string;
  value?: string;
}

// Lightweight SVG timeseries chart — no chart library. Draws a filled area
// under a smooth-ish polyline, scaled to the container width via a 0..100
// viewBox so it stays crisp at any size. Thresholds tint the stroke amber/rose
// when a percent series runs hot, matching ResourceBar's language.
export function Sparkline({ data, color, max, height = 48, label, value }: SparklineProps) {
  const gradId = useId();
  const n = data.length;

  const peak = max ?? Math.max(1, ...data) * 1.15;
  const latest = n > 0 ? data[n - 1] : 0;
  // Only re-tint when the caller gave a percent scale (max === 100).
  const stroke =
    max === 100 && latest > 90 ? "var(--rose)" :
    max === 100 && latest > 70 ? "var(--amber)" :
    color;

  const W = 100;
  const H = 100;
  const pts = data.map((v, i) => {
    const x = n <= 1 ? 0 : (i / (n - 1)) * W;
    const y = H - Math.min(H, Math.max(0, (v / peak) * H));
    return [x, y] as const;
  });

  const line = pts.map(([x, y]) => `${x.toFixed(2)},${y.toFixed(2)}`).join(" ");
  const area = n > 0
    ? `0,${H} ${line} ${W},${H}`
    : "";

  return (
    <div className="relative w-full" style={{ height }}>
      {(label || value) && (
        <div className="absolute top-0 left-0 flex items-baseline gap-1.5 pointer-events-none z-10">
          {value && <span className="font-mono text-sm font-semibold" style={{ color: stroke }}>{value}</span>}
          {label && <span className="text-[10px] text-muted uppercase tracking-wider">{label}</span>}
        </div>
      )}
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        className="w-full h-full"
        aria-hidden
      >
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={stroke} stopOpacity="0.35" />
            <stop offset="100%" stopColor={stroke} stopOpacity="0" />
          </linearGradient>
        </defs>
        {n > 0 && (
          <>
            <polygon points={area} fill={`url(#${gradId})`} />
            <polyline
              points={line}
              fill="none"
              stroke={stroke}
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          </>
        )}
      </svg>
    </div>
  );
}
