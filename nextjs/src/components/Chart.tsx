"use client";

// Dependency-free SVG chart — the visualization "plugin". Supports bar / line
// / area via one `type` prop so callers (pivot, finance) can switch renderers
// without pulling a chart library. Values may contain nulls (gaps).

export type ChartType = "bar" | "line" | "area";

interface Props {
  type: ChartType;
  labels: (string | number)[];
  values: (number | null)[];
  height?: number;
  color?: string;
  yLabel?: string;
}

const W = 720; // viewBox width; SVG scales to its container

export default function Chart({ type, labels, values, height = 220, color = "var(--emerald)", yLabel }: Props) {
  const nums = values.filter((v): v is number => v != null && isFinite(v));
  if (nums.length === 0) {
    return <div className="text-[11px] text-muted font-mono p-4">no numeric data to plot</div>;
  }
  const padL = 52, padR = 12, padT = 12, padB = 28;
  const plotW = W - padL - padR;
  const plotH = height - padT - padB;
  let min = Math.min(...nums), max = Math.max(...nums);
  if (min === max) { max = min + 1; min = min - 1; }
  // include zero baseline for bars/area so magnitudes read true
  if (type !== "line") { min = Math.min(min, 0); max = Math.max(max, 0); }

  const n = values.length;
  const x = (i: number) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const xBar = (i: number) => padL + (i + 0.5) * (plotW / n);
  const y = (v: number) => padT + plotH - ((v - min) / (max - min)) * plotH;
  const zeroY = y(0);

  const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(1) : v.toFixed(2));
  const ticks = [max, min + (max - min) * 0.5, min];

  // x labels: at most ~8, evenly spaced
  const step = Math.max(1, Math.ceil(n / 8));
  const xticks = labels.map((l, i) => ({ l, i })).filter(({ i }) => i % step === 0);

  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} className="block" preserveAspectRatio="none">
      {/* gridlines + y ticks */}
      {ticks.map((t, k) => (
        <g key={k}>
          <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--foreground)" strokeOpacity="0.06" />
          <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="monospace">{fmt(t)}</text>
        </g>
      ))}
      {type !== "line" && min < 0 && max > 0 && (
        <line x1={padL} x2={W - padR} y1={zeroY} y2={zeroY} stroke="var(--muted)" strokeOpacity="0.4" />
      )}

      {/* series */}
      {type === "bar" && values.map((v, i) =>
        v == null || !isFinite(v) ? null : (
          <rect
            key={i}
            x={xBar(i) - Math.max(1, (plotW / n) * 0.36)}
            width={Math.max(1, (plotW / n) * 0.72)}
            y={Math.min(y(v), zeroY)}
            height={Math.max(1, Math.abs(y(v) - zeroY))}
            fill={color} fillOpacity="0.8"
          />
        ),
      )}

      {(type === "line" || type === "area") && (() => {
        const pts = values.map((v, i) => (v == null || !isFinite(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[];
        if (pts.length === 0) return null;
        return (
          <>
            {type === "area" && (
              <polygon points={`${padL},${zeroY} ${pts.join(" ")} ${x(n - 1)},${zeroY}`} fill={color} fillOpacity="0.18" />
            )}
            <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.8" strokeLinejoin="round" />
          </>
        );
      })()}

      {/* x labels */}
      {xticks.map(({ l, i }) => (
        <text key={i} x={type === "bar" ? xBar(i) : x(i)} y={height - 8} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="monospace">
          {String(l).slice(0, 10)}
        </text>
      ))}
      {yLabel && <text x={6} y={padT + 4} fontSize="9" fill="var(--muted)" fontFamily="monospace">{yLabel}</text>}
    </svg>
  );
}
