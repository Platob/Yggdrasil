"use client";

// Dependency-free SVG chart — bar / line / area / candlestick with optional
// volume sub-panel. Fully prop-driven and null-safe throughout.

export type ChartType = "bar" | "line" | "area" | "candle";

interface Props {
  type: ChartType;
  labels: (string | number)[];
  values?: (number | null)[];
  band?: { min: (number | null)[]; max: (number | null)[] };
  ohlc?: { open: (number | null)[]; high: (number | null)[]; low: (number | null)[]; close: (number | null)[] };
  volume?: (number | null)[];   // renders a volume sub-panel below candles/price
  overlay?: (number | null)[];  // single moving-average line over candles
  overlays?: { values: (number | null)[]; color: string }[];  // multiple overlay lines (MAs, etc.)
  height?: number;
  color?: string;
  yLabel?: string;
}

const W = 720;
const padL = 52, padR = 12, padT = 12, padB = 28;
const VOL_H = 60;   // height reserved for the volume sub-panel
const VOL_GAP = 6;  // gap between price and volume panels

export default function Chart({
  type, labels, values, band, ohlc, volume, overlay, overlays, height = 220, color = "var(--emerald)", yLabel,
}: Props) {
  const hasVol = volume && volume.some((v) => v != null && isFinite(v));
  const priceH = hasVol ? height - VOL_H - VOL_GAP : height;

  const plotW = W - padL - padR;
  const plotH = priceH - padT - padB;

  // y-domain for the price panel
  const pool: number[] = [];
  if (type === "candle" && ohlc) {
    for (const a of [ohlc.high, ohlc.low, ohlc.open, ohlc.close])
      for (const v of a) if (v != null && isFinite(v)) pool.push(v);
    if (overlay) for (const v of overlay) if (v != null && isFinite(v)) pool.push(v);
    if (overlays) for (const ol of overlays) for (const v of ol.values) if (v != null && isFinite(v)) pool.push(v);
  } else if (values) {
    for (const v of values) if (v != null && isFinite(v)) pool.push(v);
    if (band) for (const a of [band.min, band.max]) for (const v of a) if (v != null && isFinite(v)) pool.push(v);
    if (overlays) for (const ol of overlays) for (const v of ol.values) if (v != null && isFinite(v)) pool.push(v);
  }
  if (pool.length === 0)
    return <div className="text-[11px] text-muted font-mono p-4">no numeric data to plot</div>;

  let min = Math.min(...pool), max = Math.max(...pool);
  if (min === max) { max = min + 1; min = min - 1; }
  if (type === "bar" || type === "area") { min = Math.min(min, 0); max = Math.max(max, 0); }

  const n = type === "candle" && ohlc ? ohlc.close.length : (values?.length ?? 0);
  const x = (i: number) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const xBand = (i: number) => padL + (i + 0.5) * (plotW / Math.max(1, n));
  const y = (v: number) => padT + plotH - ((v - min) / (max - min)) * plotH;
  const zeroY = y(0);
  const fmt = (v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01) ? v.toExponential(1) : v.toFixed(2));
  const step = Math.max(1, Math.ceil(n / 8));
  const xticks = labels.map((l, i) => ({ l, i })).filter(({ i }) => i % step === 0);

  // Volume panel geometry
  const volTop = priceH + VOL_GAP;
  const volPlotH = VOL_H - padB;
  const volMax = hasVol ? Math.max(...(volume!.filter((v) => v != null && isFinite(v)) as number[])) : 1;
  const yVol = (v: number) => volTop + volPlotH - (v / (volMax || 1)) * volPlotH;

  const totalH = height;

  return (
    <svg viewBox={`0 0 ${W} ${totalH}`} width="100%" height={totalH} className="block" preserveAspectRatio="none">
      {/* Price panel grid lines */}
      {[max, min + (max - min) * 0.5, min].map((t, k) => (
        <g key={k}>
          <line x1={padL} x2={W - padR} y1={y(t)} y2={y(t)} stroke="var(--foreground)" strokeOpacity="0.06" />
          <text x={padL - 6} y={y(t) + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="monospace">{fmt(t)}</text>
        </g>
      ))}

      {/* Candlesticks */}
      {type === "candle" && ohlc && ohlc.close.map((c, i) => {
        const o = ohlc.open[i], h = ohlc.high[i], l = ohlc.low[i];
        if (o == null || c == null || h == null || l == null) return null;
        const up = c >= o;
        const col = up ? "var(--emerald)" : "var(--rose)";
        const cx = xBand(i);
        const bw = Math.max(1, (plotW / Math.max(1, n)) * 0.6);
        return (
          <g key={i}>
            <line x1={cx} x2={cx} y1={y(h)} y2={y(l)} stroke={col} strokeWidth="1" />
            <rect x={cx - bw / 2} width={bw} y={Math.min(y(o), y(c))} height={Math.max(1, Math.abs(y(o) - y(c)))} fill={col} fillOpacity="0.85" />
          </g>
        );
      })}

      {/* Overlay (single MA line) over candles */}
      {type === "candle" && overlay && (() => {
        const pts = overlay.map((v, i) => (v == null || !isFinite(v) ? null : `${xBand(i)},${y(v)}`)).filter(Boolean) as string[];
        return pts.length ? <polyline points={pts.join(" ")} fill="none" stroke="var(--amber)" strokeWidth="1.4" strokeOpacity="0.9" /> : null;
      })()}
      {/* Multiple overlays over candles */}
      {type === "candle" && overlays && overlays.map((ol, oi) => {
        const pts = ol.values.map((v, i) => (v == null || !isFinite(v) ? null : `${xBand(i)},${y(v)}`)).filter(Boolean) as string[];
        return pts.length ? <polyline key={oi} points={pts.join(" ")} fill="none" stroke={ol.color} strokeWidth="1.3" strokeOpacity="0.85" /> : null;
      })}

      {/* Bars */}
      {type === "bar" && values?.map((v, i) =>
        v == null || !isFinite(v) ? null : (
          <rect key={i} x={xBand(i) - Math.max(1, (plotW / n) * 0.36)} width={Math.max(1, (plotW / n) * 0.72)}
            y={Math.min(y(v), zeroY)} height={Math.max(1, Math.abs(y(v) - zeroY))} fill={color} fillOpacity="0.8" />
        ),
      )}

      {/* Line / area (+ envelope band) */}
      {(type === "line" || type === "area") && values && (() => {
        const pts = values.map((v, i) => (v == null || !isFinite(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[];
        if (pts.length === 0) return null;
        const bandPoly = band
          ? (() => {
              const top = band.max.map((v, i) => (v == null || !isFinite(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[];
              const bot = band.min.map((v, i) => (v == null || !isFinite(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[];
              return top.length && bot.length ? `${top.join(" ")} ${bot.reverse().join(" ")}` : null;
            })()
          : null;
        return (
          <>
            {bandPoly && <polygon points={bandPoly} fill={color} fillOpacity="0.12" />}
            {type === "area" && <polygon points={`${padL},${zeroY} ${pts.join(" ")} ${x(n - 1)},${zeroY}`} fill={color} fillOpacity="0.18" />}
            <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth="1.6" strokeLinejoin="round" />
            {overlays && overlays.map((ol, oi) => {
              const opts = ol.values.map((v, i) => (v == null || !isFinite(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean) as string[];
              return opts.length ? <polyline key={oi} points={opts.join(" ")} fill="none" stroke={ol.color} strokeWidth="1.3" strokeOpacity="0.85" strokeLinejoin="round" /> : null;
            })}
          </>
        );
      })()}

      {/* X-axis labels */}
      {xticks.map(({ l, i }) => (
        <text key={i} x={type === "bar" || type === "candle" ? xBand(i) : x(i)} y={priceH - 8} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="monospace">
          {String(l).slice(0, 10)}
        </text>
      ))}
      {yLabel && <text x={6} y={padT + 4} fontSize="9" fill="var(--muted)" fontFamily="monospace">{yLabel}</text>}

      {/* Volume sub-panel */}
      {hasVol && (
        <>
          <line x1={padL} x2={W - padR} y1={volTop} y2={volTop} stroke="var(--foreground)" strokeOpacity="0.08" />
          <text x={padL - 6} y={volTop + 10} textAnchor="end" fontSize="8" fill="var(--muted)" fontFamily="monospace">vol</text>
          {volume!.map((v, i) => {
            if (v == null || !isFinite(v) || v === 0) return null;
            const up = type === "candle" && ohlc ? (ohlc.close[i] ?? 0) >= (ohlc.open[i] ?? 0) : v > 0;
            const col = up ? "var(--emerald)" : "var(--rose)";
            const bw = Math.max(1, (plotW / Math.max(1, n)) * 0.6);
            const cx = xBand(i);
            const barH = Math.max(1, volPlotH - (yVol(v) - volTop));
            return (
              <rect key={i} x={cx - bw / 2} width={bw} y={yVol(v)} height={barH} fill={col} fillOpacity="0.55" />
            );
          })}
        </>
      )}
    </svg>
  );
}
