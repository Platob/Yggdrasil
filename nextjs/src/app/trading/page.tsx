"use client";

import { useState, useCallback } from "react";
import type { OHLCVBar, MarketQuote } from "@/lib/api";
import { getMarketOHLCV, getMarketQuote, searchMarket } from "@/lib/api";
import type { MarketSearchResult } from "@/lib/api";

// ── Client-side technical indicators (no file path needed) ────────────────

function ema(prices: number[], span: number): (number | null)[] {
  const alpha = 2 / (span + 1);
  const out: (number | null)[] = Array(prices.length).fill(null);
  let prev: number | null = null;
  for (let i = 0; i < prices.length; i++) {
    const v = prices[i];
    if (!Number.isFinite(v)) continue;
    prev = prev == null ? v : alpha * v + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

function sma(prices: number[], period: number): (number | null)[] {
  return prices.map((_, i) => {
    if (i < period - 1) return null;
    const w = prices.slice(i - period + 1, i + 1);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });
}

function rsi(prices: number[], period = 14): (number | null)[] {
  const out: (number | null)[] = Array(prices.length).fill(null);
  const alpha = 1 / period;
  let avgGain = 0, avgLoss = 0;
  for (let i = 1; i < prices.length; i++) {
    const d = prices[i] - prices[i - 1];
    const g = d > 0 ? d : 0, l = d < 0 ? -d : 0;
    avgGain = alpha * g + (1 - alpha) * avgGain;
    avgLoss = alpha * l + (1 - alpha) * avgLoss;
    if (i >= period) out[i] = avgLoss > 0 ? 100 - 100 / (1 + avgGain / avgLoss) : 100;
  }
  return out;
}

function macdIndicator(prices: number[], fast = 12, slow = 26, signal = 9) {
  const ef = ema(prices, fast);
  const es = ema(prices, slow);
  const line = prices.map((_, i) => (ef[i] != null && es[i] != null ? ef[i]! - es[i]! : null));
  const filled = line.map((v, i) => v ?? (i === 0 ? 0 : null));
  const validFilled: number[] = [];
  const signalLine: (number | null)[] = Array(prices.length).fill(null);
  // Use EMA over the macd line values
  let prev: number | null = null;
  const alpha = 2 / (signal + 1);
  for (let i = 0; i < line.length; i++) {
    if (line[i] == null) continue;
    prev = prev == null ? line[i]! : alpha * line[i]! + (1 - alpha) * prev;
    signalLine[i] = prev;
  }
  const hist = line.map((m, i) => (m != null && signalLine[i] != null ? m - signalLine[i]! : null));
  return { line, signalLine, hist };
}

function bollingerBands(prices: number[], period = 20, mult = 2) {
  const mid = sma(prices, period);
  const upper: (number | null)[] = Array(prices.length).fill(null);
  const lower: (number | null)[] = Array(prices.length).fill(null);
  for (let i = period - 1; i < prices.length; i++) {
    const w = prices.slice(i - period + 1, i + 1);
    const m = w.reduce((a, b) => a + b, 0) / w.length;
    const std = Math.sqrt(w.reduce((a, b) => a + (b - m) ** 2, 0) / (w.length - 1));
    upper[i] = m + mult * std;
    lower[i] = m - mult * std;
  }
  return { upper, mid, lower };
}

// ── Finance metrics ───────────────────────────────────────────────────────

interface FinanceMetrics {
  totalReturn: number | null;
  cagr: number | null;
  annReturn: number | null;
  annVol: number | null;
  sharpe: number | null;
  sortino: number | null;
  maxDrawdown: number | null;
  calmar: number | null;
}

function computeMetrics(closes: number[], ppy = 252): FinanceMetrics {
  if (closes.length < 2) return { totalReturn: null, cagr: null, annReturn: null, annVol: null, sharpe: null, sortino: null, maxDrawdown: null, calmar: null };
  const rets: number[] = [];
  for (let i = 1; i < closes.length; i++) rets.push(closes[i] / closes[i - 1] - 1);
  const first = closes[0], last = closes[closes.length - 1];
  const n = rets.length;
  const totalReturn = first > 0 ? last / first - 1 : null;
  const years = n / ppy;
  const cagr = (first > 0 && last > 0 && years > 0) ? Math.pow(last / first, 1 / years) - 1 : null;
  const meanR = rets.reduce((a, b) => a + b, 0) / n;
  const variance = rets.reduce((a, b) => a + (b - meanR) ** 2, 0) / (n - 1);
  const stdR = Math.sqrt(variance);
  const annReturn = meanR * ppy;
  const annVol = stdR * Math.sqrt(ppy);
  const rfPerPeriod = 0;
  const downside = rets.filter(r => r < rfPerPeriod);
  const downDev = downside.length > 0
    ? Math.sqrt(downside.reduce((a, b) => a + (b - rfPerPeriod) ** 2, 0) / downside.length) * Math.sqrt(ppy)
    : 0;
  const sharpe = annVol > 0 ? annReturn / annVol : null;
  const sortino = downDev > 0 ? annReturn / downDev : null;
  // Running drawdown
  let peak = closes[0], maxDD = 0;
  for (const c of closes) {
    if (c > peak) peak = c;
    const dd = peak > 0 ? c / peak - 1 : 0;
    if (dd < maxDD) maxDD = dd;
  }
  const calmar = (cagr != null && maxDD < 0) ? cagr / Math.abs(maxDD) : null;
  return { totalReturn, cagr, annReturn, annVol, sharpe, sortino, maxDrawdown: maxDD, calmar };
}

// ── Trading signals ────────────────────────────────────────────────────────

interface Signal {
  label: string;
  type: "buy" | "sell" | "neutral" | "watch";
  detail: string;
}

function computeSignals(
  closes: number[],
  rsiVals: (number | null)[],
  macdLine: (number | null)[],
  macdSignal: (number | null)[],
  bbUpper: (number | null)[],
  bbLower: (number | null)[],
  ma20: (number | null)[],
  ma50: (number | null)[],
): Signal[] {
  const signals: Signal[] = [];
  if (closes.length < 2) return signals;
  const last = closes.length - 1;
  const price = closes[last];
  const prev = closes[last - 1];

  // RSI signal
  const rsi = rsiVals[last];
  if (rsi != null) {
    if (rsi > 70) signals.push({ label: "RSI", type: "sell", detail: `Overbought at ${rsi.toFixed(1)}` });
    else if (rsi < 30) signals.push({ label: "RSI", type: "buy", detail: `Oversold at ${rsi.toFixed(1)}` });
    else if (rsi > 60) signals.push({ label: "RSI", type: "watch", detail: `Approaching overbought (${rsi.toFixed(1)})` });
    else if (rsi < 40) signals.push({ label: "RSI", type: "watch", detail: `Approaching oversold (${rsi.toFixed(1)})` });
    else signals.push({ label: "RSI", type: "neutral", detail: `Neutral at ${rsi.toFixed(1)}` });
  }

  // MACD crossover
  if (last >= 1 && macdLine[last] != null && macdSignal[last] != null &&
      macdLine[last - 1] != null && macdSignal[last - 1] != null) {
    const crossed = (macdLine[last]! - macdSignal[last]!);
    const crossedPrev = (macdLine[last - 1]! - macdSignal[last - 1]!);
    if (crossedPrev <= 0 && crossed > 0) signals.push({ label: "MACD", type: "buy", detail: "Bullish crossover — MACD crossed above signal" });
    else if (crossedPrev >= 0 && crossed < 0) signals.push({ label: "MACD", type: "sell", detail: "Bearish crossover — MACD crossed below signal" });
    else signals.push({ label: "MACD", type: crossed > 0 ? "watch" : "neutral", detail: crossed > 0 ? "MACD above signal (bullish)" : "MACD below signal (bearish)" });
  }

  // Bollinger Band position
  const bbu = bbUpper[last], bbl = bbLower[last];
  if (bbu != null && bbl != null) {
    const pct = (price - bbl) / (bbu - bbl);
    if (price >= bbu) signals.push({ label: "Bollinger", type: "sell", detail: `Price at upper band — potential reversal` });
    else if (price <= bbl) signals.push({ label: "Bollinger", type: "buy", detail: `Price at lower band — potential bounce` });
    else if (pct > 0.8) signals.push({ label: "Bollinger", type: "watch", detail: `Near upper band (${(pct * 100).toFixed(0)}%)` });
    else if (pct < 0.2) signals.push({ label: "Bollinger", type: "watch", detail: `Near lower band (${(pct * 100).toFixed(0)}%)` });
    else signals.push({ label: "Bollinger", type: "neutral", detail: `Mid-band position (${(pct * 100).toFixed(0)}%)` });
  }

  // MA trend
  const m20 = ma20[last], m50 = ma50[last];
  if (m20 != null && m50 != null) {
    if (price > m20 && price > m50 && m20 > m50) {
      signals.push({ label: "Trend", type: "buy", detail: "Price above MA20 & MA50, MA20 > MA50 (uptrend)" });
    } else if (price < m20 && price < m50 && m20 < m50) {
      signals.push({ label: "Trend", type: "sell", detail: "Price below MA20 & MA50, MA20 < MA50 (downtrend)" });
    } else {
      signals.push({ label: "Trend", type: "neutral", detail: `Price ${price > m20 ? "above" : "below"} MA20, ${price > m50 ? "above" : "below"} MA50` });
    }
  }

  // Momentum (last 5 closes)
  if (closes.length >= 5) {
    const pct5 = (price / closes[last - 4] - 1) * 100;
    if (pct5 > 5) signals.push({ label: "Momentum", type: "watch", detail: `+${pct5.toFixed(1)}% over 5 bars (strong)` });
    else if (pct5 < -5) signals.push({ label: "Momentum", type: "watch", detail: `${pct5.toFixed(1)}% over 5 bars (weak)` });
  }

  return signals;
}

function SignalBadge({ signal }: { signal: Signal }) {
  const colors = {
    buy: "border-emerald/40 bg-emerald/8 text-emerald",
    sell: "border-rose/40 bg-rose/8 text-rose",
    watch: "border-amber/40 bg-amber/8 text-amber",
    neutral: "border-border bg-card text-muted",
  };
  const icons = {
    buy: "↑",
    sell: "↓",
    watch: "◈",
    neutral: "—",
  };
  return (
    <div className={`flex items-start gap-2 px-3 py-2 rounded-lg border text-xs font-mono ${colors[signal.type]}`}>
      <span className="shrink-0 font-bold">{icons[signal.type]}</span>
      <div>
        <span className="font-bold">{signal.label}</span>
        <span className="text-current opacity-70 ml-1.5">{signal.detail}</span>
      </div>
    </div>
  );
}

// ── Formatting helpers ─────────────────────────────────────────────────────

function fmtPct(v: number | null, decimals = 2) {
  if (v == null || !Number.isFinite(v)) return "—";
  return `${(v * 100).toFixed(decimals)}%`;
}
function fmtNum(v: number | null, decimals = 2) {
  if (v == null || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1e9) return `${(v / 1e9).toFixed(1)}B`;
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(1)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return v.toFixed(decimals);
}
function fmtPrice(v: number | null) {
  if (v == null || !Number.isFinite(v)) return "—";
  return v.toFixed(2);
}
function shortDate(iso: string) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "2-digit" });
  } catch { return iso.slice(0, 10); }
}

// ── SVG chart primitives ───────────────────────────────────────────────────

const W = 900, PL = 56, PR = 16, PT = 10, PB = 24;

function yScale(min: number, max: number, plotH: number) {
  if (min === max) { max = min + 1; min = min - 1; }
  return (v: number) => PT + plotH - ((v - min) / (max - min)) * plotH;
}

function GridLines({ min, max, plotH, ticks = 4 }: { min: number; max: number; plotH: number; ticks?: number }) {
  const toY = yScale(min, max, plotH);
  const range = max - min;
  const step = range / (ticks - 1);
  const vals = Array.from({ length: ticks }, (_, i) => min + i * step);
  const fmt = (v: number) => Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}K` : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3);
  return (
    <>
      {vals.map((v, k) => (
        <g key={k}>
          <line x1={PL} x2={W - PR} y1={toY(v)} y2={toY(v)} stroke="var(--border)" strokeOpacity="0.5" />
          <text x={PL - 6} y={toY(v) + 3} textAnchor="end" fontSize="9" fill="var(--muted)" fontFamily="monospace">{fmt(v)}</text>
        </g>
      ))}
    </>
  );
}

function XTicks({ labels, n, height, mode }: { labels: string[]; n: number; height: number; mode: "band" | "point" }) {
  const plotW = W - PL - PR;
  const step = Math.max(1, Math.ceil(n / 10));
  const ticks = labels.map((l, i) => ({ l, i })).filter(({ i }) => i % step === 0);
  const x = mode === "band"
    ? (i: number) => PL + (i + 0.5) * (plotW / Math.max(1, n))
    : (i: number) => PL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  return (
    <>
      {ticks.map(({ l, i }) => (
        <text key={i} x={x(i)} y={height - 6} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="monospace">
          {shortDate(l)}
        </text>
      ))}
    </>
  );
}

// ── OHLCV Candle + Volume chart ────────────────────────────────────────────

interface CandleChartProps {
  bars: OHLCVBar[];
  ma20?: (number | null)[];
  ma50?: (number | null)[];
  bbUpper?: (number | null)[];
  bbLower?: (number | null)[];
  height?: number;
}

function CandleChart({ bars, ma20, ma50, bbUpper, bbLower, height = 300 }: CandleChartProps) {
  if (!bars.length) return null;
  const plotW = W - PL - PR, plotH = height - PT - PB;
  const closes = bars.map(b => b.c ?? 0).filter(Boolean);
  const highs = bars.map(b => b.h ?? 0).filter(Boolean);
  const lows = bars.map(b => b.l ?? 0).filter(Boolean);
  const allVals = [...closes, ...highs, ...lows,
    ...(bbUpper?.filter(Boolean) as number[] ?? []),
    ...(bbLower?.filter(Boolean) as number[] ?? []),
    ...(ma20?.filter(Boolean) as number[] ?? []),
    ...(ma50?.filter(Boolean) as number[] ?? []),
  ].filter(Number.isFinite);
  const minY = Math.min(...allVals), maxY = Math.max(...allVals);
  const toY = yScale(minY, maxY, plotH);
  const n = bars.length;
  const bw = Math.max(1, (plotW / Math.max(1, n)) * 0.65);
  const cx = (i: number) => PL + (i + 0.5) * (plotW / Math.max(1, n));
  const labels = bars.map(b => b.t);
  const ptX = (i: number) => PL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);

  function lineOf(series: (number | null)[], color: string, strokeWidth = 1.4) {
    const pts = series.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
    return pts.length > 1 ? <polyline key={color} points={pts.join(" ")} fill="none" stroke={color} strokeWidth={strokeWidth} strokeLinejoin="round" /> : null;
  }

  function bbBand() {
    if (!bbUpper || !bbLower) return null;
    const top = bbUpper.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
    const bot = bbLower.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
    if (!top.length || !bot.length) return null;
    return <polygon points={`${top.join(" ")} ${bot.reverse().join(" ")}`} fill="var(--frost)" fillOpacity="0.06" />;
  }

  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} className="block" preserveAspectRatio="none">
      <GridLines min={minY} max={maxY} plotH={plotH} ticks={5} />
      {bbBand()}
      {bbUpper && lineOf(bbUpper, "var(--frost-dim)", 1)}
      {bbLower && lineOf(bbLower, "var(--frost-dim)", 1)}
      {ma50 && lineOf(ma50, "var(--amber)", 1.2)}
      {ma20 && lineOf(ma20, "var(--emerald)", 1.2)}
      {bars.map((b, i) => {
        const o = b.o, c = b.c, h = b.h, l = b.l;
        if (o == null || c == null || h == null || l == null) return null;
        const up = c >= o;
        const color = up ? "var(--emerald)" : "var(--rose)";
        return (
          <g key={i}>
            <line x1={cx(i)} x2={cx(i)} y1={toY(h)} y2={toY(l)} stroke={color} strokeWidth="1" />
            <rect x={cx(i) - bw / 2} width={bw} y={Math.min(toY(o), toY(c))} height={Math.max(1, Math.abs(toY(o) - toY(c)))} fill={color} fillOpacity="0.85" />
          </g>
        );
      })}
      <XTicks labels={labels} n={n} height={height} mode="band" />
    </svg>
  );
}

function VolumeChart({ bars, height = 70 }: { bars: OHLCVBar[]; height?: number }) {
  if (!bars.length) return null;
  const plotH = height - PT - PB;
  const vols = bars.map(b => b.v ?? 0);
  const maxV = Math.max(...vols, 1);
  const toY = yScale(0, maxV, plotH);
  const n = bars.length, plotW = W - PL - PR;
  const bw = Math.max(1, (plotW / Math.max(1, n)) * 0.65);
  const cx = (i: number) => PL + (i + 0.5) * (plotW / Math.max(1, n));
  const zeroY = toY(0);
  const fmtVol = (v: number) => v >= 1e9 ? `${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : `${(v / 1e3).toFixed(0)}K`;
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} className="block" preserveAspectRatio="none">
      <line x1={PL} x2={W - PR} y1={toY(maxV)} y2={toY(maxV)} stroke="var(--border)" strokeOpacity="0.3" />
      <text x={PL - 6} y={toY(maxV) + 3} textAnchor="end" fontSize="8" fill="var(--muted)" fontFamily="monospace">{fmtVol(maxV)}</text>
      {bars.map((b, i) => {
        const v = b.v ?? 0;
        const up = (b.c ?? 0) >= (b.o ?? 0);
        const top = toY(v), bot = zeroY;
        return (
          <rect key={i} x={cx(i) - bw / 2} width={bw}
            y={Math.min(top, bot)} height={Math.max(1, Math.abs(top - bot))}
            fill={up ? "var(--emerald)" : "var(--rose)"} fillOpacity="0.45" />
        );
      })}
    </svg>
  );
}

// ── RSI Panel ──────────────────────────────────────────────────────────────

function RSIChart({ values, height = 100 }: { values: (number | null)[]; height?: number }) {
  const plotH = height - PT - PB, n = values.length;
  const toY = yScale(0, 100, plotH);
  const plotW = W - PL - PR;
  const ptX = (i: number) => PL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const pts = values.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} className="block" preserveAspectRatio="none">
      {[70, 50, 30].map(level => (
        <g key={level}>
          <line x1={PL} x2={W - PR} y1={toY(level)} y2={toY(level)}
            stroke={level === 70 ? "var(--rose)" : level === 30 ? "var(--emerald)" : "var(--border)"}
            strokeOpacity={level === 50 ? "0.3" : "0.5"} strokeDasharray={level !== 50 ? "3,3" : undefined} />
          <text x={PL - 6} y={toY(level) + 3} textAnchor="end" fontSize="8" fill="var(--muted)" fontFamily="monospace">{level}</text>
        </g>
      ))}
      {pts.length > 1 && <polyline points={pts.join(" ")} fill="none" stroke="var(--amber)" strokeWidth="1.4" strokeLinejoin="round" />}
      <text x={PL + 4} y={PT + 10} fontSize="9" fill="var(--muted)" fontFamily="monospace">RSI(14)</text>
    </svg>
  );
}

// ── MACD Panel ─────────────────────────────────────────────────────────────

function MACDChart({ line, signalLine, histogram, height = 100 }: {
  line: (number | null)[]; signalLine: (number | null)[]; histogram: (number | null)[]; height?: number;
}) {
  const plotH = height - PT - PB, n = line.length;
  const allVals = [...line, ...signalLine, ...histogram].filter((v): v is number => v != null && Number.isFinite(v));
  if (!allVals.length) return null;
  const minV = Math.min(...allVals), maxV = Math.max(...allVals);
  const pad = (maxV - minV) * 0.05;
  const toY = yScale(minV - pad, maxV + pad, plotH);
  const zeroY = toY(0);
  const plotW = W - PL - PR;
  const ptX = (i: number) => PL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const bw = Math.max(1, (plotW / Math.max(1, n)) * 0.5);
  const cx = (i: number) => PL + (i + 0.5) * (plotW / Math.max(1, n));
  const linePts = line.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
  const sigPts = signalLine.map((v, i) => (v != null && Number.isFinite(v) ? `${ptX(i)},${toY(v)}` : null)).filter(Boolean) as string[];
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} className="block" preserveAspectRatio="none">
      <line x1={PL} x2={W - PR} y1={zeroY} y2={zeroY} stroke="var(--border)" strokeOpacity="0.5" />
      <text x={PL - 6} y={toY(0) + 3} textAnchor="end" fontSize="8" fill="var(--muted)" fontFamily="monospace">0</text>
      {histogram.map((v, i) => {
        if (v == null || !Number.isFinite(v)) return null;
        const top = toY(v), bot = zeroY;
        return (
          <rect key={i} x={cx(i) - bw / 2} width={bw}
            y={Math.min(top, bot)} height={Math.max(1, Math.abs(top - bot))}
            fill={v >= 0 ? "var(--emerald)" : "var(--rose)"} fillOpacity="0.5" />
        );
      })}
      {linePts.length > 1 && <polyline points={linePts.join(" ")} fill="none" stroke="var(--frost)" strokeWidth="1.4" strokeLinejoin="round" />}
      {sigPts.length > 1 && <polyline points={sigPts.join(" ")} fill="none" stroke="var(--amber)" strokeWidth="1.2" strokeLinejoin="round" />}
      <text x={PL + 4} y={PT + 10} fontSize="9" fill="var(--muted)" fontFamily="monospace">MACD(12,26,9)</text>
    </svg>
  );
}

// ── Metric card ────────────────────────────────────────────────────────────

function MetricCard({ label, value, color, suffix = "" }: { label: string; value: string; color?: string; suffix?: string }) {
  return (
    <div className="flex flex-col gap-0.5 p-3 rounded-lg bg-card border border-border">
      <span className="text-[10px] font-mono text-muted uppercase tracking-wider">{label}</span>
      <span className={`text-sm font-mono font-bold ${color ?? "text-foreground"}`}>{value}{suffix}</span>
    </div>
  );
}

// ── Search suggestions ─────────────────────────────────────────────────────

function SearchBox({ onSelect }: { onSelect: (sym: string) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<MarketSearchResult[]>([]);
  const [searching, setSearching] = useState(false);

  const search = useCallback(async (val: string) => {
    if (val.length < 1) { setResults([]); return; }
    setSearching(true);
    try {
      const res = await searchMarket(val);
      setResults(res.slice(0, 8));
    } catch { setResults([]); }
    finally { setSearching(false); }
  }, []);

  return (
    <div className="relative">
      <input
        className="w-40 px-3 py-1.5 rounded-lg bg-background-elevated border border-border text-sm font-mono text-foreground placeholder:text-muted focus:outline-none focus:border-frost/50 transition-colors"
        placeholder="Symbol…"
        value={q}
        onChange={e => { setQ(e.target.value.toUpperCase()); search(e.target.value); }}
        onKeyDown={e => { if (e.key === "Enter" && q) { onSelect(q); setResults([]); } }}
      />
      {results.length > 0 && (
        <div className="absolute top-full left-0 z-50 mt-1 w-72 rounded-lg bg-card border border-border shadow-xl overflow-hidden">
          {results.map(r => (
            <button key={r.symbol} className="w-full flex items-center justify-between px-3 py-2 hover:bg-card-hover text-left transition-colors"
              onClick={() => { setQ(r.symbol); setResults([]); onSelect(r.symbol); }}>
              <span className="text-sm font-mono font-bold text-frost">{r.symbol}</span>
              <span className="text-xs text-muted truncate max-w-[160px]">{r.name}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Indicator toggle button ────────────────────────────────────────────────

function ToggleBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick}
      className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all border ${active
        ? "border-frost/50 bg-frost/10 text-frost"
        : "border-border text-muted hover:border-border-accent hover:text-foreground-dim"}`}>
      {label}
    </button>
  );
}

// ── Period / Interval configs ──────────────────────────────────────────────

const PERIODS = [
  { label: "1M", period: "1mo", interval: "1d" },
  { label: "3M", period: "3mo", interval: "1d" },
  { label: "6M", period: "6mo", interval: "1d" },
  { label: "1Y", period: "1y", interval: "1d" },
  { label: "2Y", period: "2y", interval: "1wk" },
  { label: "5Y", period: "5y", interval: "1wk" },
];

// ── Watchlist helpers ──────────────────────────────────────────────────────

const WATCHLIST_KEY = "ygg-trading-watchlist";
const DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA", "TSLA", "BTC-USD", "ETH-USD"];

function loadWatchlist(): string[] {
  try { return JSON.parse(localStorage.getItem(WATCHLIST_KEY) ?? "") || DEFAULT_WATCHLIST; }
  catch { return DEFAULT_WATCHLIST; }
}
function saveWatchlist(list: string[]) {
  try { localStorage.setItem(WATCHLIST_KEY, JSON.stringify(list)); } catch { /* ignore */ }
}

function Watchlist({ active, onSelect, onAdd, onRemove }: {
  active: string; onSelect: (s: string) => void; onAdd: (s: string) => void; onRemove: (s: string) => void;
}) {
  const [list, setList] = useState<string[]>(DEFAULT_WATCHLIST);
  const [adding, setAdding] = useState("");
  // Load from localStorage on mount
  const [mounted, setMounted] = useState(false);
  if (typeof window !== "undefined" && !mounted) { setList(loadWatchlist()); setMounted(true); }

  const add = () => {
    const sym = adding.trim().toUpperCase();
    if (!sym || list.includes(sym)) { setAdding(""); return; }
    const next = [...list, sym];
    setList(next); saveWatchlist(next); setAdding(""); onAdd(sym);
  };
  const remove = (sym: string) => {
    const next = list.filter(s => s !== sym);
    setList(next); saveWatchlist(next); onRemove(sym);
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="text-[10px] font-mono text-muted uppercase tracking-wider shrink-0">Watchlist</span>
      {list.map(sym => (
        <div key={sym} className="group flex items-center gap-0 rounded overflow-hidden border border-border bg-card">
          <button onClick={() => onSelect(sym)}
            className={`px-2.5 py-1 text-[11px] font-mono transition-colors ${sym === active ? "bg-frost/10 text-frost" : "text-muted hover:text-foreground"}`}>
            {sym}
          </button>
          <button onClick={() => remove(sym)} title="Remove"
            className="hidden group-hover:flex px-1 py-1 text-muted hover:text-rose transition-colors text-[10px]">×</button>
        </div>
      ))}
      <div className="flex items-center gap-1">
        <input className="w-20 px-2 py-0.5 rounded border border-border bg-background-elevated text-[11px] font-mono text-foreground placeholder:text-muted focus:outline-none focus:border-frost/50"
          placeholder="+SYM" value={adding}
          onChange={e => setAdding(e.target.value.toUpperCase())}
          onKeyDown={e => { if (e.key === "Enter") add(); }} />
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function TradingPage() {
  const [symbol, setSymbol] = useState("AAPL");
  const [periodIdx, setPeriodIdx] = useState(3);
  const [bars, setBars] = useState<OHLCVBar[]>([]);
  const [quote, setQuote] = useState<MarketQuote | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Indicator toggles
  const [showMA, setShowMA] = useState(true);
  const [showBB, setShowBB] = useState(false);
  const [showRSI, setShowRSI] = useState(true);
  const [showMACD, setShowMACD] = useState(false);

  const load = useCallback(async (sym: string, pIdx: number) => {
    if (!sym) return;
    setLoading(true);
    setError(null);
    const { period, interval } = PERIODS[pIdx];
    try {
      const [ohlcv, q] = await Promise.allSettled([
        getMarketOHLCV(sym, period, interval),
        getMarketQuote(sym),
      ]);
      if (ohlcv.status === "fulfilled") setBars(ohlcv.value.bars);
      else { setBars([]); throw new Error((ohlcv.reason as Error).message); }
      if (q.status === "fulfilled") setQuote(q.value);
      else setQuote(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  // Derived indicator values from close prices
  const closes = bars.map(b => b.c ?? 0);
  const ma20 = closes.length ? sma(closes, 20) : [];
  const ma50 = closes.length ? sma(closes, 50) : [];
  const bb = closes.length ? bollingerBands(closes) : null;
  const rsiVals = closes.length ? rsi(closes) : [];
  const macdData = closes.length ? macdIndicator(closes) : null;
  const metrics = closes.length >= 10 ? computeMetrics(closes, PERIODS[periodIdx].interval === "1wk" ? 52 : 252) : null;
  const signals = closes.length >= 20 ? computeSignals(
    closes, rsiVals,
    macdData?.line ?? [], macdData?.signalLine ?? [],
    bb?.upper ?? [], bb?.lower ?? [],
    ma20, ma50,
  ) : [];

  const changeColor = quote?.change_pct != null
    ? quote.change_pct >= 0 ? "text-emerald" : "text-rose"
    : "text-muted";

  return (
    <div className="flex flex-col gap-4 p-4 min-h-0 animate-in">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--frost)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
            <polyline points="16 7 22 7 22 13" />
          </svg>
          <h1 className="text-lg font-bold tracking-tight text-foreground">Trading</h1>
        </div>

        {/* Symbol search */}
        <SearchBox onSelect={sym => { setSymbol(sym); load(sym, periodIdx); }} />

        {/* Period buttons */}
        <div className="flex gap-1">
          {PERIODS.map((p, i) => (
            <button key={p.label} onClick={() => { setPeriodIdx(i); load(symbol, i); }}
              className={`px-2.5 py-1 rounded text-[11px] font-mono transition-all border ${i === periodIdx
                ? "border-frost/50 bg-frost/10 text-frost"
                : "border-border text-muted hover:border-border-accent hover:text-foreground-dim"}`}>
              {p.label}
            </button>
          ))}
        </div>

        {/* Load button */}
        <button
          onClick={() => load(symbol, periodIdx)}
          disabled={loading}
          className="px-3 py-1.5 rounded-lg bg-frost/10 border border-frost/30 text-frost text-xs font-mono hover:bg-frost/20 transition-colors disabled:opacity-40"
        >
          {loading ? "Loading…" : "Load"}
        </button>

        {/* Quote strip */}
        {quote && (
          <div className="flex items-baseline gap-2 ml-2">
            <span className="text-xs font-mono text-muted">{quote.symbol}</span>
            <span className="text-xl font-bold font-mono text-foreground">${fmtPrice(quote.price)}</span>
            <span className={`text-sm font-mono ${changeColor}`}>
              {quote.change != null ? (quote.change >= 0 ? "+" : "") + fmtPrice(quote.change) : ""}
              {" "}({quote.change_pct != null ? (quote.change_pct >= 0 ? "+" : "") + quote.change_pct.toFixed(2) + "%" : ""})
            </span>
          </div>
        )}
      </div>

      {/* ── Watchlist ─────────────────────────────────────────── */}
      <Watchlist
        active={symbol}
        onSelect={sym => { setSymbol(sym); load(sym, periodIdx); }}
        onAdd={sym => { setSymbol(sym); load(sym, periodIdx); }}
        onRemove={() => {}}
      />

      {/* ── Error ─────────────────────────────────────────────── */}
      {error && (
        <div className="rounded-lg border border-rose/30 bg-rose/5 px-4 py-3 text-sm font-mono text-rose">
          {error.includes("yfinance") ? (
            <>Market data requires yfinance: <code className="text-amber">pip install yfinance</code> on the node</>
          ) : error}
        </div>
      )}

      {/* ── Loading spinner ───────────────────────────────────── */}
      {loading && (
        <div className="flex items-center gap-2 text-sm text-muted font-mono">
          <span className="w-3 h-3 border border-frost border-t-transparent rounded-full animate-spin" />
          Fetching {symbol}…
        </div>
      )}

      {/* ── Empty state ───────────────────────────────────────── */}
      {!loading && !error && bars.length === 0 && (
        <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
            <polyline points="16 7 22 7 22 13" />
          </svg>
          <p className="text-sm text-muted font-mono">Enter a symbol and click <span className="text-frost">Load</span> to fetch market data</p>
          <p className="text-xs text-muted/60 font-mono">Powered by yfinance — requires the market service to be enabled on the node</p>
        </div>
      )}

      {/* ── Charts ────────────────────────────────────────────── */}
      {bars.length > 0 && (
        <>
          {/* Indicator toggles */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-[10px] font-mono text-muted uppercase tracking-wider">Overlays</span>
            <ToggleBtn label="MA 20/50" active={showMA} onClick={() => setShowMA(v => !v)} />
            <ToggleBtn label="Bollinger" active={showBB} onClick={() => setShowBB(v => !v)} />
            <span className="text-[10px] font-mono text-muted uppercase tracking-wider ml-2">Indicators</span>
            <ToggleBtn label="RSI" active={showRSI} onClick={() => setShowRSI(v => !v)} />
            <ToggleBtn label="MACD" active={showMACD} onClick={() => setShowMACD(v => !v)} />
            <span className="text-[10px] font-mono text-muted ml-2">
              {bars.length.toLocaleString()} bars · {PERIODS[periodIdx].interval}
            </span>
          </div>

          {/* Legend */}
          {(showMA || showBB) && (
            <div className="flex gap-4 text-[10px] font-mono">
              {showMA && <><span className="text-emerald">— SMA20</span><span className="text-amber">— SMA50</span></>}
              {showBB && <span className="text-frost-dim">— Bollinger(20,2)</span>}
            </div>
          )}

          {/* Main candle chart */}
          <div className="rounded-xl border border-border overflow-hidden bg-card">
            <CandleChart
              bars={bars}
              ma20={showMA ? ma20 : undefined}
              ma50={showMA ? ma50 : undefined}
              bbUpper={showBB ? bb?.upper : undefined}
              bbLower={showBB ? bb?.lower : undefined}
              height={300}
            />
          </div>

          {/* Volume */}
          <div className="rounded-xl border border-border overflow-hidden bg-card -mt-2">
            <VolumeChart bars={bars} height={70} />
          </div>

          {/* RSI */}
          {showRSI && (
            <div className="rounded-xl border border-border overflow-hidden bg-card">
              <RSIChart values={rsiVals} height={100} />
            </div>
          )}

          {/* MACD */}
          {showMACD && macdData && (
            <div className="rounded-xl border border-border overflow-hidden bg-card">
              <MACDChart line={macdData.line} signalLine={macdData.signalLine} histogram={macdData.hist} height={110} />
            </div>
          )}

          {/* Signals */}
          {signals.length > 0 && (
            <div>
              <p className="text-[10px] font-mono text-muted uppercase tracking-wider mb-2">Signals</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-1.5">
                {signals.map((s, i) => <SignalBadge key={i} signal={s} />)}
              </div>
            </div>
          )}

          {/* Metrics */}
          {metrics && (
            <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
              <MetricCard label="Total Return" value={fmtPct(metrics.totalReturn)}
                color={(metrics.totalReturn ?? 0) >= 0 ? "text-emerald" : "text-rose"} />
              <MetricCard label="CAGR" value={fmtPct(metrics.cagr)}
                color={(metrics.cagr ?? 0) >= 0 ? "text-emerald" : "text-rose"} />
              <MetricCard label="Ann. Return" value={fmtPct(metrics.annReturn)}
                color={(metrics.annReturn ?? 0) >= 0 ? "text-emerald" : "text-rose"} />
              <MetricCard label="Ann. Vol" value={fmtPct(metrics.annVol)} />
              <MetricCard label="Sharpe" value={metrics.sharpe != null ? metrics.sharpe.toFixed(2) : "—"}
                color={metrics.sharpe != null ? (metrics.sharpe >= 1 ? "text-emerald" : metrics.sharpe >= 0 ? "text-amber" : "text-rose") : undefined} />
              <MetricCard label="Sortino" value={metrics.sortino != null ? metrics.sortino.toFixed(2) : "—"}
                color={metrics.sortino != null ? (metrics.sortino >= 1 ? "text-emerald" : metrics.sortino >= 0 ? "text-amber" : "text-rose") : undefined} />
              <MetricCard label="Max Drawdown" value={fmtPct(metrics.maxDrawdown)}
                color="text-rose" />
              <MetricCard label="Calmar" value={metrics.calmar != null ? metrics.calmar.toFixed(2) : "—"}
                color={metrics.calmar != null ? (metrics.calmar >= 0.5 ? "text-emerald" : "text-muted") : undefined} />
            </div>
          )}

          {/* Quote details */}
          {quote && (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
              <MetricCard label="Open" value={`$${fmtPrice(quote.open)}`} />
              <MetricCard label="Day High" value={`$${fmtPrice(quote.day_high)}`} color="text-emerald" />
              <MetricCard label="Day Low" value={`$${fmtPrice(quote.day_low)}`} color="text-rose" />
              <MetricCard label="Volume" value={fmtNum(quote.volume, 0)} />
              <MetricCard label="Market Cap" value={fmtNum(quote.market_cap, 1)} />
              <MetricCard label="Currency" value={quote.currency ?? "USD"} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
