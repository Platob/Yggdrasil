"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  node as api,
  type FxRateEntry,
  type FxHistoryPoint,
  type WatchlistEntry,
} from "@/lib/api";
import { formatRelative } from "@/lib/time";

// ── Icons ──────────────────────────────────────────────────────
const TrendUpIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /><polyline points="17 6 23 6 23 12" />
  </svg>
);
const TrendDownIcon = () => (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 18 13.5 8.5 8.5 13.5 1 6" /><polyline points="17 18 23 18 23 12" />
  </svg>
);
const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
  </svg>
);
const TrashIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    <path d="M10 11v6" /><path d="M14 11v6" /><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
  </svg>
);
const RefreshIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" />
    <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
  </svg>
);
const ArrowRightIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="5" y1="12" x2="19" y2="12" /><polyline points="12 5 19 12 12 19" />
  </svg>
);
const ChartIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" />
    <line x1="6" y1="20" x2="6" y2="14" />
  </svg>
);

// ── Mini sparkline ─────────────────────────────────────────────
function Sparkline({
  points,
  width = 80,
  height = 28,
  positive,
}: {
  points: number[];
  width?: number;
  height?: number;
  positive: boolean;
}) {
  if (points.length < 2) return null;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const xs = points.map((_, i) => (i / (points.length - 1)) * width);
  const ys = points.map((v) => height - ((v - min) / range) * height * 0.85 - height * 0.075);
  const d = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const color = positive ? "var(--success)" : "var(--destructive)";
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="opacity-80">
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Rate card ──────────────────────────────────────────────────
function RateCard({
  rate,
  history,
  prev,
  onRemove,
  onClick,
  selected,
}: {
  rate: FxRateEntry;
  history?: FxHistoryPoint[];
  prev?: FxRateEntry;
  onRemove: (pair: string) => void;
  onClick: (pair: string) => void;
  selected: boolean;
}) {
  const change = prev ? rate.value - prev.value : null;
  const pct = prev && prev.value !== 0 ? ((rate.value - prev.value) / prev.value) * 100 : null;
  const positive = change == null ? true : change >= 0;
  const sparkPoints = history ? history.map((p) => p.value) : [];

  return (
    <div
      onClick={() => onClick(rate.pair)}
      className="nordic-card p-4 cursor-pointer transition-all duration-150 group"
      style={{
        borderColor: selected ? "var(--primary)" : undefined,
        background: selected ? "rgba(242,107,58,0.06)" : undefined,
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div>
          <span className="font-mono font-bold text-base text-foreground tracking-wide">
            {rate.pair}
          </span>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="font-mono text-xl font-semibold" style={{ color: "var(--foreground)" }}>
              {rate.value.toFixed(4)}
            </span>
            {pct !== null && (
              <span
                className="flex items-center gap-0.5 text-[11px] font-semibold px-1.5 py-0.5 rounded"
                style={{
                  color: positive ? "var(--success)" : "var(--destructive)",
                  background: positive ? "color-mix(in srgb, var(--success) 12%, transparent)" : "color-mix(in srgb, var(--destructive) 12%, transparent)",
                }}
              >
                {positive ? <TrendUpIcon /> : <TrendDownIcon />}
                {pct > 0 ? "+" : ""}{pct.toFixed(3)}%
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          {sparkPoints.length > 1 && (
            <Sparkline points={sparkPoints} positive={positive} />
          )}
          <button
            onClick={(e) => { e.stopPropagation(); onRemove(rate.pair); }}
            className="opacity-0 group-hover:opacity-100 transition-opacity p-1 rounded hover:text-destructive"
            style={{ color: "var(--muted)" }}
            title="Remove from watchlist"
          >
            <TrashIcon />
          </button>
        </div>
      </div>
      <div className="flex items-center justify-between text-[11px] text-muted mt-1">
        <span>{rate.source} → {rate.target}</span>
        {change !== null && (
          <span style={{ color: positive ? "var(--success)" : "var(--destructive)" }}>
            {change > 0 ? "+" : ""}{change.toFixed(5)}
          </span>
        )}
      </div>
    </div>
  );
}

// ── History chart ──────────────────────────────────────────────
function HistoryChart({ points, pair }: { points: FxHistoryPoint[]; pair: string }) {
  if (points.length < 2) {
    return (
      <div className="flex items-center justify-center h-32 text-muted text-sm">
        Not enough data
      </div>
    );
  }
  const values = points.map((p) => p.value);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 0.0001;
  const W = 560, H = 120;
  const xs = values.map((_, i) => (i / (values.length - 1)) * W);
  const ys = values.map((v) => H - ((v - min) / range) * H * 0.88 - H * 0.06);
  const path = xs.map((x, i) => `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${ys[i].toFixed(1)}`).join(" ");
  const areaPath = path + ` L ${W} ${H} L 0 ${H} Z`;

  const positive = values[values.length - 1] >= values[0];
  const color = positive ? "var(--success)" : "var(--destructive)";

  const labelCount = Math.min(6, points.length);
  const labelStep = Math.floor((points.length - 1) / (labelCount - 1));
  const labels = Array.from({ length: labelCount }, (_, i) => {
    const idx = Math.min(i * labelStep, points.length - 1);
    return { x: xs[idx], label: new Date(points[idx].from_timestamp).toLocaleDateString("en", { month: "short", day: "numeric" }) };
  });

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H + 22}`} className="w-full" style={{ minWidth: "280px" }}>
        <defs>
          <linearGradient id="chartGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.18" />
            <stop offset="100%" stopColor={color} stopOpacity="0.01" />
          </linearGradient>
        </defs>
        <path d={areaPath} fill="url(#chartGrad)" />
        <path d={path} fill="none" stroke={color} strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
        {labels.map(({ x, label }, i) => (
          <text key={i} x={x} y={H + 16} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="monospace">
            {label}
          </text>
        ))}
        <text x="2" y="12" fontSize="9" fill="var(--muted)" fontFamily="monospace">{max.toFixed(4)}</text>
        <text x="2" y={H - 4} fontSize="9" fill="var(--muted)" fontFamily="monospace">{min.toFixed(4)}</text>
      </svg>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────
export default function MarketPage() {
  const [rates, setRates] = useState<FxRateEntry[]>([]);
  const [prevRates, setPrevRates] = useState<Map<string, FxRateEntry>>(new Map());
  const [history, setHistory] = useState<Map<string, FxHistoryPoint[]>>(new Map());
  const [watchlist, setWatchlist] = useState<WatchlistEntry[]>([]);
  const [selectedPair, setSelectedPair] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<string | null>(null);

  // Converter
  const [convAmount, setConvAmount] = useState("100");
  const [convFrom, setConvFrom] = useState("EUR");
  const [convTo, setConvTo] = useState("USD");
  const [convResult, setConvResult] = useState<number | null>(null);
  const [convLoading, setConvLoading] = useState(false);
  const [convError, setConvError] = useState<string | null>(null);

  // Add pair
  const [addPair, setAddPair] = useState("");
  const [addError, setAddError] = useState<string | null>(null);

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadRates = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true);
    else setRefreshing(true);
    setError(null);
    try {
      const wlData = await api.getWatchlist();
      setWatchlist(wlData.pairs);
      const rateData = await api.getWatchlistRates();
      setPrevRates((prev) => {
        const next = new Map(prev);
        for (const r of rates) next.set(r.pair, r);
        return next;
      });
      setRates(rateData.rates);
      setFetchedAt(rateData.fetched_at);
      if (!selectedPair && rateData.rates.length > 0) {
        setSelectedPair(rateData.rates[0].pair);
      }
    } catch (e) {
      setError(`Market data unavailable: ${e}`);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadHistory = useCallback(async (pair: string) => {
    if (history.has(pair)) return;
    try {
      const data = await api.getFxHistory(pair, 30);
      setHistory((prev) => new Map(prev).set(pair, data.points));
    } catch {
      // non-critical
    }
  }, [history]);

  useEffect(() => {
    loadRates();
    pollingRef.current = setInterval(() => loadRates(true), 60_000);
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [loadRates]);

  useEffect(() => {
    if (selectedPair) loadHistory(selectedPair);
  }, [selectedPair, loadHistory]);

  async function handleRemove(pair: string) {
    try {
      await api.removeFromWatchlist(pair);
      if (selectedPair === pair) setSelectedPair(null);
      await loadRates(true);
    } catch (e) {
      setError(`Failed to remove ${pair}: ${e}`);
    }
  }

  async function handleAdd(e: React.FormEvent) {
    e.preventDefault();
    setAddError(null);
    const p = addPair.trim().toUpperCase();
    if (!p) return;
    try {
      await api.addToWatchlist(p);
      setAddPair("");
      await loadRates(true);
    } catch (e) {
      setAddError(`Failed to add ${p}: ${e}`);
    }
  }

  async function handleConvert(e: React.FormEvent) {
    e.preventDefault();
    setConvError(null);
    setConvResult(null);
    setConvLoading(true);
    try {
      const amt = parseFloat(convAmount);
      if (isNaN(amt) || amt <= 0) throw new Error("Enter a valid positive amount");
      const data = await api.convertFx(amt, convFrom, convTo);
      setConvResult(data.result);
    } catch (e) {
      setConvError(String(e));
    } finally {
      setConvLoading(false);
    }
  }

  const selectedRate = rates.find((r) => r.pair === selectedPair) ?? null;
  const selectedHistory = selectedPair ? (history.get(selectedPair) ?? []) : [];

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
            <ChartIcon />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Market</h1>
            <p className="text-sm text-muted mt-0.5">
              FX rates · live watchlist · converter
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {fetchedAt && (
            <span className="text-[11px] text-muted font-mono">
              Updated {formatRelative(fetchedAt)}
            </span>
          )}
          {error && (
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
              <div className="status-dot pending" />
              <span className="text-xs font-medium text-warning">{error}</span>
            </div>
          )}
          <button
            onClick={() => loadRates(true)}
            disabled={refreshing}
            className="btn-ghost text-xs flex items-center gap-1.5"
          >
            <span className={refreshing ? "animate-spin" : ""}><RefreshIcon /></span>
            Refresh
          </button>
        </div>
      </div>

      {/* Layout: rates grid + detail panel */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: rate cards */}
        <div className="xl:col-span-2 space-y-4">
          {/* Add pair form */}
          <form onSubmit={handleAdd} className="flex items-center gap-2">
            <input
              type="text"
              value={addPair}
              onChange={(e) => setAddPair(e.target.value)}
              placeholder="EUR/USD, GBP/JPY…"
              className="input-nordic text-sm flex-1 font-mono uppercase"
              maxLength={10}
            />
            <button type="submit" className="btn-primary text-xs flex items-center gap-1.5">
              <PlusIcon /> Add pair
            </button>
          </form>
          {addError && (
            <p className="text-xs text-destructive">{addError}</p>
          )}

          {loading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="nordic-card p-4 animate-pulse h-24" />
              ))}
            </div>
          ) : rates.length === 0 ? (
            <div className="nordic-card p-8 text-center">
              <ChartIcon />
              <p className="text-sm text-muted mt-3">No pairs in watchlist</p>
              <p className="text-xs text-muted mt-1">Add pairs like EUR/USD above</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {rates.map((r) => (
                <RateCard
                  key={r.pair}
                  rate={r}
                  history={history.get(r.pair)}
                  prev={prevRates.get(r.pair)}
                  onRemove={handleRemove}
                  onClick={setSelectedPair}
                  selected={selectedPair === r.pair}
                />
              ))}
            </div>
          )}
        </div>

        {/* Right: detail + converter */}
        <div className="space-y-4">
          {/* Selected pair detail */}
          {selectedRate ? (
            <div className="nordic-card p-4 space-y-3">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-foreground font-mono">{selectedRate.pair}</h2>
                <span className="text-xs text-muted">{selectedRate.sampling}</span>
              </div>
              <HistoryChart points={selectedHistory} pair={selectedRate.pair} />
              {selectedHistory.length === 0 && (
                <p className="text-xs text-muted text-center py-4">Loading history…</p>
              )}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="space-y-1">
                  <p className="text-muted">Current</p>
                  <p className="font-mono font-semibold text-foreground">{selectedRate.value.toFixed(5)}</p>
                </div>
                {selectedHistory.length > 0 && (() => {
                  const vals = selectedHistory.map((p) => p.value);
                  return (
                    <>
                      <div className="space-y-1">
                        <p className="text-muted">30d High</p>
                        <p className="font-mono text-foreground" style={{ color: "var(--success)" }}>{Math.max(...vals).toFixed(5)}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-muted">30d Low</p>
                        <p className="font-mono text-foreground" style={{ color: "var(--destructive)" }}>{Math.min(...vals).toFixed(5)}</p>
                      </div>
                      <div className="space-y-1">
                        <p className="text-muted">30d Avg</p>
                        <p className="font-mono text-foreground">{(vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(5)}</p>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>
          ) : (
            <div className="nordic-card p-4 text-center text-sm text-muted">
              Select a pair to see detail
            </div>
          )}

          {/* Converter */}
          <div className="nordic-card p-4 space-y-3">
            <h2 className="text-sm font-semibold text-foreground">Converter</h2>
            <form onSubmit={handleConvert} className="space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <label className="block text-[11px] text-muted mb-1">Amount</label>
                  <input
                    type="number"
                    min="0.000001"
                    step="any"
                    value={convAmount}
                    onChange={(e) => setConvAmount(e.target.value)}
                    className="input-nordic w-full text-sm font-mono"
                  />
                </div>
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1">
                  <label className="block text-[11px] text-muted mb-1">From</label>
                  <input
                    type="text"
                    value={convFrom}
                    onChange={(e) => setConvFrom(e.target.value.toUpperCase())}
                    maxLength={3}
                    className="input-nordic w-full text-sm font-mono uppercase"
                    placeholder="EUR"
                  />
                </div>
                <div className="mt-4 text-muted"><ArrowRightIcon /></div>
                <div className="flex-1">
                  <label className="block text-[11px] text-muted mb-1">To</label>
                  <input
                    type="text"
                    value={convTo}
                    onChange={(e) => setConvTo(e.target.value.toUpperCase())}
                    maxLength={3}
                    className="input-nordic w-full text-sm font-mono uppercase"
                    placeholder="USD"
                  />
                </div>
              </div>
              <button type="submit" className="btn-primary text-sm w-full" disabled={convLoading}>
                {convLoading ? "Converting…" : "Convert"}
              </button>
            </form>
            {convError && <p className="text-xs text-destructive">{convError}</p>}
            {convResult !== null && (
              <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 text-center">
                <p className="text-[11px] text-muted mb-0.5">
                  {convAmount} {convFrom} =
                </p>
                <p className="text-lg font-mono font-bold" style={{ color: "var(--primary)" }}>
                  {convResult.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })} {convTo}
                </p>
              </div>
            )}
          </div>

          {/* Watchlist summary */}
          {watchlist.length > 0 && (
            <div className="nordic-card p-4 space-y-2">
              <h2 className="text-xs font-medium text-muted uppercase tracking-wider">Watchlist</h2>
              {watchlist.map((w) => (
                <div key={w.pair} className="flex items-center justify-between text-xs">
                  <span className="font-mono text-foreground">{w.pair}</span>
                  <button
                    onClick={() => handleRemove(w.pair)}
                    className="text-muted hover:text-destructive transition-colors"
                  >
                    <TrashIcon />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
