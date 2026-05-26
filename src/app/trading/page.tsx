"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { bot, type PricesResponse, type PortfolioResponse, type TechnicalIndicators, type PriceAlert } from "@/lib/api";

// ── Color palette ────────────────────────────────────────────
const SYMBOLS_FX = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"];
const SYMBOLS_CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD"];
const ALL_SYMBOLS = [...SYMBOLS_FX, ...SYMBOLS_CRYPTO];

const SYMBOL_COLORS: Record<string, string> = {
  "EUR/USD": "#5b9bd5",
  "GBP/USD": "#4ade80",
  "USD/JPY": "#fbbf24",
  "USD/CHF": "#f26b3a",
  "AUD/USD": "#a78bfa",
  "USD/CAD": "#fb7185",
  "BTC-USD": "#f59e0b",
  "ETH-USD": "#8b5cf6",
  "SOL-USD": "#06b6d4",
};

// ── Mini sparkline canvas ────────────────────────────────────
const SPARK_SIZE = 40;

function Spark({ data, color }: { data: number[]; color: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || data.length < 2) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = 80 * dpr;
    canvas.height = SPARK_SIZE * dpr;
    ctx.scale(dpr, dpr);
    const w = 80, h = SPARK_SIZE;
    ctx.clearRect(0, 0, w, h);
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const pad = 3;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = pad + (i / (data.length - 1)) * (w - pad * 2);
      const y = h - pad - ((v - min) / range) * (h - pad * 2);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.stroke();
  }, [data, color]);

  return <canvas ref={ref} style={{ width: 80, height: SPARK_SIZE, display: "block" }} />;
}

// ── Price card ───────────────────────────────────────────────
function PriceCard({
  symbol,
  price,
  stale,
  history,
  change,
}: {
  symbol: string;
  price: number;
  stale: boolean;
  history: number[];
  change: number;
}) {
  const color = SYMBOL_COLORS[symbol] ?? "#f26b3a";
  const isCrypto = symbol.endsWith("-USD");
  const fmt = isCrypto
    ? price >= 1000 ? price.toLocaleString("en-US", { minimumFractionDigits: 0, maximumFractionDigits: 0 })
      : price.toFixed(2)
    : price.toFixed(4);

  return (
    <div
      className="rounded-xl p-3 flex flex-col gap-1 transition-all"
      style={{
        background: "var(--card)",
        border: `1px solid ${stale ? "var(--warning)" : "var(--border)"}`,
      }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color }}>
          {symbol}
        </span>
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded"
          style={{
            background: change >= 0 ? "var(--success)22" : "var(--destructive)22",
            color: change >= 0 ? "var(--success)" : "var(--destructive)",
          }}
        >
          {change >= 0 ? "+" : ""}{change.toFixed(3)}%
        </span>
      </div>
      <div className="flex items-end justify-between gap-2">
        <span className="font-mono font-bold text-base text-foreground">{fmt}</span>
        <Spark data={history} color={color} />
      </div>
      {stale && (
        <span className="text-[9px] text-warning">stale data</span>
      )}
    </div>
  );
}

// ── RSI gauge ────────────────────────────────────────────────
function RsiGauge({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted font-mono text-sm">—</span>;
  const color = value > 70 ? "var(--destructive)" : value < 30 ? "var(--success)" : "var(--accent)";
  const label = value > 70 ? "Overbought" : value < 30 ? "Oversold" : "Neutral";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-2 rounded-full bg-border overflow-hidden">
        <div style={{ width: `${value}%`, background: color, height: "100%", borderRadius: 9999, transition: "width 0.5s" }} />
      </div>
      <span className="font-mono text-sm" style={{ color }}>{value.toFixed(1)}</span>
      <span className="text-muted text-xs">{label}</span>
    </div>
  );
}

// ── P&L badge ────────────────────────────────────────────────
function PnlBadge({ value }: { value: number | null }) {
  if (value === null) return <span className="text-muted">—</span>;
  return (
    <span
      className="font-mono text-sm font-semibold"
      style={{ color: value >= 0 ? "var(--success)" : "var(--destructive)" }}
    >
      {value >= 0 ? "+" : ""}${value.toFixed(2)}
    </span>
  );
}

// ── Add Position modal ────────────────────────────────────────
function AddPositionModal({ onAdd, onClose }: {
  onAdd: (symbol: string, qty: number, cost: number, currency: string) => void;
  onClose: () => void;
}) {
  const [symbol, setSymbol] = useState("BTC-USD");
  const [qty, setQty] = useState("0.1");
  const [cost, setCost] = useState("50000");
  const [currency, setCurrency] = useState("USD");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.6)" }}>
      <div className="rounded-xl p-6 w-80 space-y-4" style={{ background: "var(--card-elevated)", border: "1px solid var(--border-accent)" }}>
        <h3 className="font-semibold text-foreground">Add / Update Position</h3>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-muted mb-1 block">Symbol</label>
            <select value={symbol} onChange={e => setSymbol(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary">
              {ALL_SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted mb-1 block">Quantity</label>
            <input value={qty} onChange={e => setQty(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary" type="number" step="any" />
          </div>
          <div>
            <label className="text-xs text-muted mb-1 block">Avg Cost (per unit)</label>
            <input value={cost} onChange={e => setCost(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary" type="number" step="any" />
          </div>
          <div>
            <label className="text-xs text-muted mb-1 block">Currency</label>
            <input value={currency} onChange={e => setCurrency(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary" />
          </div>
        </div>
        <div className="flex gap-2 pt-1">
          <button onClick={onClose} className="flex-1 py-1.5 rounded text-sm text-muted border border-border hover:bg-card-hover transition-colors">Cancel</button>
          <button
            onClick={() => { onAdd(symbol, parseFloat(qty), parseFloat(cost), currency); onClose(); }}
            className="flex-1 py-1.5 rounded text-sm font-medium transition-colors"
            style={{ background: "var(--primary)", color: "#fff" }}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Add Alert modal ──────────────────────────────────────────
function AddAlertModal({ onAdd, onClose }: {
  onAdd: (symbol: string, condition: "above" | "below", price: number) => void;
  onClose: () => void;
}) {
  const [symbol, setSymbol] = useState("BTC-USD");
  const [condition, setCondition] = useState<"above" | "below">("above");
  const [price, setPrice] = useState("100000");

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: "rgba(0,0,0,0.6)" }}>
      <div className="rounded-xl p-6 w-72 space-y-4" style={{ background: "var(--card-elevated)", border: "1px solid var(--border-accent)" }}>
        <h3 className="font-semibold text-foreground">Set Price Alert</h3>
        <div className="space-y-3">
          <div>
            <label className="text-xs text-muted mb-1 block">Symbol</label>
            <select value={symbol} onChange={e => setSymbol(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary">
              {ALL_SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted mb-1 block">Condition</label>
            <select value={condition} onChange={e => setCondition(e.target.value as "above" | "below")} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary">
              <option value="above">Above</option>
              <option value="below">Below</option>
            </select>
          </div>
          <div>
            <label className="text-xs text-muted mb-1 block">Price</label>
            <input value={price} onChange={e => setPrice(e.target.value)} className="w-full bg-card border border-border rounded px-2 py-1.5 text-sm font-mono focus:outline-none focus:border-primary" type="number" step="any" />
          </div>
        </div>
        <div className="flex gap-2 pt-1">
          <button onClick={onClose} className="flex-1 py-1.5 rounded text-sm text-muted border border-border hover:bg-card-hover transition-colors">Cancel</button>
          <button
            onClick={() => { onAdd(symbol, condition, parseFloat(price)); onClose(); }}
            className="flex-1 py-1.5 rounded text-sm font-medium transition-colors"
            style={{ background: "var(--primary)", color: "#fff" }}
          >
            Set Alert
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Section header ────────────────────────────────────────────
function SectionHead({ title, action }: { title: string; action?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-3">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-muted">{title}</h2>
      {action}
    </div>
  );
}

// ── Mock fallback data ────────────────────────────────────────
function makeMockPrices(): PricesResponse {
  const now = new Date().toISOString();
  return {
    timestamp: now,
    prices: {
      "EUR/USD": { symbol: "EUR/USD", price: 1.0842 + (Math.random() - 0.5) * 0.002, currency: "USD", source: "mock", timestamp: now, stale: false },
      "GBP/USD": { symbol: "GBP/USD", price: 1.2714 + (Math.random() - 0.5) * 0.003, currency: "USD", source: "mock", timestamp: now, stale: false },
      "USD/JPY": { symbol: "USD/JPY", price: 148.32 + (Math.random() - 0.5) * 0.5, currency: "JPY", source: "mock", timestamp: now, stale: false },
      "USD/CHF": { symbol: "USD/CHF", price: 0.9021 + (Math.random() - 0.5) * 0.002, currency: "CHF", source: "mock", timestamp: now, stale: false },
      "AUD/USD": { symbol: "AUD/USD", price: 0.6512 + (Math.random() - 0.5) * 0.002, currency: "USD", source: "mock", timestamp: now, stale: false },
      "USD/CAD": { symbol: "USD/CAD", price: 1.3641 + (Math.random() - 0.5) * 0.003, currency: "CAD", source: "mock", timestamp: now, stale: false },
      "BTC-USD": { symbol: "BTC-USD", price: 94200 + (Math.random() - 0.5) * 800, currency: "USD", source: "mock", timestamp: now, stale: false },
      "ETH-USD": { symbol: "ETH-USD", price: 3512 + (Math.random() - 0.5) * 60, currency: "USD", source: "mock", timestamp: now, stale: false },
      "SOL-USD": { symbol: "SOL-USD", price: 182 + (Math.random() - 0.5) * 5, currency: "USD", source: "mock", timestamp: now, stale: false },
    },
  };
}

// ── Main component ────────────────────────────────────────────
const HISTORY_CAP = 60;

export default function TradingDashboard() {
  const [prices, setPrices] = useState<PricesResponse | null>(null);
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [indicators, setIndicators] = useState<TechnicalIndicators | null>(null);
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState<string>("BTC-USD");
  const [demoMode, setDemoMode] = useState(false);
  const [showAddPosition, setShowAddPosition] = useState(false);
  const [showAddAlert, setShowAddAlert] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(5000);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  // Price history per symbol (for sparklines + change%)
  const historyRef = useRef<Record<string, number[]>>({});

  const prevPricesRef = useRef<Record<string, number>>({});

  const fetchPrices = useCallback(async () => {
    try {
      const data = await bot.getPrices(ALL_SYMBOLS);
      setPrices(data);
      setLastUpdate(new Date().toLocaleTimeString());
      // Append to history
      for (const [sym, q] of Object.entries(data.prices)) {
        if (!historyRef.current[sym]) historyRef.current[sym] = [];
        historyRef.current[sym].push(q.price);
        if (historyRef.current[sym].length > HISTORY_CAP) historyRef.current[sym].shift();
        prevPricesRef.current[sym] = q.price;
      }
    } catch {
      setDemoMode(true);
      const mock = makeMockPrices();
      setPrices(mock);
      setLastUpdate(new Date().toLocaleTimeString());
      for (const [sym, q] of Object.entries(mock.prices)) {
        if (!historyRef.current[sym]) historyRef.current[sym] = [];
        historyRef.current[sym].push(q.price);
        if (historyRef.current[sym].length > HISTORY_CAP) historyRef.current[sym].shift();
      }
    }
  }, []);

  const fetchPortfolio = useCallback(async () => {
    try {
      const p = await bot.getPortfolio();
      setPortfolio(p);
    } catch {
      // ignore if bot unavailable
    }
  }, []);

  const fetchIndicators = useCallback(async (symbol: string) => {
    try {
      const ind = await bot.getIndicators(symbol);
      setIndicators(ind);
    } catch {
      // ignore
    }
  }, []);

  const fetchAlerts = useCallback(async () => {
    try {
      const a = await bot.getAlerts();
      setAlerts(a.alerts);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    fetchPrices();
    fetchPortfolio();
    fetchAlerts();
    const iv = setInterval(() => {
      fetchPrices();
      fetchPortfolio();
    }, refreshInterval);
    return () => clearInterval(iv);
  }, [fetchPrices, fetchPortfolio, fetchAlerts, refreshInterval]);

  useEffect(() => {
    fetchIndicators(selectedSymbol);
  }, [selectedSymbol, fetchIndicators]);

  const computeChange = (symbol: string, currentPrice: number): number => {
    const hist = historyRef.current[symbol];
    if (!hist || hist.length < 2) return 0;
    const first = hist[0];
    if (first === 0) return 0;
    return ((currentPrice - first) / first) * 100;
  };

  const handleAddPosition = async (symbol: string, qty: number, cost: number, currency: string) => {
    try {
      const updated = await bot.upsertPosition({ symbol, quantity: qty, avg_cost: cost, currency });
      setPortfolio(updated);
    } catch {
      // demo fallback
      const mockPos = {
        symbol, quantity: qty, avg_cost: cost, currency,
        current_price: prices?.prices[symbol]?.price ?? null,
        pnl: null, pnl_pct: null,
      };
      setPortfolio(prev => prev ? {
        ...prev,
        positions: [...prev.positions.filter(p => p.symbol !== symbol), mockPos],
        total_pnl: 0,
        total_value: 0,
        timestamp: new Date().toISOString(),
      } : {
        positions: [mockPos],
        total_value: 0,
        total_pnl: 0,
        currency: "USD",
        timestamp: new Date().toISOString(),
      });
    }
  };

  const handleRemovePosition = async (symbol: string) => {
    try {
      const updated = await bot.removePosition(symbol);
      setPortfolio(updated);
    } catch {
      setPortfolio(prev => prev ? { ...prev, positions: prev.positions.filter(p => p.symbol !== symbol) } : null);
    }
  };

  const handleAddAlert = async (symbol: string, condition: "above" | "below", price: number) => {
    try {
      const alert = await bot.createAlert({ symbol, condition, price });
      setAlerts(prev => [...prev, alert]);
    } catch {
      const mockAlert: PriceAlert = {
        id: Date.now().toString(),
        symbol, condition, price,
        created_at: new Date().toISOString(),
        triggered_at: null,
      };
      setAlerts(prev => [...prev, mockAlert]);
    }
  };

  const handleRemoveAlert = async (id: string) => {
    try { await bot.removeAlert(id); } catch { /* ignore */ }
    setAlerts(prev => prev.filter(a => a.id !== id));
  };

  return (
    <div className="p-6 space-y-6 animate-in max-w-[1400px]">
      {/* ── Header ── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-foreground">Trading</h1>
          <p className="text-sm text-muted mt-0.5">
            {demoMode ? "Demo mode — bot offline" : `Live data · updated ${lastUpdate ?? "—"}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">Refresh:</span>
            <select
              value={refreshInterval}
              onChange={e => setRefreshInterval(Number(e.target.value))}
              className="bg-card border border-border rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-primary"
            >
              <option value={2000}>2s</option>
              <option value={5000}>5s</option>
              <option value={10000}>10s</option>
              <option value={30000}>30s</option>
            </select>
          </div>
          <div
            className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium"
            style={{
              background: demoMode ? "var(--warning)22" : "var(--success)22",
              color: demoMode ? "var(--warning)" : "var(--success)",
              border: `1px solid ${demoMode ? "var(--warning)44" : "var(--success)44"}`,
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: demoMode ? "var(--warning)" : "var(--success)" }} />
            {demoMode ? "Demo" : "Live"}
          </div>
        </div>
      </div>

      {/* ── FX Prices ── */}
      <section>
        <SectionHead title="FX Majors" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {SYMBOLS_FX.map(sym => {
            const q = prices?.prices[sym];
            const hist = historyRef.current[sym] ?? [];
            return (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className="text-left transition-all"
                style={{ outline: selectedSymbol === sym ? `2px solid ${SYMBOL_COLORS[sym]}` : "none", borderRadius: "0.75rem" }}
              >
                <PriceCard
                  symbol={sym}
                  price={q?.price ?? 0}
                  stale={q?.stale ?? false}
                  history={hist}
                  change={q ? computeChange(sym, q.price) : 0}
                />
              </button>
            );
          })}
        </div>
      </section>

      {/* ── Crypto Prices ── */}
      <section>
        <SectionHead title="Crypto" />
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {SYMBOLS_CRYPTO.map(sym => {
            const q = prices?.prices[sym];
            const hist = historyRef.current[sym] ?? [];
            return (
              <button
                key={sym}
                onClick={() => setSelectedSymbol(sym)}
                className="text-left transition-all"
                style={{ outline: selectedSymbol === sym ? `2px solid ${SYMBOL_COLORS[sym]}` : "none", borderRadius: "0.75rem" }}
              >
                <PriceCard
                  symbol={sym}
                  price={q?.price ?? 0}
                  stale={q?.stale ?? false}
                  history={hist}
                  change={q ? computeChange(sym, q.price) : 0}
                />
              </button>
            );
          })}
        </div>
      </section>

      {/* ── Indicators + Portfolio ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Technical Indicators */}
        <div className="rounded-xl p-4 space-y-3" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
          <SectionHead
            title={`Indicators — ${selectedSymbol}`}
            action={
              <select
                value={selectedSymbol}
                onChange={e => setSelectedSymbol(e.target.value)}
                className="bg-card border border-border rounded px-2 py-1 text-xs font-mono focus:outline-none focus:border-primary"
              >
                {ALL_SYMBOLS.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            }
          />
          {indicators ? (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg p-2.5" style={{ background: "var(--background-elevated)" }}>
                  <span className="text-[10px] text-muted block mb-1">SMA 20</span>
                  <span className="font-mono text-sm text-foreground">
                    {indicators.sma_20 !== null ? indicators.sma_20.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="rounded-lg p-2.5" style={{ background: "var(--background-elevated)" }}>
                  <span className="text-[10px] text-muted block mb-1">SMA 50</span>
                  <span className="font-mono text-sm text-foreground">
                    {indicators.sma_50 !== null ? indicators.sma_50.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="rounded-lg p-2.5" style={{ background: "var(--background-elevated)" }}>
                  <span className="text-[10px] text-muted block mb-1">EMA 20</span>
                  <span className="font-mono text-sm text-foreground">
                    {indicators.ema_20 !== null ? indicators.ema_20.toFixed(4) : "—"}
                  </span>
                </div>
                <div className="rounded-lg p-2.5" style={{ background: "var(--background-elevated)" }}>
                  <span className="text-[10px] text-muted block mb-1">Current Price</span>
                  <span className="font-mono text-sm text-foreground">
                    {indicators.price !== null ? indicators.price.toFixed(4) : "—"}
                  </span>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-[10px] text-muted mb-1">
                  <span>RSI 14</span>
                  <span>Oversold · Neutral · Overbought</span>
                </div>
                <RsiGauge value={indicators.rsi_14} />
              </div>
            </div>
          ) : (
            <div className="text-center py-6 text-muted text-sm">Select a symbol to view indicators</div>
          )}
        </div>

        {/* Portfolio */}
        <div className="rounded-xl p-4" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
          <SectionHead
            title="Portfolio"
            action={
              <button
                onClick={() => setShowAddPosition(true)}
                className="text-[11px] font-medium px-2.5 py-1 rounded-lg transition-colors"
                style={{ background: "var(--primary)22", color: "var(--primary)", border: "1px solid var(--primary)44" }}
              >
                + Add
              </button>
            }
          />
          {portfolio && portfolio.positions.length > 0 ? (
            <>
              <div className="space-y-1 mb-3 max-h-[200px] overflow-y-auto">
                <div className="grid grid-cols-[1fr_80px_80px_80px_24px] gap-1 text-[10px] text-muted uppercase tracking-wider pb-1 border-b border-border mb-1">
                  <span>Symbol</span>
                  <span className="text-right">Qty</span>
                  <span className="text-right">Avg Cost</span>
                  <span className="text-right">P&amp;L</span>
                  <span />
                </div>
                {portfolio.positions.map(pos => (
                  <div key={pos.symbol} className="grid grid-cols-[1fr_80px_80px_80px_24px] gap-1 items-center py-1 hover:bg-card-hover rounded transition-colors px-1">
                    <span className="font-mono text-xs font-semibold" style={{ color: SYMBOL_COLORS[pos.symbol] ?? "var(--foreground)" }}>
                      {pos.symbol}
                    </span>
                    <span className="font-mono text-xs text-right text-foreground-dim">{pos.quantity}</span>
                    <span className="font-mono text-xs text-right text-foreground-dim">{pos.avg_cost.toFixed(2)}</span>
                    <PnlBadge value={pos.pnl} />
                    <button
                      onClick={() => handleRemovePosition(pos.symbol)}
                      className="text-muted hover:text-destructive transition-colors text-xs"
                    >×</button>
                  </div>
                ))}
              </div>
              <div className="border-t border-border pt-2 flex justify-between text-sm">
                <span className="text-muted">Total P&amp;L</span>
                <PnlBadge value={portfolio.total_pnl} />
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 gap-2">
              <span className="text-muted text-sm">No positions yet</span>
              <button
                onClick={() => setShowAddPosition(true)}
                className="text-xs px-3 py-1.5 rounded-lg font-medium transition-colors"
                style={{ background: "var(--primary)", color: "#fff" }}
              >
                Add first position
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Alerts ── */}
      <section>
        <SectionHead
          title="Price Alerts"
          action={
            <button
              onClick={() => setShowAddAlert(true)}
              className="text-[11px] font-medium px-2.5 py-1 rounded-lg transition-colors"
              style={{ background: "var(--primary)22", color: "var(--primary)", border: "1px solid var(--primary)44" }}
            >
              + Alert
            </button>
          }
        />
        {alerts.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {alerts.map(alert => (
              <div
                key={alert.id}
                className="rounded-xl p-3 flex items-center justify-between gap-2"
                style={{
                  background: "var(--card)",
                  border: `1px solid ${alert.triggered_at ? "var(--success)" : "var(--border)"}`,
                }}
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-semibold" style={{ color: SYMBOL_COLORS[alert.symbol] ?? "var(--foreground)" }}>
                      {alert.symbol}
                    </span>
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{
                        background: alert.condition === "above" ? "var(--success)22" : "var(--destructive)22",
                        color: alert.condition === "above" ? "var(--success)" : "var(--destructive)",
                      }}
                    >
                      {alert.condition} {alert.price.toLocaleString()}
                    </span>
                  </div>
                  {alert.triggered_at && (
                    <span className="text-[10px] text-success">Triggered</span>
                  )}
                </div>
                <button onClick={() => handleRemoveAlert(alert.id)} className="text-muted hover:text-destructive transition-colors shrink-0">×</button>
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded-xl p-6 text-center text-muted text-sm" style={{ background: "var(--card)", border: "1px solid var(--border)" }}>
            No alerts set — prices will notify you when thresholds are hit.
          </div>
        )}
      </section>

      {/* Modals */}
      {showAddPosition && (
        <AddPositionModal
          onAdd={handleAddPosition}
          onClose={() => setShowAddPosition(false)}
        />
      )}
      {showAddAlert && (
        <AddAlertModal
          onAdd={handleAddAlert}
          onClose={() => setShowAddAlert(false)}
        />
      )}
    </div>
  );
}
