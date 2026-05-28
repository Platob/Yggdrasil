"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  getPrices, getPortfolio, getOrders, placeOrder, cancelOrder,
  getSignals, getAlerts, createAlert, deleteAlert,
  type PriceQuote, type PortfolioSummary, type Order, type TradingSignal,
  type PriceAlert, type OrderSide,
} from "@/lib/api";

const SIGNAL_COLOR: Record<TradingSignal["signal"], string> = {
  strong_buy: "#16a34a",
  buy: "#4ade80",
  hold: "#9ca3af",
  sell: "#f97316",
  strong_sell: "#dc2626",
};

function fmtMoney(n: number, decimals = 2): string {
  return new Intl.NumberFormat("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(n);
}

function fmtPct(n: number): string {
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

function pnlColor(n: number): string {
  return n > 0 ? "#4ade80" : n < 0 ? "#f87171" : "var(--muted)";
}

// ── Top P&L strip ──────────────────────────────────────────────
function PortfolioStrip({ p }: { p: PortfolioSummary | null }) {
  if (!p) return null;
  const items = [
    { label: "Total Value", value: `$${fmtMoney(p.total_value)}`, color: "var(--foreground)" },
    { label: "Cash", value: `$${fmtMoney(p.cash)}`, color: "var(--foreground)" },
    { label: "Equity", value: `$${fmtMoney(p.equity)}`, color: "var(--foreground)" },
    { label: "P&L", value: `${p.total_pnl >= 0 ? "+" : ""}$${fmtMoney(p.total_pnl)}`, color: pnlColor(p.total_pnl) },
    { label: "P&L %", value: fmtPct(p.total_pnl_pct), color: pnlColor(p.total_pnl) },
  ];
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      {items.map(it => (
        <div key={it.label} className="nordic-card p-4">
          <div className="text-[10px] uppercase tracking-wider text-muted">{it.label}</div>
          <div className="text-xl font-mono font-semibold mt-1" style={{ color: it.color }}>{it.value}</div>
        </div>
      ))}
    </div>
  );
}

// ── Price grid ─────────────────────────────────────────────────
function PriceGrid({
  prices, onSelect, selected,
}: {
  prices: PriceQuote[];
  onSelect: (sym: string) => void;
  selected: string | null;
}) {
  return (
    <div className="nordic-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs font-medium text-muted uppercase tracking-wider">Watched Symbols</h3>
        <span className="text-[10px] text-muted">{prices.length} symbols</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2">
        {prices.map(q => {
          const up = q.change_pct >= 0;
          const active = selected === q.symbol;
          return (
            <button
              key={q.symbol}
              onClick={() => onSelect(q.symbol)}
              className="text-left rounded-lg p-3 transition-colors border"
              style={{
                background: active ? "rgba(242,107,58,0.08)" : "var(--card-bg)",
                borderColor: active ? "rgba(242,107,58,0.3)" : "var(--border)",
              }}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-sm font-semibold text-foreground">{q.symbol}</span>
                <span className="text-[10px] uppercase font-medium" style={{ color: up ? "#4ade80" : "#f87171" }}>
                  {up ? "▲" : "▼"} {fmtPct(q.change_pct)}
                </span>
              </div>
              <div className="font-mono text-base mt-1.5" style={{ color: up ? "#4ade80" : "#f87171" }}>
                ${fmtMoney(q.price, q.price > 100 ? 2 : 4)}
              </div>
              <div className="text-[10px] text-muted mt-0.5">Vol {Math.round(q.volume / 1000)}K</div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ── Positions table ────────────────────────────────────────────
function PositionsTable({ p }: { p: PortfolioSummary | null }) {
  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Positions</h3>
      {!p || p.positions.length === 0 ? (
        <div className="text-center py-6 text-sm text-muted">No open positions</div>
      ) : (
        <div className="space-y-1">
          <div className="grid grid-cols-[1fr_60px_80px_80px_80px_70px] gap-2 text-[10px] text-muted uppercase tracking-wider pb-2 border-b border-border">
            <span>Symbol</span>
            <span className="text-right">Qty</span>
            <span className="text-right">Avg</span>
            <span className="text-right">Current</span>
            <span className="text-right">P&L</span>
            <span className="text-right">%</span>
          </div>
          {p.positions.map(pos => (
            <div key={pos.id} className="grid grid-cols-[1fr_60px_80px_80px_80px_70px] gap-2 py-1.5 text-sm hover:bg-card-hover rounded transition-colors">
              <span className="font-mono font-semibold text-foreground">{pos.symbol}</span>
              <span className="font-mono text-muted text-right">{pos.qty}</span>
              <span className="font-mono text-muted text-right">${fmtMoney(pos.avg_price)}</span>
              <span className="font-mono text-foreground text-right">${fmtMoney(pos.current_price)}</span>
              <span className="font-mono text-right" style={{ color: pnlColor(pos.pnl) }}>
                {pos.pnl >= 0 ? "+" : ""}${fmtMoney(pos.pnl)}
              </span>
              <span className="font-mono text-right" style={{ color: pnlColor(pos.pnl) }}>
                {fmtPct(pos.pnl_pct)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Quick order form ───────────────────────────────────────────
function OrderForm({
  symbols, defaultSymbol, onPlaced,
}: {
  symbols: string[];
  defaultSymbol: string;
  onPlaced: () => void;
}) {
  // Derived state from props: when defaultSymbol changes (parent click),
  // reset the user override. Tracking the previous prop value avoids the
  // "setState in effect" cascade.
  const [override, setOverride] = useState<string | null>(null);
  const [prevDefault, setPrevDefault] = useState(defaultSymbol);
  if (prevDefault !== defaultSymbol) {
    setPrevDefault(defaultSymbol);
    setOverride(null);
  }
  const symbol = override ?? defaultSymbol;
  const setSymbol = (s: string) => setOverride(s);
  const [side, setSide] = useState<OrderSide>("buy");
  const [qty, setQty] = useState(1);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      await placeOrder({ symbol, side, qty });
      onPlaced();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="nordic-card p-4 space-y-3">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider">Quick Order</h3>
      <div>
        <label className="text-[10px] uppercase tracking-wider text-muted block mb-1">Symbol</label>
        <select
          value={symbol}
          onChange={e => setSymbol(e.target.value)}
          className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none"
          style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        >
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          onClick={() => setSide("buy")}
          className="py-2 rounded-md text-sm font-semibold transition-colors"
          style={{
            background: side === "buy" ? "rgba(74,222,128,0.15)" : "var(--card-bg)",
            border: `1px solid ${side === "buy" ? "#4ade80" : "var(--border)"}`,
            color: side === "buy" ? "#4ade80" : "var(--muted)",
          }}
        >
          BUY
        </button>
        <button
          type="button"
          onClick={() => setSide("sell")}
          className="py-2 rounded-md text-sm font-semibold transition-colors"
          style={{
            background: side === "sell" ? "rgba(248,113,113,0.15)" : "var(--card-bg)",
            border: `1px solid ${side === "sell" ? "#f87171" : "var(--border)"}`,
            color: side === "sell" ? "#f87171" : "var(--muted)",
          }}
        >
          SELL
        </button>
      </div>
      <div>
        <label className="text-[10px] uppercase tracking-wider text-muted block mb-1">Quantity</label>
        <input
          type="number"
          min={0.0001}
          step={0.1}
          value={qty}
          onChange={e => setQty(Number(e.target.value))}
          className="w-full px-3 py-2 rounded-md text-sm font-mono outline-none"
          style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        />
      </div>
      <button
        type="submit"
        disabled={busy || qty <= 0}
        className="w-full py-2.5 rounded-md font-semibold text-sm transition-colors disabled:opacity-50"
        style={{
          background: side === "buy"
            ? "linear-gradient(135deg, #16a34a, #4ade80)"
            : "linear-gradient(135deg, #dc2626, #f87171)",
          color: "#fff",
        }}
      >
        {busy ? "Placing…" : `${side.toUpperCase()} ${qty} ${symbol}`}
      </button>
      {err && <div className="text-xs text-red-400">{err}</div>}
    </form>
  );
}

// ── Signals panel ──────────────────────────────────────────────
function SignalsPanel({ signals }: { signals: TradingSignal[] }) {
  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Signals</h3>
      <div className="space-y-1.5 max-h-80 overflow-y-auto">
        {signals.map(s => (
          <div key={s.symbol} className="flex items-center justify-between py-1.5 px-2 rounded hover:bg-card-hover">
            <span className="font-mono text-sm text-foreground">{s.symbol}</span>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted">{(s.confidence * 100).toFixed(0)}%</span>
              <span
                className="text-[10px] uppercase font-semibold px-2 py-0.5 rounded"
                style={{ background: SIGNAL_COLOR[s.signal] + "30", color: SIGNAL_COLOR[s.signal] }}
              >
                {s.signal.replace("_", " ")}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Alerts panel ───────────────────────────────────────────────
function AlertsPanel({
  alerts, symbols, onRefresh,
}: {
  alerts: PriceAlert[];
  symbols: string[];
  onRefresh: () => void;
}) {
  const [symbol, setSymbol] = useState(symbols[0] ?? "AAPL");
  const [condition, setCondition] = useState<"above" | "below">("above");
  const [threshold, setThreshold] = useState(0);

  async function add() {
    await createAlert(symbol, condition, threshold);
    setThreshold(0);
    onRefresh();
  }

  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Alerts</h3>
      <div className="flex gap-1.5 mb-3">
        <select
          value={symbol}
          onChange={e => setSymbol(e.target.value)}
          className="flex-1 px-2 py-1.5 text-xs font-mono rounded outline-none"
          style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        >
          {symbols.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <select
          value={condition}
          onChange={e => setCondition(e.target.value as "above" | "below")}
          className="px-2 py-1.5 text-xs rounded outline-none"
          style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        >
          <option value="above">above</option>
          <option value="below">below</option>
        </select>
        <input
          type="number"
          value={threshold}
          onChange={e => setThreshold(Number(e.target.value))}
          placeholder="$"
          className="w-20 px-2 py-1.5 text-xs font-mono rounded outline-none"
          style={{ background: "var(--card-bg)", border: "1px solid var(--border)", color: "var(--foreground)" }}
        />
        <button onClick={add} className="px-3 py-1.5 text-xs rounded bg-primary/15 text-primary hover:bg-primary/25">
          Add
        </button>
      </div>
      <div className="space-y-1 max-h-40 overflow-y-auto">
        {alerts.length === 0 && <div className="text-xs text-muted text-center py-2">No alerts</div>}
        {alerts.map(a => (
          <div key={a.id} className="flex items-center justify-between text-xs py-1 px-2 rounded hover:bg-card-hover">
            <span className="font-mono">
              {a.symbol} {a.condition} ${a.threshold}
              {a.triggered && <span className="ml-2 text-amber-400">TRIGGERED</span>}
            </span>
            <button onClick={async () => { await deleteAlert(a.id); onRefresh(); }} className="text-muted hover:text-red-400">
              ✕
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Orders table ───────────────────────────────────────────────
function OrdersTable({ orders, onRefresh }: { orders: Order[]; onRefresh: () => void }) {
  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Recent Orders</h3>
      <div className="space-y-1 max-h-60 overflow-y-auto">
        {orders.length === 0 && <div className="text-xs text-muted text-center py-2">No orders yet</div>}
        {orders.slice(0, 12).map(o => (
          <div key={o.id} className="grid grid-cols-[60px_70px_80px_60px_80px_30px] gap-2 text-xs items-center py-1 hover:bg-card-hover rounded px-1">
            <span className="font-mono font-semibold">{o.symbol}</span>
            <span style={{ color: o.side === "buy" ? "#4ade80" : "#f87171" }}>{o.side.toUpperCase()}</span>
            <span className="font-mono">{o.qty}</span>
            <span className="text-muted">{o.order_type}</span>
            <span className="text-muted">{o.status}</span>
            {o.status === "pending" && (
              <button onClick={async () => { await cancelOrder(o.id); onRefresh(); }} className="text-muted hover:text-red-400">✕</button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────
export default function TradingPage() {
  const [prices, setPrices] = useState<PriceQuote[]>([]);
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [orders, setOrders] = useState<Order[]>([]);
  const [signals, setSignals] = useState<TradingSignal[]>([]);
  const [alerts, setAlerts] = useState<PriceAlert[]>([]);
  const [selected, setSelected] = useState<string>("AAPL");
  const [demoMode, setDemoMode] = useState(false);

  const symbols = useMemo(() => prices.map(p => p.symbol), [prices]);

  const refreshAll = useCallback(async () => {
    try {
      const [pr, po, or, si, al] = await Promise.all([
        getPrices(), getPortfolio(), getOrders(), getSignals(), getAlerts(),
      ]);
      setPrices(pr); setPortfolio(po); setOrders(or); setSignals(si); setAlerts(al);
      setDemoMode(false);
    } catch {
      setDemoMode(true);
    }
  }, []);

  useEffect(() => {
    // refreshAll runs async fetches then setState — the rule cannot see
    // through the async boundary, but the call itself does not setState
    // synchronously. Subscription/polling is exactly what useEffect is for.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refreshAll();
    const id = setInterval(refreshAll, 3000);
    return () => clearInterval(id);
  }, [refreshAll]);

  return (
    <div className="p-6 space-y-4 animate-in">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-foreground">Trading</h1>
          <p className="text-sm text-muted mt-0.5">Paper-trading dashboard · simulated prices</p>
        </div>
        {demoMode && (
          <span className="text-xs px-3 py-1 rounded-full bg-warning/10 text-warning border border-warning/20">
            Offline · backend unreachable
          </span>
        )}
      </div>

      <PortfolioStrip p={portfolio} />

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
        <div className="space-y-4">
          <PriceGrid prices={prices} onSelect={setSelected} selected={selected} />
          <PositionsTable p={portfolio} />
          <OrdersTable orders={orders} onRefresh={refreshAll} />
        </div>
        <div className="space-y-4">
          <OrderForm symbols={symbols.length > 0 ? symbols : ["AAPL"]} defaultSymbol={selected} onPlaced={refreshAll} />
          <SignalsPanel signals={signals} />
          <AlertsPanel alerts={alerts} symbols={symbols.length > 0 ? symbols : ["AAPL"]} onRefresh={refreshAll} />
        </div>
      </div>
    </div>
  );
}
