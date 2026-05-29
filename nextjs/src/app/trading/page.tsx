"use client";

import { useEffect, useRef, useState } from "react";
import {
  createSignalStream,
  deletePosition,
  emitSignal,
  getPortfolio,
  getSignals,
  upsertPosition,
} from "@/lib/api";
import type { Portfolio, Position, TradeSignal } from "@/lib/types";

type Direction = "buy" | "sell" | "hold";

function directionStyle(d: Direction): { bg: string; text: string; border: string } {
  if (d === "buy") return { bg: "rgba(52,211,153,0.12)", text: "var(--emerald)", border: "rgba(52,211,153,0.3)" };
  if (d === "sell") return { bg: "rgba(244,63,94,0.12)", text: "var(--rose)", border: "rgba(244,63,94,0.3)" };
  return { bg: "rgba(103,232,249,0.08)", text: "var(--frost)", border: "rgba(103,232,249,0.2)" };
}

function pnlColor(pnl: number | null): string {
  if (pnl == null) return "var(--muted)";
  if (pnl > 0) return "var(--emerald)";
  if (pnl < 0) return "var(--rose)";
  return "var(--foreground-dim)";
}

function timeAgo(ts: string): string {
  try {
    const diff = Date.now() - new Date(ts).getTime();
    const s = Math.floor(diff / 1000);
    if (s < 60) return `${s}s ago`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ago`;
    return `${Math.floor(m / 60)}h ago`;
  } catch {
    return "";
  }
}

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "var(--emerald)" : pct >= 50 ? "var(--amber)" : "var(--rose)";
  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-1 rounded-full bg-white/[0.06] overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[10px] font-mono" style={{ color }}>{pct}%</span>
    </div>
  );
}

function SignalRow({ sig, flash }: { sig: TradeSignal; flash: boolean }) {
  const ds = directionStyle(sig.direction as Direction);
  return (
    <div
      className="flex items-center gap-3 px-3 py-2 rounded-lg transition-colors"
      style={{
        background: flash ? "rgba(103,232,249,0.06)" : "rgba(255,255,255,0.02)",
        transition: "background 0.8s",
      }}
    >
      <span className="text-[10px] font-mono text-muted w-14 shrink-0">{timeAgo(sig.created_at)}</span>
      <span
        className="text-xs font-bold font-mono px-1.5 py-0.5 rounded shrink-0"
        style={{ background: ds.bg, color: ds.text, border: `1px solid ${ds.border}` }}
      >
        {sig.direction.toUpperCase()}
      </span>
      <span className="text-sm font-bold font-mono text-frost shrink-0 w-20">{sig.symbol}</span>
      <span className="text-xs text-foreground-dim flex-1 truncate">{sig.name}</span>
      {sig.price != null && (
        <span className="text-xs font-mono text-foreground-dim shrink-0">
          ${sig.price.toFixed(2)}
        </span>
      )}
      <ConfidenceBar value={sig.confidence} />
    </div>
  );
}

function PositionRow({ pos, onDelete }: { pos: Position; onDelete: (s: string) => void }) {
  const pnlColor_ = pnlColor(pos.pnl ?? null);
  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors group">
      <span className="text-sm font-bold font-mono text-frost w-20 shrink-0">{pos.symbol}</span>
      <span className="text-xs font-mono text-foreground-dim shrink-0">
        {pos.qty > 0 ? "+" : ""}{pos.qty}
      </span>
      <span className="text-xs font-mono text-foreground-dim shrink-0">
        avg ${pos.avg_price.toFixed(2)}
      </span>
      {pos.current_price != null && (
        <span className="text-xs font-mono text-foreground-dim shrink-0">
          now ${pos.current_price.toFixed(2)}
        </span>
      )}
      <span className="flex-1" />
      {pos.pnl != null && (
        <span className="text-xs font-bold font-mono shrink-0" style={{ color: pnlColor_ }}>
          {pos.pnl >= 0 ? "+" : ""}${pos.pnl.toFixed(2)}
          {pos.pnl_pct != null && ` (${pos.pnl_pct > 0 ? "+" : ""}${pos.pnl_pct.toFixed(1)}%)`}
        </span>
      )}
      <button
        onClick={() => onDelete(pos.symbol)}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-muted hover:text-rose ml-2"
        title="Remove position"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
        </svg>
      </button>
    </div>
  );
}

export default function TradingPage() {
  const [signals, setSignals] = useState<TradeSignal[]>([]);
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null);
  const [symbolFilter, setSymbolFilter] = useState("");
  const [flashIds, setFlashIds] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);

  // new signal form
  const [sigName, setSigName] = useState("");
  const [sigSymbol, setSigSymbol] = useState("");
  const [sigDir, setSigDir] = useState<Direction>("buy");
  const [sigPrice, setSigPrice] = useState("");
  const [sigConf, setSigConf] = useState("1.0");
  const [sigSaving, setSigSaving] = useState(false);

  // new position form
  const [posSymbol, setPosSymbol] = useState("");
  const [posQty, setPosQty] = useState("");
  const [posAvg, setPosAvg] = useState("");
  const [posCurrent, setPosCurrent] = useState("");
  const [posSaving, setPosSaving] = useState(false);

  const sseRef = useRef<EventSource | null>(null);

  useEffect(() => {
    Promise.all([
      getSignals(undefined, 100),
      getPortfolio(),
    ]).then(([sigs, port]) => {
      setSignals(sigs.signals);
      setPortfolio(port.portfolio);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    const es = createSignalStream();
    sseRef.current = es;
    es.onmessage = (e) => {
      try {
        const sig: TradeSignal = JSON.parse(e.data);
        setSignals((prev) => [sig, ...prev.slice(0, 199)]);
        setFlashIds((prev) => {
          const next = new Set(prev);
          next.add(sig.id);
          return next;
        });
        setTimeout(() => {
          setFlashIds((prev) => {
            const next = new Set(prev);
            next.delete(sig.id);
            return next;
          });
        }, 1200);
      } catch {
        // ignore malformed events
      }
    };
    return () => {
      es.close();
      sseRef.current = null;
    };
  }, []);

  const handleEmitSignal = async () => {
    const name = sigName.trim();
    const symbol = sigSymbol.trim().toUpperCase();
    if (!name || !symbol) return;
    setSigSaving(true);
    try {
      const res = await emitSignal({
        name,
        symbol,
        direction: sigDir,
        confidence: parseFloat(sigConf) || 1.0,
        price: sigPrice ? parseFloat(sigPrice) : null,
      });
      setSigName("");
      setSigSymbol("");
      setSigDir("buy");
      setSigPrice("");
      setSigConf("1.0");
      // SSE will push the new signal; also add immediately for responsiveness
      setSignals((prev) => [res.signal, ...prev.slice(0, 199)]);
    } finally {
      setSigSaving(false);
    }
  };

  const handleUpsertPosition = async () => {
    const symbol = posSymbol.trim().toUpperCase();
    const qty = parseFloat(posQty);
    const avg = parseFloat(posAvg);
    if (!symbol || isNaN(qty) || isNaN(avg)) return;
    setPosSaving(true);
    try {
      await upsertPosition(symbol, {
        qty,
        avg_price: avg,
        current_price: posCurrent ? parseFloat(posCurrent) : null,
      });
      const port = await getPortfolio();
      setPortfolio(port.portfolio);
      setPosSymbol("");
      setPosQty("");
      setPosAvg("");
      setPosCurrent("");
    } finally {
      setPosSaving(false);
    }
  };

  const handleDeletePosition = async (symbol: string) => {
    await deletePosition(symbol);
    const port = await getPortfolio();
    setPortfolio(port.portfolio);
  };

  const filtered = symbolFilter
    ? signals.filter((s) => s.symbol.includes(symbolFilter.toUpperCase()))
    : signals;

  const totalPnl = portfolio?.total_pnl ?? 0;
  const totalPnlPct = portfolio?.total_pnl_pct ?? 0;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="w-6 h-6 border-2 border-frost/30 border-t-frost rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="relative p-6 space-y-6 overflow-y-auto h-screen animate-in">
      <div className="aurora-bg" />

      {/* Header */}
      <div className="relative flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground glow-frost">Trading</h1>
          <p className="text-sm text-muted mt-1">Signal feed &middot; Portfolio tracker</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          <span className="text-[11px] font-mono text-muted">SSE Live</span>
        </div>
      </div>

      {/* Portfolio summary */}
      <div className="relative grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="runic-card p-4">
          <p className="text-[10px] text-muted uppercase tracking-widest font-medium">Positions</p>
          <p className="text-3xl font-bold font-mono text-frost mt-1">{portfolio?.positions.length ?? 0}</p>
        </div>
        <div className="runic-card p-4">
          <p className="text-[10px] text-muted uppercase tracking-widest font-medium">Total P&amp;L</p>
          <p className="text-3xl font-bold font-mono mt-1" style={{ color: pnlColor(totalPnl) }}>
            {totalPnl >= 0 ? "+" : ""}${totalPnl.toFixed(2)}
          </p>
        </div>
        <div className="runic-card p-4">
          <p className="text-[10px] text-muted uppercase tracking-widest font-medium">P&amp;L %</p>
          <p className="text-3xl font-bold font-mono mt-1" style={{ color: pnlColor(totalPnl) }}>
            {totalPnlPct >= 0 ? "+" : ""}{totalPnlPct.toFixed(2)}%
          </p>
        </div>
        <div className="runic-card p-4">
          <p className="text-[10px] text-muted uppercase tracking-widest font-medium">Signals</p>
          <p className="text-3xl font-bold font-mono text-frost mt-1">{signals.length}</p>
        </div>
      </div>

      <div className="relative grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Signal feed */}
        <div className="glass-card p-5 space-y-3">
          <div className="flex items-center gap-3">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex-1 flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              Signal Feed
            </h2>
            <input
              type="text"
              placeholder="Filter symbol…"
              value={symbolFilter}
              onChange={(e) => setSymbolFilter(e.target.value)}
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30 w-28"
            />
          </div>

          {/* New signal form */}
          <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-white/[0.02] border border-white/[0.05]">
            <input
              type="text" placeholder="Symbol (e.g. BTC)" value={sigSymbol}
              onChange={(e) => setSigSymbol(e.target.value)}
              className="col-span-2 bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <input
              type="text" placeholder="Signal name" value={sigName}
              onChange={(e) => setSigName(e.target.value)}
              className="col-span-2 bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <select
              value={sigDir} onChange={(e) => setSigDir(e.target.value as Direction)}
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
              <option value="hold">Hold</option>
            </select>
            <input
              type="number" placeholder="Confidence (0-1)" value={sigConf}
              onChange={(e) => setSigConf(e.target.value)} step="0.1" min="0" max="1"
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <input
              type="number" placeholder="Price (optional)" value={sigPrice}
              onChange={(e) => setSigPrice(e.target.value)} step="0.01"
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <button
              onClick={handleEmitSignal} disabled={sigSaving || !sigName || !sigSymbol}
              className="px-3 py-1 rounded text-xs font-medium bg-frost/15 text-frost border border-frost/25 hover:bg-frost/25 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              {sigSaving ? "Emitting…" : "Emit Signal"}
            </button>
          </div>

          {/* Signal list */}
          <div className="space-y-1 max-h-72 overflow-y-auto">
            {filtered.length === 0 ? (
              <p className="text-xs text-muted/60 italic py-4 text-center">No signals yet</p>
            ) : filtered.map((sig) => (
              <SignalRow key={sig.id} sig={sig} flash={flashIds.has(sig.id)} />
            ))}
          </div>
        </div>

        {/* Portfolio */}
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="2" y="7" width="20" height="14" rx="2" />
              <path d="M16 7V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v2" />
            </svg>
            Portfolio
          </h2>

          {/* Add position form */}
          <div className="grid grid-cols-2 gap-2 p-3 rounded-lg bg-white/[0.02] border border-white/[0.05]">
            <input
              type="text" placeholder="Symbol" value={posSymbol}
              onChange={(e) => setPosSymbol(e.target.value)}
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <input
              type="number" placeholder="Quantity" value={posQty}
              onChange={(e) => setPosQty(e.target.value)} step="0.001"
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <input
              type="number" placeholder="Avg price" value={posAvg}
              onChange={(e) => setPosAvg(e.target.value)} step="0.01"
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <input
              type="number" placeholder="Current price" value={posCurrent}
              onChange={(e) => setPosCurrent(e.target.value)} step="0.01"
              className="bg-white/[0.04] border border-white/[0.08] rounded px-2 py-1 text-xs font-mono text-foreground outline-none focus:border-frost/30"
            />
            <button
              onClick={handleUpsertPosition} disabled={posSaving || !posSymbol || !posQty || !posAvg}
              className="col-span-2 px-3 py-1 rounded text-xs font-medium bg-emerald/10 text-emerald border border-emerald/25 hover:bg-emerald/20 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              {posSaving ? "Saving…" : "Add / Update Position"}
            </button>
          </div>

          {/* Position list */}
          <div className="space-y-1 max-h-72 overflow-y-auto">
            {(portfolio?.positions.length ?? 0) === 0 ? (
              <p className="text-xs text-muted/60 italic py-4 text-center">No positions yet</p>
            ) : portfolio?.positions.map((pos) => (
              <PositionRow key={pos.symbol} pos={pos} onDelete={handleDeletePosition} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
