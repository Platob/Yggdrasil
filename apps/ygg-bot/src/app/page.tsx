"use client";

import { useCallback, useEffect, useState } from "react";
import { AiChat } from "@/components/AiChat";
import { FxTicker } from "@/components/FxTicker";
import { Header } from "@/components/Header";
import { MarketFeed } from "@/components/MarketFeed";
import { PriceChart } from "@/components/PriceChart";
import { SignalCard, SignalPlaceholder } from "@/components/SignalCard";
import {
  api,
  type FxRate,
  type HealthResponse,
  type PricePoint,
  type Signal,
  type StatsResponse,
} from "@/lib/api";

const ZONES = ["DE_LU", "FR", "NL", "GB", "BE"];
const REFRESH_MS = 30_000;

export default function Dashboard() {
  const [zone, setZone] = useState("DE_LU");
  const [days, setDays] = useState(7);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [fxRates, setFxRates] = useState<FxRate[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loadingPrices, setLoadingPrices] = useState(false);
  const [connected, setConnected] = useState(false);

  const fetchAll = useCallback(async () => {
    try {
      const [h, s] = await Promise.all([api.health(), api.stats()]);
      setHealth(h);
      setStats(s);
      setConnected(true);
    } catch {
      setConnected(false);
    }

    setLoadingPrices(true);
    try {
      const [p, f, sig] = await Promise.all([
        api.prices(zone, days),
        api.fx(),
        api.signals(zone, days),
      ]);
      setPrices(p.prices);
      setFxRates(f.rates);
      setSignals(sig.signals);
    } finally {
      setLoadingPrices(false);
    }
  }, [zone, days]);

  useEffect(() => {
    fetchAll();
    const t = setInterval(fetchAll, REFRESH_MS);
    return () => clearInterval(t);
  }, [fetchAll]);

  return (
    <div className="flex flex-col min-h-screen">
      <Header
        uptime={health?.uptime_s}
        wsConnections={health?.ws_connections}
        connected={connected}
      />

      {/* Zone + days controls */}
      <div className="flex items-center gap-4 px-6 py-2 border-b border-border bg-surface text-xs">
        <span className="text-muted">zone</span>
        {ZONES.map((z) => (
          <button
            key={z}
            onClick={() => setZone(z)}
            className={`px-2 py-0.5 rounded transition-colors ${
              z === zone
                ? "bg-accent text-white"
                : "text-muted hover:text-text"
            }`}
          >
            {z}
          </button>
        ))}
        <span className="text-muted ml-4">days</span>
        {[1, 7, 30].map((d) => (
          <button
            key={d}
            onClick={() => setDays(d)}
            className={`px-2 py-0.5 rounded transition-colors ${
              d === days
                ? "bg-accent text-white"
                : "text-muted hover:text-text"
            }`}
          >
            {d}d
          </button>
        ))}
        <button
          onClick={fetchAll}
          className="ml-auto text-muted hover:text-text transition-colors px-2 py-0.5"
          title="refresh"
        >
          ↻
        </button>
      </div>

      {/* Stats bar */}
      {stats && (
        <div className="flex gap-6 px-6 py-2 text-xs text-muted border-b border-border">
          <span>reqs {stats.requests_total.toLocaleString()}</span>
          <span>cache {stats.cache_hits}↑ {stats.cache_misses}↓</span>
          <span>ws-msgs {stats.ws_messages_sent}</span>
        </div>
      )}

      {/* Main grid */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
        {/* Left column */}
        <div className="lg:col-span-2 flex flex-col gap-4">
          <PriceChart prices={prices} zone={zone} isLoading={loadingPrices} />
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <FxTicker rates={fxRates} base="EUR" />
            <MarketFeed />
          </div>
        </div>

        {/* Right column */}
        <div className="flex flex-col gap-4">
          {signals.length > 0
            ? signals.map((s, i) => <SignalCard key={i} signal={s} />)
            : <SignalPlaceholder />
          }
          <AiChat />
        </div>
      </main>
    </div>
  );
}
