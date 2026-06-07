"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CandlestickChart } from "lucide-react";
import Chart from "@/components/Chart";
import { getAssets, getCandles } from "@/lib/api";
import type { AssetInfo, Candle } from "@/lib/types";
import { fmtNum, fmtPrice } from "@/lib/format";
import { ErrorBanner, Panel, Spinner } from "@/components/ui";

const INTERVALS = ["1m", "5m", "1h", "4h", "1d"];
const LIMIT = 200;

export default function MarketPage() {
  const [assets, setAssets] = useState<AssetInfo[]>([]);
  const [symbol, setSymbol] = useState<string>("");
  const [interval, setInterval_] = useState<string>("1h");
  const [candles, setCandles] = useState<Candle[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getAssets()
      .then((list) => {
        setAssets(list);
        if (list.length && !symbol) setSymbol(list[0].symbol);
      })
      .catch((err) => setError((err as Error).message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const load = useCallback(async () => {
    if (!symbol) return;
    try {
      const res = await getCandles(symbol, interval, LIMIT);
      setCandles(res.candles);
      setError(null);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [symbol, interval]);

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    load();
    const id = window.setInterval(load, 10000);
    return () => window.clearInterval(id);
  }, [load, symbol]);

  const stats = useMemo(() => {
    if (!candles.length) return null;
    const last = candles[candles.length - 1];
    let high = -Infinity;
    let low = Infinity;
    let volume = 0;
    for (const c of candles) {
      if (c.high > high) high = c.high;
      if (c.low < low) low = c.low;
      volume += c.volume;
    }
    return { price: last.close, high, low, volume };
  }, [candles]);

  return (
    <div className="p-4 md:p-6 pt-20 md:pt-6 space-y-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <CandlestickChart className="text-green-400" size={22} />
          <h1 className="text-xl font-semibold text-gray-100">Market</h1>
        </div>
        <div className="flex items-center gap-3">
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-gray-900 border border-gray-800 rounded-md px-3 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-green-500"
          >
            {assets.map((a) => (
              <option key={a.symbol} value={a.symbol}>
                {a.symbol} — {a.name}
              </option>
            ))}
          </select>
          <div className="flex rounded-md border border-gray-800 overflow-hidden">
            {INTERVALS.map((iv) => (
              <button
                key={iv}
                onClick={() => setInterval_(iv)}
                className={`px-3 py-1.5 text-sm transition-colors ${
                  iv === interval
                    ? "bg-gray-800 text-green-400"
                    : "bg-gray-900 text-gray-400 hover:text-gray-200"
                }`}
              >
                {iv}
              </button>
            ))}
          </div>
        </div>
      </header>

      {error && <ErrorBanner message={error} />}

      <Panel
        title={symbol ? `${symbol} · ${interval}` : "Chart"}
        action={
          stats && (
            <span className="font-mono text-sm text-gray-200">
              {fmtPrice(stats.price)}
            </span>
          )
        }
      >
        <div className="p-4">
          {loading && !candles.length ? (
            <Spinner label="Loading candles…" />
          ) : (
            <Chart candles={candles} height={440} />
          )}
        </div>
      </Panel>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Stat label="Last Price" value={stats ? fmtPrice(stats.price) : "—"} />
        <Stat label="24h High" value={stats ? fmtPrice(stats.high) : "—"} />
        <Stat label="24h Low" value={stats ? fmtPrice(stats.low) : "—"} />
        <Stat label="Volume" value={stats ? fmtNum(stats.volume, 0) : "—"} />
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="mt-1 text-lg font-mono font-semibold text-gray-100">{value}</div>
    </div>
  );
}
