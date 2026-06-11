"use client";
import { useEffect, useState } from "react";
import { LineChart } from "@/components/Chart";

interface FxQuote {
  source: string; target: string; value: number; date: string; sampling?: string;
}

const PAIRS = [
  ["EUR", "USD"], ["EUR", "GBP"], ["EUR", "JPY"],
  ["USD", "EUR"], ["USD", "GBP"], ["USD", "JPY"],
];

function pairLabel(src: string, tgt: string) { return `${src}/${tgt}`; }

export default function FxPage() {
  const [latest, setLatest] = useState<FxQuote[]>([]);
  const [source, setSource] = useState("EUR");
  const [target, setTarget] = useState("USD");
  const [start, setStart] = useState(() => {
    const d = new Date(); d.setMonth(d.getMonth() - 3);
    return d.toISOString().slice(0, 10);
  });
  const [end] = useState(() => new Date().toISOString().slice(0, 10));
  const [series, setSeries] = useState<FxQuote[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch("/api/v2/fx/latest")
      .then(r => r.json())
      .then(d => setLatest(d.quotes ?? []))
      .catch(() => {});
  }, []);

  const loadSeries = async () => {
    setLoading(true); setError("");
    try {
      const r = await fetch(
        `/api/v2/fx/timeseries?source=${source}&target=${target}&start=${start}&end=${end}`
      );
      if (!r.ok) throw new Error(await r.text());
      const d = await r.json();
      setSeries(d.quotes ?? []);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const prices = series.map(q => q.value);

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">FX Rates</h1>
      <p className="text-gray-400 text-sm">
        Live rates via <a className="text-emerald-400 hover:underline" href="https://frankfurter.dev" target="_blank" rel="noreferrer">Frankfurter</a> (ECB data, ~170 pairs, free).
      </p>

      {/* Snapshot grid */}
      {latest.length > 0 && (
        <div>
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Latest snapshot</div>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
            {latest.map(q => (
              <div key={`${q.source}${q.target}`}
                className="bg-gray-900 border border-gray-800 rounded-lg p-3">
                <div className="text-gray-500 text-xs mb-1">{pairLabel(q.source, q.target)}</div>
                <div className="font-mono font-semibold">{q.value.toFixed(4)}</div>
                <div className="text-gray-600 text-xs mt-0.5">{q.date}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {latest.length === 0 && (
        <div className="text-gray-500 text-sm bg-gray-900 border border-gray-800 rounded-lg p-4">
          FX service offline — no backend configured or network unavailable.
        </div>
      )}

      {/* Time series chart */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <div className="text-xs text-gray-500 uppercase tracking-wider">Historical series</div>
        <div className="flex gap-2 flex-wrap items-center text-sm">
          <select value={source} onChange={e => setSource(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm">
            {["EUR", "USD", "GBP", "JPY", "CHF"].map(c =>
              <option key={c} value={c}>{c}</option>)}
          </select>
          <span className="text-gray-500">→</span>
          <select value={target} onChange={e => setTarget(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm">
            {["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"].map(c =>
              <option key={c} value={c}>{c}</option>)}
          </select>
          <input type="date" value={start} onChange={e => setStart(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-sm" />
          <button onClick={loadSeries} disabled={loading}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-4 py-1.5 rounded-lg text-sm font-medium">
            {loading ? "Loading…" : "Load"}
          </button>
        </div>
        {error && <div className="text-red-400 text-sm">{error}</div>}
        {prices.length > 0 && (
          <div>
            <div className="text-xs text-gray-500 mb-1">
              {pairLabel(source, target)} · {series.length} days · last {prices[prices.length - 1].toFixed(4)}
            </div>
            <LineChart data={prices} color="#6366f1" fill height={200} />
          </div>
        )}
      </div>
    </div>
  );
}
