"use client";
import { useCallback, useEffect, useRef, useState } from "react";

interface Tick {
  ts: number; price: number; ema9: number | null; ema21: number | null;
  rsi14: number | null; signal: number;
}

function signalLabel(s: number) {
  if (s > 0) return { text: "BUY", cls: "text-emerald-400 bg-emerald-950/50" };
  if (s < 0) return { text: "SELL", cls: "text-red-400 bg-red-950/50" };
  return { text: "HOLD", cls: "text-gray-400 bg-gray-800" };
}

const W = 600;
const PH = 180;
const RH = 70;

function MiniChart({ ticks }: { ticks: Tick[] }) {
  if (ticks.length < 2) return null;
  const prices = ticks.map(t => t.price);
  const ema9 = ticks.map(t => t.ema9);
  const ema21 = ticks.map(t => t.ema21);
  const rsi = ticks.map(t => t.rsi14);
  const n = ticks.length;

  const pMin = Math.min(...prices);
  const pMax = Math.max(...prices);
  const pRange = pMax - pMin || 1;
  const px = (i: number) => (i / Math.max(n - 1, 1)) * W;
  const py = (v: number) => PH - ((v - pMin) / pRange) * PH;
  const rsiY = (v: number | null) => v == null ? null : RH - (v / 100) * RH;

  const linePath = (vals: (number | null)[], y: (v: number) => number) => {
    const pts = vals.map((v, i) => v == null ? null : `${px(i)},${y(v)}`).filter(Boolean) as string[];
    return pts.length ? `M ${pts.join(" L ")}` : "";
  };

  return (
    <div className="space-y-2">
      <svg viewBox={`0 0 ${W} ${PH}`} className="w-full" style={{ height: PH }}>
        <path d={linePath(ema9, py)} fill="none" stroke="#f59e0b" strokeWidth="1" opacity="0.7" />
        <path d={linePath(ema21, py)} fill="none" stroke="#ec4899" strokeWidth="1" opacity="0.7" />
        <path d={linePath(prices, py)} fill="none" stroke="#6366f1" strokeWidth="1.5" />
        {/* Latest tick */}
        {ticks.length > 0 && (
          <circle cx={px(n - 1)} cy={py(prices[n - 1])} r="3"
            fill={ticks[n - 1].signal > 0 ? "#10b981" : ticks[n - 1].signal < 0 ? "#ef4444" : "#6366f1"} />
        )}
      </svg>
      <svg viewBox={`0 0 ${W} ${RH}`} className="w-full" style={{ height: RH }}>
        <rect x={0} y={RH - (70 / 100) * RH} width={W} height={(40 / 100) * RH} fill="#fbbf24" fillOpacity={0.07} />
        <line x1={0} y1={RH - (70 / 100) * RH} x2={W} y2={RH - (70 / 100) * RH}
          stroke="#fbbf24" strokeWidth="0.5" strokeDasharray="3 3" />
        <line x1={0} y1={RH - (30 / 100) * RH} x2={W} y2={RH - (30 / 100) * RH}
          stroke="#fbbf24" strokeWidth="0.5" strokeDasharray="3 3" />
        <path d={linePath(rsi, v => rsiY(v)!)} fill="none" stroke="#22d3ee" strokeWidth="1.5" />
      </svg>
    </div>
  );
}

export default function LivePage() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("close");
  const [interval, setInterval_] = useState(500);
  const [ticks, setTicks] = useState<Tick[]>([]);
  const [status, setStatus] = useState<"idle" | "connecting" | "streaming" | "done" | "error">("idle");
  const [error, setError] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const WINDOW = 200;

  const stop = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setStatus(s => s === "streaming" ? "done" : s);
  }, []);

  const start = useCallback(() => {
    if (!path) { setError("Enter a file path"); return; }
    wsRef.current?.close();
    setTicks([]);
    setError("");
    setStatus("connecting");

    // Connect to the same host as the page, upgrading to ws(s)://
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    // In dev, proxy through Next.js or connect to backend directly on port 8100
    const host = window.location.hostname;
    const port = 8100;
    const ws = new WebSocket(`${proto}://${host}:${port}/ws/v2/trading/stream`);
    wsRef.current = ws;

    ws.onopen = () => {
      setStatus("streaming");
      ws.send(JSON.stringify({ path, column, interval_ms: interval }));
    };
    ws.onmessage = (e) => {
      const tick: Tick = JSON.parse(e.data);
      setTicks(prev => {
        const next = [...prev, tick];
        return next.length > WINDOW ? next.slice(next.length - WINDOW) : next;
      });
    };
    ws.onerror = () => { setError("WebSocket connection failed"); setStatus("error"); };
    ws.onclose = () => { if (status !== "error") setStatus("done"); };
  }, [path, column, interval, status]);

  useEffect(() => () => { wsRef.current?.close(); }, []);

  const latest = ticks[ticks.length - 1];
  const sig = latest ? signalLabel(latest.signal) : null;

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-bold">Live Stream</h1>
      <p className="text-gray-400 text-sm">
        Replays a local dataset as a real-time indicator feed over WebSocket.
        Each tick emits: price, EMA 9/21, RSI(14), composite signal.
      </p>

      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 space-y-3">
        <div className="flex gap-2 flex-wrap">
          <input value={path} onChange={e => setPath(e.target.value)} placeholder="Path to parquet…"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm flex-1 min-w-48" />
          <input value={column} onChange={e => setColumn(e.target.value)} placeholder="Column"
            className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm w-28" />
          <select value={interval} onChange={e => setInterval_(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-2 text-sm">
            <option value={100}>100ms</option>
            <option value={250}>250ms</option>
            <option value={500}>500ms</option>
            <option value={1000}>1s</option>
          </select>
        </div>
        <div className="flex gap-2">
          <button onClick={start} disabled={status === "streaming"}
            className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 px-5 py-1.5 rounded-lg text-sm font-medium">
            {status === "connecting" ? "Connecting…" : "Start"}
          </button>
          <button onClick={stop} disabled={status !== "streaming"}
            className="bg-gray-700 hover:bg-gray-600 disabled:opacity-40 px-4 py-1.5 rounded-lg text-sm">
            Stop
          </button>
          <span className={`text-xs self-center px-2 py-1 rounded ${
            status === "streaming" ? "text-emerald-400 bg-emerald-950/50" :
            status === "done" ? "text-gray-500 bg-gray-800" :
            status === "error" ? "text-red-400 bg-red-950/50" :
            "text-gray-500 bg-gray-800"
          }`}>{status}</span>
        </div>
        {error && <div className="text-red-400 text-sm">{error}</div>}
      </div>

      {latest && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {[
            ["Price", latest.price.toFixed(4)],
            ["EMA 9", latest.ema9?.toFixed(4) ?? "–"],
            ["RSI(14)", latest.rsi14?.toFixed(1) ?? "–"],
            ["Ticks", ticks.length],
          ].map(([k, v]) => (
            <div key={k} className="bg-gray-900 border border-gray-800 rounded-lg p-2.5">
              <div className="text-gray-500 text-xs mb-0.5">{k}</div>
              <div className="font-mono font-semibold text-sm">{v}</div>
            </div>
          ))}
        </div>
      )}

      {latest && sig && (
        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-bold ${sig.cls}`}>
          <span className={`w-2 h-2 rounded-full ${latest.signal > 0 ? "bg-emerald-400" : latest.signal < 0 ? "bg-red-400" : "bg-gray-500"} animate-pulse`} />
          {sig.text} signal
        </div>
      )}

      {ticks.length > 1 && (
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">
            Price + EMA 9/21 (top) · RSI(14) (bottom) · last {ticks.length} ticks
          </div>
          <MiniChart ticks={ticks} />
        </div>
      )}
    </div>
  );
}
