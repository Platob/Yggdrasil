"use client";

import { useEffect, useRef, useState } from "react";
import { api, type FxRate, type EnergyResponse } from "@/lib/api";

const FX_PAIRS = "EUR/USD,EUR/GBP,EUR/JPY,USD/JPY,GBP/USD,USD/CHF,EUR/CHF,AUD/USD,NZD/USD,USD/CAD";
const ZONES = ["DE_LU", "FR", "NL", "BE", "AT", "ES", "GB", "PL"];

function FxTable({ rates }: { rates: FxRate[] }) {
  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ fontWeight: 600, marginBottom: 14, fontSize: 14 }}>FX Rates</div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ color: "var(--text-muted)", textAlign: "left" }}>
            <th style={{ paddingBottom: 8, fontWeight: 500 }}>Pair</th>
            <th style={{ paddingBottom: 8, fontWeight: 500, textAlign: "right" }}>Rate</th>
            <th style={{ paddingBottom: 8, fontWeight: 500, textAlign: "right" }}>Date</th>
          </tr>
        </thead>
        <tbody>
          {rates.map((r) => (
            <tr key={r.pair} style={{ borderTop: "1px solid var(--border)" }}>
              <td style={{ padding: "9px 0", fontWeight: 600 }}>{r.pair}</td>
              <td style={{
                padding: "9px 0", textAlign: "right", fontFamily: "monospace",
                color: r.error ? "var(--red)" : "var(--green)",
              }}>
                {r.rate != null ? r.rate.toFixed(5) : r.error ?? "—"}
              </td>
              <td style={{ padding: "9px 0", textAlign: "right", color: "var(--text-muted)" }}>
                {r.date ?? "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EnergyPanel({ zone, onZoneChange }: { zone: string; onZoneChange: (z: string) => void }) {
  const [data, setData] = useState<EnergyResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.energy(zone).then(setData).catch((e) => {
      setData({ zone, series: "day_ahead_prices", data: [], error: String(e) });
    }).finally(() => setLoading(false));
  }, [zone]);

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
        <div style={{ fontWeight: 600, fontSize: 14 }}>Energy Prices (ENTSO-E)</div>
        <select
          value={zone}
          onChange={(e) => onZoneChange(e.target.value)}
          style={{
            background: "var(--border)", border: "1px solid var(--border)", color: "var(--text)",
            borderRadius: 4, padding: "4px 8px", fontSize: 12, cursor: "pointer",
          }}
        >
          {ZONES.map((z) => <option key={z} value={z}>{z}</option>)}
        </select>
      </div>
      {loading && <div style={{ color: "var(--text-muted)" }}>Loading…</div>}
      {data?.error && (
        <div style={{ color: "var(--yellow)", fontSize: 12, padding: "12px", background: "rgba(245,158,11,0.08)", borderRadius: 6 }}>
          {data.error.includes("ENTSOE_API_TOKEN")
            ? "Set ENTSOE_API_TOKEN env var to enable energy market data from the ENTSO-E Transparency Platform."
            : data.error}
        </div>
      )}
      {!loading && !data?.error && data?.data && data.data.length > 0 && (
        <MiniSparkline data={data.data} />
      )}
    </div>
  );
}

function MiniSparkline({ data }: { data: Record<string, unknown>[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || data.length === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const values = data
      .map((d) => {
        const v = Object.values(d).find((x) => typeof x === "number") as number | undefined;
        return v;
      })
      .filter((v): v is number => v !== undefined);

    if (values.length < 2) return;

    const w = canvas.width;
    const h = canvas.height;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;

    ctx.clearRect(0, 0, w, h);
    ctx.beginPath();
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 1.5;

    values.forEach((v, i) => {
      const x = (i / (values.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Fill under line
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(59,130,246,0.08)";
    ctx.fill();
  }, [data]);

  return (
    <div>
      <canvas ref={canvasRef} width={600} height={120} style={{ width: "100%", height: 120 }} />
      <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-muted)", fontSize: 11, marginTop: 4 }}>
        <span>{data.length} data points</span>
      </div>
    </div>
  );
}

function OhlcChart({ path, column }: { path: string; column: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    api.analysis.ohlc(path, column, 120)
      .then((data) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        const w = canvas.width, h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        const all = [...data.open, ...data.high, ...data.low, ...data.close];
        const min = Math.min(...all), max = Math.max(...all);
        const range = max - min || 1;
        const n = data.bars;
        const cw = Math.max(2, Math.floor(w / n) - 1);

        for (let i = 0; i < n; i++) {
          const x = (i / n) * w + cw / 2;
          const o = h - ((data.open[i] - min) / range) * (h - 16) - 8;
          const c = h - ((data.close[i] - min) / range) * (h - 16) - 8;
          const hi = h - ((data.high[i] - min) / range) * (h - 16) - 8;
          const lo = h - ((data.low[i] - min) / range) * (h - 16) - 8;
          const color = c <= o ? "#10b981" : "#ef4444";

          ctx.strokeStyle = color;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, hi);
          ctx.lineTo(x, lo);
          ctx.stroke();

          ctx.fillStyle = color;
          ctx.fillRect(x - cw / 2, Math.min(o, c), cw, Math.abs(c - o) || 1);
        }
        setLoading(false);
      })
      .catch((e) => { setError(String(e)); setLoading(false); });
  }, [path, column]);

  if (error) return <div style={{ color: "var(--text-muted)", fontSize: 12 }}>No OHLC data: {error}</div>;
  if (loading) return <div style={{ color: "var(--text-muted)" }}>Loading chart…</div>;

  return <canvas ref={canvasRef} width={600} height={180} style={{ width: "100%", height: 180 }} />;
}

export default function MarketPage() {
  const [rates, setRates] = useState<FxRate[]>([]);
  const [zone, setZone] = useState("DE_LU");
  const [loading, setLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<string>("");

  const refresh = () => {
    setLoading(true);
    api.fx(FX_PAIRS)
      .then((r) => { setRates(r.rates); setLastUpdate(new Date().toLocaleTimeString()); })
      .catch(() => null)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 30_000);
    return () => clearInterval(t);
  }, []);

  return (
    <div style={{ padding: 28 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Market Data</h1>
          <div style={{ color: "var(--text-muted)", fontSize: 12, marginTop: 4 }}>
            {loading ? "Refreshing…" : `Updated ${lastUpdate} · auto-refresh 30s`}
          </div>
        </div>
        <button
          onClick={refresh}
          style={{
            background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff",
            padding: "7px 14px", fontSize: 12, cursor: "pointer", fontWeight: 500,
          }}
        >
          ↻ Refresh
        </button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
        <FxTable rates={rates} />
        <EnergyPanel zone={zone} onZoneChange={setZone} />
      </div>

      <div className="card" style={{ padding: 20 }}>
        <div style={{ fontWeight: 600, marginBottom: 14, fontSize: 14 }}>
          OHLC Analysis
          <span style={{ color: "var(--text-muted)", fontWeight: 400, fontSize: 12, marginLeft: 8 }}>
            Upload a parquet file and enter its path below
          </span>
        </div>
        <OhlcInputForm />
      </div>
    </div>
  );
}

function OhlcInputForm() {
  const [path, setPath] = useState("");
  const [column, setColumn] = useState("price");
  const [submitted, setSubmitted] = useState<{ path: string; column: string } | null>(null);

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="data.parquet"
          style={{
            flex: 2, background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6,
            color: "var(--text)", padding: "7px 10px", fontSize: 12,
          }}
        />
        <input
          value={column}
          onChange={(e) => setColumn(e.target.value)}
          placeholder="column"
          style={{
            flex: 1, background: "var(--bg)", border: "1px solid var(--border)", borderRadius: 6,
            color: "var(--text)", padding: "7px 10px", fontSize: 12,
          }}
        />
        <button
          onClick={() => path && setSubmitted({ path, column })}
          style={{
            background: "var(--accent)", border: "none", borderRadius: 6, color: "#fff",
            padding: "7px 14px", fontSize: 12, cursor: "pointer",
          }}
        >
          Load
        </button>
      </div>
      {submitted && <OhlcChart path={submitted.path} column={submitted.column} />}
      {!submitted && (
        <div style={{ color: "var(--text-muted)", fontSize: 12, padding: "20px 0" }}>
          Enter a parquet file path from the node home directory to render an OHLC chart.
        </div>
      )}
    </div>
  );
}
