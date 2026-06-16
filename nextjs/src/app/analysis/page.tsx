"use client";
import { useState } from "react";
import OhlcChart from "@/components/OhlcChart";

interface OhlcBar { open: number; high: number; low: number; close: number; }

function genDemoOhlc(n = 80): OhlcBar[] {
  const bars: OhlcBar[] = [];
  let price = 1.0850;
  for (let i = 0; i < n; i++) {
    const change = (Math.random() - 0.49) * 0.003;
    const open = price;
    const close = price + change;
    const hi = Math.max(open, close) + Math.random() * 0.001;
    const lo = Math.min(open, close) - Math.random() * 0.001;
    bars.push({ open, high: hi, low: lo, close });
    price = close;
  }
  return bars;
}

export default function AnalysisPage() {
  const [ohlcBars] = useState<OhlcBar[]>(genDemoOhlc(80));
  const [pair, setPair] = useState("EUR/USD");
  const [tfm, setTfm] = useState("1h");

  const last = ohlcBars[ohlcBars.length - 1];
  const first = ohlcBars[0];
  const change = last && first ? ((last.close - first.open) / first.open) * 100 : 0;
  const isUp = change >= 0;

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto" }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>Analysis</h1>
        <p style={{ color: "var(--muted)", marginTop: 4, fontSize: 12 }}>
          Polars lazy scan · OHLC resampling · Projection pushdown · Forecasting
        </p>
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <select value={pair} onChange={e => setPair(e.target.value)} style={{ fontSize: 14, fontWeight: 600 }}>
              {["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD"].map(p => (
                <option key={p}>{p}</option>
              ))}
            </select>
            <div style={{ display: "flex", gap: 4 }}>
              {["1m", "5m", "15m", "1h", "4h", "1d"].map(tf => (
                <button key={tf} onClick={() => setTfm(tf)} style={{
                  padding: "3px 8px",
                  fontSize: 11,
                  fontWeight: 600,
                  borderRadius: 5,
                  border: "1px solid",
                  borderColor: tf === tfm ? "var(--accent)" : "var(--border)",
                  background: tf === tfm ? "rgba(59,130,246,0.15)" : "transparent",
                  color: tf === tfm ? "var(--accent)" : "var(--muted)",
                  cursor: "pointer",
                }}>
                  {tf}
                </button>
              ))}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            {last && (
              <span style={{ fontSize: 20, fontWeight: 700, fontVariantNumeric: "tabular-nums" }}>
                {last.close.toFixed(5)}
              </span>
            )}
            <span className={`badge ${isUp ? "badge-green" : "badge-red"}`}>
              {isUp ? "+" : ""}{change.toFixed(2)}%
            </span>
          </div>
        </div>

        <OhlcChart bars={ohlcBars} height={240} />

        {last && (
          <div style={{ display: "flex", gap: 24, marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
            {[
              ["Open", first?.open.toFixed(5)],
              ["High", Math.max(...ohlcBars.map(b => b.high)).toFixed(5)],
              ["Low",  Math.min(...ohlcBars.map(b => b.low)).toFixed(5)],
              ["Close", last.close.toFixed(5)],
              ["Bars", ohlcBars.length],
              ["TF", tfm],
            ].map(([k, v]) => (
              <div key={String(k)}>
                <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>{k}</div>
                <div style={{ fontSize: 13, fontWeight: 600, fontVariantNumeric: "tabular-nums", marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Analysis cards */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Polars Lazy Scan</div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8 }}>
            <div>• Projection pushdown: 2 of N columns</div>
            <div>• Streaming group-by aggregation</div>
            <div>• Parquet columnar I/O skip</div>
            <div style={{ marginTop: 8, padding: "8px 10px", background: "rgba(16,185,129,0.08)", borderRadius: 6, border: "1px solid rgba(16,185,129,0.2)" }}>
              <span style={{ color: "var(--green)", fontWeight: 600 }}>3–8× faster</span> vs eager full-table read
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>OHLC Resampling</div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8 }}>
            <div>• Bucket-based candlestick assembly</div>
            <div>• Adaptive downsample to target points</div>
            <div>• Streaming — no full load required</div>
            <div style={{ marginTop: 8, padding: "8px 10px", background: "rgba(59,130,246,0.08)", borderRadius: 6, border: "1px solid rgba(59,130,246,0.2)" }}>
              <span style={{ color: "var(--accent)", fontWeight: 600 }}>1M rows → 120 bars</span> in ~40ms
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>ML Forecasting</div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8 }}>
            <div>• Auto-selects: XGBoost → GBR → Ridge</div>
            <div>• Lag features + rolling mean + sin/cos period</div>
            <div>• Per-group forecasting with RMSE reporting</div>
            <div style={{ marginTop: 8, padding: "8px 10px", background: "rgba(139,92,246,0.08)", borderRadius: 6, border: "1px solid rgba(139,92,246,0.2)" }}>
              <span style={{ color: "#8b5cf6", fontWeight: 600 }}>48h ahead</span> per group
            </div>
          </div>
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Cross-Tab Pivot</div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8 }}>
            <div>• Reads only row/col/measure columns</div>
            <div>• Bounded grid (no memory explosion)</div>
            <div>• Sum / mean / min / max / count</div>
            <div style={{ marginTop: 8, padding: "8px 10px", background: "rgba(245,158,11,0.08)", borderRadius: 6, border: "1px solid rgba(245,158,11,0.2)" }}>
              <span style={{ color: "var(--yellow)", fontWeight: 600 }}>Projection aware</span> — skips unused cols
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
