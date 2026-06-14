"use client";

import { useEffect, useState } from "react";
import { api, type BackendInfo, type StatsResponse, type FxRate } from "@/lib/api";
import Link from "next/link";

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card" style={{ padding: "16px 20px" }}>
      <div style={{ color: "var(--text-muted)", fontSize: 11, textTransform: "uppercase", letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, marginTop: 6, color: "var(--text)" }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function RateRow({ rate }: { rate: FxRate }) {
  const val = rate.rate != null ? rate.rate.toFixed(4) : "—";
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "8px 0", borderBottom: "1px solid var(--border)",
    }}>
      <span style={{ fontWeight: 600, color: "var(--text)" }}>{rate.pair}</span>
      <span style={{ fontFamily: "monospace", color: rate.error ? "var(--red)" : "var(--green)" }}>{val}</span>
    </div>
  );
}

export default function Dashboard() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [fx, setFx] = useState<FxRate[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [uptime, setUptime] = useState<string>("—");

  useEffect(() => {
    api.ping().then(() => setOnline(true)).catch(() => setOnline(false));
    api.backend().then(setBackend).catch(() => null);
    api.stats().then((s) => {
      setStats(s);
      const secs = Math.floor(Date.now() / 1000 - s.uptime);
      const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60);
      setUptime(`${h}h ${m}m`);
    }).catch(() => null);
    api.fx("EUR/USD,EUR/GBP,USD/JPY").then((r) => setFx(r.rates)).catch(() => null);
  }, []);

  return (
    <div style={{ padding: 28 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Dashboard</h1>
        <div style={{ color: "var(--text-muted)", marginTop: 4, fontSize: 12 }}>
          {online === null ? "Connecting…" : online ? "● Node online" : "○ Node offline"}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        <StatCard label="Status" value={online ? "Online" : "Offline"} sub={backend?.version ? `v${backend.version}` : undefined} />
        <StatCard label="Python" value={stats?.python ?? "—"} sub={stats?.platform} />
        <StatCard label="Uptime" value={uptime} />
        <StatCard label="Node" value={backend?.name ?? "—"} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
            FX Rates
            <Link href="/market" style={{ color: "var(--accent)", fontSize: 11, textDecoration: "none" }}>View all →</Link>
          </div>
          {fx.length === 0 ? (
            <div style={{ color: "var(--text-muted)" }}>Loading…</div>
          ) : (
            fx.map((r) => <RateRow key={r.pair} rate={r} />)
          )}
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, display: "flex", justifyContent: "space-between" }}>
            Quick Links
          </div>
          {[
            { href: "/market", label: "Market Data", desc: "FX rates, energy prices, OHLC charts" },
            { href: "/chat", label: "Loki AI", desc: "Natural language queries against your data" },
            { href: "/data", label: "Data Browser", desc: "Explore parquet files and run analysis" },
          ].map((item) => (
            <Link key={item.href} href={item.href} style={{ display: "block", textDecoration: "none" }}>
              <div style={{
                padding: "10px 12px", marginBottom: 8, borderRadius: 6,
                background: "rgba(59,130,246,0.05)", border: "1px solid rgba(59,130,246,0.15)",
                transition: "border-color 0.15s",
              }}>
                <div style={{ fontWeight: 500, color: "var(--accent)" }}>{item.label}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 2 }}>{item.desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
