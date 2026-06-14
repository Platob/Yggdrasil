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
      padding: "7px 0", borderBottom: "1px solid var(--border)",
    }}>
      <span style={{ fontWeight: 600, fontSize: 12 }}>{rate.pair}</span>
      <span style={{ fontFamily: "monospace", fontSize: 12, color: rate.error ? "var(--red)" : "var(--green)" }}>{val}</span>
    </div>
  );
}

function CryptoRow({ coin }: { coin: { id: string; price: number | null; change_24h: number | null } }) {
  const ch = coin.change_24h;
  const chColor = ch == null ? "var(--text-muted)" : ch >= 0 ? "var(--green)" : "var(--red)";
  const chSign = ch != null && ch >= 0 ? "+" : "";
  return (
    <div style={{
      display: "flex", justifyContent: "space-between", alignItems: "center",
      padding: "7px 0", borderBottom: "1px solid var(--border)",
    }}>
      <span style={{ fontWeight: 600, fontSize: 12, textTransform: "capitalize" }}>{coin.id}</span>
      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
        <span style={{ fontFamily: "monospace", fontSize: 12, color: "var(--text)" }}>
          {coin.price != null ? `$${coin.price.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "—"}
        </span>
        {ch != null && (
          <span style={{ fontSize: 11, color: chColor }}>{chSign}{ch.toFixed(2)}%</span>
        )}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [backend, setBackend] = useState<BackendInfo | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [fx, setFx] = useState<FxRate[]>([]);
  const [crypto, setCrypto] = useState<{ id: string; price: number | null; change_24h: number | null }[]>([]);
  const [online, setOnline] = useState<boolean | null>(null);
  const [uptime, setUptime] = useState<string>("—");
  const [node, setNode] = useState<{ cpu_percent: number; mem_percent: number } | null>(null);

  useEffect(() => {
    api.ping().then(() => setOnline(true)).catch(() => setOnline(false));
    api.backend().then(setBackend).catch(() => null);
    api.stats().then((s) => {
      setStats(s);
      const secs = Math.floor(Date.now() / 1000 - s.uptime);
      const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60);
      setUptime(`${h}h ${m}m`);
    }).catch(() => null);
    api.marketSummary()
      .then((s) => {
        setFx(s.fx ?? []);
        setCrypto(s.crypto ?? []);
        setNode(s.node ?? null);
      })
      .catch(() => {
        api.fx("EUR/USD,EUR/GBP,USD/JPY").then((r) => setFx(r.rates)).catch(() => null);
      });
  }, []);

  return (
    <div style={{ padding: 28 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Dashboard</h1>
        <div style={{ color: "var(--text-muted)", marginTop: 4, fontSize: 12 }}>
          {online === null ? "Connecting…" : online ? "● Node online" : "○ Node offline — run: ygg node serve"}
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
        <StatCard label="Status" value={online ? "Online" : "Offline"} sub={backend?.version ? `v${backend.version}` : undefined} />
        <StatCard label="Python" value={stats?.python ?? "—"} sub={stats?.platform} />
        <StatCard label="Uptime" value={uptime} />
        <StatCard label="CPU / Mem" value={node ? `${node.cpu_percent.toFixed(0)}% / ${node.mem_percent.toFixed(0)}%` : "—"} />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13, display: "flex", justifyContent: "space-between" }}>
            FX Rates
            <Link href="/market" style={{ color: "var(--accent)", fontSize: 11, textDecoration: "none" }}>View all →</Link>
          </div>
          {fx.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading…</div>
          ) : (
            fx.map((r) => <RateRow key={r.pair} rate={r} />)
          )}
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13, display: "flex", justifyContent: "space-between" }}>
            Crypto
            <Link href="/market" style={{ color: "var(--accent)", fontSize: 11, textDecoration: "none" }}>View all →</Link>
          </div>
          {crypto.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: 12 }}>Loading…</div>
          ) : (
            crypto.map((c) => <CryptoRow key={c.id} coin={c} />)
          )}
        </div>

        <div className="card" style={{ padding: 20 }}>
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Quick Actions</div>
          {[
            { href: "/market", label: "Market Data", desc: "FX, crypto, energy, OHLC charts" },
            { href: "/chat", label: "✦ Loki AI", desc: "Natural language market queries" },
            { href: "/data", label: "Data Browser", desc: "Parquet files · analysis · indicators" },
          ].map((item) => (
            <Link key={item.href} href={item.href} style={{ display: "block", textDecoration: "none" }}>
              <div style={{
                padding: "10px 12px", marginBottom: 8, borderRadius: 6,
                background: "rgba(59,130,246,0.05)", border: "1px solid rgba(59,130,246,0.15)",
              }}>
                <div style={{ fontWeight: 500, color: "var(--accent)", fontSize: 12 }}>{item.label}</div>
                <div style={{ color: "var(--text-muted)", fontSize: 11, marginTop: 2 }}>{item.desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
