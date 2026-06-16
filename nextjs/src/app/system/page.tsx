"use client";
import { useState, useEffect } from "react";
import StatCard from "@/components/StatCard";

interface Health { status: string; uptime: number; }
interface Stats { uptime: number; requests: number; memory_mb: number; }
interface Backend { node_id: string; version: string; engines: string[]; status: string; }
interface AuditEntry { id: number; action: string; resource_type: string; resource_id: unknown; detail: string; timestamp: string; }

export default function SystemPage() {
  const apiBase = process.env.NEXT_PUBLIC_API_URL ?? "";
  const [health, setHealth] = useState<Health | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [backend, setBackend] = useState<Backend | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      const results = await Promise.allSettled([
        fetch(`${apiBase}/api/v2/health`).then(r => r.json()),
        fetch(`${apiBase}/api/v2/stats`).then(r => r.json()),
        fetch(`${apiBase}/api/v2/backend`).then(r => r.json()),
        fetch(`${apiBase}/api/v2/audit?limit=20`).then(r => r.json()),
      ]);
      if (results[0].status === "fulfilled") setHealth(results[0].value as Health);
      if (results[1].status === "fulfilled") setStats(results[1].value as Stats);
      if (results[2].status === "fulfilled") setBackend(results[2].value as Backend);
      if (results[3].status === "fulfilled") setAudit(((results[3].value as { entries: AuditEntry[] }).entries) ?? []);
      setLoading(false);
    };
    void load();
    const id = setInterval(() => void load(), 10_000);
    return () => clearInterval(id);
  }, [apiBase]);

  const uptimeMins = stats ? Math.floor(stats.uptime / 60) : null;

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em", margin: 0 }}>System</h1>
        <p style={{ color: "var(--muted)", marginTop: 4, fontSize: 12 }}>
          Node monitoring, audit log, environments · 10s auto-refresh
        </p>
      </div>

      <div style={{ display: "flex", gap: 12, marginBottom: 24, flexWrap: "wrap" }}>
        <StatCard label="Status" value={health?.status === "ok" ? "OK" : loading ? "…" : "Offline"} trend={health?.status === "ok" ? "up" : "down"} sub="FastAPI node" />
        <StatCard label="Uptime" value={uptimeMins != null ? `${uptimeMins}m` : "—"} sub={stats ? `${stats.uptime.toFixed(0)}s total` : undefined} />
        <StatCard label="Requests" value={stats?.requests ?? "—"} sub="since start" />
        <StatCard label="Memory" value={stats ? `${stats.memory_mb.toFixed(0)} MB` : "—"} sub="RSS" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Backend Info</div>
          {backend ? (
            <div style={{ fontSize: 12, display: "flex", flexDirection: "column", gap: 6 }}>
              <Row k="Node ID" v={backend.node_id} />
              <Row k="Version" v={`v${backend.version}`} />
              <Row k="Status" v={backend.status} />
              <Row k="AI Engines" v={backend.engines.join(", ") || "—"} />
            </div>
          ) : (
            <div style={{ color: "var(--muted)", fontSize: 12 }}>{loading ? "Loading…" : "API not reachable"}</div>
          )}
        </div>

        <div className="card">
          <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13 }}>Performance Notes</div>
          <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8 }}>
            <div>• Audit log: append-only JSONL (single open fd)</div>
            <div>• File reads: bounded at 4 MB cap</div>
            <div>• Monitor: psutil w/ TTL cache</div>
            <div>• Messenger: asyncio.Lock in-memory store</div>
            <div>• Analysis: polars lazy scan + pushdown</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div style={{ fontWeight: 600, marginBottom: 12, fontSize: 13, display: "flex", justifyContent: "space-between" }}>
          <span>Audit Log</span>
          <span style={{ fontSize: 11, color: "var(--muted)" }}>{audit.length} entries</span>
        </div>
        {audit.length === 0 ? (
          <div style={{ color: "var(--muted)", fontSize: 12, textAlign: "center", padding: 20 }}>
            {loading ? "Loading…" : "No audit entries"}
          </div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ borderBottom: "1px solid var(--border)" }}>
                  {["Time", "Action", "Resource", "ID", "Detail"].map(h => (
                    <th key={h} style={{ padding: "6px 10px", textAlign: "left", color: "var(--muted)", fontWeight: 600, fontSize: 10, textTransform: "uppercase" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {audit.map((e, i) => (
                  <tr key={i} style={{ borderBottom: "1px solid rgba(31,41,55,0.5)" }}>
                    <td style={{ padding: "6px 10px", color: "var(--muted)", whiteSpace: "nowrap" }}>
                      {new Date(e.timestamp).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: "6px 10px", fontWeight: 600 }}>
                      <span className={`badge ${e.action === "delete" ? "badge-red" : e.action === "create" ? "badge-green" : "badge-blue"}`}>
                        {e.action}
                      </span>
                    </td>
                    <td style={{ padding: "6px 10px" }}>{e.resource_type}</td>
                    <td style={{ padding: "6px 10px", color: "var(--muted)", fontVariantNumeric: "tabular-nums" }}>{String(e.resource_id)}</td>
                    <td style={{ padding: "6px 10px", color: "var(--muted)", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{e.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid var(--border)" }}>
      <span style={{ color: "var(--muted)" }}>{k}</span>
      <span style={{ fontWeight: 600 }}>{v}</span>
    </div>
  );
}
