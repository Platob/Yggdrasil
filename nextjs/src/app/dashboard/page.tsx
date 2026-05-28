"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { getStats, getAudit } from "@/lib/api";
import type { ClusterStats, AuditEntry } from "@/lib/types";

// ── Format uptime as "2d 4h" or "12m" etc. ───────────────────
function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

// ── Relative time ────────────────────────────────────────────
function timeAgo(ts: string): string {
  try {
    const now = Date.now();
    const then = new Date(ts).getTime();
    const diffMs = now - then;
    if (diffMs < 0) return "just now";
    const diffSec = Math.floor(diffMs / 1000);
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    const diffDay = Math.floor(diffHr / 24);
    return `${diffDay}d ago`;
  } catch {
    return "";
  }
}

// ── Big number card with count-up flash on change ───────────
function BigStatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const [flash, setFlash] = useState(false);
  const prevRef = useRef<number>(value);
  useEffect(() => {
    if (prevRef.current !== value) {
      setFlash(true);
      const t = setTimeout(() => setFlash(false), 600);
      prevRef.current = value;
      return () => clearTimeout(t);
    }
  }, [value]);
  return (
    <div className="glass-card p-5 flex-1 min-w-[140px]">
      <p
        className="text-4xl font-bold font-mono transition-colors duration-500"
        style={{ color: flash ? "var(--frost)" : color || "var(--foreground)" }}
      >
        {value}
      </p>
      <p className="text-[10px] text-muted uppercase tracking-widest font-medium mt-2">
        {label}
      </p>
    </div>
  );
}

// ── Live metric card with optional progress bar ─────────────
function MetricCard({
  label,
  value,
  sub,
  percent,
  color,
}: {
  label: string;
  value: string;
  sub?: string;
  percent?: number;
  color?: string;
}) {
  return (
    <div className="glass-card p-4 flex-1 min-w-[140px]">
      <p className="text-[10px] text-muted uppercase tracking-widest font-medium">
        {label}
      </p>
      <p
        className="text-2xl font-bold font-mono mt-2"
        style={{ color: color || "var(--foreground)" }}
      >
        {value}
      </p>
      {sub && (
        <p className="text-[10px] text-foreground-dim font-mono mt-0.5">{sub}</p>
      )}
      {percent != null && (
        <div className="mt-2 h-1 rounded-full bg-white/[0.06] overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{
              width: `${Math.min(100, Math.max(0, percent))}%`,
              background: color || "var(--frost)",
            }}
          />
        </div>
      )}
    </div>
  );
}

// ── Operation badge colour map ──────────────────────────────
function opStyle(op: string): { bg: string; text: string } {
  if (op === "create") return { bg: "rgba(52,211,153,0.1)", text: "var(--emerald)" };
  if (op === "delete") return { bg: "rgba(244,63,94,0.1)", text: "var(--rose)" };
  return { bg: "rgba(103,232,249,0.1)", text: "var(--frost)" };
}

export default function DashboardPage() {
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchStats = useCallback(async () => {
    try {
      const s = await getStats();
      setStats(s);
      setError(false);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchAudit = useCallback(async () => {
    try {
      const a = await getAudit();
      setAudit(a.entries);
    } catch {
      // Silently ignore audit errors
    }
  }, []);

  useEffect(() => {
    fetchStats();
    fetchAudit();
  }, [fetchStats, fetchAudit]);

  // Poll stats every 3s
  useEffect(() => {
    const id = setInterval(fetchStats, 3000);
    return () => clearInterval(id);
  }, [fetchStats]);

  // Poll audit every 5s
  useEffect(() => {
    const id = setInterval(fetchAudit, 5000);
    return () => clearInterval(id);
  }, [fetchAudit]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  if (error || !stats) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in space-y-4">
          <div className="w-12 h-12 rounded-full bg-rose/10 flex items-center justify-center mx-auto">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--rose)" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="15" y1="9" x2="9" y2="15" />
              <line x1="9" y1="9" x2="15" y2="15" />
            </svg>
          </div>
          <p className="text-sm text-muted">Backend unreachable</p>
        </div>
      </div>
    );
  }

  const cpuColor = stats.cpu_percent > 80 ? "var(--rose)" : stats.cpu_percent > 50 ? "var(--amber)" : "var(--emerald)";
  const memColor = stats.memory_percent > 80 ? "var(--rose)" : stats.memory_percent > 50 ? "var(--amber)" : "var(--emerald)";

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-screen animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Dashboard</h1>
          <p className="text-sm text-muted mt-1">
            <span className="font-mono">{stats.node_id}</span> overview
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          <span className="text-[11px] font-mono text-muted">Live - 3s</span>
        </div>
      </div>

      {/* Big numbers row */}
      <div className="flex flex-wrap gap-3">
        <BigStatCard label="Functions" value={stats.func_count} color="var(--frost)" />
        <BigStatCard label="Environments" value={stats.env_count} color="var(--emerald)" />
        <BigStatCard label="DAGs" value={stats.dag_count} color="var(--frost)" />
        <BigStatCard label="Scheduled" value={stats.scheduled_dags} color="var(--amber)" />
        <BigStatCard
          label="Active Runs"
          value={stats.active_runs}
          color={stats.active_runs > 0 ? "var(--amber)" : undefined}
        />
        <BigStatCard label="Total Runs" value={stats.total_runs} />
      </div>

      {/* Live metric cards */}
      <div className="flex flex-wrap gap-3">
        <MetricCard
          label="CPU"
          value={`${stats.cpu_percent.toFixed(1)}%`}
          percent={stats.cpu_percent}
          color={cpuColor}
        />
        <MetricCard
          label="Memory"
          value={`${stats.memory_percent.toFixed(1)}%`}
          percent={stats.memory_percent}
          color={memColor}
        />
        <MetricCard
          label="Uptime"
          value={formatUptime(stats.uptime)}
          color="var(--frost)"
        />
        <MetricCard
          label="Peers"
          value={String(stats.peer_count)}
          sub={stats.peer_count === 1 ? "node" : "nodes"}
          color={stats.peer_count > 0 ? "var(--frost)" : "var(--muted)"}
        />
        <MetricCard
          label="GPUs"
          value={String(stats.gpu_count)}
          color={stats.gpu_count > 0 ? "var(--emerald)" : "var(--muted)"}
        />
      </div>

      {/* Activity feed */}
      <div className="glass-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <polyline points="12 6 12 12 16 14" />
          </svg>
          Activity Feed
          <span className="ml-auto text-foreground-dim font-mono">{audit.length}</span>
        </h2>
        {audit.length === 0 ? (
          <p className="text-xs text-muted/60 italic py-4">No activity yet</p>
        ) : (
          <div className="space-y-1">
            {audit.slice(0, 20).map((entry, i) => {
              const s = opStyle(entry.operation);
              return (
                <div
                  key={`${entry.asset_id}-${entry.timestamp}-${i}`}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
                >
                  <span className="text-[10px] font-mono text-muted shrink-0 w-16">
                    {timeAgo(entry.timestamp)}
                  </span>
                  <span
                    className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded shrink-0"
                    style={{ background: s.bg, color: s.text }}
                  >
                    {entry.operation}
                  </span>
                  <span className="text-xs text-foreground-dim font-mono shrink-0">
                    {entry.asset_type}
                  </span>
                  <span className="text-xs text-muted truncate flex-1">
                    {entry.detail}
                  </span>
                  <span className="text-[10px] font-mono text-muted/60 shrink-0">
                    #{entry.asset_id}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
