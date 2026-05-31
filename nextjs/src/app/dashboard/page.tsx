"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { getStats, getAudit, createFunc, getHealth } from "@/lib/api";
import type { ClusterStats, AuditEntry, HealthResponse } from "@/lib/types";

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
    <div className="runic-card p-5 flex-1 min-w-[140px]">
      <p
        className={`text-4xl font-bold font-mono transition-all duration-500 gradient-frost ${flash ? "glow-frost" : ""}`}
        style={{ color: color }}
      >
        {value}
      </p>
      <p className="text-[10px] text-muted uppercase tracking-widest font-medium mt-2">
        {label}
      </p>
    </div>
  );
}

// ── Quick Action button ─────────────────────────────────────
function QuickAction({
  href,
  onClick,
  icon,
  label,
}: {
  href?: string;
  onClick?: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  const inner = (
    <div className="runic-card p-4 flex items-center gap-3 cursor-pointer transition-all hover:bg-white/[0.05]">
      <span className="text-frost shrink-0">{icon}</span>
      <span className="text-xs font-medium text-foreground-dim">{label}</span>
    </div>
  );
  if (href) {
    return (
      <Link href={href} className="flex-1 min-w-[160px]">
        {inner}
      </Link>
    );
  }
  return (
    <button onClick={onClick} className="flex-1 min-w-[160px] text-left">
      {inner}
    </button>
  );
}

// ── Live metric card with optional progress bar + trend ──────
function MetricCard({
  label,
  value,
  sub,
  percent,
  color,
  trend,
}: {
  label: string;
  value: string;
  sub?: string;
  percent?: number;
  color?: string;
  trend?: number;  // positive = up (good), negative = down, 0 = flat
}) {
  const trendIcon = trend == null ? null
    : trend > 0 ? "↑" : trend < 0 ? "↓" : "→";
  const trendColor = trend == null ? undefined
    : trend > 0 ? "var(--emerald)" : trend < 0 ? "var(--rose)" : "var(--muted)";
  return (
    <div className="glass-card p-4 flex-1 min-w-[140px]">
      <p className="text-[10px] text-muted uppercase tracking-widest font-medium">
        {label}
      </p>
      <div className="flex items-baseline gap-2 mt-2">
        <p
          className="text-2xl font-bold font-mono"
          style={{ color: color || "var(--foreground)" }}
        >
          {value}
        </p>
        {trendIcon && (
          <span className="text-sm font-mono font-bold" style={{ color: trendColor }}>{trendIcon}</span>
        )}
      </div>
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

// ── Health colour map ──────────────────────────────────────
function healthStyle(status: string): { dot: string; text: string; label: string } {
  if (status === "healthy" || status === "ok") {
    return { dot: "var(--emerald)", text: "var(--emerald)", label: "Healthy" };
  }
  if (status === "degraded" || status === "warning") {
    return { dot: "var(--amber)", text: "var(--amber)", label: "Degraded" };
  }
  return { dot: "var(--rose)", text: "var(--rose)", label: "Down" };
}

// ── Mini Sparkline (cluster pulse) ─────────────────────────
function PulseSparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length === 0) return null;
  const max = Math.max(...values, 1);
  return (
    <div className="flex items-end gap-[3px] h-10">
      {values.map((v, i) => (
        <div
          key={i}
          className="flex-1 rounded-sm transition-all duration-300"
          style={{
            height: `${Math.max(3, (v / max) * 40)}px`,
            background: color,
            opacity: 0.3 + (i / Math.max(1, values.length - 1)) * 0.7,
            boxShadow: i === values.length - 1 ? `0 0 6px ${color}` : "none",
          }}
        />
      ))}
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const [prevStats, setPrevStats] = useState<ClusterStats | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [pulseHistory, setPulseHistory] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // New Function modal
  const [newFuncOpen, setNewFuncOpen] = useState(false);
  const [newFuncName, setNewFuncName] = useState("");
  const [newFuncCode, setNewFuncCode] = useState("def hello():\n    return 'world'\n");
  const [newFuncSaving, setNewFuncSaving] = useState(false);
  const [newFuncError, setNewFuncError] = useState("");

  const fetchStats = useCallback(async () => {
    try {
      const s = await getStats();
      setStats((prev) => { setPrevStats(prev); return s; });
      // Keep a rolling 30-sample history of active_runs for the pulse widget.
      setPulseHistory((prev) => [...prev.slice(-29), s.active_runs]);
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

  const fetchHealth = useCallback(async () => {
    try {
      const h = await getHealth();
      setHealth(h);
    } catch {
      setHealth({ status: "down", node_id: "", checks: {} });
    }
  }, []);

  const handleCreateFunc = async () => {
    const name = newFuncName.trim();
    if (!name) {
      setNewFuncError("Name is required");
      return;
    }
    setNewFuncSaving(true);
    setNewFuncError("");
    try {
      await createFunc({ name, code: newFuncCode });
      setNewFuncOpen(false);
      setNewFuncName("");
      setNewFuncCode("def hello():\n    return 'world'\n");
      fetchStats();
      fetchAudit();
    } catch (err) {
      setNewFuncError(err instanceof Error ? err.message : "Create failed");
    } finally {
      setNewFuncSaving(false);
    }
  };

  useEffect(() => {
    fetchStats();
    fetchAudit();
    fetchHealth();
  }, [fetchStats, fetchAudit, fetchHealth]);

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

  // Poll health every 10s
  useEffect(() => {
    const id = setInterval(fetchHealth, 10000);
    return () => clearInterval(id);
  }, [fetchHealth]);

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
    <div className="relative p-6 space-y-6 overflow-y-auto h-screen animate-in">
      {/* Ambient aurora behind the page content */}
      <div className="aurora-bg" />

      {/* Header */}
      <div className="relative flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground glow-frost">Dashboard</h1>
          <p className="text-sm text-muted mt-1">
            <span className="font-mono">{stats.node_id}</span> overview
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Health badge */}
          {health && (() => {
            const hs = healthStyle(health.status);
            return (
              <div
                className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]"
                title={Object.entries(health.checks)
                  .map(([k, v]) => `${k}: ${v.status}`)
                  .join(" • ")}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ background: hs.dot, boxShadow: `0 0 8px ${hs.dot}` }}
                />
                <span className="text-[11px] font-mono" style={{ color: hs.text }}>
                  {hs.label}
                </span>
              </div>
            );
          })()}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
            <span className="w-1.5 h-1.5 rounded-full status-online" />
            <span className="text-[11px] font-mono text-muted">Live - 3s</span>
          </div>
        </div>
      </div>

      {/* Cluster Pulse widget — recent active_runs sparkline */}
      <div className="relative runic-card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            Cluster Pulse
          </h2>
          <div className="flex items-baseline gap-2">
            <span
              className="text-2xl font-bold font-mono gradient-frost"
              style={{ color: stats.active_runs > 0 ? "var(--amber)" : undefined }}
            >
              {stats.active_runs}
            </span>
            <span className="text-[10px] text-muted uppercase tracking-widest">active</span>
          </div>
        </div>
        <PulseSparkline
          values={pulseHistory}
          color={stats.active_runs > 0 ? "var(--amber)" : "var(--frost)"}
        />
        <p className="text-[10px] text-muted/60 font-mono mt-2 text-right">
          last {pulseHistory.length} samples &middot; 3s interval
        </p>
      </div>

      {/* Quick Actions row */}
      <div className="relative flex flex-wrap gap-3">
        <QuickAction
          onClick={() => {
            setNewFuncOpen(true);
            setNewFuncError("");
          }}
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
          }
          label="New Function"
        />
        <QuickAction
          href="/dags"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="5" cy="6" r="3" />
              <circle cx="19" cy="6" r="3" />
              <circle cx="12" cy="18" r="3" />
              <path d="M7.5 8l3 7M16.5 8l-3 7" />
            </svg>
          }
          label="New DAG"
        />
        <QuickAction
          href="/files"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
            </svg>
          }
          label="Browse Files"
        />
        <QuickAction
          href="/chat"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
          }
          label="Open Chat"
        />
      </div>

      {/* Big numbers row */}
      <div className="relative flex flex-wrap gap-3">
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
      <div className="relative flex flex-wrap gap-3">
        <MetricCard
          label="CPU"
          value={`${stats.cpu_percent.toFixed(1)}%`}
          percent={stats.cpu_percent}
          color={cpuColor}
          trend={prevStats ? Math.sign(stats.cpu_percent - prevStats.cpu_percent) * -1 : undefined}
        />
        <MetricCard
          label="Memory"
          value={`${stats.memory_percent.toFixed(1)}%`}
          percent={stats.memory_percent}
          color={memColor}
          trend={prevStats ? Math.sign(stats.memory_percent - prevStats.memory_percent) * -1 : undefined}
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
          trend={prevStats ? Math.sign(stats.peer_count - prevStats.peer_count) : undefined}
        />
        <MetricCard
          label="GPUs"
          value={String(stats.gpu_count)}
          color={stats.gpu_count > 0 ? "var(--emerald)" : "var(--muted)"}
        />
      </div>

      {/* Activity feed */}
      <div className="relative glass-card p-5 space-y-3">
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

      {/* ── New Function modal ─────────────────────────────── */}
      {newFuncOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          onClick={() => setNewFuncOpen(false)}
        >
          <div
            className="runic-card p-6 w-full max-w-2xl space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold gradient-frost">New Function</h2>
              <button
                onClick={() => setNewFuncOpen(false)}
                className="text-muted hover:text-foreground transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Name</label>
              <input
                type="text"
                value={newFuncName}
                onChange={(e) => setNewFuncName(e.target.value)}
                placeholder="my_function"
                autoFocus
                className="w-full bg-white/[0.04] border border-white/[0.08] rounded-lg px-3 py-2 text-sm font-mono text-foreground placeholder-muted/50 outline-none focus:border-frost/30 transition-colors"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] text-muted uppercase tracking-wider font-medium">Code</label>
              {/* Code editor pane — fake gutter (line numbers) + monospace dark textarea. */}
              <div className="rounded-lg overflow-hidden border border-white/[0.08] bg-[#0a0a14]">
                {/* Editor toolbar */}
                <div className="flex items-center justify-between px-3 py-1.5 border-b border-white/[0.06] bg-white/[0.02]">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-rose/60" />
                    <span className="w-2 h-2 rounded-full bg-amber/60" />
                    <span className="w-2 h-2 rounded-full bg-emerald/60" />
                    <span className="ml-2 text-[10px] uppercase tracking-widest font-bold text-frost">
                      Python
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-muted/60">
                    {newFuncCode.split("\n").length} lines &middot; 80 cols
                  </span>
                </div>
                {/* Gutter + textarea */}
                <div className="relative flex">
                  {/* Line numbers gutter */}
                  <div
                    aria-hidden
                    className="select-none text-right text-[11px] font-mono text-muted/40 leading-[1.5] py-3 pl-3 pr-2 border-r border-white/[0.05] bg-black/30"
                    style={{ minWidth: "2.4rem" }}
                  >
                    {newFuncCode.split("\n").map((_, i) => (
                      <div key={i}>{i + 1}</div>
                    ))}
                  </div>
                  <textarea
                    value={newFuncCode}
                    onChange={(e) => setNewFuncCode(e.target.value)}
                    rows={12}
                    spellCheck={false}
                    wrap="off"
                    cols={80}
                    className="flex-1 bg-transparent px-3 py-3 text-[12px] font-mono text-foreground placeholder-muted/50 outline-none resize-none leading-[1.5] overflow-x-auto"
                    style={{
                      caretColor: "var(--frost)",
                      tabSize: 4,
                    }}
                  />
                </div>
              </div>
            </div>

            {newFuncError && <p className="text-xs text-rose font-mono">{newFuncError}</p>}

            <div className="flex justify-end gap-2">
              <button
                onClick={() => setNewFuncOpen(false)}
                className="px-4 py-2 text-sm rounded-lg bg-white/[0.04] hover:bg-white/[0.06] border border-white/[0.08]"
              >
                Cancel
              </button>
              <button
                onClick={handleCreateFunc}
                disabled={newFuncSaving}
                className="px-4 py-2 text-sm rounded-lg bg-frost/20 text-frost border border-frost/30 hover:bg-frost/30 disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {newFuncSaving ? "Creating..." : "Create"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
