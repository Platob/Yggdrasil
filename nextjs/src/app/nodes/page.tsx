"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import Link from "next/link";
import { NodeCard } from "@/components/NodeCard";
import { ResourceBar } from "@/components/ResourceBar";
import {
  getBackend,
  getPeers,
  getEnvs,
  getFuncs,
  getUsers,
  getAudit,
  getDags,
  getRuns,
  getStats,
  createBackendStream,
} from "@/lib/api";
import type {
  NodeBackend,
  NodeMeta,
  PyEnvEntry,
  PyFuncEntry,
  BackendStreamEvent,
  UserCard,
  AuditEntry,
  DAGEntry,
  PyFuncRunEntry,
  ClusterStats,
} from "@/lib/types";

// ── Helper: convert NodeBackend to NodeMeta for the self card ──
function backendToMeta(b: NodeBackend): NodeMeta {
  return {
    node_id: b.node_id,
    host: b.hostname,
    port: 8100,
    role: b.role,
    version: "",
    lat: null,
    lon: null,
    cpu_percent: b.cpu_percent,
    memory_percent:
      b.memory_total_mb > 0
        ? (b.memory_used_mb / b.memory_total_mb) * 100
        : 0,
    active_runs: b.active_runs,
    gpu_count: b.gpus.length,
  };
}

// ── Status badge for environments ────────────────────────────
function EnvStatusBadge({ status }: { status: string }) {
  const colors: Record<string, { bg: string; text: string }> = {
    ready:   { bg: "rgba(52,211,153,0.1)",  text: "var(--emerald)" },
    pending: { bg: "rgba(251,191,36,0.1)",  text: "var(--amber)" },
    error:   { bg: "rgba(244,63,94,0.1)",   text: "var(--rose)" },
  };
  const c = colors[status] || colors.pending;
  return (
    <span
      className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
      style={{ background: c.bg, color: c.text }}
    >
      {status}
    </span>
  );
}

// ── Relative time helper ─────────────────────────────────────
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

// ── Mini sparkline (5-6 thin bars) ──────────────────────────
function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length === 0) return null;
  const max = Math.max(...values, 1);
  return (
    <div className="flex items-end gap-[2px] h-4 mt-1">
      {values.slice(-6).map((v, i) => (
        <div
          key={i}
          className="w-[4px] rounded-sm transition-all duration-300"
          style={{
            height: `${Math.max(2, (v / max) * 16)}px`,
            background: color,
            opacity: 0.5 + (i / values.length) * 0.5,
          }}
        />
      ))}
    </div>
  );
}

// ── KPI card ─────────────────────────────────────────────────
function KpiCard({ label, value, sub, color, pulse, sparkValues }: {
  label: string;
  value: string;
  sub?: string;
  color?: string;
  pulse?: boolean;
  sparkValues?: number[];
}) {
  return (
    <div className="glass-card p-4 flex-1 min-w-[140px]">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-muted uppercase tracking-wider font-medium">{label}</span>
        {pulse && (
          <span className="relative flex h-2 w-2">
            <span className="absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: color || "var(--amber)", animation: "pulse-frost 1.5s ease-in-out infinite" }} />
            <span className="relative inline-flex rounded-full h-2 w-2" style={{ background: color || "var(--amber)" }} />
          </span>
        )}
      </div>
      <p className="text-xl font-bold font-mono mt-1" style={{ color: color || "var(--foreground)" }}>
        {value}
      </p>
      {sub && <p className="text-[11px] text-foreground-dim font-mono mt-0.5">{sub}</p>}
      {sparkValues && sparkValues.length > 1 && <Sparkline values={sparkValues} color={color || "var(--frost)"} />}
    </div>
  );
}

export default function NodesPage() {
  const [backend, setBackend] = useState<NodeBackend | null>(null);
  const [peers, setPeers] = useState<NodeMeta[]>([]);
  const [envs, setEnvs] = useState<PyEnvEntry[]>([]);
  const [funcs, setFuncs] = useState<PyFuncEntry[]>([]);
  const [dagList, setDagList] = useState<DAGEntry[]>([]);
  const [activeRuns, setActiveRuns] = useState<PyFuncRunEntry[]>([]);
  const [users, setUsers] = useState<UserCard[]>([]);
  const [auditEntries, setAuditEntries] = useState<AuditEntry[]>([]);
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const [cpuHistory, setCpuHistory] = useState<number[]>([]);
  const [memHistory, setMemHistory] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [sseConnected, setSseConnected] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // ── Initial data fetch ──────────────────────────────────────
  const fetchAll = useCallback(async () => {
    try {
      const [backendRes, peersRes, envsRes, funcsRes, dagsRes, runsRes, usersRes, auditRes, statsRes] = await Promise.allSettled([
        getBackend(),
        getPeers(),
        getEnvs(),
        getFuncs(),
        getDags(),
        getRuns(),
        getUsers(),
        getAudit(),
        getStats(),
      ]);
      if (statsRes.status === "fulfilled") setStats(statsRes.value);
      if (backendRes.status === "fulfilled") {
        setBackend(backendRes.value.backend);
        setCpuHistory((prev) => [...prev.slice(-5), backendRes.value.backend.cpu_percent]);
        setMemHistory((prev) => {
          const b = backendRes.value.backend;
          const pct = b.memory_total_mb > 0 ? (b.memory_used_mb / b.memory_total_mb) * 100 : 0;
          return [...prev.slice(-5), pct];
        });
      }
      if (peersRes.status === "fulfilled") setPeers(peersRes.value.peers);
      if (envsRes.status === "fulfilled") setEnvs(envsRes.value.envs);
      if (funcsRes.status === "fulfilled") setFuncs(funcsRes.value.funcs);
      if (dagsRes.status === "fulfilled") setDagList(dagsRes.value.dags);
      if (runsRes.status === "fulfilled") setActiveRuns(runsRes.value.runs.filter((r) => r.status === "running"));
      if (usersRes.status === "fulfilled") setUsers(usersRes.value.users);
      if (auditRes.status === "fulfilled") setAuditEntries(auditRes.value.entries);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // ── SSE stream for real-time backend updates ────────────────
  useEffect(() => {
    const es = createBackendStream();
    eventSourceRef.current = es;

    es.onopen = () => setSseConnected(true);

    es.onmessage = (event) => {
      try {
        const data: BackendStreamEvent = JSON.parse(event.data);
        if (data.backend) {
          setBackend(data.backend);
          setCpuHistory((prev) => [...prev.slice(-5), data.backend.cpu_percent]);
          setMemHistory((prev) => {
            const b = data.backend;
            const pct = b.memory_total_mb > 0 ? (b.memory_used_mb / b.memory_total_mb) * 100 : 0;
            return [...prev.slice(-5), pct];
          });
        }
      } catch {
        // Silently ignore parse errors from SSE
      }
    };

    es.onerror = () => {
      setSseConnected(false);
    };

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  // ── Build the combined node list ────────────────────────────
  const selfNode: NodeMeta | null = backend ? backendToMeta(backend) : null;
  const allNodes: { node: NodeMeta; isSelf: boolean }[] = [];
  if (selfNode) {
    allNodes.push({ node: selfNode, isSelf: true });
  }
  for (const peer of peers) {
    allNodes.push({ node: peer, isSelf: false });
  }

  // ── Aggregated KPIs ────────────────────────────────────────
  // Prefer the consolidated /api/v2/stats payload when available — it's one
  // call instead of fanning out across backend+peers+dags client-side.
  const totalNodes = stats ? stats.peer_count + 1 : allNodes.length;
  const totalCpuCores = backend?.cpu_count ?? 0;
  const clusterCpuPercent = stats?.cpu_percent ?? (
    allNodes.length > 0
      ? allNodes.map((n) => n.node.cpu_percent).reduce((a, b) => a + b, 0) / allNodes.length
      : 0
  );
  const totalMemoryMb = backend?.memory_total_mb ?? 0;
  const totalGpus = stats?.gpu_count ?? ((backend?.gpus.length ?? 0) + peers.reduce((sum, p) => sum + p.gpu_count, 0));
  const totalActiveRuns = stats?.active_runs ?? allNodes.reduce((sum, n) => sum + n.node.active_runs, 0);
  const totalDags = stats?.dag_count ?? dagList.length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Connecting to Yggdrasil...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen">
      {/* ── Main content ─────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6 animate-in">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground">Nodes</h1>
            <p className="text-sm text-muted mt-1">
              {allNodes.length} node{allNodes.length !== 1 ? "s" : ""} in the mesh
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* SSE status */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  sseConnected ? "status-online" : "status-offline"
                }`}
              />
              <span className="text-[11px] font-mono text-muted">
                {sseConnected ? "Live" : "Polling"}
              </span>
            </div>
            {/* Refresh button */}
            <button
              onClick={fetchAll}
              className="
                px-3 py-1.5 rounded-lg text-xs font-medium
                text-frost/70 hover:text-frost
                bg-frost/5 hover:bg-frost/10
                border border-frost/10 hover:border-frost/20
                transition-all duration-150
              "
            >
              <svg className="inline-block mr-1.5 -mt-0.5" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 11-2.12-9.36L23 10" />
              </svg>
              Refresh
            </button>
          </div>
        </div>

        {/* Cluster KPIs */}
        <div className="flex flex-wrap gap-3">
          <KpiCard label="Total Nodes" value={String(totalNodes)} color="var(--frost)" />
          <KpiCard label="CPU Cores" value={String(totalCpuCores)} sub={`${clusterCpuPercent.toFixed(1)}% avg`} />
          <KpiCard
            label="Cluster CPU"
            value={`${clusterCpuPercent.toFixed(1)}%`}
            color={clusterCpuPercent > 80 ? "var(--rose)" : clusterCpuPercent > 50 ? "var(--amber)" : "var(--emerald)"}
            sparkValues={cpuHistory}
          />
          <KpiCard
            label="Total Memory"
            value={totalMemoryMb >= 1024 ? `${(totalMemoryMb / 1024).toFixed(1)} GB` : `${totalMemoryMb} MB`}
            sparkValues={memHistory}
            color="var(--emerald)"
          />
          <KpiCard label="GPUs" value={String(totalGpus)} color={totalGpus > 0 ? "var(--emerald)" : "var(--muted)"} />
          <KpiCard
            label="Active Runs"
            value={String(totalActiveRuns)}
            color={totalActiveRuns > 0 ? "var(--amber)" : "var(--muted)"}
            pulse={totalActiveRuns > 0}
          />
          <KpiCard
            label="DAGs"
            value={String(totalDags)}
            color={totalDags > 0 ? "var(--frost)" : "var(--muted)"}
          />
        </div>

        {/* Active Runs — task manager view */}
        {activeRuns.length > 0 && (
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Running
              <span className="relative flex h-2 w-2 ml-1">
                <span className="absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: "var(--amber)", animation: "pulse-frost 1.5s ease-in-out infinite" }} />
                <span className="relative inline-flex rounded-full h-2 w-2" style={{ background: "var(--amber)" }} />
              </span>
              <span className="ml-auto text-[10px] font-mono text-foreground-dim">
                {activeRuns.length}
              </span>
            </h2>
            <div className="space-y-2">
              {activeRuns.slice(0, 8).map((run) => {
                const funcName = funcs.find((f) => f.id === run.func_id)?.name ?? `func#${run.func_id}`;
                const progress = run.progress != null ? run.progress : null;
                return (
                  <div key={run.id} className="px-3 py-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04] space-y-1.5">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="relative flex h-1.5 w-1.5">
                          <span className="absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: "var(--amber)", animation: "pulse-frost 1.5s ease-in-out infinite" }} />
                          <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: "var(--amber)" }} />
                        </span>
                        <span className="text-xs font-mono font-medium text-foreground">{funcName}</span>
                      </div>
                      <span className="text-[10px] font-mono text-muted">#{run.id}</span>
                    </div>
                    <div className="h-1 rounded-full bg-white/[0.06] overflow-hidden">
                      {progress != null ? (
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{
                            width: `${Math.min(100, Math.max(0, progress))}%`,
                            background: "linear-gradient(90deg, var(--amber), var(--amber)cc)",
                            boxShadow: "0 0 8px var(--amber-glow)",
                          }}
                        />
                      ) : (
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: "100%",
                            background: "linear-gradient(90deg, transparent 0%, var(--amber) 50%, transparent 100%)",
                            backgroundSize: "200% 100%",
                            animation: "shimmer 1.5s ease-in-out infinite",
                          }}
                        />
                      )}
                    </div>
                    {progress != null && (
                      <p className="text-[10px] font-mono text-muted text-right">{progress.toFixed(0)}%</p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Recent Activity */}
        {auditEntries.length > 0 && (
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              Recent Activity
            </h2>
            <div className="space-y-1">
              {auditEntries.slice(0, 5).map((entry, i) => (
                <div
                  key={`${entry.asset_id}-${entry.timestamp}-${i}`}
                  className="flex items-center gap-3 px-3 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors"
                >
                  <span className="text-[10px] font-mono text-muted shrink-0 w-14">
                    {timeAgo(entry.timestamp)}
                  </span>
                  <span
                    className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded shrink-0"
                    style={{
                      background: entry.operation === "create" ? "rgba(52,211,153,0.1)"
                        : entry.operation === "delete" ? "rgba(244,63,94,0.1)"
                        : "rgba(103,232,249,0.1)",
                      color: entry.operation === "create" ? "var(--emerald)"
                        : entry.operation === "delete" ? "var(--rose)"
                        : "var(--frost)",
                    }}
                  >
                    {entry.operation}
                  </span>
                  <span className="text-xs text-foreground-dim font-mono">{entry.asset_type}</span>
                  <span className="text-xs text-muted truncate flex-1">{entry.detail}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Node grid */}
        {allNodes.length === 0 ? (
          <div className="text-center py-20">
            <p className="text-muted text-sm">No nodes found. Is the backend running?</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {allNodes.map(({ node, isSelf }) => (
              <Link key={node.node_id} href={`/nodes/${node.node_id}`} className="block">
                <NodeCard node={node} isSelf={isSelf} />
              </Link>
            ))}
          </div>
        )}

        {/* Backend details (when available) */}
        {backend && (
          <div className="glass-card p-5 space-y-3">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">
              Local Backend
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <InfoCell label="Hostname" value={backend.hostname} mono />
              <InfoCell label="Platform" value={backend.platform} />
              <InfoCell label="Python" value={backend.python_version} mono />
              <InfoCell label="CPU Cores" value={String(backend.cpu_count)} />
              <InfoCell
                label="Memory"
                value={`${(backend.memory_used_mb / 1024).toFixed(1)} / ${(backend.memory_total_mb / 1024).toFixed(1)} GB`}
                mono
              />
              <InfoCell
                label="Disk"
                value={`${(backend.disk_used_mb / 1024).toFixed(1)} / ${(backend.disk_total_mb / 1024).toFixed(1)} GB`}
                mono
              />
              <InfoCell label="Active Runs" value={String(backend.active_runs)} highlight />
              <InfoCell label="Total Runs" value={String(backend.total_runs)} />
            </div>
          </div>
        )}
      </div>

      {/* ── Right sidebar: functions + environments + users ──── */}
      <div className="w-72 border-l border-border shrink-0 overflow-y-auto bg-background-elevated/50">
        <div className="p-4 space-y-6">
          {/* Users */}
          <div>
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted mb-3 flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4-4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 00-3-3.87" />
                <path d="M16 3.13a4 4 0 010 7.75" />
              </svg>
              Users
              <span className="ml-auto text-[10px] font-mono text-foreground-dim">
                {users.length}
              </span>
            </h3>
            {users.length === 0 ? (
              <p className="text-xs text-muted/60 italic">No users connected</p>
            ) : (
              <div className="space-y-1.5">
                {users.map((user) => (
                  <div
                    key={user.user_id}
                    className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04] hover:border-white/[0.08] transition-all"
                  >
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full shrink-0 ${user.online ? "status-online" : "status-offline"}`} />
                      <span className="text-xs font-mono font-medium text-foreground truncate">
                        {user.first_name || user.key}
                      </span>
                      <span className="text-[10px] text-muted ml-auto capitalize shrink-0">{user.role}</span>
                    </div>
                    {user.email && (
                      <p className="text-[10px] text-muted mt-0.5 truncate pl-4">{user.email}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Functions */}
          <div>
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted mb-3 flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              Functions
              <span className="ml-auto text-[10px] font-mono text-foreground-dim">
                {funcs.length}
              </span>
            </h3>
            {funcs.length === 0 ? (
              <p className="text-xs text-muted/60 italic">No functions registered</p>
            ) : (
              <div className="space-y-1.5">
                {funcs.map((f) => (
                  <div
                    key={f.id}
                    className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04] hover:border-white/[0.08] transition-all"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-medium text-foreground truncate">
                        {f.name}
                      </span>
                      <span className="text-[10px] text-muted font-mono ml-2 shrink-0">
                        {f.run_count} runs
                      </span>
                    </div>
                    {f.description && (
                      <p className="text-[11px] text-muted mt-0.5 truncate">
                        {f.description}
                      </p>
                    )}
                    {f.dependencies.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {f.dependencies.slice(0, 3).map((dep) => (
                          <span
                            key={dep}
                            className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-frost/5 text-frost/60"
                          >
                            {dep}
                          </span>
                        ))}
                        {f.dependencies.length > 3 && (
                          <span className="text-[9px] text-muted">
                            +{f.dependencies.length - 3}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Environments */}
          <div>
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted mb-3 flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
                <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                <line x1="12" y1="22.08" x2="12" y2="12" />
              </svg>
              Environments
              <span className="ml-auto text-[10px] font-mono text-foreground-dim">
                {envs.length}
              </span>
            </h3>
            {envs.length === 0 ? (
              <p className="text-xs text-muted/60 italic">No environments created</p>
            ) : (
              <div className="space-y-1.5">
                {envs.map((env) => (
                  <div
                    key={env.id}
                    className="px-3 py-2 rounded-lg bg-white/[0.02] border border-white/[0.04] hover:bg-white/[0.04] hover:border-white/[0.08] transition-all"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-medium text-foreground truncate">
                        {env.name}
                      </span>
                      <EnvStatusBadge status={env.status} />
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-[10px] text-muted font-mono">
                        Python {env.python_version}
                      </span>
                      {env.dependencies.length > 0 && (
                        <span className="text-[10px] text-muted">
                          {env.dependencies.length} dep{env.dependencies.length !== 1 ? "s" : ""}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Small info cell component ────────────────────────────────
function InfoCell({
  label,
  value,
  mono,
  highlight,
}: {
  label: string;
  value: string;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <span className="text-[10px] text-muted uppercase tracking-wider">{label}</span>
      <p
        className={`text-sm truncate mt-0.5 ${mono ? "font-mono text-xs" : ""} ${
          highlight ? "text-frost" : "text-foreground"
        }`}
      >
        {value}
      </p>
    </div>
  );
}
