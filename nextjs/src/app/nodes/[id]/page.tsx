"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { ResourceBar } from "@/components/ResourceBar";
import { Sparkline } from "@/components/Sparkline";
import { getBackend, getEnvs, getEnvPackages, getFuncs, getRuns, getNodeCard, getDags, getDagRuns, bulkDeleteFuncs, setEnvVars, deleteEnvVar, createBackendStream, cancelRun } from "@/lib/api";
import type { NodeBackend, NodeCard, PyEnvEntry, PyEnvPackages, PyFuncEntry, PyFuncRunEntry, DAGEntry, DAGRunEntry } from "@/lib/types";

// Rolling window of live samples for the timeseries charts (~1 min at 1s).
const MAX_SAMPLES = 90;

// Refresh-rate options for the live stream. 0 = paused.
const REFRESH_OPTIONS: { label: string; sec: number }[] = [
  { label: "0.5s", sec: 0.5 },
  { label: "1s", sec: 1 },
  { label: "2s", sec: 2 },
  { label: "5s", sec: 5 },
  { label: "Paused", sec: 0 },
];

interface MetricSample {
  cpu: number;
  mem: number;
  disk: number;
  sentRate: number;   // MB/s
  recvRate: number;   // MB/s
  active: number;
}

function formatRate(mbPerSec: number): string {
  if (mbPerSec >= 1) return `${mbPerSec.toFixed(1)} MB/s`;
  return `${(mbPerSec * 1024).toFixed(0)} KB/s`;
}

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function formatBytes(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(0)} MB`;
}

// ── Run status badge ────────────────────────────────────────
const RUN_STATUS_STYLES: Record<string, { bg: string; text: string }> = {
  completed: { bg: "rgba(52,211,153,0.1)",  text: "var(--emerald)" },
  success:   { bg: "rgba(52,211,153,0.1)",  text: "var(--emerald)" },
  failed:    { bg: "rgba(244,63,94,0.1)",   text: "var(--rose)" },
  error:     { bg: "rgba(244,63,94,0.1)",   text: "var(--rose)" },
  running:   { bg: "rgba(251,191,36,0.1)",  text: "var(--amber)" },
  pending:   { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
  queued:    { bg: "rgba(255,255,255,0.05)", text: "var(--muted)" },
};

function RunStatusBadge({ status }: { status: string }) {
  const s = RUN_STATUS_STYLES[status] || RUN_STATUS_STYLES.pending;
  return (
    <span
      className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded inline-flex items-center gap-1"
      style={{ background: s.bg, color: s.text }}
    >
      {status === "running" && (
        <span className="relative flex h-1.5 w-1.5">
          <span className="absolute inline-flex h-full w-full rounded-full opacity-75" style={{ background: s.text, animation: "pulse-frost 1.5s ease-in-out infinite" }} />
          <span className="relative inline-flex rounded-full h-1.5 w-1.5" style={{ background: s.text }} />
        </span>
      )}
      {status}
    </span>
  );
}

function formatDuration(sec: number | null): string {
  if (sec == null) return "--";
  if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

function timeAgo(ts: string | null): string {
  if (!ts) return "--";
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
    return "--";
  }
}

export default function NodeDetailPage() {
  const params = useParams();
  const nodeId = params.id as string;

  const [backend, setBackend] = useState<NodeBackend | null>(null);
  const [card, setCard] = useState<NodeCard | null>(null);
  const [envs, setEnvs] = useState<PyEnvEntry[]>([]);
  const [funcs, setFuncs] = useState<PyFuncEntry[]>([]);
  const [runs, setRuns] = useState<PyFuncRunEntry[]>([]);
  const [dagList, setDagList] = useState<DAGEntry[]>([]);
  const [dagRuns, setDagRuns] = useState<Record<number, DAGRunEntry[]>>({});
  const [expandedDagId, setExpandedDagId] = useState<number | null>(null);
  const [loadingDagRuns, setLoadingDagRuns] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const [selectedFuncIds, setSelectedFuncIds] = useState<Set<number>>(new Set());
  const [deletingFuncs, setDeletingFuncs] = useState(false);

  // ── Live metrics + refresh control ──────────────────────────────
  const [refreshSec, setRefreshSec] = useState(1);
  const [samples, setSamples] = useState<MetricSample[]>([]);
  const [live, setLive] = useState(false);
  const [cancellingRuns, setCancellingRuns] = useState<Set<number>>(new Set());
  // Last raw snapshot, kept in a ref so we can derive network throughput
  // (cumulative bytes → per-interval rate) without re-subscribing.
  const lastNetRef = useRef<{ sent: number; recv: number; t: number } | null>(null);
  // Lazily-loaded, per-env library listings (the backend TTL-caches the
  // underlying ``pip list``, so we only fetch on expand). Keyed by env
  // name — the int64 id can't survive JSON.parse in JS losslessly.
  const [expandedEnvName, setExpandedEnvName] = useState<string | null>(null);
  const [envPackages, setEnvPackages] = useState<Record<string, PyEnvPackages>>({});
  const [loadingEnvPkgs, setLoadingEnvPkgs] = useState(false);

  // env-var editing: draft {key,val} per env name
  const [evDraft, setEvDraft] = useState<Record<string, { k: string; v: string }>>({});

  const applyEnvVars = useCallback((name: string, env_vars: Record<string, string>) => {
    setEnvs((prev) => prev.map((e) => (e.name === name ? { ...e, env_vars } : e)));
  }, []);

  const addEnvVar = useCallback(async (name: string) => {
    const d = evDraft[name];
    if (!d || !d.k.trim()) return;
    try {
      const res = await setEnvVars(name, { [d.k.trim()]: d.v });
      applyEnvVars(name, res.env_vars);
      setEvDraft((prev) => ({ ...prev, [name]: { k: "", v: "" } }));
    } catch { /* surfaced by status elsewhere */ }
  }, [evDraft, applyEnvVars]);

  const removeEnvVar = useCallback(async (name: string, key: string) => {
    try {
      const res = await deleteEnvVar(name, key);
      applyEnvVars(name, res.env_vars);
    } catch { /* ignore */ }
  }, [applyEnvVars]);

  const toggleEnv = useCallback(async (envName: string) => {
    if (expandedEnvName === envName) {
      setExpandedEnvName(null);
      return;
    }
    setExpandedEnvName(envName);
    if (!envPackages[envName]) {
      setLoadingEnvPkgs(true);
      try {
        const pkgs = await getEnvPackages(envName);
        setEnvPackages((prev) => ({ ...prev, [envName]: pkgs }));
      } catch {
        /* leave unset; UI shows nothing extra */
      } finally {
        setLoadingEnvPkgs(false);
      }
    }
  }, [expandedEnvName, envPackages]);

  const fetchAll = useCallback(async () => {
    try {
      const [backendRes, cardRes, envsRes, funcsRes, runsRes, dagsRes] = await Promise.allSettled([
        getBackend(),
        getNodeCard(),
        getEnvs(),
        getFuncs(),
        getRuns(),
        getDags(),
      ]);
      if (backendRes.status === "fulfilled") setBackend(backendRes.value.backend);
      if (cardRes.status === "fulfilled") setCard(cardRes.value);
      if (envsRes.status === "fulfilled") setEnvs(envsRes.value.envs);
      if (funcsRes.status === "fulfilled") setFuncs(funcsRes.value.funcs);
      if (runsRes.status === "fulfilled") setRuns(runsRes.value.runs);
      if (dagsRes.status === "fulfilled") setDagList(dagsRes.value.dags);

      // Check if at least one call succeeded
      const anySuccess = [backendRes, cardRes].some((r) => r.status === "fulfilled");
      if (!anySuccess) setError(true);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  // Live backend stream at the chosen cadence: updates the resource bars,
  // appends to the rolling timeseries, and derives network throughput.
  // Re-subscribes whenever the refresh rate changes; paused (0) tears down.
  useEffect(() => {
    if (refreshSec <= 0) {
      setLive(false);
      lastNetRef.current = null;
      return;
    }
    const es = createBackendStream(refreshSec);
    es.onmessage = (ev) => {
      let snap: NodeBackend;
      try {
        snap = JSON.parse(ev.data) as NodeBackend;
      } catch {
        return;
      }
      setLive(true);
      setBackend(snap);

      const mem = snap.memory_total_mb ? (snap.memory_used_mb / snap.memory_total_mb) * 100 : 0;
      const disk = snap.disk_total_mb ? (snap.disk_used_mb / snap.disk_total_mb) * 100 : 0;
      const now = Date.now();
      const prev = lastNetRef.current;
      let sentRate = 0;
      let recvRate = 0;
      if (prev) {
        const dt = Math.max(0.001, (now - prev.t) / 1000);
        sentRate = Math.max(0, (snap.network.bytes_sent - prev.sent) / 1048576 / dt);
        recvRate = Math.max(0, (snap.network.bytes_recv - prev.recv) / 1048576 / dt);
      }
      lastNetRef.current = { sent: snap.network.bytes_sent, recv: snap.network.bytes_recv, t: now };

      setSamples((prevS) => {
        const next = [...prevS, {
          cpu: snap.cpu_percent,
          mem,
          disk,
          sentRate,
          recvRate,
          active: snap.active_runs,
        }];
        return next.length > MAX_SAMPLES ? next.slice(next.length - MAX_SAMPLES) : next;
      });
    };
    es.onerror = () => setLive(false);
    return () => es.close();
  }, [refreshSec]);

  // Poll the run list at the chosen cadence so the task manager stays live.
  useEffect(() => {
    if (refreshSec <= 0) return;
    const id = setInterval(async () => {
      try {
        const res = await getRuns();
        setRuns(res.runs);
      } catch { /* transient — keep last good list */ }
    }, Math.max(1000, refreshSec * 1000));
    return () => clearInterval(id);
  }, [refreshSec]);

  const handleCancelRun = useCallback(async (runId: number) => {
    setCancellingRuns((prev) => new Set(prev).add(runId));
    try {
      const res = await cancelRun(runId);
      setRuns((prev) => prev.map((r) => (r.id === runId ? res.run : r)));
    } catch { /* surfaced on next poll */ }
    finally {
      setCancellingRuns((prev) => {
        const next = new Set(prev);
        next.delete(runId);
        return next;
      });
    }
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Loading node details...</p>
        </div>
      </div>
    );
  }

  if (error || (!backend && !card)) {
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
          <Link
            href="/nodes"
            className="inline-block text-xs text-frost hover:text-frost-dim transition-colors"
          >
            Back to Nodes
          </Link>
        </div>
      </div>
    );
  }

  const memPercent = backend
    ? (backend.memory_used_mb / backend.memory_total_mb) * 100
    : 0;
  const diskPercent = backend
    ? (backend.disk_used_mb / backend.disk_total_mb) * 100
    : 0;

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-screen animate-in">
      {/* Back link + header */}
      <div className="flex items-center gap-4">
        <Link
          href="/nodes"
          className="flex items-center gap-1.5 text-xs text-frost/70 hover:text-frost transition-colors"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Nodes
        </Link>
        <div className="flex items-center gap-3">
          <span className="w-2.5 h-2.5 rounded-full status-online" />
          <h1 className="text-xl font-bold font-mono text-foreground">{nodeId}</h1>
        </div>

        {/* Live indicator + refresh-rate selector */}
        <div className="ml-auto flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-[11px] font-mono text-muted">
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: live && refreshSec > 0 ? "var(--emerald)" : "var(--muted)",
                boxShadow: live && refreshSec > 0 ? "0 0 6px var(--emerald)" : "none",
                animation: live && refreshSec > 0 ? "pulse-frost 1.5s ease-in-out infinite" : "none",
              }}
            />
            {refreshSec > 0 ? (live ? "live" : "connecting…") : "paused"}
          </span>
          <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
            {REFRESH_OPTIONS.map((opt) => {
              const active = refreshSec === opt.sec;
              return (
                <button
                  key={opt.label}
                  onClick={() => setRefreshSec(opt.sec)}
                  className={`px-2 py-1 rounded-md text-[10px] font-mono font-semibold uppercase tracking-wider transition-colors ${
                    active ? "bg-frost/15 text-frost" : "text-muted hover:text-foreground-dim"
                  }`}
                >
                  {opt.label}
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Network info row */}
      {card && (
        <div className="glass-card p-5">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted mb-3">Network Info</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 text-sm">
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Role</span>
              <p className="text-sm font-mono mt-0.5 capitalize text-frost">{card.role}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Version</span>
              <p className="text-sm font-mono mt-0.5">v{card.version}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Uptime</span>
              <p className="text-sm font-mono mt-0.5">{formatUptime(card.uptime_seconds)}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">URL</span>
              <p className="text-sm font-mono mt-0.5 text-foreground-dim truncate">{card.url}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Platform</span>
              <p className="text-sm font-mono mt-0.5">{card.platform}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Python</span>
              <p className="text-sm font-mono mt-0.5">{card.python_version}</p>
            </div>
          </div>
          {card.peers.length > 0 && (
            <div className="mt-4 pt-3 border-t border-white/[0.04]">
              <span className="text-[10px] text-muted uppercase tracking-wider">Connected Peers</span>
              <div className="flex flex-wrap gap-2 mt-2">
                {card.peers.map((p) => (
                  <span key={p} className="text-[10px] font-mono px-2 py-1 rounded bg-white/[0.03] border border-white/[0.06] text-foreground-dim">
                    {p}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Resource bars */}
      {backend && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="glass-card p-5 space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted">CPU</h3>
            <ResourceBar
              label="Utilization"
              value={backend.cpu_percent}
              color="var(--frost)"
              detail={`${backend.cpu_count} cores`}
            />
          </div>
          <div className="glass-card p-5 space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted">Memory</h3>
            <ResourceBar
              label="Used"
              value={memPercent}
              color="var(--emerald)"
              detail={`${formatBytes(backend.memory_used_mb)} / ${formatBytes(backend.memory_total_mb)}`}
            />
          </div>
          <div className="glass-card p-5 space-y-4">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-muted">Disk</h3>
            <ResourceBar
              label="Used"
              value={diskPercent}
              color="var(--amber)"
              detail={`${formatBytes(backend.disk_used_mb)} / ${formatBytes(backend.disk_total_mb)}`}
            />
          </div>
        </div>
      )}

      {/* Live timeseries metrics */}
      {backend && (
        <div className="glass-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              Live Metrics
            </h2>
            <span className="text-[10px] text-muted font-mono">
              {samples.length > 0 ? `${samples.length} samples · ${refreshSec > 0 ? `${refreshSec}s` : "paused"}` : "waiting for stream…"}
            </span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <Sparkline
                data={samples.map((s) => s.cpu)}
                color="var(--frost)"
                max={100}
                label="CPU"
                value={`${backend.cpu_percent.toFixed(0)}%`}
                height={56}
              />
            </div>
            <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <Sparkline
                data={samples.map((s) => s.mem)}
                color="var(--emerald)"
                max={100}
                label="Memory"
                value={`${memPercent.toFixed(0)}%`}
                height={56}
              />
            </div>
            <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <Sparkline
                data={samples.map((s) => s.sentRate + s.recvRate)}
                color="var(--amber)"
                label="Net I/O"
                value={samples.length > 0 ? formatRate(samples[samples.length - 1].sentRate + samples[samples.length - 1].recvRate) : "--"}
                height={56}
              />
            </div>
            <div className="p-3 rounded-lg bg-white/[0.02] border border-white/[0.04]">
              <Sparkline
                data={samples.map((s) => s.active)}
                color="var(--frost)"
                label="Active Runs"
                value={`${backend.active_runs}`}
                height={56}
              />
            </div>
          </div>
        </div>
      )}

      {/* GPUs */}
      {backend && backend.gpus.length > 0 && (
        <div className="glass-card p-5 space-y-4">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted">GPUs</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {backend.gpus.map((gpu) => (
              <div key={gpu.index} className="p-4 rounded-lg bg-white/[0.02] border border-white/[0.04] space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-medium text-foreground">{gpu.name}</span>
                  <span className="text-[10px] text-muted">GPU {gpu.index}</span>
                </div>
                <ResourceBar
                  label="Utilization"
                  value={gpu.utilization_percent}
                  color="var(--frost)"
                />
                <ResourceBar
                  label="VRAM"
                  value={(gpu.memory_used_mb / gpu.memory_total_mb) * 100}
                  color="var(--emerald)"
                  detail={`${formatBytes(gpu.memory_used_mb)} / ${formatBytes(gpu.memory_total_mb)}`}
                />
                {gpu.power_limit_w > 0 && (
                  <ResourceBar
                    label="Power"
                    value={(gpu.power_draw_w / gpu.power_limit_w) * 100}
                    color="var(--amber)"
                    detail={`${gpu.power_draw_w.toFixed(0)}W / ${gpu.power_limit_w.toFixed(0)}W`}
                  />
                )}
                <div className="flex items-center gap-4 text-[11px] text-muted">
                  <span className="flex items-center gap-1.5">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z" />
                    </svg>
                    {gpu.temperature_c}C
                  </span>
                  {gpu.power_draw_w > 0 && (
                    <span className="flex items-center gap-1.5">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                      </svg>
                      {gpu.power_draw_w.toFixed(0)}W
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Functions and Environments */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Functions */}
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
            </svg>
            Functions
            {selectedFuncIds.size > 0 && (
              <button
                onClick={async () => {
                  if (deletingFuncs) return;
                  setDeletingFuncs(true);
                  try {
                    const ids = Array.from(selectedFuncIds);
                    await bulkDeleteFuncs(ids);
                    setSelectedFuncIds(new Set());
                    // Refresh the function list
                    const res = await getFuncs();
                    setFuncs(res.funcs);
                  } catch {
                    // Bulk delete failed silently
                  } finally {
                    setDeletingFuncs(false);
                  }
                }}
                disabled={deletingFuncs}
                className="
                  px-2.5 py-1 rounded text-[10px] font-semibold uppercase tracking-wider
                  bg-rose/10 text-rose border border-rose/20
                  hover:bg-rose/20 hover:border-rose/40
                  disabled:opacity-30 disabled:cursor-not-allowed
                  transition-all duration-150
                "
              >
                {deletingFuncs ? "Deleting..." : `Delete ${selectedFuncIds.size}`}
              </button>
            )}
            <span className="ml-auto text-foreground-dim font-mono">{funcs.length}</span>
          </h2>
          {funcs.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No functions registered</p>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {funcs.map((f) => {
                const checked = selectedFuncIds.has(f.id);
                return (
                  <div
                    key={f.id}
                    className={`
                      px-3 py-2.5 rounded-lg border transition-colors
                      ${checked
                        ? "bg-rose/5 border-rose/20"
                        : "bg-white/[0.02] border-white/[0.04]"}
                    `}
                  >
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => {
                          setSelectedFuncIds((prev) => {
                            const next = new Set(prev);
                            if (next.has(f.id)) next.delete(f.id);
                            else next.add(f.id);
                            return next;
                          });
                        }}
                        className="
                          shrink-0 w-3.5 h-3.5 rounded
                          appearance-none bg-white/[0.04] border border-white/[0.12]
                          checked:bg-rose checked:border-rose
                          cursor-pointer transition-colors
                          relative
                        "
                        style={{
                          backgroundImage: checked
                            ? "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23050510' stroke-width='3'%3E%3Cpolyline points='20 6 9 17 4 12'/%3E%3C/svg%3E\")"
                            : undefined,
                          backgroundSize: "75%",
                          backgroundPosition: "center",
                          backgroundRepeat: "no-repeat",
                        }}
                      />
                      <span className="text-xs font-mono font-medium text-foreground flex-1 truncate">{f.name}</span>
                      <span className="text-[10px] text-muted font-mono shrink-0">{f.run_count} runs</span>
                    </div>
                    {f.description && (
                      <p className="text-[11px] text-muted mt-0.5 pl-5">{f.description}</p>
                    )}
                    {f.dependencies.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5 pl-5">
                        {f.dependencies.map((dep) => (
                          <span key={dep} className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-frost/5 text-frost/60">
                            {dep}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Environments */}
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
            Environments
            <span className="ml-auto text-foreground-dim font-mono">{envs.length}</span>
          </h2>
          {envs.length === 0 ? (
            <p className="text-xs text-muted/60 italic py-4">No environments created</p>
          ) : (
            <div className="space-y-2 max-h-72 overflow-y-auto">
              {envs.map((env) => {
                const expanded = expandedEnvName === env.name;
                const pkgs = envPackages[env.name];
                return (
                <div
                  key={env.id}
                  className="px-3 py-2.5 rounded-lg bg-white/[0.02] border border-white/[0.04]"
                >
                  <button
                    type="button"
                    onClick={() => toggleEnv(env.name)}
                    className="w-full text-left"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono font-medium text-foreground flex items-center gap-1.5">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" className="text-muted transition-transform" style={{ transform: expanded ? "rotate(90deg)" : "none" }}>
                          <polyline points="9 18 15 12 9 6" />
                        </svg>
                        {env.name}
                      </span>
                      <span
                        className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                        style={{
                          background: env.status === "ready" ? "rgba(52,211,153,0.1)" : env.status === "error" ? "rgba(244,63,94,0.1)" : "rgba(251,191,36,0.1)",
                          color: env.status === "ready" ? "var(--emerald)" : env.status === "error" ? "var(--rose)" : "var(--amber)",
                        }}
                      >
                        {env.status}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-1 text-[10px] text-muted font-mono">
                      <span>Python {pkgs?.python_version ?? env.python_version}</span>
                      <span>·</span>
                      <span>{pkgs ? `${pkgs.package_count} libraries` : `${env.dependencies.length} deps`}</span>
                    </div>
                  </button>
                  {expanded && (
                    <div className="mt-2 pt-2 border-t border-white/[0.06]">
                      {loadingEnvPkgs && !pkgs ? (
                        <p className="text-[10px] text-muted/60 italic py-1">Loading libraries…</p>
                      ) : pkgs && pkgs.error ? (
                        <p className="text-[10px] text-[var(--rose)] font-mono py-1">{pkgs.error}</p>
                      ) : pkgs && pkgs.packages.length > 0 ? (
                        <div className="space-y-0.5 max-h-44 overflow-y-auto">
                          {pkgs.packages.map((p) => (
                            <div key={p.name} className="flex items-center justify-between text-[10px] font-mono">
                              <span className="text-foreground-dim">{p.name}</span>
                              <span className="text-muted">{p.version}</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-[10px] text-muted/60 italic py-1">No libraries installed</p>
                      )}

                      {/* Environment variables */}
                      <div className="mt-3 pt-2 border-t border-white/[0.06]">
                        <p className="text-[9px] font-bold uppercase tracking-widest text-muted mb-1.5">Env vars</p>
                        {Object.keys(env.env_vars ?? {}).length === 0 ? (
                          <p className="text-[10px] text-muted/60 italic">None set</p>
                        ) : (
                          <div className="space-y-0.5 mb-1.5">
                            {Object.entries(env.env_vars).map(([k, v]) => (
                              <div key={k} className="flex items-center justify-between text-[10px] font-mono group">
                                <span className="text-foreground-dim">{k}</span>
                                <span className="flex items-center gap-2">
                                  <span className="text-muted truncate max-w-[120px]" title={v}>{v}</span>
                                  <button className="text-[var(--rose)] opacity-60 hover:opacity-100" onClick={() => removeEnvVar(env.name, k)} title="remove">✕</button>
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                        <div className="flex items-center gap-1">
                          <input
                            className="flex-1 min-w-0 bg-white/[0.03] border border-white/[0.06] rounded px-1.5 py-1 text-[10px] font-mono"
                            placeholder="KEY"
                            value={evDraft[env.name]?.k ?? ""}
                            onChange={(e) => setEvDraft((p) => ({ ...p, [env.name]: { k: e.target.value, v: p[env.name]?.v ?? "" } }))}
                          />
                          <input
                            className="flex-1 min-w-0 bg-white/[0.03] border border-white/[0.06] rounded px-1.5 py-1 text-[10px] font-mono"
                            placeholder="value"
                            value={evDraft[env.name]?.v ?? ""}
                            onChange={(e) => setEvDraft((p) => ({ ...p, [env.name]: { k: p[env.name]?.k ?? "", v: e.target.value } }))}
                          />
                          <button className="text-[10px] px-2 py-1 rounded bg-white/[0.05] border border-white/[0.08] hover:bg-white/[0.1]" onClick={() => addEnvVar(env.name)}>add</button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Task Manager — live run table with cancel controls */}
      {runs.length > 0 && (
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Task Manager
            {runs.some((r) => r.status === "running" || r.status === "pending") && (
              <span
                className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded inline-flex items-center gap-1"
                style={{ background: "rgba(251,191,36,0.1)", color: "var(--amber)" }}
              >
                {runs.filter((r) => r.status === "running" || r.status === "pending").length} active
              </span>
            )}
            <span className="ml-auto text-foreground-dim font-mono">{runs.length}</span>
          </h2>
          {/* Table header */}
          <div className="grid grid-cols-[90px_1fr_90px_100px_64px] gap-3 px-3 py-2 text-[10px] text-muted uppercase tracking-widest font-medium border-b border-white/[0.06]">
            <span>Status</span>
            <span>Function</span>
            <span className="text-right">Duration</span>
            <span className="text-right">Started</span>
            <span className="text-right">Action</span>
          </div>
          <div className="space-y-0.5 max-h-72 overflow-y-auto">
            {runs.slice(0, 20).map((run) => {
              const funcName = funcs.find((f) => f.id === run.func_id)?.name ?? `func#${run.func_id}`;
              const isRunning = run.status === "running";
              const cancellable = run.status === "running" || run.status === "pending";
              const cancelling = cancellingRuns.has(run.id);
              const progress = run.progress != null ? run.progress : null;
              return (
                <div key={run.id} className="space-y-0">
                  <div className="grid grid-cols-[90px_1fr_90px_100px_64px] gap-3 items-center px-3 py-2 rounded-lg bg-white/[0.02] hover:bg-white/[0.04] transition-colors">
                    <RunStatusBadge status={run.status} />
                    <span className="text-xs font-mono text-foreground truncate">{funcName}</span>
                    <span className="text-xs font-mono text-muted text-right">{formatDuration(run.duration)}</span>
                    <span className="text-[11px] font-mono text-muted text-right">{timeAgo(run.started_at)}</span>
                    <div className="flex justify-end">
                      {cancellable ? (
                        <button
                          onClick={() => handleCancelRun(run.id)}
                          disabled={cancelling}
                          className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-rose/10 text-rose border border-rose/20 hover:bg-rose/20 hover:border-rose/40 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
                          title="Cancel this run"
                        >
                          {cancelling ? "…" : "Stop"}
                        </button>
                      ) : (
                        <span className="text-[10px] text-muted/40 font-mono">--</span>
                      )}
                    </div>
                  </div>
                  {isRunning && (
                    <div className="mx-3 mb-1">
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
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* DAGs */}
      {dagList.length > 0 && (
        <div className="glass-card p-5 space-y-3">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="5" cy="6" r="3" />
              <circle cx="19" cy="6" r="3" />
              <circle cx="12" cy="18" r="3" />
              <path d="M7.5 8l3 7M16.5 8l-3 7" />
            </svg>
            DAGs
            <span className="ml-auto text-foreground-dim font-mono">{dagList.length}</span>
          </h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {dagList.map((dag) => (
              <div key={dag.id} className="rounded-lg bg-white/[0.02] border border-white/[0.04] overflow-hidden">
                <button
                  onClick={async () => {
                    if (expandedDagId === dag.id) {
                      setExpandedDagId(null);
                      return;
                    }
                    setExpandedDagId(dag.id);
                    if (!dagRuns[dag.id]) {
                      setLoadingDagRuns(true);
                      try {
                        const res = await getDagRuns(dag.id);
                        setDagRuns((prev) => ({ ...prev, [dag.id]: res.runs }));
                      } catch {
                        setDagRuns((prev) => ({ ...prev, [dag.id]: [] }));
                      } finally {
                        setLoadingDagRuns(false);
                      }
                    }
                  }}
                  className="w-full text-left px-3 py-2.5 hover:bg-white/[0.02] transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-medium text-foreground">{dag.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-muted font-mono">{dag.steps.length} steps</span>
                      <span className="text-[10px] text-muted font-mono">{dag.run_count} runs</span>
                      {dag.schedule_active && (
                        <span
                          className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded"
                          style={{ background: "rgba(103,232,249,0.1)", color: "var(--frost)" }}
                        >
                          Scheduled
                        </span>
                      )}
                      <svg
                        width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2"
                        className={`transition-transform ${expandedDagId === dag.id ? "rotate-90" : ""}`}
                      >
                        <polyline points="9 18 15 12 9 6" />
                      </svg>
                    </div>
                  </div>
                  {dag.description && (
                    <p className="text-[11px] text-muted mt-0.5">{dag.description}</p>
                  )}
                </button>
                {expandedDagId === dag.id && (
                  <div className="border-t border-white/[0.04] px-3 py-2">
                    {loadingDagRuns ? (
                      <div className="flex items-center gap-2 py-2">
                        <div className="w-3 h-3 border-2 border-frost/30 border-t-frost rounded-full spin-slow" />
                        <span className="text-[10px] text-muted font-mono">Loading runs...</span>
                      </div>
                    ) : (dagRuns[dag.id] || []).length === 0 ? (
                      <p className="text-[10px] text-muted/60 italic py-2">No runs yet</p>
                    ) : (
                      <div className="space-y-1">
                        {(dagRuns[dag.id] || []).slice(0, 5).map((run) => (
                          <div key={run.id} className="flex items-center gap-3 py-1.5">
                            <RunStatusBadge status={run.status} />
                            <span className="text-[10px] font-mono text-foreground-dim">#{run.id}</span>
                            <span className="text-[10px] font-mono text-muted ml-auto">{formatDuration(run.duration)}</span>
                            <span className="text-[10px] font-mono text-muted">{timeAgo(run.started_at)}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Run stats */}
      {backend && (
        <div className="glass-card p-5">
          <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted mb-3">Run Statistics</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Active</span>
              <p className="text-lg font-mono font-bold mt-0.5 text-frost">{backend.active_runs}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Total</span>
              <p className="text-lg font-mono font-bold mt-0.5">{backend.total_runs}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Network Sent</span>
              <p className="text-lg font-mono font-bold mt-0.5 text-foreground-dim">{formatBytes(backend.network.bytes_sent / 1048576)}</p>
            </div>
            <div>
              <span className="text-[10px] text-muted uppercase tracking-wider">Network Recv</span>
              <p className="text-lg font-mono font-bold mt-0.5 text-foreground-dim">{formatBytes(backend.network.bytes_recv / 1048576)}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
