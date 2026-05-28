"use client";

import { useEffect, useState, useRef } from "react";
import {
  bot,
  type NodeInfo,
  type NodeBackend,
  type MetricsRecentRun,
} from "@/lib/api";
import { YggdrasilLogo } from "@/components/logo";

// ── Types ────────────────────────────────────────────────────
interface SystemMetrics {
  cpu: number;        // 0-100
  ram: number;        // 0-100
  ramUsed: number;    // GB
  ramTotal: number;   // GB
  gpu: number;        // 0-100
  gpuMemUsed: number; // GB
  gpuMemTotal: number;// GB
  gpuName: string;
}

const HISTORY_SIZE = 60;

function snapshotToMetrics(snap: NodeBackend): SystemMetrics {
  const ram = snap.memory_total_mb > 0
    ? (snap.memory_used_mb / snap.memory_total_mb) * 100
    : 0;
  const gpu = snap.gpus[0];
  return {
    cpu: snap.cpu_percent,
    ram,
    ramUsed: snap.memory_used_mb / 1024,
    ramTotal: snap.memory_total_mb / 1024,
    gpu: gpu ? gpu.utilization_percent : 0,
    gpuMemUsed: gpu ? gpu.memory_used_mb / 1024 : 0,
    gpuMemTotal: gpu ? gpu.memory_total_mb / 1024 : 0,
    gpuName: gpu ? gpu.name : "No GPU detected",
  };
}

// ── Time Plot Component ──────────────────────────────────────

function TimePlot({
  data,
  color,
  label,
  value,
  unit,
  secondaryValue,
  secondaryLabel,
}: {
  data: number[];
  color: string;
  label: string;
  value: number;
  unit: string;
  secondaryValue?: string;
  secondaryLabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const padding = { top: 8, right: 8, bottom: 8, left: 8 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
    }

    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, color + "40");
    gradient.addColorStop(1, color + "00");

    ctx.beginPath();
    ctx.moveTo(padding.left, h - padding.bottom);
    data.forEach((val, i) => {
      const x = padding.left + (i / (HISTORY_SIZE - 1)) * plotW;
      const y = padding.top + plotH - (val / 100) * plotH;
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.lineTo(padding.left + plotW, h - padding.bottom);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    data.forEach((val, i) => {
      const x = padding.left + (i / (HISTORY_SIZE - 1)) * plotW;
      const y = padding.top + plotH - (val / 100) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();

    if (data.length > 0) {
      const lastVal = data[data.length - 1];
      const x = padding.left + plotW;
      const y = padding.top + plotH - (lastVal / 100) * plotH;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = color + "60";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [data, color]);

  return (
    <div className="nordic-card p-4 flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-medium text-muted uppercase tracking-wider">{label}</span>
        <div className="text-right">
          <span className="text-lg font-mono font-semibold" style={{ color }}>{value.toFixed(1)}{unit}</span>
          {secondaryValue && (
            <span className="text-[10px] text-muted ml-2">{secondaryLabel}: {secondaryValue}</span>
          )}
        </div>
      </div>
      <div className="flex-1 min-h-[100px]">
        <canvas ref={canvasRef} className="w-full h-full" style={{ display: "block" }} />
      </div>
    </div>
  );
}

// ── Recent Runs (replaces fake process list) ─────────────────

function statusColor(status: string): string {
  if (status === "succeeded" || status === "completed") return "var(--success)";
  if (status === "failed" || status === "error") return "var(--destructive)";
  if (status === "running" || status === "pending") return "var(--warning)";
  return "var(--muted)";
}

function RecentRunsList({ runs }: { runs: MetricsRecentRun[] }) {
  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">
        Recent Runs <span className="text-muted/60">({runs.length})</span>
      </h3>
      <div className="space-y-1">
        <div className="grid grid-cols-[80px_1fr_80px_80px] gap-2 text-[10px] text-muted uppercase tracking-wider pb-2 border-b border-border">
          <span>Run</span>
          <span>Func</span>
          <span className="text-right">Status</span>
          <span className="text-right">Duration</span>
        </div>
        <div className="max-h-[200px] overflow-y-auto space-y-0.5">
          {runs.length === 0 ? (
            <div className="py-6 text-center text-xs text-muted">No recent runs.</div>
          ) : (
            runs.map((r) => (
              <div
                key={r.id}
                className="grid grid-cols-[80px_1fr_80px_80px] gap-2 py-1.5 text-sm hover:bg-card-hover rounded transition-colors"
              >
                <span className="font-mono text-muted truncate">#{r.id}</span>
                <span className="font-mono text-foreground truncate">fn#{r.func_id}</span>
                <span
                  className="font-mono text-right text-xs"
                  style={{ color: statusColor(r.status) }}
                >
                  {r.status}
                </span>
                <span className="font-mono text-right text-muted">
                  {r.duration === null ? "-" : `${(r.duration * 1000).toFixed(0)}ms`}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Dashboard ───────────────────────────────────────────
export default function BotDashboard() {
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [snap, setSnap] = useState<NodeBackend | null>(null);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [recentRuns, setRecentRuns] = useState<MetricsRecentRun[]>([]);

  // History buffers — pushed as snapshots arrive over SSE.
  const [cpuHistory, setCpuHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));
  const [ramHistory, setRamHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));
  const [gpuHistory, setGpuHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));

  // 1) Identity card from /api/hello. Demo mode if the node isn't reachable.
  useEffect(() => {
    bot.getNodeInfo()
      .then((info) => {
        setNode(info);
        setDemoMode(false);
      })
      .catch(() => {
        setDemoMode(true);
        setNode({
          node_id: "offline",
          host: "127.0.0.1",
          port: 8100,
          version: "—",
          uptime: 0,
          channels: [],
          functions: [],
        });
      });
  }, []);

  // 2) Live backend snapshots via SSE. Auto-falls back to demo mode on error.
  useEffect(() => {
    let zeroedOut = false;
    const cleanup = bot.streamBackend(
      (s) => {
        setSnap(s);
        setDemoMode(false);
        const m = snapshotToMetrics(s);
        setMetrics(m);
        setCpuHistory((prev) => [...prev.slice(1), m.cpu]);
        setRamHistory((prev) => [...prev.slice(1), m.ram]);
        setGpuHistory((prev) => [...prev.slice(1), m.gpu]);
      },
      () => {
        // Stream dropped — flip into demo mode without injecting fake data.
        setDemoMode(true);
        if (!zeroedOut) {
          zeroedOut = true;
          setMetrics((prev) => prev ?? {
            cpu: 0, ram: 0, ramUsed: 0, ramTotal: 0,
            gpu: 0, gpuMemUsed: 0, gpuMemTotal: 0,
            gpuName: "Backend offline",
          });
        }
      },
    );
    return cleanup;
  }, []);

  // 3) Recent runs (every 5s). Not critical — fails silently into [].
  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const m = await bot.getMetrics();
        if (!cancelled) setRecentRuns(m.recent_runs);
      } catch {
        if (!cancelled) setRecentRuns([]);
      }
    }
    load();
    const interval = setInterval(load, 5000);
    return () => { cancelled = true; clearInterval(interval); };
  }, []);

  if (!node || !metrics) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <YggdrasilLogo size={48} className="mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Connecting to Yggdrasil...</p>
        </div>
      </div>
    );
  }

  const currentUser = typeof window !== "undefined" ? (process.env.USER || "user") : "user";

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Demo banner */}
      {demoMode && (
        <div
          className="px-4 py-2 rounded-lg text-xs font-medium"
          style={{
            background: "rgba(251,191,36,0.10)",
            border: "1px solid rgba(251,191,36,0.30)",
            color: "#fbbf24",
          }}
        >
          Backend offline — demo mode. Live metrics paused; no fake data is being injected.
        </div>
      )}

      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <YggdrasilLogo size={28} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Bot Control</h1>
            <div className="flex items-center gap-3 text-sm text-muted mt-0.5">
              <span className="font-mono">{node.node_id}</span>
              <span className="text-border">|</span>
              <span>{currentUser}@{node.host}</span>
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center gap-4">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${demoMode ? "bg-warning/10 border border-warning/20" : "bg-success/10 border border-success/20"}`}>
            <div className={`status-dot ${demoMode ? "pending" : "online"}`} />
            <span className={`text-xs font-medium ${demoMode ? "text-warning" : "text-success"}`}>
              {demoMode ? "Demo Mode" : "Online"}
            </span>
          </div>
        </div>
      </div>

      {/* Resource Usage Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <TimePlot
          data={cpuHistory}
          color="#f26b3a"
          label="CPU Usage"
          value={metrics.cpu}
          unit="%"
        />
        <TimePlot
          data={ramHistory}
          color="#5b9bd5"
          label="Memory"
          value={metrics.ram}
          unit="%"
          secondaryValue={`${metrics.ramUsed.toFixed(1)}/${metrics.ramTotal.toFixed(1)}GB`}
          secondaryLabel="Used"
        />
        <TimePlot
          data={gpuHistory}
          color="#4ade80"
          label="GPU"
          value={metrics.gpu}
          unit="%"
          secondaryValue={metrics.gpuName}
          secondaryLabel=""
        />
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RecentRunsList runs={recentRuns} />

        {/* System Info */}
        <div className="nordic-card p-4">
          <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">System Info</h3>
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
            <InfoRow label="Node ID" value={node.node_id} mono />
            <InfoRow label="Version" value={node.version} mono />
            <InfoRow label="Host" value={`${node.host}:${node.port}`} mono />
            <InfoRow label="Uptime" value={formatUptime(snap?.uptime_seconds ?? node.uptime)} />
            <InfoRow label="Hostname" value={snap?.hostname ?? "-"} mono />
            <InfoRow label="Platform" value={snap?.platform ?? "-"} mono />
            <InfoRow label="Python" value={snap?.python_version ?? "-"} mono />
            <InfoRow label="CPU cores" value={String(snap?.cpu_count ?? "-")} />
            <InfoRow label="RAM Total" value={`${metrics.ramTotal.toFixed(1)} GB`} />
            <InfoRow label="GPU" value={metrics.gpuName} />
            <InfoRow label="GPU Memory" value={metrics.gpuMemTotal > 0 ? `${metrics.gpuMemUsed.toFixed(1)}/${metrics.gpuMemTotal.toFixed(1)} GB` : "-"} />
            <InfoRow label="Active runs" value={String(snap?.active_runs ?? 0)} />
            <InfoRow label="Total runs" value={String(snap?.total_runs ?? 0)} />
            <InfoRow label="Functions" value={`${node.functions.length} registered`} />
          </div>
        </div>
      </div>
    </div>
  );
}

function InfoRow({ label, value, mono, primary }: { label: string; value: string; mono?: boolean; primary?: boolean }) {
  return (
    <div>
      <span className="text-muted text-xs">{label}</span>
      <p className={`truncate ${mono ? "font-mono text-xs" : ""} ${primary ? "text-primary" : "text-foreground"}`}>{value}</p>
    </div>
  );
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}
