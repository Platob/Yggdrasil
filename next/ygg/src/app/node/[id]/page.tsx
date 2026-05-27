"use client";

import { useEffect, useState, useRef, useCallback, use } from "react";
import { node as api, type NodeInfo, type MonitorResponse, type ProcessInfo as APIProcessInfo } from "@/lib/api";
import { YggdrasilLogo } from "@/components/logo";
import Link from "next/link";

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

interface ProcessInfo {
  pid: number;
  name: string;
  cpu: number;
  ram: number;
  status: string;
}

function metricsFromMonitor(m: MonitorResponse["snapshot"]): SystemMetrics {
  return {
    cpu: m.cpu_percent,
    ram: m.memory_percent,
    ramUsed: m.memory_used_mb / 1024,
    ramTotal: m.memory_total_mb / 1024,
    gpu: 0,
    gpuMemUsed: 0,
    gpuMemTotal: 0,
    gpuName: "N/A",
  };
}

function procsFromMonitor(procs: APIProcessInfo[]): ProcessInfo[] {
  return procs.map((p) => ({
    pid: p.pid,
    name: p.name,
    cpu: p.cpu_percent,
    ram: p.memory_mb,
    status: p.status,
  }));
}

// ── Time Plot Component ──────────────────────────────────────
const HISTORY_SIZE = 60; // 60 data points

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

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
    }

    // Fill gradient
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, color + "40");
    gradient.addColorStop(1, color + "00");

    // Draw filled area
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

    // Draw line
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

    // Current value dot
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

// ── Process List Component ───────────────────────────────────
function ProcessList({ processes }: { processes: ProcessInfo[] }) {
  return (
    <div className="nordic-card p-4">
      <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">Active Processes</h3>
      <div className="space-y-1">
        <div className="grid grid-cols-[1fr_60px_80px_70px] gap-2 text-[10px] text-muted uppercase tracking-wider pb-2 border-b border-border">
          <span>Name</span>
          <span className="text-right">PID</span>
          <span className="text-right">CPU %</span>
          <span className="text-right">RAM MB</span>
        </div>
        <div className="max-h-[200px] overflow-y-auto space-y-0.5">
          {processes.map((p) => (
            <div key={p.pid} className="grid grid-cols-[1fr_60px_80px_70px] gap-2 py-1.5 text-sm hover:bg-card-hover rounded transition-colors">
              <span className="font-mono text-foreground truncate">{p.name}</span>
              <span className="font-mono text-muted text-right">{p.pid}</span>
              <span className="font-mono text-right" style={{ color: p.cpu > 50 ? "var(--destructive)" : p.cpu > 20 ? "var(--warning)" : "var(--success)" }}>
                {p.cpu.toFixed(1)}%
              </span>
              <span className="font-mono text-muted text-right">{p.ram.toFixed(0)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main Node Detail Page ────────────────────────────────────
export default function NodeDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [isLocal, setIsLocal] = useState(false);
  const [refreshRate, setRefreshRate] = useState(1000);
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);
  const [processes, setProcesses] = useState<ProcessInfo[]>([]);

  // History buffers
  const [cpuHistory, setCpuHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));
  const [ramHistory, setRamHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));
  const [gpuHistory, setGpuHistory] = useState<number[]>(Array(HISTORY_SIZE).fill(0));

  // Fetch node info — check if this is self or a peer
  useEffect(() => {
    async function loadNode() {
      try {
        const self = await api.getNodeInfo();
        if (self.node_id === id) {
          // LOCAL node — use direct API calls (no peer network overhead)
          setNode(self);
          setIsLocal(true);
        } else {
          // REMOTE node — display info from the peers list
          setIsLocal(false);
          try {
            const peersData = await api.getPeers();
            const peer = peersData.peers.find((p) => p.node_id === id);
            if (peer) {
              setNode(peer);
            } else {
              // Unknown node — show fallback
              setDemoMode(true);
              setNode({
                node_id: id,
                host: "unknown",
                port: 0,
                version: "unknown",
                uptime: 0,
                channels: [],
                functions: [],
                lat: null,
                lon: null,
              });
            }
          } catch {
            // Peers endpoint unavailable — use placeholder
            setDemoMode(true);
            setNode({
              node_id: id,
              host: "unknown",
              port: 0,
              version: "unknown",
              uptime: 0,
              channels: [],
              functions: [],
              lat: null,
              lon: null,
            });
          }
        }
      } catch {
        // Node unavailable — enter demo mode
        setDemoMode(true);
        setNode({
          node_id: id,
          host: "localhost",
          port: 8100,
          version: "0.1.0-demo",
          uptime: 3600,
          channels: ["general"],
          functions: ["echo", "ping", "execute"],
          lat: null,
          lon: null,
        });
      }
    }
    loadNode();
  }, [id]);

  // Periodic metrics fetch — only for local nodes (live resource monitoring)
  const fetchMetrics = useCallback(async () => {
    try {
      const monData = await api.getMonitor(60);
      const m = metricsFromMonitor(monData.snapshot);
      setMetrics(m);
      setCpuHistory((prev) => [...prev.slice(1), m.cpu]);
      setRamHistory((prev) => [...prev.slice(1), m.ram]);
      setGpuHistory((prev) => [...prev.slice(1), m.gpu]);
      setProcesses(procsFromMonitor(monData.snapshot.processes || []));
    } catch {
      // Demo fallback when monitor unavailable
      const m: SystemMetrics = { cpu: Math.random() * 60 + 10, ram: Math.random() * 40 + 30, ramUsed: 8 + Math.random() * 6, ramTotal: 32, gpu: 0, gpuMemUsed: 0, gpuMemTotal: 0, gpuName: "N/A" };
      setMetrics(m);
      setCpuHistory((prev) => [...prev.slice(1), m.cpu]);
      setRamHistory((prev) => [...prev.slice(1), m.ram]);
      setGpuHistory((prev) => [...prev.slice(1), 0]);
    }
  }, []);

  useEffect(() => {
    if (isLocal || demoMode) {
      // Local node: show live metrics
      fetchMetrics();
      const interval = setInterval(fetchMetrics, refreshRate);
      return () => clearInterval(interval);
    } else {
      // Remote node: show static placeholder metrics (last-known info from discovery)
      setMetrics({
        cpu: 0, ram: 0, ramUsed: 0, ramTotal: 0,
        gpu: 0, gpuMemUsed: 0, gpuMemTotal: 0, gpuName: "N/A (remote)",
      });
      setProcesses([]);
    }
  }, [refreshRate, fetchMetrics, isLocal, demoMode]);

  if (!node || !metrics) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <YggdrasilLogo size={48} className="mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Loading node data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Link
            href="/node"
            className="w-8 h-8 rounded-lg bg-card border border-border flex items-center justify-center text-muted hover:text-foreground hover:border-primary transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </Link>
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <YggdrasilLogo size={28} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Node: <span className="font-mono">{node.node_id}</span></h1>
            <div className="flex items-center gap-3 text-sm text-muted mt-0.5">
              <span className="font-mono">{node.host}:{node.port}</span>
              <span className="text-border">|</span>
              <span>v{node.version}</span>
            </div>
          </div>
        </div>

        {/* Status + Refresh Rate */}
        <div className="flex items-center gap-4">
          {/* Local/Remote badge */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-semibold uppercase tracking-wide ${
            isLocal
              ? "bg-success/10 border border-success/20 text-success"
              : "bg-orange-500/10 border border-orange-500/20 text-orange-400"
          }`}>
            <span className={`w-1.5 h-1.5 rounded-full ${isLocal ? "bg-success" : "bg-orange-400"}`} />
            {isLocal ? "Local" : "Remote"}
          </div>

          {/* Refresh rate — only for local nodes with live metrics */}
          {isLocal && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted">Refresh:</span>
              <select
                value={refreshRate}
                onChange={(e) => setRefreshRate(Number(e.target.value))}
                className="bg-card border border-border rounded px-2 py-1 text-xs font-mono text-foreground focus:outline-none focus:border-primary"
              >
                <option value={500}>500ms</option>
                <option value={1000}>1s</option>
                <option value={2000}>2s</option>
                <option value={5000}>5s</option>
              </select>
            </div>
          )}

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full ${demoMode ? "bg-warning/10 border border-warning/20" : "bg-success/10 border border-success/20"}`}>
            <div className={`status-dot ${demoMode ? "pending" : "online"}`} />
            <span className={`text-xs font-medium ${demoMode ? "text-warning" : "text-success"}`}>
              {demoMode ? "Demo Mode" : "Online"}
            </span>
          </div>
        </div>
      </div>

      {/* Resource Usage Charts — only for local nodes */}
      {isLocal || demoMode ? (
        <>
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
              secondaryValue={`${metrics.ramUsed.toFixed(1)}/${metrics.ramTotal}GB`}
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
            {/* Process List */}
            <ProcessList processes={processes} />

            {/* System Info */}
            <div className="nordic-card p-4">
              <h3 className="text-xs font-medium text-muted uppercase tracking-wider mb-3">System Info</h3>
              <div className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                <InfoRow label="Node ID" value={node.node_id} mono />
                <InfoRow label="Version" value={node.version} mono />
                <InfoRow label="Host" value={`${node.host}:${node.port}`} mono />
                <InfoRow label="Uptime" value={formatUptime(node.uptime)} />
                <InfoRow label="Latitude" value={node.lat != null ? String(node.lat) : "N/A"} mono primary />
                <InfoRow label="Longitude" value={node.lon != null ? String(node.lon) : "N/A"} mono primary />
                <InfoRow label="RAM Total" value={`${metrics.ramTotal} GB`} />
                <InfoRow label="GPU" value={metrics.gpuName} />
                <InfoRow label="GPU Memory" value={`${metrics.gpuMemUsed.toFixed(1)}/${metrics.gpuMemTotal} GB`} />
                <InfoRow label="Functions" value={`${node.functions.length} registered`} />
              </div>
            </div>
          </div>
        </>
      ) : (
        /* Remote node: show last-known discovery info */
        <div className="space-y-4">
          <div className="nordic-card p-5 border-orange-500/20">
            <div className="flex items-center gap-2 mb-3">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-orange-400">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              <span className="text-xs font-medium text-orange-400 uppercase tracking-wider">Remote Node - Last Known Info</span>
            </div>
            <p className="text-xs text-muted mb-4">
              Live resource metrics are not available for remote nodes. Showing discovery information only.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-3 text-sm">
              <InfoRow label="Node ID" value={node.node_id} mono />
              <InfoRow label="Version" value={node.version} mono />
              <InfoRow label="Host" value={`${node.host}:${node.port}`} mono />
              <InfoRow label="Uptime" value={formatUptime(node.uptime)} />
              <InfoRow label="Latitude" value={node.lat != null ? String(node.lat) : "N/A"} mono primary />
              <InfoRow label="Longitude" value={node.lon != null ? String(node.lon) : "N/A"} mono primary />
              <InfoRow label="Channels" value={`${node.channels.length}`} />
              <InfoRow label="Functions" value={`${node.functions.length} registered`} />
            </div>
          </div>
        </div>
      )}
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
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}
