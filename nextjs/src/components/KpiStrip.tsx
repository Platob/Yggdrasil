"use client";

// Top-of-layout live KPI strip. Subscribes to /api/v2/backend/stream (SSE)
// for CPU/RAM/run counters and polls /api/v2/stats every 5s for asset/peer
// counts (which aren't in the SSE snapshot). Renders inline so children
// pages can scroll under it without remounting.
//
// Hidden when SSE never connects, so the legacy chromeless `/` route stays
// distraction-free until the user navigates to a real page.

import { useEffect, useState, useRef } from "react";
import { usePathname } from "next/navigation";
import { getStats } from "@/lib/api";
import type { NodeBackend, ClusterStats } from "@/lib/types";

interface Kpi {
  label: string;
  value: string;
  color: string;
  sub?: string;
}

export function KpiStrip() {
  const pathname = usePathname() || "/";
  const [snap, setSnap] = useState<NodeBackend | null>(null);
  const [stats, setStats] = useState<ClusterStats | null>(null);
  const [history, setHistory] = useState<number[]>([]);
  const lastRunsRef = useRef(0);
  const lastTsRef = useRef(Date.now());
  const [runsPerMin, setRunsPerMin] = useState(0);

  // Hide on the marketing home so the globe owns the screen.
  const hide = pathname === "/" || pathname.startsWith("/nodes/");

  useEffect(() => {
    if (hide) return;
    const es = new EventSource("/api/v2/backend/stream");
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as NodeBackend;
        setSnap(data);
        // Sliding CPU history sparkline
        setHistory((prev) => [...prev.slice(-29), data.cpu_percent]);
        // Runs/min derivative — clip to 0 to avoid jitter when counters reset
        const now = Date.now();
        const dt = (now - lastTsRef.current) / 1000;
        if (dt > 0.5) {
          const drun = Math.max(0, data.total_runs - lastRunsRef.current);
          setRunsPerMin((drun / dt) * 60);
          lastRunsRef.current = data.total_runs;
          lastTsRef.current = now;
        }
      } catch {
        /* ignore non-JSON keepalives */
      }
    };
    return () => es.close();
  }, [hide]);

  useEffect(() => {
    if (hide) return;
    let alive = true;
    const tick = async () => {
      try {
        const s = await getStats();
        if (alive) setStats(s);
      } catch {
        /* ignore */
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, [hide]);

  if (hide || !snap) return null;

  const memPct = snap.memory_total_mb > 0
    ? (snap.memory_used_mb / snap.memory_total_mb) * 100
    : 0;

  const cpuColor = snap.cpu_percent > 80 ? "var(--rose)" : snap.cpu_percent > 50 ? "var(--amber)" : "var(--frost)";
  const memColor = memPct > 80 ? "var(--rose)" : memPct > 50 ? "var(--amber)" : "var(--emerald)";

  const kpis: Kpi[] = [
    { label: "CPU", value: `${snap.cpu_percent.toFixed(1)}%`, color: cpuColor },
    { label: "RAM", value: `${memPct.toFixed(0)}%`, color: memColor, sub: `${(snap.memory_used_mb / 1024).toFixed(1)} GB` },
    { label: "Runs", value: String(snap.total_runs), color: "var(--foreground)", sub: `${runsPerMin.toFixed(1)}/min` },
    { label: "Active", value: String(snap.active_runs), color: snap.active_runs > 0 ? "var(--amber)" : "var(--muted)" },
    { label: "Funcs", value: String(stats?.func_count ?? "-"), color: "var(--frost)" },
    { label: "Peers", value: String(stats?.peer_count ?? "-"), color: "var(--emerald)" },
    { label: "GPUs", value: String(stats?.gpu_count ?? snap.gpus.length), color: snap.gpus.length > 0 ? "var(--emerald)" : "var(--muted)" },
  ];

  const sparkMax = Math.max(...history, 1);

  return (
    <div
      className="sticky top-0 z-40 backdrop-blur-md bg-black/40 border-b border-white/[0.06] px-4 py-1.5 flex items-center gap-4 text-[11px] font-mono overflow-x-auto"
      style={{ height: 32 }}
    >
      {/* CPU sparkline */}
      <div className="flex items-center gap-2 shrink-0">
        <span className="text-muted uppercase tracking-widest text-[9px]">cpu</span>
        <div className="flex items-end gap-[1px] h-4 w-20">
          {history.map((v, i) => (
            <div
              key={i}
              className="flex-1 rounded-sm"
              style={{
                height: `${Math.max(2, (v / sparkMax) * 16)}px`,
                background: cpuColor,
                opacity: 0.35 + (i / Math.max(1, history.length - 1)) * 0.65,
              }}
            />
          ))}
        </div>
      </div>

      <span className="text-muted/30">|</span>

      {kpis.map((k) => (
        <div key={k.label} className="flex items-baseline gap-1.5 shrink-0">
          <span className="text-[9px] uppercase tracking-widest text-muted">{k.label}</span>
          <span className="font-bold" style={{ color: k.color }}>{k.value}</span>
          {k.sub && <span className="text-muted/70 text-[10px]">{k.sub}</span>}
        </div>
      ))}

      <div className="ml-auto flex items-center gap-2 shrink-0">
        <span
          className="w-1.5 h-1.5 rounded-full status-online"
          title="streaming /api/v2/backend/stream"
        />
        <span className="text-muted/70 text-[10px] uppercase tracking-widest">live</span>
      </div>
    </div>
  );
}
