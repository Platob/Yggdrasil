"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { getTopology } from "@/lib/api";
import type { TopologyResponse, TopologyNode } from "@/lib/types";
import { BrainMesh } from "@/components/BrainMesh";

// ── Load-based color: frost (idle) -> amber (loaded) -> rose (overloaded)
function loadColor(cpu: number): string {
  if (cpu >= 80) return "var(--rose)";
  if (cpu >= 50) return "var(--amber)";
  return "var(--frost)";
}

function loadGlow(cpu: number): string {
  if (cpu >= 80) return "var(--rose-glow)";
  if (cpu >= 50) return "var(--amber-glow)";
  return "var(--frost-glow)";
}

// ── Big stat card with count-up flash on change ───────────────
function BigStatCard({
  label,
  value,
  color,
  suffix,
}: {
  label: string;
  value: number;
  color?: string;
  suffix?: string;
}) {
  return (
    <div className="runic-card p-5 flex-1 min-w-[180px]">
      <p className="text-[10px] text-muted uppercase tracking-widest font-medium">
        {label}
      </p>
      <p
        className="text-4xl font-bold font-mono mt-2 gradient-frost"
        style={{ color }}
      >
        {value.toFixed(value % 1 === 0 ? 0 : 1)}
        {suffix && <span className="text-xl ml-1 text-foreground-dim">{suffix}</span>}
      </p>
    </div>
  );
}


export default function TopologyPage() {
  const router = useRouter();
  const [topology, setTopology] = useState<TopologyResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchTopology = useCallback(async () => {
    try {
      const t = await getTopology();
      setTopology(t);
      setError(false);
    } catch {
      setError(true);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchTopology();
    const id = setInterval(fetchTopology, 5000);
    return () => clearInterval(id);
  }, [fetchTopology]);

  const handleNodeClick = useCallback(
    (node: TopologyNode) => {
      router.push(`/nodes/${node.node_id}`);
    },
    [router],
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center animate-in">
          <div className="w-8 h-8 border-2 border-frost/30 border-t-frost rounded-full spin-slow mx-auto mb-4" />
          <p className="text-sm text-muted font-mono">Mapping topology...</p>
        </div>
      </div>
    );
  }

  if (error || !topology) {
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

  return (
    <div className="relative p-6 space-y-6 overflow-y-auto h-screen animate-in">
      {/* Aurora ambient layer */}
      <div className="aurora-bg" />

      {/* Header */}
      <div className="relative flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-foreground glow-frost">Topology</h1>
          <p className="text-sm text-muted mt-1">
            {topology.nodes.length} node{topology.nodes.length !== 1 ? "s" : ""} in the mesh
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/[0.03] border border-white/[0.06]">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          <span className="text-[11px] font-mono text-muted">Live - 5s</span>
        </div>
      </div>

      {/* Top stat row */}
      <div className="relative flex flex-wrap gap-3">
        <BigStatCard
          label="Cluster CPU Avg"
          value={topology.total_cpu_percent}
          suffix="%"
          color={
            topology.total_cpu_percent > 80
              ? "var(--rose)"
              : topology.total_cpu_percent > 50
              ? "var(--amber)"
              : undefined
          }
        />
        <BigStatCard
          label="Total Active Runs"
          value={topology.total_active_runs}
          color={topology.total_active_runs > 0 ? "var(--amber)" : undefined}
        />
        <BigStatCard
          label="Total GPUs"
          value={topology.total_gpus}
          color={topology.total_gpus > 0 ? "var(--emerald)" : undefined}
        />
      </div>

      {/* Neural Brain Graph */}
      <div className="relative runic-card p-5">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2 mb-3">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="4" cy="6" r="1.5" />
            <circle cx="4" cy="12" r="1.5" />
            <circle cx="4" cy="18" r="1.5" />
            <circle cx="12" cy="9" r="1.5" />
            <circle cx="12" cy="15" r="1.5" />
            <circle cx="20" cy="12" r="1.5" />
            <path d="M5.5 6L10.5 9M5.5 12L10.5 9M5.5 12L10.5 15M5.5 18L10.5 15M13.5 9L18.5 12M13.5 15L18.5 12" />
          </svg>
          Brain Cell Mesh
          <span className="ml-auto text-foreground-dim font-mono">
            3D synaptic network — drag to rotate · scroll to zoom · click a cell to inspect
          </span>
        </h2>
        <BrainMesh nodes={topology.nodes} onNodeClick={handleNodeClick} />
      </div>

      {/* Nodes table */}
      <div className="relative runic-card p-5 space-y-3">
        <h2 className="text-[10px] font-bold uppercase tracking-widest text-muted flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <line x1="3" y1="9" x2="21" y2="9" />
            <line x1="9" y1="21" x2="9" y2="9" />
          </svg>
          Nodes
          <span className="ml-auto text-foreground-dim font-mono">{topology.nodes.length}</span>
        </h2>

        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-[10px] uppercase tracking-widest text-muted">
                <th className="text-left font-medium pb-2 px-2">Node ID</th>
                <th className="text-left font-medium pb-2 px-2">Role</th>
                <th className="text-left font-medium pb-2 px-2">Host</th>
                <th className="text-right font-medium pb-2 px-2">CPU%</th>
                <th className="text-right font-medium pb-2 px-2">Mem%</th>
                <th className="text-right font-medium pb-2 px-2">Active</th>
                <th className="text-right font-medium pb-2 px-2">GPUs</th>
              </tr>
            </thead>
            <tbody>
              {topology.nodes.map((node) => {
                const cpuC = loadColor(node.cpu_percent);
                const memC = loadColor(node.memory_percent);
                return (
                  <tr
                    key={node.node_id}
                    onClick={() => handleNodeClick(node)}
                    className="cursor-pointer transition-colors hover:bg-white/[0.04] border-t border-white/[0.04]"
                  >
                    <td className="px-2 py-2 font-mono text-foreground">
                      <div className="flex items-center gap-2">
                        {node.self && (
                          <span className="text-[9px] font-bold uppercase tracking-wider text-frost bg-frost/10 px-1.5 py-0.5 rounded">
                            Self
                          </span>
                        )}
                        <span className="truncate max-w-[180px]">{node.node_id}</span>
                      </div>
                    </td>
                    <td className="px-2 py-2 font-mono capitalize text-foreground-dim">
                      {node.role}
                    </td>
                    <td className="px-2 py-2 font-mono text-muted">
                      {node.host}:{node.port}
                    </td>
                    <td className="px-2 py-2 text-right font-mono" style={{ color: cpuC }}>
                      {node.cpu_percent.toFixed(1)}
                    </td>
                    <td className="px-2 py-2 text-right font-mono" style={{ color: memC }}>
                      {node.memory_percent.toFixed(1)}
                    </td>
                    <td
                      className="px-2 py-2 text-right font-mono"
                      style={{ color: node.active_runs > 0 ? "var(--amber)" : "var(--muted)" }}
                    >
                      {node.active_runs}
                    </td>
                    <td
                      className="px-2 py-2 text-right font-mono"
                      style={{ color: node.gpu_count > 0 ? "var(--emerald)" : "var(--muted)" }}
                    >
                      {node.gpu_count}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
