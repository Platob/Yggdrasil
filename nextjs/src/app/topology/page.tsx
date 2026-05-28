"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import { getTopology } from "@/lib/api";
import type { TopologyResponse, TopologyNode } from "@/lib/types";

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

// ── Neural Network Brain Layout ───────────────────────────────
// Multi-layer architecture:
//   Layer 0 (input/sensors): peer nodes feeding data in
//   Layer 1 (cortex/hidden): self node + virtual processing neurons
//   Layer 2 (output/effectors): synthetic neurons fan-out
interface BrainNeuron {
  id: string;
  x: number;
  y: number;
  r: number;
  layer: 0 | 1 | 2;
  activation: number; // 0-1, drives glow intensity
  node?: TopologyNode; // null for synthetic hidden/output neurons
  label: string;
  color: string;
}

interface Synapse {
  from: string;
  to: string;
  weight: number; // 0-1, drives stroke opacity
  pulseDelay: number;
}

function buildBrain(nodes: TopologyNode[], width: number, height: number): {
  neurons: BrainNeuron[];
  synapses: Synapse[];
} {
  const peers = nodes.filter((n) => !n.self);
  const selfNode = nodes.find((n) => n.self);

  // Layer X positions
  const inputX = width * 0.12;
  const hiddenX1 = width * 0.38;
  const hiddenX2 = width * 0.62;
  const outputX = width * 0.88;

  const neurons: BrainNeuron[] = [];
  const synapses: Synapse[] = [];

  // Layer 0: peer inputs (or synthetic if no peers)
  const inputs = peers.length > 0 ? peers : [];
  const inputCount = Math.max(3, inputs.length);
  for (let i = 0; i < inputCount; i++) {
    const node = i < inputs.length ? inputs[i] : undefined;
    const y = (height / (inputCount + 1)) * (i + 1);
    const activation = node ? Math.min(1, (node.cpu_percent + node.active_runs * 20) / 100) : 0.15 + Math.random() * 0.2;
    neurons.push({
      id: node?.node_id ?? `input-${i}`,
      x: inputX,
      y,
      r: node ? 14 + Math.min(8, node.active_runs * 2) : 8,
      layer: 0,
      activation,
      node,
      label: node?.node_id ?? "",
      color: node ? loadColor(node.cpu_percent) : "var(--frost)",
    });
  }

  // Layer 1a: cortex (self + hidden neurons left)
  const hiddenLeft = 5;
  for (let i = 0; i < hiddenLeft; i++) {
    const y = (height / (hiddenLeft + 1)) * (i + 1);
    const isSelf = i === Math.floor(hiddenLeft / 2);
    const activation = isSelf && selfNode
      ? Math.min(1, (selfNode.cpu_percent + selfNode.active_runs * 25) / 100)
      : 0.3 + Math.random() * 0.4;
    neurons.push({
      id: isSelf ? selfNode?.node_id ?? "self" : `hidden-l-${i}`,
      x: hiddenX1,
      y,
      r: isSelf ? 24 : 11,
      layer: 1,
      activation,
      node: isSelf ? selfNode : undefined,
      label: isSelf ? selfNode?.node_id ?? "self" : "",
      color: isSelf ? "var(--frost)" : "var(--emerald)",
    });
  }

  // Layer 1b: hidden right
  const hiddenRight = 6;
  for (let i = 0; i < hiddenRight; i++) {
    const y = (height / (hiddenRight + 1)) * (i + 1);
    neurons.push({
      id: `hidden-r-${i}`,
      x: hiddenX2,
      y,
      r: 9,
      layer: 1,
      activation: 0.2 + Math.random() * 0.5,
      label: "",
      color: "var(--frost)",
    });
  }

  // Layer 2: output neurons (representing assets/runs)
  const outputCount = 4;
  const outputLabels = ["FUNCS", "DAGS", "ENVS", "RUNS"];
  const outputColors = ["var(--frost)", "var(--emerald)", "var(--amber)", "var(--rose)"];
  for (let i = 0; i < outputCount; i++) {
    const y = (height / (outputCount + 1)) * (i + 1);
    neurons.push({
      id: `output-${i}`,
      x: outputX,
      y,
      r: 13,
      layer: 2,
      activation: 0.4 + Math.random() * 0.4,
      label: outputLabels[i],
      color: outputColors[i],
    });
  }

  // Build synapses: input -> hidden-left -> hidden-right -> output
  // Use a sparsity factor — not every neuron connects to every other
  const inputs0 = neurons.filter((n) => n.layer === 0);
  const hiddenL = neurons.filter((n) => n.layer === 1 && n.x === hiddenX1);
  const hiddenR = neurons.filter((n) => n.layer === 1 && n.x === hiddenX2);
  const outputs = neurons.filter((n) => n.layer === 2);

  let synIdx = 0;
  for (const i of inputs0) {
    for (const h of hiddenL) {
      // Sparse connect: only 70% of edges
      if (Math.random() > 0.3) {
        synapses.push({
          from: i.id,
          to: h.id,
          weight: 0.3 + i.activation * 0.6,
          pulseDelay: (synIdx++ * 0.13) % 4,
        });
      }
    }
  }
  for (const h1 of hiddenL) {
    for (const h2 of hiddenR) {
      if (Math.random() > 0.4) {
        synapses.push({
          from: h1.id,
          to: h2.id,
          weight: 0.25 + h1.activation * 0.5,
          pulseDelay: (synIdx++ * 0.13) % 4,
        });
      }
    }
  }
  for (const h2 of hiddenR) {
    for (const o of outputs) {
      if (Math.random() > 0.3) {
        synapses.push({
          from: h2.id,
          to: o.id,
          weight: 0.3 + o.activation * 0.5,
          pulseDelay: (synIdx++ * 0.13) % 4,
        });
      }
    }
  }

  return { neurons, synapses };
}

// ── Brain Network SVG ─────────────────────────────────────────
function NetworkGraph({
  nodes,
  onNodeClick,
}: {
  nodes: TopologyNode[];
  onNodeClick: (node: TopologyNode) => void;
}) {
  const width = 900;
  const height = 520;

  const { neurons, synapses } = useMemo(
    () => buildBrain(nodes, width, height),
    // Re-shuffle hidden layers only when peer count changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [nodes.length, nodes.map((n) => n.node_id).join(",")]
  );
  const neuronById = useMemo(() => {
    const m = new Map<string, BrainNeuron>();
    for (const n of neurons) m.set(n.id, n);
    return m;
  }, [neurons]);

  if (nodes.length === 0) {
    return (
      <div className="flex items-center justify-center h-[520px] text-muted/60 text-sm italic">
        No nodes in the mesh
      </div>
    );
  }

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full h-auto"
      style={{ maxHeight: "560px" }}
    >
      <defs>
        <radialGradient id="brain-bg" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="rgba(103,232,249,0.06)" />
          <stop offset="60%" stopColor="rgba(52,211,153,0.03)" />
          <stop offset="100%" stopColor="transparent" />
        </radialGradient>
        {/* Neuron glow filter */}
        <filter id="neuron-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>
      <rect width={width} height={height} fill="url(#brain-bg)" />

      {/* Layer column labels */}
      <text x={width * 0.12} y={20} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="ui-monospace, monospace" letterSpacing="0.2em">
        INPUT · SENSORS
      </text>
      <text x={width * 0.5} y={20} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="ui-monospace, monospace" letterSpacing="0.2em">
        CORTEX · HIDDEN
      </text>
      <text x={width * 0.88} y={20} textAnchor="middle" fontSize="9" fill="var(--muted)" fontFamily="ui-monospace, monospace" letterSpacing="0.2em">
        OUTPUT · ASSETS
      </text>

      {/* Synapses (drawn first, behind neurons) */}
      {synapses.map((syn, i) => {
        const from = neuronById.get(syn.from);
        const to = neuronById.get(syn.to);
        if (!from || !to) return null;
        const baseOpacity = 0.08 + syn.weight * 0.25;
        return (
          <g key={`syn-${i}`}>
            {/* Static line */}
            <line
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="var(--frost)"
              strokeWidth={0.6 + syn.weight}
              opacity={baseOpacity}
            />
            {/* Animated pulse along the synapse */}
            <circle r={1.6} fill={to.color} opacity="0.9">
              <animateMotion
                dur={`${2.2 + (syn.weight * 1.5)}s`}
                begin={`${syn.pulseDelay}s`}
                repeatCount="indefinite"
                path={`M ${from.x},${from.y} L ${to.x},${to.y}`}
              />
              <animate
                attributeName="opacity"
                values="0;1;1;0"
                keyTimes="0;0.1;0.9;1"
                dur={`${2.2 + (syn.weight * 1.5)}s`}
                begin={`${syn.pulseDelay}s`}
                repeatCount="indefinite"
              />
            </circle>
          </g>
        );
      })}

      {/* Neurons */}
      {neurons.map((n) => {
        const isInteractive = n.node !== undefined;
        const innerOpacity = 0.15 + n.activation * 0.4;
        return (
          <g
            key={n.id}
            transform={`translate(${n.x}, ${n.y})`}
            style={{ cursor: isInteractive ? "pointer" : "default" }}
            onClick={() => n.node && onNodeClick(n.node)}
          >
            {/* Glow halo */}
            <circle
              r={n.r * 1.5}
              fill={n.color}
              opacity={n.activation * 0.15}
              filter="url(#neuron-glow)"
            />
            {/* Outer ring */}
            <circle
              r={n.r}
              fill={`rgba(255,255,255,${innerOpacity * 0.15})`}
              stroke={n.color}
              strokeWidth={isInteractive ? 2 : 1}
              opacity={0.5 + n.activation * 0.5}
            />
            {/* Core */}
            <circle
              r={Math.max(2, n.r * 0.35)}
              fill={n.color}
              opacity={0.6 + n.activation * 0.4}
            >
              {n.layer === 1 && (
                <animate
                  attributeName="r"
                  values={`${Math.max(2, n.r * 0.35)};${Math.max(3, n.r * 0.55)};${Math.max(2, n.r * 0.35)}`}
                  dur={`${1.5 + n.activation}s`}
                  repeatCount="indefinite"
                />
              )}
            </circle>
            {/* Label */}
            {n.label && (
              <text
                y={n.r + 14}
                textAnchor="middle"
                fontSize="10"
                fontFamily="ui-monospace, monospace"
                fill={n.color}
                fontWeight={isInteractive ? "600" : "400"}
                opacity={0.8}
              >
                {n.label.length > 16 ? n.label.slice(0, 14) + "…" : n.label}
              </text>
            )}
            {/* Self/peer activity badge */}
            {n.node && (
              <text
                y={n.r + 26}
                textAnchor="middle"
                fontSize="9"
                fontFamily="ui-monospace, monospace"
                fill="var(--foreground-dim)"
              >
                {n.node.cpu_percent.toFixed(0)}% · {n.node.active_runs} runs
              </text>
            )}
          </g>
        );
      })}
    </svg>
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
          Neural Mesh
          <span className="ml-auto text-foreground-dim font-mono">
            synapses pulse with cluster activity · click a neuron to inspect
          </span>
        </h2>
        <NetworkGraph nodes={topology.nodes} onNodeClick={handleNodeClick} />
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
