"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { WorldGlobe, type BotNode, type ArcDef } from "@/components/world-globe";
import { bot, type TopologyResponse, type TopologyNode } from "@/lib/api";

// ─── Demo fallback (only used when the backend is unreachable) ───────────────
// Kept small and explicitly labelled — never blended with real data.
const DEMO_FALLBACK_NODES: BotNode[] = [
  { id: "demo-paris",  label: "Paris (demo)",  lat: 48.8566, lng: 2.3522,    status: "pending", version: "demo", uptime: 0 },
  { id: "demo-nyc",    label: "NYC (demo)",    lat: 40.7128, lng: -74.006,   status: "pending", version: "demo", uptime: 0 },
  { id: "demo-tokyo",  label: "Tokyo (demo)",  lat: 35.6762, lng: 139.6503,  status: "pending", version: "demo", uptime: 0 },
  { id: "demo-sydney", label: "Sydney (demo)", lat: -33.8688, lng: 151.2093, status: "pending", version: "demo", uptime: 0 },
];

const DEMO_FALLBACK_ARCS: ArcDef[] = [
  { fromId: "demo-paris", toId: "demo-nyc" },
  { fromId: "demo-nyc",   toId: "demo-tokyo" },
  { fromId: "demo-tokyo", toId: "demo-sydney" },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatUptime(s: number) {
  if (s === 0) return "Offline";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

// Stable hash → lat/lng on a small ring when the node has no geo yet.
// Keeps the globe usable even before IP geolocation resolves.
function hashRingCoords(nodeId: string, index: number): { lat: number; lng: number } {
  let h = 0;
  for (let i = 0; i < nodeId.length; i++) h = (h * 31 + nodeId.charCodeAt(i)) | 0;
  const angle = ((Math.abs(h) % 360) + index * 47) % 360;
  const lat = (((Math.abs(h) >> 8) % 60) - 30) * 0.5;
  const lng = angle - 180;
  return { lat, lng };
}

function topologyToBotNode(n: TopologyNode, idx: number): BotNode {
  const hasGeo = typeof n.lat === "number" && typeof n.lon === "number";
  const coords = hasGeo
    ? { lat: n.lat as number, lng: n.lon as number }
    : hashRingCoords(n.node_id, idx);
  // Topology never returns "offline" peers (the network service expires them).
  // Pending = registered but idle (no active runs); online = doing work.
  const status: BotNode["status"] = n.active_runs > 0 ? "online" : "pending";
  return {
    id: n.node_id,
    label: n.self ? `${n.node_id} (this node)` : n.node_id,
    lat: coords.lat,
    lng: coords.lng,
    status,
    version: n.role,
    uptime: Math.round(n.cpu_percent),
  };
}

const STATUS_COLOR: Record<BotNode["status"], string> = {
  online: "#4ade80",
  pending: "#fbbf24",
  offline: "#ef4444",
};

// ─── Node Detail Panel ────────────────────────────────────────────────────────

function NodePanel({
  node,
  topo,
  onClose,
}: {
  node: BotNode;
  topo: TopologyNode | null;
  onClose: () => void;
}) {
  const col = STATUS_COLOR[node.status];
  const rows: [string, string][] = topo
    ? [
        ["Node ID", topo.node_id],
        ["Role", topo.role],
        ["Host", `${topo.host}:${topo.port}`],
        ["CPU", `${topo.cpu_percent.toFixed(1)}%`],
        ["Memory", `${topo.memory_percent.toFixed(1)}%`],
        ["Active runs", String(topo.active_runs)],
        ["GPUs", String(topo.gpu_count)],
        ["Latitude", node.lat.toFixed(4)],
        ["Longitude", node.lng.toFixed(4)],
      ]
    : [
        ["Node ID", node.id],
        ["Status", node.status],
        ["Version", node.version ?? "-"],
        ["Uptime", formatUptime(node.uptime ?? 0)],
        ["Latitude", node.lat.toFixed(4)],
        ["Longitude", node.lng.toFixed(4)],
      ];

  return (
    <div
      className="pointer-events-auto w-full rounded-xl p-4 backdrop-blur-md"
      style={{ background: "rgba(8,8,14,0.92)", border: "1px solid rgba(255,255,255,0.08)" }}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
            style={{ background: col, boxShadow: `0 0 6px ${col}` }}
          />
          <span className="font-semibold text-white text-sm truncate">{node.label}</span>
        </div>
        <button
          onClick={onClose}
          className="text-white/30 hover:text-white/70 transition-colors p-1 -mr-1"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M18 6 6 18M6 6l12 12" />
          </svg>
        </button>
      </div>
      <div className="grid grid-cols-2 gap-y-2.5 gap-x-4 text-xs">
        {rows.map(([k, v]) => (
          <div key={k}>
            <div className="text-white/30 mb-0.5">{k}</div>
            <div className="font-mono text-white/80 truncate">{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Stats Strip ─────────────────────────────────────────────────────────────

function StatsStrip({
  nodes,
  collapsed,
  onToggle,
  demoMode,
  loading,
}: {
  nodes: BotNode[];
  collapsed: boolean;
  onToggle: () => void;
  demoMode: boolean;
  loading: boolean;
}) {
  const online  = nodes.filter((n) => n.status === "online").length;
  const pending = nodes.filter((n) => n.status === "pending").length;
  const offline = nodes.filter((n) => n.status === "offline").length;

  return (
    <div
      className="pointer-events-auto rounded-xl backdrop-blur-md overflow-hidden"
      style={{ background: "rgba(8,8,14,0.88)", border: "1px solid rgba(255,255,255,0.07)" }}
    >
      {/* Mobile toggle header */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-2.5 md:hidden"
      >
        <div className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: demoMode ? "#fbbf24" : "#4ade80" }}
          />
          <span className="text-xs text-white/50 font-mono">
            {loading ? "loading..." : `${online}/${nodes.length} active`}
          </span>
        </div>
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2"
          className={`text-white/30 transition-transform duration-200 ${collapsed ? "" : "rotate-180"}`}
        >
          <path d="M18 15l-6-6-6 6" />
        </svg>
      </button>

      {/* Stats row */}
      <div className={`px-4 pb-3 gap-5 ${collapsed ? "hidden md:flex" : "flex"} flex-wrap items-center pt-1 md:pt-3`}>
        {demoMode && (
          <span
            className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider"
            style={{ background: "rgba(251,191,36,0.15)", color: "#fbbf24", border: "1px solid rgba(251,191,36,0.4)" }}
          >
            DEMO · Backend offline
          </span>
        )}
        {([
          ["active",  online,  "#4ade80"],
          ["idle",    pending, "#fbbf24"],
          ["offline", offline, "#ef4444"],
        ] as const).map(([label, count, color]) => (
          <div key={label} className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
            <span className="text-xs text-white/40 capitalize">{label}</span>
            <span className="text-xs font-bold font-mono" style={{ color }}>{count}</span>
          </div>
        ))}
        <div className="h-3 w-px bg-white/10" />
        <span className="text-xs text-white/40">
          Total <span className="font-bold text-white ml-1">{nodes.length}</span>
        </span>
        <div className="ml-auto hidden md:flex items-center gap-1.5">
          {([["Bot", "/bot"], ["Messages", "/msg"], ["Network", "/bot/network"]] as const).map(([name, href]) => (
            <Link
              key={name}
              href={href}
              className="px-2.5 py-1 rounded-full text-[11px] font-medium text-white/50 hover:text-white border border-white/[0.06] hover:border-white/20 transition-all"
            >
              {name}
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Logo SVG (inline to avoid layout shift) ─────────────────────────────────

function LogoMark() {
  return (
    <svg width="22" height="22" viewBox="0 0 150 150" fill="#f26b3a" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <path fillRule="evenodd" d="M57.2,36.22c-2.41-6.61-4.38-13.26-5.36-20.21-.29-1.94-.25-2,1.62-2.65,3.77-1.31,7.62-2.34,11.57-2.95,1.83-.28,1.92-.22,2.17,1.65,1.04,7.89,3.26,15.44,6.62,22.65,.38,.82,.59,2.15,1.46,2.19,1.11,.05,1.26-1.36,1.66-2.21,3.35-7.13,5.55-14.6,6.56-22.42,.27-2.11,.33-2.16,2.46-1.85,3.96,.59,7.78,1.71,11.56,2.99,1.72,.59,1.69,.66,1.42,2.53-.9,6.25-2.53,12.31-4.67,18.24-.24,.67-.98,1.46-.26,2.09,.57,.5,1.29-.15,1.9-.4,6.45-2.67,12.07-6.64,16.97-11.6q1.74-1.72,3.68-.1c2.89,2.41,5.66,4.94,8.09,7.82,1.22,1.46,1.21,1.48-.05,2.81-9.64,10.14-21.57,16.63-35.2,19.67-1.4,.42-1.5,.73-.43,1.82,3.55,3.63,7.48,6.81,11.7,9.62,1.32,.88,1.37,.84,2.9-.6,6.5-6.08,13.83-10.9,21.92-14.58,2.37-1.08,4.81-2,7.26-2.9,1.66-.61,1.93-.47,2.61,1.18,1.53,3.69,2.84,7.45,3.67,11.36,.38,1.76,.35,1.84-1.44,2.48-3.91,1.38-7.68,3.09-11.27,5.17-2.32,1.35-4.56,2.83-6.71,4.43-.6,.44-1.55,.8-1.45,1.63,.12,1.02,1.28,.95,2.02,1.19,5.5,1.74,11.13,2.83,16.89,3.28,4.33,.33,3.49,.2,3.15,3.87-.32,3.44-.99,6.83-1.96,10.14-.52,1.77-.57,1.84-2.4,1.7-22.06-1.63-42.83-11.44-58.39-27.1-2.21-2.2-1.8-2.45-4.18-.06-11.99,12.09-26.24,20.29-42.71,24.59-4.98,1.3-10.04,2.16-15.17,2.55-2.37,.18-2.33,.12-3-2.09-1.07-3.54-1.6-7.18-2.01-10.84-.28-2.44-.23-2.52,2.35-2.7,5.61-.39,11.11-1.32,16.49-2.93,.95-.29,1.9-.6,2.84-.93,.81-.29,.9-.82,.27-1.36-5.7-4.56-12.18-8.03-19.05-10.49-2.03-.71-2.05-.74-1.57-2.77,.88-3.74,2.01-7.4,3.57-10.91,.87-1.96,.9-2.01,2.89-1.32,10.82,3.72,20.74,9.8,29.16,17.53,1.47,1.33,1.46,1.3,3.09,.22,4.2-2.83,8.13-6.04,11.6-9.73,1.03-1.24-.9-1.55-1.76-1.74-13.5-3.1-25.07-9.78-34.6-19.81-.57-.58-.68-1.16-.1-1.81,3.01-3.39,6.23-6.55,9.8-9.36,.74-.58,1.24-.24,1.77,.31,5.17,5.46,11.41,9.9,18.42,12.66,.41,.16,.83,.42,1.61,.04" />
    </svg>
  );
}

// ─── Welcome Page ────────────────────────────────────────────────────────────

export default function WelcomePage() {
  const [topology, setTopology] = useState<TopologyResponse | null>(null);
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [statsCollapsed, setStatsCollapsed] = useState(true);

  // Fetch topology once on mount, then refresh every 10s.
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function load() {
      try {
        const t = await bot.getTopology();
        if (cancelled) return;
        setTopology(t);
        setDemoMode(false);
        setLoading(false);
      } catch {
        if (cancelled) return;
        setDemoMode(true);
        setTopology(null);
        setLoading(false);
      }
      if (!cancelled) timer = setTimeout(load, 10_000);
    }
    load();

    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Map topology → globe data. In demo mode use the small fallback set.
  const { nodes, arcs } = useMemo(() => {
    if (demoMode || !topology) {
      return { nodes: DEMO_FALLBACK_NODES, arcs: DEMO_FALLBACK_ARCS };
    }
    const nodesMapped: BotNode[] = topology.nodes.map(topologyToBotNode);
    // Arcs = self → each peer
    const self = topology.nodes.find((n) => n.self);
    const peers = topology.nodes.filter((n) => !n.self);
    const arcsMapped: ArcDef[] = self
      ? peers.map((p) => ({ fromId: self.node_id, toId: p.node_id }))
      : [];
    return { nodes: nodesMapped, arcs: arcsMapped };
  }, [topology, demoMode]);

  const selectedNode = nodes.find((n) => n.id === selectedId) ?? null;
  const selectedTopo = topology?.nodes.find((n) => n.node_id === selectedId) ?? null;
  const hasPeers = (topology?.nodes.length ?? 0) > 1;

  function handleSelect(id: string) {
    setSelectedId((prev) => (prev === id ? null : id));
    setStatsCollapsed(true);
  }

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#030306] font-sans">

      <WorldGlobe
        nodes={nodes}
        arcs={arcs}
        selectedId={selectedId}
        onSelect={handleSelect}
        className="absolute inset-0 w-full h-full"
      />

      <div
        className="absolute inset-x-0 top-0 h-28 pointer-events-none"
        style={{ background: "linear-gradient(to bottom, rgba(3,3,6,0.7) 0%, transparent 100%)" }}
      />

      <nav className="absolute inset-x-0 top-0 flex items-center justify-between px-4 md:px-6 py-3 z-10">
        <div className="flex items-center gap-2">
          <LogoMark />
          <span className="font-bold text-white tracking-widest text-[11px] uppercase hidden sm:block">
            Yggdrasil
          </span>
        </div>

        <div className="flex items-center gap-2">
          <div className="hidden md:flex items-center gap-1.5 mr-1">
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: demoMode ? "#fbbf24" : "#4ade80",
                animation: demoMode ? "none" : "pulse 2s infinite",
              }}
            />
            <span className="text-[11px] text-white/40 font-mono">
              {demoMode ? "demo" : `${nodes.length} node${nodes.length === 1 ? "" : "s"}`}
            </span>
          </div>
          <Link
            href="/bot"
            className="text-[11px] font-medium px-3 py-1.5 rounded-full text-white/60 hover:text-white transition-all"
            style={{ border: "1px solid rgba(255,255,255,0.1)" }}
          >
            Dashboard
          </Link>
        </div>
      </nav>

      <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between px-3 md:px-5 pb-4 pt-16 md:pt-5">

        <div className="flex justify-end md:mt-0">
          {selectedNode && (
            <div className="pointer-events-auto w-full max-w-[17rem]">
              <NodePanel node={selectedNode} topo={selectedTopo} onClose={() => setSelectedId(null)} />
            </div>
          )}
        </div>

        <StatsStrip
          nodes={nodes}
          collapsed={statsCollapsed}
          onToggle={() => setStatsCollapsed((c) => !c)}
          demoMode={demoMode}
          loading={loading}
        />
      </div>

      {/* Empty state — backend reachable but no peers registered yet */}
      {!demoMode && !loading && !hasPeers && (
        <p className="absolute bottom-20 md:bottom-5 left-1/2 -translate-x-1/2 text-[11px] text-white/30 pointer-events-none whitespace-nowrap z-10">
          This node has no peers yet. Run another node and POST /api/hello.
        </p>
      )}

      {!selectedNode && hasPeers && (
        <p className="absolute bottom-20 md:bottom-5 left-1/2 -translate-x-1/2 text-[10px] text-white/20 pointer-events-none whitespace-nowrap z-10 hidden md:block">
          Drag to rotate · Scroll to zoom · Click a node
        </p>
      )}
    </div>
  );
}
