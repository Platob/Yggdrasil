"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { WorldGlobe, type BotNode, type ArcDef } from "@/components/world-globe";
import { node } from "@/lib/api";

// ─── Data ────────────────────────────────────────────────────────────────────

const BOT_NODES: BotNode[] = [
  { id: "paris-01",     label: "Paris",         lat: 48.8566,  lng: 2.3522,    status: "online",  version: "0.1.0", uptime: 86400 },
  { id: "nyc-01",       label: "New York",      lat: 40.7128,  lng: -74.006,   status: "online",  version: "0.1.0", uptime: 72000 },
  { id: "sf-01",        label: "San Francisco", lat: 37.7749,  lng: -122.4194, status: "online",  version: "0.1.0", uptime: 43200 },
  { id: "tokyo-01",     label: "Tokyo",         lat: 35.6762,  lng: 139.6503,  status: "online",  version: "0.1.0", uptime: 36000 },
  { id: "sydney-01",    label: "Sydney",        lat: -33.8688, lng: 151.2093,  status: "pending", version: "0.1.0", uptime: 18000 },
  { id: "london-01",    label: "London",        lat: 51.5074,  lng: -0.1278,   status: "online",  version: "0.1.0", uptime: 54000 },
  { id: "singapore-01", label: "Singapore",     lat: 1.3521,   lng: 103.8198,  status: "online",  version: "0.1.0", uptime: 28800 },
  { id: "brazil-01",    label: "São Paulo",     lat: -23.5505, lng: -46.6333,  status: "online",  version: "0.1.0", uptime: 21600 },
  { id: "dubai-01",     label: "Dubai",         lat: 25.2048,  lng: 55.2708,   status: "online",  version: "0.1.0", uptime: 14400 },
  { id: "mumbai-01",    label: "Mumbai",        lat: 19.076,   lng: 72.8777,   status: "online",  version: "0.1.0", uptime: 32400 },
  { id: "berlin-01",    label: "Berlin",        lat: 52.52,    lng: 13.405,    status: "online",  version: "0.1.0", uptime: 64800 },
  { id: "toronto-01",   label: "Toronto",       lat: 43.6532,  lng: -79.3832,  status: "offline", version: "0.0.9", uptime: 0 },
];

const ARCS: ArcDef[] = [
  { fromId: "paris-01",     toId: "nyc-01"       },
  { fromId: "nyc-01",       toId: "sf-01"        },
  { fromId: "sf-01",        toId: "tokyo-01"     },
  { fromId: "tokyo-01",     toId: "singapore-01" },
  { fromId: "singapore-01", toId: "sydney-01"    },
  { fromId: "london-01",    toId: "dubai-01"     },
  { fromId: "dubai-01",     toId: "mumbai-01"    },
  { fromId: "paris-01",     toId: "london-01"    },
  { fromId: "berlin-01",    toId: "paris-01"     },
  { fromId: "brazil-01",    toId: "nyc-01"       },
  { fromId: "tokyo-01",     toId: "sydney-01"    },
  { fromId: "singapore-01", toId: "mumbai-01"    },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatUptime(s: number) {
  if (s === 0) return "Offline";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${m}m` : `${m}m`;
}

const STATUS_COLOR: Record<BotNode["status"], string> = {
  online: "#4ade80",
  pending: "#fbbf24",
  offline: "#ef4444",
};

// ─── Node Detail Panel ────────────────────────────────────────────────────────

function NodePanel({ node, onClose }: { node: BotNode; onClose: () => void }) {
  const col = STATUS_COLOR[node.status];
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
          <span className="font-semibold text-white text-sm">{node.label}</span>
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
        {([
          ["Node ID",   node.id],
          ["Status",    node.status],
          ["Version",   node.version ?? "—"],
          ["Uptime",    formatUptime(node.uptime ?? 0)],
          ["Latitude",  node.lat.toFixed(4)],
          ["Longitude", node.lng.toFixed(4)],
        ] as [string, string][]).map(([k, v]) => (
          <div key={k}>
            <div className="text-white/30 mb-0.5">{k}</div>
            <div
              className="font-mono text-white/80 truncate capitalize"
              style={k === "Status" ? { color: col } : {}}
            >
              {v}
            </div>
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
}: {
  nodes: BotNode[];
  collapsed: boolean;
  onToggle: () => void;
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
          <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
          <span className="text-xs text-white/50 font-mono">{online}/{nodes.length} online</span>
        </div>
        <svg
          width="12" height="12" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2"
          className={`text-white/30 transition-transform duration-200 ${collapsed ? "" : "rotate-180"}`}
        >
          <path d="M18 15l-6-6-6 6" />
        </svg>
      </button>

      {/* Stats row — always on md+, collapsible on mobile */}
      <div className={`px-4 pb-3 gap-5 ${collapsed ? "hidden md:flex" : "flex"} flex-wrap items-center pt-1 md:pt-3`}>
        {([
          ["online",  online,  "#4ade80"],
          ["pending", pending, "#fbbf24"],
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
          {([["Nodes", "/node"], ["Messages", "/msg"], ["Network", "/node/network"]] as const).map(([name, href]) => (
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

// ─── Welcome Page ────────────────────────────────────────────────────────────

export default function WelcomePage() {
  const router = useRouter();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [statsCollapsed, setStatsCollapsed] = useState(true);
  const [botNodes, setBotNodes] = useState<BotNode[]>(BOT_NODES);

  // Fetch real node data on mount
  useEffect(() => {
    async function loadNodes() {
      try {
        const [self, peersData] = await Promise.all([
          node.getNodeInfo(),
          node.getPeers(),
        ]);
        // Convert NodeInfo to BotNode format
        const allNodes = [self, ...peersData.peers]
          .filter(n => n.lat != null && n.lon != null)
          .map(n => ({
            id: n.node_id,
            label: n.node_id,
            lat: n.lat!,
            lng: n.lon!,
            status: "online" as const,
            version: n.version,
            uptime: n.uptime,
          }));
        if (allNodes.length > 0) setBotNodes(allNodes);
      } catch { /* keep demo data */ }
    }
    loadNodes();
  }, []);

  const selectedNode = botNodes.find((n) => n.id === selectedId) ?? null;

  function handleSelect(id: string) {
    setSelectedId((prev) => (prev === id ? null : id));
    // On mobile, collapse the stats drawer when a node is selected
    setStatsCollapsed(true);
  }

  function handleNavigate(id: string) {
    router.push(`/node/${encodeURIComponent(id)}`);
  }

  return (
    <div className="relative w-full h-[calc(100vh-0px)] -m-6 overflow-hidden bg-[#030306] font-sans">

      {/* Full-bleed globe */}
      <WorldGlobe
        nodes={botNodes}
        arcs={ARCS}
        selectedId={selectedId}
        onSelect={handleSelect}
        onNavigate={handleNavigate}
        className="absolute inset-0 w-full h-full"
      />

      {/* ── Overlays: node panel + stats — pointer-events handled per-child ── */}
      <div className="absolute inset-0 z-10 pointer-events-none flex flex-col justify-between px-3 md:px-5 pb-4 pt-5">

        {/* Node detail panel — top-right on md, top on mobile */}
        <div className="flex justify-end md:mt-0">
          {selectedNode && (
            <div className="pointer-events-auto w-full max-w-[17rem]">
              <NodePanel node={selectedNode} onClose={() => setSelectedId(null)} />
              <Link
                href={`/node/${encodeURIComponent(selectedNode.id)}`}
                className="mt-2 flex items-center justify-center gap-1.5 w-full py-2 rounded-lg text-xs font-medium transition-all"
                style={{
                  background: "rgba(242,107,58,0.12)",
                  border: "1px solid rgba(242,107,58,0.25)",
                  color: "#f26b3a",
                }}
              >
                View Node &rarr;
              </Link>
            </div>
          )}
        </div>

        {/* Stats strip — bottom */}
        <StatsStrip
          nodes={botNodes}
          collapsed={statsCollapsed}
          onToggle={() => setStatsCollapsed((c) => !c)}
        />
      </div>

      {/* Hint text — only when nothing selected, fades at bottom-center */}
      {!selectedNode && (
        <p className="absolute bottom-20 md:bottom-5 left-1/2 -translate-x-1/2 text-[10px] text-white/20 pointer-events-none whitespace-nowrap z-10 hidden md:block">
          Drag to rotate · Scroll to zoom · Click a node
        </p>
      )}
    </div>
  );
}
