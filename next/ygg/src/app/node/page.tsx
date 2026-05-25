"use client";

import { useEffect, useState } from "react";
import { node, type NodeInfo, type FunctionEntry, type EnvironmentEntry, type RunEntry } from "@/lib/api";
import { YggdrasilLogo } from "@/components/logo";
import { formatRelative, formatDuration } from "@/lib/time";
import Link from "next/link";

// -- Haversine distance helper --
function haversine(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // km
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

// -- Format uptime --
function formatUptime(s: number): string {
  const sec = Math.floor(s);
  if (sec < 60) return `${sec}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m`;
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  return `${h}h ${m}m`;
}

// -- Node Card Component --
function NodeCard({ node: nodeInfo, distance }: { node: NodeInfo; distance?: number }) {
  return (
    <Link
      href={`/bot/${encodeURIComponent(nodeInfo.node_id)}`}
      className="bg-card border border-border rounded-xl p-4 hover:border-primary/50 hover:bg-card-hover transition-all duration-200 group"
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div className="status-dot online" />
          <span className="font-mono text-sm text-foreground font-medium group-hover:text-primary transition-colors truncate max-w-[180px]">
            {nodeInfo.node_id}
          </span>
        </div>
        <span className="text-[10px] font-mono text-muted bg-border/50 px-1.5 py-0.5 rounded">
          v{nodeInfo.version}
        </span>
      </div>

      <div className="space-y-1.5 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-muted">Host</span>
          <span className="font-mono text-foreground">{nodeInfo.host}:{nodeInfo.port}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted">Uptime</span>
          <span className="text-foreground">{formatUptime(nodeInfo.uptime)}</span>
        </div>
        {nodeInfo.lat != null && nodeInfo.lon != null && (
          <div className="flex items-center justify-between">
            <span className="text-muted">Location</span>
            <span className="font-mono text-primary text-[11px]">
              {nodeInfo.lat.toFixed(2)}, {nodeInfo.lon.toFixed(2)}
            </span>
          </div>
        )}
        {distance != null && (
          <div className="flex items-center justify-between">
            <span className="text-muted">Distance</span>
            <span className="font-mono text-primary text-[11px]">
              {distance < 1 ? `${(distance * 1000).toFixed(0)} m` : distance < 100 ? `${distance.toFixed(1)} km` : `${distance.toFixed(0)} km`}
            </span>
          </div>
        )}
        <div className="flex items-center justify-between pt-1">
          <span className="text-muted">Functions</span>
          <span className="text-foreground">{nodeInfo.functions.length}</span>
        </div>
      </div>
    </Link>
  );
}

// -- Main Network Overview Dashboard --
export default function NetworkOverview() {
  const [selfNode, setSelfNode] = useState<NodeInfo | null>(null);
  const [peers, setPeers] = useState<NodeInfo[]>([]);
  const [functions, setFunctions] = useState<FunctionEntry[]>([]);
  const [environments, setEnvironments] = useState<EnvironmentEntry[]>([]);
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadNetwork() {
      setLoading(true);
      setError(null);

      try {
        // Fetch self info
        const self = await node.getNodeInfo();
        setSelfNode(self);

        // Fetch peers
        try {
          const peersData = await node.getPeers();
          setPeers(peersData.peers);
        } catch {
          setPeers([]);
        }

        // Fetch functions, environments, runs in parallel
        const [fnData, envData, runData] = await Promise.allSettled([
          node.listFunctions(),
          node.listEnvironments(),
          node.listRuns(),
        ]);
        if (fnData.status === "fulfilled") setFunctions(fnData.value.functions);
        if (envData.status === "fulfilled") setEnvironments(envData.value.environments);
        if (runData.status === "fulfilled") setRuns(runData.value.runs);
      } catch {
        setError("Node unavailable - showing demo data");
        const demoSelf: NodeInfo = {
          node_id: "ygg-node-alpha",
          host: "192.168.1.10",
          port: 8100,
          version: "0.3.1",
          uptime: 86400,
          channels: ["general", "alerts"],
          functions: ["echo", "ping", "execute", "deploy"],
          lat: 48.8566,
          lon: 2.3522,
        };
        const demoPeers: NodeInfo[] = [
          { node_id: "ygg-node-beta", host: "10.0.2.15", port: 8100, version: "0.3.1", uptime: 43200, channels: ["general"], functions: ["echo", "ping"], lat: 51.5074, lon: -0.1278 },
          { node_id: "ygg-node-gamma", host: "172.16.0.5", port: 8100, version: "0.3.0", uptime: 7200, channels: ["general", "data"], functions: ["echo", "stream"], lat: 40.7128, lon: -74.0060 },
          { node_id: "ygg-node-delta", host: "10.0.3.22", port: 8100, version: "0.3.1", uptime: 172800, channels: ["general"], functions: ["echo", "ping", "monitor"], lat: 35.6762, lon: 139.6503 },
          { node_id: "ygg-node-epsilon", host: "192.168.5.8", port: 8100, version: "0.2.9", uptime: 3600, channels: ["alerts"], functions: ["echo"], lat: 52.5200, lon: 13.4050 },
          { node_id: "ygg-node-zeta", host: "10.0.1.100", port: 8100, version: "0.3.1", uptime: 259200, channels: ["general", "trading"], functions: ["echo", "ping", "trade", "analyze"], lat: 37.7749, lon: -122.4194 },
        ];
        setSelfNode(demoSelf);
        setPeers(demoPeers);
      }

      setLoading(false);
    }
    loadNetwork();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <YggdrasilLogo size={48} className="mx-auto mb-4" />
          <p className="text-primary font-mono text-sm pulse-primary">Scanning network...</p>
        </div>
      </div>
    );
  }

  // All nodes = self + peers
  const allNodes: NodeInfo[] = selfNode ? [selfNode, ...peers] : peers;

  // Closest neighbors: sorted by haversine distance from self
  const closestNeighbors = (() => {
    if (!selfNode || selfNode.lat == null || selfNode.lon == null) return [];
    return peers
      .filter((p) => p.lat != null && p.lon != null)
      .map((p) => ({
        node: p,
        distance: haversine(selfNode.lat!, selfNode.lon!, p.lat!, p.lon!),
      }))
      .sort((a, b) => a.distance - b.distance)
      .slice(0, 5);
  })();

  // Build function lookup for runs
  const fnLookup = new Map(functions.map((f) => [f.id, f]));

  // Recent runs (last 10)
  const recentRuns = [...runs]
    .sort((a, b) => {
      const ta = a.started_at ? new Date(a.started_at).getTime() : 0;
      const tb = b.started_at ? new Date(b.started_at).getTime() : 0;
      return tb - ta;
    })
    .slice(0, 10);

  return (
    <div className="p-6 space-y-8 animate-in">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
            <YggdrasilLogo size={28} />
          </div>
          <div>
            <h1 className="text-xl font-bold text-foreground">Network Overview</h1>
            <p className="text-sm text-muted mt-0.5">
              {allNodes.length} node{allNodes.length !== 1 ? "s" : ""} in the network
            </p>
          </div>
        </div>

        {error && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-warning/10 border border-warning/20">
            <div className="status-dot pending" />
            <span className="text-xs font-medium text-warning">{error}</span>
          </div>
        )}
      </div>

      {/* All Nodes Grid */}
      <section>
        <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">All Nodes</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {allNodes.map((n) => (
            <NodeCard key={n.node_id} node={n} />
          ))}
        </div>
      </section>

      {/* Closest Neighbors */}
      {closestNeighbors.length > 0 && (
        <section>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">Closest Neighbors</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {closestNeighbors.map(({ node: n, distance }) => (
              <NodeCard key={n.node_id} node={n} distance={distance} />
            ))}
          </div>
        </section>
      )}

      {/* All Functions */}
      {functions.length > 0 && (
        <section>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">All Functions</h2>
          <div className="nordic-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-[10px] uppercase tracking-wider text-muted">
                    <th className="text-left px-4 py-2 font-medium">Name</th>
                    <th className="text-left px-4 py-2 font-medium">Language</th>
                    <th className="text-right px-4 py-2 font-medium">Runs</th>
                    <th className="text-right px-4 py-2 font-medium">Last Used</th>
                    <th className="text-left px-4 py-2 font-medium">Environment</th>
                  </tr>
                </thead>
                <tbody>
                  {functions.map((fn) => (
                    <tr key={fn.id} className="border-b border-border/50 hover:bg-card-hover transition-colors">
                      <td className="px-4 py-2.5">
                        <Link href={`/node/functions/${fn.id}`} className="font-mono text-xs text-foreground hover:text-primary transition-colors">
                          {fn.name}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5">
                        <span className="text-[10px] font-mono text-muted bg-border/50 px-1.5 py-0.5 rounded">{fn.language}</span>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-foreground-dim">{fn.run_count}</td>
                      <td className="px-4 py-2.5 text-right text-xs text-muted">{formatRelative(fn.last_used_at)}</td>
                      <td className="px-4 py-2.5">
                        {fn.environment_id != null ? (
                          <span className="text-xs font-mono text-primary">
                            {environments.find((e) => e.id === fn.environment_id)?.name ?? `#${fn.environment_id}`}
                          </span>
                        ) : (
                          <span className="text-xs text-muted">default</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* All Environments */}
      {environments.length > 0 && (
        <section>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">All Environments</h2>
          <div className="nordic-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-[10px] uppercase tracking-wider text-muted">
                    <th className="text-left px-4 py-2 font-medium">Name</th>
                    <th className="text-left px-4 py-2 font-medium">Python</th>
                    <th className="text-center px-4 py-2 font-medium">Status</th>
                    <th className="text-right px-4 py-2 font-medium">Deps</th>
                  </tr>
                </thead>
                <tbody>
                  {environments.map((env) => (
                    <tr key={env.id} className="border-b border-border/50 hover:bg-card-hover transition-colors">
                      <td className="px-4 py-2.5">
                        <Link href={`/node/environments/${env.id}`} className="font-mono text-xs text-foreground hover:text-primary transition-colors">
                          {env.name}
                        </Link>
                      </td>
                      <td className="px-4 py-2.5 font-mono text-xs text-foreground-dim">{env.python_version}</td>
                      <td className="px-4 py-2.5 text-center">
                        <div className="flex items-center justify-center gap-1.5">
                          <span className={`w-2 h-2 rounded-full ${env.status === "ready" ? "bg-success" : env.status === "building" ? "bg-warning" : "bg-destructive"}`} />
                          <span className="text-[10px] text-muted">{env.status}</span>
                        </div>
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-xs text-foreground-dim">{env.dependencies.length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}

      {/* Recent Runs */}
      {recentRuns.length > 0 && (
        <section>
          <h2 className="text-xs font-medium text-muted uppercase tracking-wider mb-4">Recent Runs</h2>
          <div className="nordic-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-[10px] uppercase tracking-wider text-muted">
                    <th className="text-left px-4 py-2 font-medium">Function</th>
                    <th className="text-center px-4 py-2 font-medium">Status</th>
                    <th className="text-right px-4 py-2 font-medium">Duration</th>
                    <th className="text-right px-4 py-2 font-medium">Started</th>
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.map((run) => {
                    const fn = fnLookup.get(run.function_id);
                    return (
                      <tr key={run.id} className="border-b border-border/50 hover:bg-card-hover transition-colors">
                        <td className="px-4 py-2.5">
                          <Link href={`/node/functions/${run.function_id}`} className="font-mono text-xs text-foreground hover:text-primary transition-colors">
                            {fn?.name ?? `#${run.function_id}`}
                          </Link>
                        </td>
                        <td className="px-4 py-2.5 text-center">
                          <span
                            className="text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded inline-block"
                            style={{
                              color: run.status === "completed" ? "var(--success)" : run.status === "running" ? "var(--warning)" : "var(--destructive)",
                              background: `color-mix(in srgb, ${run.status === "completed" ? "var(--success)" : run.status === "running" ? "var(--warning)" : "var(--destructive)"} 15%, transparent)`,
                            }}
                          >
                            {run.status}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-right font-mono text-xs text-foreground-dim">{formatDuration(run.duration)}</td>
                        <td className="px-4 py-2.5 text-right text-xs text-muted">{formatRelative(run.started_at)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
