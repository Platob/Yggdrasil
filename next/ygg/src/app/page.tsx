"use client";

import { useEffect, useState } from "react";
import { getNodeInfo, getRegistry, getChannels, type NodeInfo, type ChannelInfo } from "@/lib/api";

export default function Dashboard() {
  const [node, setNode] = useState<NodeInfo | null>(null);
  const [registry, setRegistry] = useState<Record<string, string>>({});
  const [channels, setChannels] = useState<ChannelInfo[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([getNodeInfo(), getRegistry(), getChannels()])
      .then(([n, r, c]) => { setNode(n); setRegistry(r); setChannels(c); })
      .catch((e) => setError(e.message));
  }, []);

  if (error) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <div className="text-accent-red text-lg font-mono mb-2">Connection failed</div>
          <p className="text-muted text-sm">{error}</p>
          <p className="text-muted text-xs mt-4">Start the bot: <code className="text-gold">ygg bot serve</code></p>
        </div>
      </div>
    );
  }

  if (!node) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-gold pulse-gold font-mono">connecting...</div>
      </div>
    );
  }

  const fns = Object.entries(registry);
  const uptime = node.uptime > 3600
    ? `${(node.uptime / 3600).toFixed(1)}h`
    : `${(node.uptime / 60).toFixed(0)}m`;

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-in">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted text-sm mt-1">Node overview</p>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <StatCard label="Node" value={node.node_id} />
        <StatCard label="Uptime" value={uptime} />
        <StatCard label="Version" value={node.version} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gold mb-3">Channels ({channels.length})</h2>
          {channels.length === 0 ? (
            <p className="text-muted text-xs">No channels</p>
          ) : (
            <div className="space-y-2">
              {channels.map((ch) => (
                <div key={ch.name} className="flex items-center justify-between text-sm">
                  <span className="font-mono text-foreground">#{ch.name}</span>
                  <span className="text-muted text-xs">{ch.message_count} msgs &middot; {ch.members.length} members</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="bg-card border border-border rounded-xl p-5">
          <h2 className="text-sm font-semibold text-gold mb-3">Functions ({fns.length})</h2>
          {fns.length === 0 ? (
            <p className="text-muted text-xs">No @remote functions registered</p>
          ) : (
            <div className="space-y-2 max-h-48 overflow-auto">
              {fns.map(([key, sig]) => (
                <div key={key} className="text-xs">
                  <span className="font-mono text-foreground">{key}</span>
                  <span className="text-muted ml-2">{sig}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-card border border-border rounded-xl p-4">
      <p className="text-[10px] uppercase tracking-widest text-muted mb-1">{label}</p>
      <p className="text-sm font-mono text-foreground truncate">{value}</p>
    </div>
  );
}
