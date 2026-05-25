"use client";

import { useEffect, useState } from "react";
import { getNodeInfo, getRegistry, getChannels, type NodeInfo, type ChannelInfo } from "@/lib/api";
import { YggdrasilLogo } from "@/components/logo";

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
          <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-destructive/10 flex items-center justify-center">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-destructive">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-destructive mb-2">Connection Failed</h2>
          <p className="text-muted text-sm mb-4">{error}</p>
          <div className="inline-flex items-center gap-2 bg-card border border-border rounded-lg px-4 py-2">
            <span className="text-xs text-muted">Start the bot:</span>
            <code className="text-xs text-primary font-mono">ygg bot serve</code>
          </div>
        </div>
      </div>
    );
  }

  if (!node) {
    return (
      <div className="flex items-center justify-center h-[80vh]">
        <div className="text-center animate-in">
          <YggdrasilLogo size={48} className="text-primary mx-auto mb-4 glow-primary" />
          <p className="text-primary font-mono text-sm pulse-primary">Connecting to Yggdrasil...</p>
        </div>
      </div>
    );
  }

  const fns = Object.entries(registry);
  const uptime = node.uptime > 3600
    ? `${(node.uptime / 3600).toFixed(1)}h`
    : `${(node.uptime / 60).toFixed(0)}m`;

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">Dashboard</h1>
          <p className="text-muted text-sm mt-1">Node overview and system status</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 bg-success/10 border border-success/20 rounded-full">
          <div className="status-dot online" />
          <span className="text-xs font-medium text-success">Online</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard 
          label="Node ID" 
          value={node.node_id} 
          icon={
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          }
        />
        <StatCard 
          label="Uptime" 
          value={uptime}
          icon={
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          }
        />
        <StatCard 
          label="Version" 
          value={node.version}
          icon={
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
            </svg>
          }
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Channels Card */}
        <div className="nordic-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
              Channels
            </h2>
            <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-0.5 rounded-full">
              {channels.length}
            </span>
          </div>
          {channels.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted text-sm">No channels available</p>
            </div>
          ) : (
            <div className="space-y-2">
              {channels.map((ch) => (
                <div 
                  key={ch.name} 
                  className="flex items-center justify-between p-3 bg-background rounded-lg border border-border hover:border-border-accent transition-colors"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-primary font-mono text-xs">#</span>
                    <span className="font-medium text-sm text-foreground">{ch.name}</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted">
                    <span className="flex items-center gap-1">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                      </svg>
                      {ch.message_count}
                    </span>
                    <span className="flex items-center gap-1">
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75" />
                      </svg>
                      {ch.members.length}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Functions Card */}
        <div className="nordic-card p-5">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-foreground flex items-center gap-2">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                <polyline points="16 18 22 12 16 6" />
                <polyline points="8 6 2 12 8 18" />
              </svg>
              Remote Functions
            </h2>
            <span className="text-xs font-mono text-primary bg-primary/10 px-2 py-0.5 rounded-full">
              {fns.length}
            </span>
          </div>
          {fns.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted text-sm">No @remote functions registered</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-64 overflow-auto">
              {fns.map(([key, sig]) => (
                <div 
                  key={key} 
                  className="p-3 bg-background rounded-lg border border-border hover:border-border-accent transition-colors"
                >
                  <div className="font-mono text-xs text-foreground mb-1">{key}</div>
                  <div className="font-mono text-[10px] text-muted truncate">{sig}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ 
  label, 
  value, 
  icon 
}: { 
  label: string; 
  value: string; 
  icon: React.ReactNode;
}) {
  return (
    <div className="nordic-card p-4 flex items-start gap-4">
      <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center text-primary shrink-0">
        {icon}
      </div>
      <div className="min-w-0 flex-1">
        <p className="text-[10px] uppercase tracking-widest text-muted mb-1">{label}</p>
        <p className="text-sm font-mono text-foreground truncate">{value}</p>
      </div>
    </div>
  );
}
