"use client";

interface HeaderProps {
  uptime?: number;
  wsConnections?: number;
  connected: boolean;
}

export function Header({ uptime, wsConnections, connected }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-border bg-surface">
      <div className="flex items-center gap-3">
        <span className="text-accent font-semibold text-base tracking-tight">YGG BOT</span>
        <span className="text-muted text-xs">trading + ai</span>
      </div>
      <div className="flex items-center gap-5 text-xs text-muted">
        {uptime !== undefined && (
          <span>uptime {uptime.toFixed(0)}s</span>
        )}
        {wsConnections !== undefined && (
          <span>{wsConnections} ws</span>
        )}
        <span className="flex items-center gap-1.5">
          <span
            className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-green animate-pulse" : "bg-red"}`}
          />
          {connected ? "live" : "offline"}
        </span>
      </div>
    </header>
  );
}
