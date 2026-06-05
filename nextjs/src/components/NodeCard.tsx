"use client";

import type { NodeMeta, NodeRole } from "@/lib/types";
import { ResourceBar } from "./ResourceBar";

// ── Role badge colors ────────────────────────────────────────
const ROLE_STYLES: Record<NodeRole, { bg: string; text: string; border: string }> = {
  driver:   { bg: "rgba(103,232,249,0.1)",  text: "var(--frost)",   border: "rgba(103,232,249,0.2)" },
  executor: { bg: "rgba(52,211,153,0.1)",   text: "var(--emerald)", border: "rgba(52,211,153,0.2)" },
  hybrid:   { bg: "rgba(251,191,36,0.1)",   text: "var(--amber)",   border: "rgba(251,191,36,0.2)" },
};

interface NodeCardProps {
  node: NodeMeta;
  isSelf?: boolean;
}

export function NodeCard({ node, isSelf }: NodeCardProps) {
  const roleStyle = ROLE_STYLES[node.role] || ROLE_STYLES.hybrid;
  const isOnline = node.cpu_percent >= 0; // presence in peers list implies online
  const nodeUrl = `http://${node.host}:${node.port}`;

  return (
    <div
      className={`
        glass-card p-5 space-y-4 relative overflow-hidden
        ${isSelf ? "ring-1 ring-frost/20" : ""}
      `}
    >
      {/* Self badge */}
      {isSelf && (
        <div className="absolute top-3 right-3">
          <span className="text-[10px] font-bold uppercase tracking-wider text-frost bg-frost/10 px-2 py-0.5 rounded-full border border-frost/20">
            Self
          </span>
        </div>
      )}

      {/* Header: status + node ID */}
      <div className="flex items-start gap-3">
        <div className="relative mt-0.5">
          <span
            className={`block w-2.5 h-2.5 rounded-full ${isOnline ? "status-online" : "status-offline"}`}
          />
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="font-mono text-sm font-semibold text-foreground truncate">
            {node.node_id}
          </h3>
          <p className="text-xs text-muted font-mono mt-0.5">
            {node.host}:{node.port}
          </p>
        </div>
      </div>

      {/* Role badge */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold uppercase tracking-wider border"
          style={{
            background: roleStyle.bg,
            color: roleStyle.text,
            borderColor: roleStyle.border,
          }}
        >
          {node.role}
        </span>
        {node.version && (
          <span className="text-[11px] text-muted font-mono">v{node.version}</span>
        )}
        {node.gpu_count > 0 && (
          <span className="inline-flex items-center gap-1 text-[11px] text-foreground-dim">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="4" y="4" width="16" height="16" rx="2" />
              <rect x="9" y="9" width="6" height="6" />
            </svg>
            {node.gpu_count} GPU{node.gpu_count > 1 ? "s" : ""}
          </span>
        )}
      </div>

      {/* Resource bars */}
      <div className="space-y-3">
        <ResourceBar
          label="CPU"
          value={node.cpu_percent}
          color="var(--frost)"
        />
        <ResourceBar
          label="Memory"
          value={node.memory_percent}
          color="var(--emerald)"
        />
      </div>

      {/* Footer: location + link */}
      <div className="flex items-center justify-between pt-1 border-t border-white/[0.04]">
        <div className="text-[11px] text-muted">
          {node.lat != null && node.lon != null ? (
            <span className="font-mono">
              {node.lat.toFixed(2)}, {node.lon.toFixed(2)}
            </span>
          ) : (
            <span>Location unknown</span>
          )}
        </div>
        {!isSelf && (
          <a
            href={nodeUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-frost/60 hover:text-frost transition-colors font-medium"
          >
            Open
            <svg className="inline-block ml-1 -mt-0.5" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6" />
              <polyline points="15 3 21 3 21 9" />
              <line x1="10" y1="14" x2="21" y2="3" />
            </svg>
          </a>
        )}
      </div>
    </div>
  );
}
