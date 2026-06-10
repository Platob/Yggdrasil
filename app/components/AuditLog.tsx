"use client";

import type { AuditEntry } from "@/lib/types";

interface AuditLogProps {
  entries: AuditEntry[];
  loading?: boolean;
  error?: string | null;
}

function formatTs(entry: AuditEntry): string {
  const raw = entry.timestamp ?? entry.ts;
  if (!raw) return "—";
  try {
    return new Date(raw).toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  } catch {
    return String(raw);
  }
}

function entryLabel(entry: AuditEntry): string {
  return (
    (entry.action ?? entry.event ?? entry.message ?? JSON.stringify(entry)).slice(0, 80)
  );
}

function levelColor(entry: AuditEntry): string {
  const level = (entry.level ?? "").toLowerCase();
  if (level === "error") return "text-red-400";
  if (level === "warn" || level === "warning") return "text-yellow-400";
  if (level === "info") return "text-[#60a5fa]";
  return "text-gray-400";
}

export default function AuditLog({ entries, loading, error }: AuditLogProps) {
  return (
    <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] flex flex-col h-full min-h-0">
      <div className="px-4 py-3 border-b border-[#1e1e2e]">
        <span className="text-xs font-mono uppercase tracking-widest text-gray-500">
          Audit Log
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-2">
        {error && (
          <div className="text-red-400 text-xs font-mono p-2">{error}</div>
        )}
        {loading && (
          <div className="flex flex-col gap-2 p-2">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="h-5 rounded bg-[#1e1e2e] animate-pulse" />
            ))}
          </div>
        )}
        {!loading && !error && entries.length === 0 && (
          <div className="text-gray-600 text-xs font-mono p-2">No audit entries.</div>
        )}
        {!loading &&
          !error &&
          entries.map((entry, i) => (
            <div
              key={i}
              className="flex items-start gap-3 px-2 py-1.5 rounded hover:bg-[#1a1a24] transition-colors"
            >
              <span className="text-[10px] font-mono text-gray-600 shrink-0 pt-0.5 w-16">
                {formatTs(entry)}
              </span>
              <span className={`text-xs font-mono truncate ${levelColor(entry)}`}>
                {entryLabel(entry)}
              </span>
              {entry.user && (
                <span className="text-[10px] font-mono text-gray-600 shrink-0 ml-auto">
                  {String(entry.user)}
                </span>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}
