"use client";

import { useQuery } from "@tanstack/react-query";
import { getHealth, getStats, getBackend, getAudit } from "@/lib/api";
import StatusCard from "@/components/StatusCard";
import AuditLog from "@/components/AuditLog";
import { useState, useEffect } from "react";

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${h}h ${m}m`;
}

function StatusDot({ healthy }: { healthy: boolean | null }) {
  if (healthy === null) {
    return <span className="inline-block w-2 h-2 rounded-full bg-gray-600 animate-pulse" />;
  }
  return (
    <span
      className={`inline-block w-2 h-2 rounded-full ${
        healthy ? "bg-green-400 shadow-[0_0_6px_#4ade80]" : "bg-red-500 shadow-[0_0_6px_#f87171]"
      }`}
    />
  );
}

export default function DashboardPage() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const healthQ = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 5000,
    retry: false,
  });

  const statsQ = useQuery({
    queryKey: ["stats"],
    queryFn: getStats,
    refetchInterval: 5000,
    retry: false,
  });

  const backendQ = useQuery({
    queryKey: ["backend"],
    queryFn: getBackend,
    refetchInterval: 30000,
    retry: false,
  });

  const auditQ = useQuery({
    queryKey: ["audit"],
    queryFn: () => getAudit(20),
    refetchInterval: 5000,
    retry: false,
  });

  const healthy =
    healthQ.data?.status === "healthy"
      ? true
      : healthQ.isError
        ? false
        : null;

  const backendUnreachable = statsQ.isError && healthQ.isError;

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 max-w-6xl mx-auto flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold font-mono tracking-tight text-white">
              YGG <span className="text-[#3b82f6]">Trading</span>
            </h1>
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full border border-[#1e1e2e] bg-[#13131a]">
              <StatusDot healthy={healthy} />
              <span className="text-xs font-mono text-gray-400">
                {healthy === null ? "checking..." : healthy ? "online" : "offline"}
              </span>
            </div>
          </div>
          <div className="text-sm font-mono text-gray-500">
            {now.toLocaleTimeString("en-US", { hour12: false })}
          </div>
        </div>

        {/* Offline banner */}
        {backendUnreachable && (
          <div className="rounded-lg border border-red-800 bg-red-950/30 px-4 py-3 flex items-center gap-2">
            <span className="text-red-400 text-sm font-mono">
              Backend unreachable at {process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8100"} — displaying cached or empty data
            </span>
          </div>
        )}

        {/* Status cards */}
        <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
          <StatusCard
            label="Node ID"
            value={statsQ.data?.node_id ?? "—"}
            loading={statsQ.isLoading}
            accent
          />
          <StatusCard
            label="Uptime"
            value={statsQ.data ? formatUptime(statsQ.data.uptime_s) : "—"}
            loading={statsQ.isLoading}
          />
          <StatusCard
            label="Messages"
            value={statsQ.data?.messages?.toLocaleString() ?? "—"}
            loading={statsQ.isLoading}
          />
          <StatusCard
            label="Functions"
            value={statsQ.data?.functions?.toLocaleString() ?? "—"}
            loading={statsQ.isLoading}
          />
        </div>

        {/* Health + Backend row */}
        <div className="grid grid-cols-2 gap-4">
          <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4 flex flex-col gap-2">
            <span className="text-xs font-mono uppercase tracking-widest text-gray-500">Health</span>
            {healthQ.isLoading ? (
              <div className="h-6 w-24 rounded bg-[#1e1e2e] animate-pulse" />
            ) : (
              <span
                className={`text-sm font-mono font-semibold ${
                  healthy ? "text-green-400" : healthy === false ? "text-red-400" : "text-gray-500"
                }`}
              >
                {healthQ.data?.status ?? (healthQ.isError ? "unreachable" : "unknown")}
              </span>
            )}
          </div>
          <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4 flex flex-col gap-2">
            <span className="text-xs font-mono uppercase tracking-widest text-gray-500">Backend</span>
            {backendQ.isLoading ? (
              <div className="h-6 w-32 rounded bg-[#1e1e2e] animate-pulse" />
            ) : (
              <span className="text-sm font-mono text-gray-200">
                {backendQ.data
                  ? `${backendQ.data.backend} v${backendQ.data.version}`
                  : "—"}
              </span>
            )}
          </div>
        </div>

        {/* Audit log */}
        <div className="flex-1" style={{ minHeight: "320px" }}>
          <AuditLog
            entries={auditQ.data ?? []}
            loading={auditQ.isLoading}
            error={auditQ.isError ? "Failed to load audit log" : null}
          />
        </div>
      </div>
    </div>
  );
}
