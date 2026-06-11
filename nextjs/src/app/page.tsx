"use client";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StatusBadge } from "@/components/StatusBadge";

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <div className="text-gray-400 text-xs uppercase tracking-wider">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
    </div>
  );
}

export default function DashboardPage() {
  const [stats, setStats] = useState<{ uptime_s: number; requests: number } | null>(null);
  const [status, setStatus] = useState<"online" | "offline" | "loading">("loading");

  useEffect(() => {
    api.ping()
      .then(() => setStatus("online"))
      .catch(() => setStatus("offline"));
    api.stats().then(setStats).catch(() => {});
  }, []);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Dashboard</h1>
      <div className="mb-6">
        <StatusBadge status={status} />
      </div>
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Uptime" value={`${Math.floor(stats.uptime_s / 60)}m`} />
          <StatCard label="Requests" value={stats.requests} />
        </div>
      )}
    </div>
  );
}
