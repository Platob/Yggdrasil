import type { ReactNode } from "react";

interface StatusCardProps {
  label: string;
  value: ReactNode;
  sub?: string;
  accent?: boolean;
  loading?: boolean;
}

export default function StatusCard({ label, value, sub, accent, loading }: StatusCardProps) {
  return (
    <div className="rounded-xl border border-[#1e1e2e] bg-[#13131a] p-4 flex flex-col gap-2">
      <span className="text-xs font-mono uppercase tracking-widest text-gray-500">{label}</span>
      {loading ? (
        <div className="h-8 w-32 rounded bg-[#1e1e2e] animate-pulse" />
      ) : (
        <span
          className={`text-2xl font-mono font-bold truncate ${
            accent ? "text-[#60a5fa]" : "text-white"
          }`}
        >
          {value}
        </span>
      )}
      {sub && <span className="text-xs text-gray-600 font-mono">{sub}</span>}
    </div>
  );
}
