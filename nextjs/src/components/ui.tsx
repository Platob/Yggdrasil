"use client";

import { pnlColor } from "@/lib/format";

export function Panel({
  title,
  children,
  className = "",
  action,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}) {
  return (
    <div
      className={`bg-gray-900 border border-gray-800 rounded-lg ${className}`}
    >
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <h2 className="text-sm font-medium text-gray-300">{title}</h2>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function Kpi({
  label,
  value,
  sub,
  colorBySign,
  signValue,
}: {
  label: string;
  value: string;
  sub?: string;
  colorBySign?: boolean;
  signValue?: number;
}) {
  const valueColor =
    colorBySign && signValue !== undefined ? pnlColor(signValue) : "text-gray-100";
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg px-4 py-3">
      <div className="text-xs text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`mt-1 text-2xl font-mono font-semibold ${valueColor}`}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-xs text-gray-500 font-mono">{sub}</div>}
    </div>
  );
}

export function SideBadge({ side }: { side: string }) {
  const buy = side?.toLowerCase() === "buy" || side?.toLowerCase() === "long";
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium uppercase ${
        buy ? "bg-green-500/15 text-green-400" : "bg-red-500/15 text-red-400"
      }`}
    >
      {side}
    </span>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-red-900 bg-red-950/40 px-4 py-3 text-sm text-red-300">
      {message}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-gray-500 text-sm py-8 justify-center">
      <span className="inline-block w-3 h-3 border-2 border-gray-600 border-t-green-400 rounded-full animate-spin" />
      {label}
    </div>
  );
}
