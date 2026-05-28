"use client";

interface ResourceBarProps {
  label: string;
  value: number;       // 0-100
  color: string;       // tailwind color token or CSS color
  detail?: string;     // e.g. "8.2 / 32 GB"
}

export function ResourceBar({ label, value, color, detail }: ResourceBarProps) {
  const clamped = Math.min(100, Math.max(0, value));
  // Determine bar color class based on thresholds
  const barColor =
    clamped > 90 ? "var(--rose)" :
    clamped > 70 ? "var(--amber)" :
    color;

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted uppercase tracking-wider font-medium">{label}</span>
        <div className="flex items-center gap-2">
          {detail && <span className="text-foreground-dim font-mono text-[11px]">{detail}</span>}
          <span className="font-mono font-semibold" style={{ color: barColor }}>
            {clamped.toFixed(1)}%
          </span>
        </div>
      </div>
      <div className="h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700 ease-out"
          style={{
            width: `${clamped}%`,
            background: `linear-gradient(90deg, ${barColor}, ${barColor}cc)`,
            boxShadow: `0 0 8px ${barColor}40`,
          }}
        />
      </div>
    </div>
  );
}
