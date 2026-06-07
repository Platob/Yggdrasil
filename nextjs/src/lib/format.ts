function tsToDate(ts: number): Date {
  // Backend timestamps may be in seconds or milliseconds — normalize to ms.
  return new Date(ts < 1e12 ? ts * 1000 : ts);
}

export function fmtPrice(n: number, decimals = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `$${n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

export function fmtNum(n: number, decimals = 2): string {
  if (n == null || Number.isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtPnl(n: number): string {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n >= 0 ? "+" : "-";
  return `${sign}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

export function fmtPct(n: number): string {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

export function fmtTime(ts: number): string {
  return tsToDate(ts).toLocaleTimeString("en-US", { hour12: false });
}

export function fmtDate(ts: number): string {
  const d = tsToDate(ts);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

export function pnlColor(n: number): string {
  if (n == null || Number.isNaN(n)) return "text-gray-400";
  return n >= 0 ? "text-green-400" : "text-red-400";
}
