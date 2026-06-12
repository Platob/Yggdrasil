import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function fmt(value: number, decimals = 2): string {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function fmtPct(value: number): string {
  const sign = value >= 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}%`;
}

export function fmtCompact(value: number): string {
  if (Math.abs(value) >= 1e12) return `${(value / 1e12).toFixed(1)}T`;
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(2);
}

export function signalColor(direction: string): string {
  switch (direction) {
    case "strong_buy":  return "text-emerald-400";
    case "buy":         return "text-green-400";
    case "neutral":     return "text-slate-400";
    case "sell":        return "text-orange-400";
    case "strong_sell": return "text-red-400";
    default:            return "text-slate-400";
  }
}

export function signalBadge(direction: string): string {
  switch (direction) {
    case "strong_buy":  return "bg-emerald-500/20 text-emerald-300 border-emerald-500/30";
    case "buy":         return "bg-green-500/20 text-green-300 border-green-500/30";
    case "neutral":     return "bg-slate-500/20 text-slate-300 border-slate-500/30";
    case "sell":        return "bg-orange-500/20 text-orange-300 border-orange-500/30";
    case "strong_sell": return "bg-red-500/20 text-red-300 border-red-500/30";
    default:            return "bg-slate-500/20 text-slate-300 border-slate-500/30";
  }
}

export function signalLabel(direction: string): string {
  return direction.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
