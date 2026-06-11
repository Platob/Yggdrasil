"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV = [
  { href: "/", label: "Dashboard", icon: "⊞" },
  { href: "/trading", label: "Trading", icon: "📈" },
  { href: "/scan", label: "Signal Scan", icon: "⚡" },
  { href: "/portfolio", label: "Portfolio", icon: "◎" },
  { href: "/fx", label: "FX Rates", icon: "💱" },
  { href: "/analysis", label: "Analysis", icon: "🔬" },
  { href: "/files", label: "Files", icon: "📁" },
  { href: "/saga", label: "Saga SQL", icon: "⚙" },
];

export function Sidebar() {
  const path = usePathname();
  return (
    <nav className="w-52 bg-gray-900 border-r border-gray-800 flex flex-col py-6 px-3 gap-1 shrink-0">
      <div className="px-3 mb-6 text-lg font-bold text-emerald-400">Yggdrasil</div>
      {NAV.map((n) => (
        <Link key={n.href} href={n.href}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors
            ${path === n.href ? "bg-emerald-900/50 text-emerald-300" : "text-gray-400 hover:text-gray-100 hover:bg-gray-800"}`}>
          <span>{n.icon}</span>{n.label}
        </Link>
      ))}
    </nav>
  );
}
