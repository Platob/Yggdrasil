"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { YggLogoIcon } from "./YggLogo";

function ThemeToggle({ collapsed }: { collapsed: boolean }) {
  const [light, setLight] = useState(false);
  useEffect(() => { setLight(document.documentElement.getAttribute("data-theme") === "light"); }, []);
  const toggle = () => {
    const next = !light;
    setLight(next);
    document.documentElement.setAttribute("data-theme", next ? "light" : "dark");
    if (!next) document.documentElement.removeAttribute("data-theme");
    try { localStorage.setItem("ygg-theme", next ? "light" : "dark"); } catch { /* ignore */ }
  };
  return (
    <button
      onClick={toggle}
      title={light ? "Switch to dark" : "Switch to light"}
      className={`flex items-center gap-2 ${collapsed ? "justify-center w-full" : ""} text-muted hover:text-foreground transition-colors`}
    >
      {light ? (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" /></svg>
      ) : (
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
      )}
      {!collapsed && <span className="text-[11px] font-mono">{light ? "Light" : "Dark"}</span>}
    </button>
  );
}

const Icons = {
  excel: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="18" height="18" rx="2" />
      <path d="M3 9h18M3 15h18M9 3v18M15 3v18" />
    </svg>
  ),
  dashboard: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="9" rx="1" />
      <rect x="14" y="3" width="7" height="5" rx="1" />
      <rect x="14" y="12" width="7" height="9" rx="1" />
      <rect x="3" y="16" width="7" height="5" rx="1" />
    </svg>
  ),
  home: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
      <polyline points="9 22 9 12 15 12 15 22" />
    </svg>
  ),
  nodes: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="2" width="6" height="6" rx="1" />
      <rect x="16" y="2" width="6" height="6" rx="1" />
      <rect x="9" y="16" width="6" height="6" rx="1" />
      <path d="M5 8v3a1 1 0 001 1h4m4 0h4a1 1 0 001-1V8" />
      <path d="M12 12v4" />
    </svg>
  ),
  dags: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="5" cy="6" r="3" />
      <circle cx="19" cy="6" r="3" />
      <circle cx="12" cy="18" r="3" />
      <path d="M7.5 8l3 7M16.5 8l-3 7" />
    </svg>
  ),
  chat: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
    </svg>
  ),
  files: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
    </svg>
  ),
  metrics: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" />
      <line x1="12" y1="20" x2="12" y2="4" />
      <line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  ),
  topology: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="5" r="2" />
      <circle cx="5" cy="19" r="2" />
      <circle cx="19" cy="19" r="2" />
      <path d="M12 7v3M12 13L7 17M12 13l5 4" />
      <circle cx="12" cy="12" r="1" fill="currentColor" />
    </svg>
  ),
  functions: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 4h4a3 3 0 013 3M7 12h7" />
      <path d="M11 4c-2 0-3 1.2-3 4v8c0 2.8-1 4-3 4" />
    </svg>
  ),
  saga: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v6c0 1.66 3.58 3 8 3s8-1.34 8-3V5" />
      <path d="M4 11v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" />
    </svg>
  ),
  trading: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 17l5-5 4 3 6-7" />
      <path d="M16 5h5v5" />
      <line x1="3" y1="21" x2="21" y2="21" strokeOpacity="0.4" />
    </svg>
  ),
};

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: Icons.dashboard, exact: true },
  { href: "/", label: "Home", icon: Icons.home, exact: true },
  { href: "/metrics", label: "Metrics", icon: Icons.metrics },
  { href: "/nodes", label: "Nodes", icon: Icons.nodes },
  { href: "/topology", label: "Topology", icon: Icons.topology },
  { href: "/functions", label: "Functions", icon: Icons.functions },
  { href: "/dags", label: "DAGs", icon: Icons.dags },
  { href: "/chat", label: "Chat", icon: Icons.chat },
  { href: "/files", label: "Files", icon: Icons.files },
  { href: "/saga", label: "Saga", icon: Icons.saga },
  { href: "/trading", label: "Trading", icon: Icons.trading },
  { href: "/excel", label: "Excel", icon: Icons.excel },
];

export function Sidebar({ collapsed, onToggle }: { collapsed: boolean; onToggle: () => void }) {
  const pathname = usePathname();

  return (
    <aside className={`fixed left-0 top-0 h-full ${collapsed ? "w-[60px]" : "w-[200px]"} flex flex-col z-50 bg-sidebar-bg border-r border-sidebar-border transition-[width] duration-200`}>
      {/* Collapse toggle — straddles the right edge, vertically centered */}
      <button
        onClick={onToggle}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        className="absolute top-1/2 -right-3 -translate-y-1/2 z-50 w-6 h-6 rounded-full bg-card border border-border-accent text-muted hover:text-frost hover:border-frost/40 flex items-center justify-center shadow-md"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ transform: collapsed ? "rotate(180deg)" : "none" }}>
          <polyline points="15 18 9 12 15 6" />
        </svg>
      </button>

      {/* Logo header */}
      <div className={`flex items-center h-14 border-b border-sidebar-border shrink-0 ${collapsed ? "justify-center px-0" : "gap-2.5 px-4"}`}>
        <Link href="/" className="flex items-center gap-2.5" title="Yggdrasil">
          <YggLogoIcon size={26} />
          {!collapsed && (
            <div className="flex flex-col leading-tight">
              <span className="font-bold text-xs tracking-[0.15em] uppercase text-foreground">Yggdrasil</span>
              <span className="text-[8px] tracking-[0.2em] uppercase text-muted/70">Living Brain</span>
            </div>
          )}
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map((item) => {
          const active = item.exact ? pathname === item.href : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              title={item.label}
              className={`
                flex items-center ${collapsed ? "justify-center" : "gap-3"} px-3 py-2 rounded-lg text-sm font-medium transition-all duration-150
                ${active
                  ? "bg-frost/10 text-frost border-l-2 border-frost"
                  : "text-foreground-dim hover:text-foreground hover:bg-white/[0.03] border-l-2 border-transparent"}
              `}
            >
              <span className={active ? "text-frost" : "text-muted"}>{item.icon}</span>
              {!collapsed && item.label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom: theme toggle + status */}
      <div className={`py-3 border-t border-sidebar-border flex flex-col gap-2.5 ${collapsed ? "px-0 items-center" : "px-4"}`}>
        <ThemeToggle collapsed={collapsed} />
        <div className={`flex items-center gap-2 ${collapsed ? "justify-center" : ""}`} title="Connected">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          {!collapsed && <span className="text-[11px] text-muted font-mono">Connected</span>}
        </div>
      </div>
    </aside>
  );
}
