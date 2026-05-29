"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { YggLogoIcon } from "./YggLogo";

const Icons = {
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
  trading: (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17" />
      <polyline points="16 7 22 7 22 13" />
    </svg>
  ),
};

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: Icons.dashboard, exact: true },
  { href: "/", label: "Home", icon: Icons.home, exact: true },
  { href: "/trading", label: "Trading", icon: Icons.trading },
  { href: "/metrics", label: "Metrics", icon: Icons.metrics },
  { href: "/nodes", label: "Nodes", icon: Icons.nodes },
  { href: "/topology", label: "Topology", icon: Icons.topology },
  { href: "/dags", label: "DAGs", icon: Icons.dags },
  { href: "/chat", label: "Chat", icon: Icons.chat },
  { href: "/files", label: "Files", icon: Icons.files },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-[200px] flex flex-col z-50 bg-sidebar-bg border-r border-sidebar-border">
      {/* Logo header */}
      <div className="flex items-center gap-2.5 h-14 px-4 border-b border-sidebar-border shrink-0">
        <Link href="/" className="flex items-center gap-2.5">
          <YggLogoIcon size={26} />
          <div className="flex flex-col leading-tight">
            <span className="font-bold text-xs tracking-[0.15em] uppercase text-foreground">
              Yggdrasil
            </span>
            <span className="text-[8px] tracking-[0.2em] uppercase text-muted/70">
              Living Brain
            </span>
          </div>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-4 px-2 space-y-1">
        {NAV_ITEMS.map((item) => {
          const active = item.exact
            ? pathname === item.href
            : pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium
                transition-all duration-150
                ${
                  active
                    ? "bg-frost/10 text-frost border-l-2 border-frost"
                    : "text-foreground-dim hover:text-foreground hover:bg-white/[0.03] border-l-2 border-transparent"
                }
              `}
            >
              <span className={active ? "text-frost" : "text-muted"}>
                {item.icon}
              </span>
              {item.label}
            </Link>
          );
        })}
      </nav>

      {/* Bottom status */}
      <div className="px-4 py-3 border-t border-sidebar-border">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full status-online" />
          <span className="text-[11px] text-muted font-mono">Connected</span>
        </div>
      </div>
    </aside>
  );
}
