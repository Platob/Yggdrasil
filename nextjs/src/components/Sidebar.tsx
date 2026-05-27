"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { YggLogoIcon } from "./YggLogo";

const Icons = {
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
};

const NAV_ITEMS = [
  { href: "/", label: "Home", icon: Icons.home, exact: true },
  { href: "/nodes", label: "Nodes", icon: Icons.nodes },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-[200px] flex flex-col z-50 bg-sidebar-bg border-r border-sidebar-border">
      {/* Logo header */}
      <div className="flex items-center gap-2.5 h-14 px-4 border-b border-sidebar-border shrink-0">
        <Link href="/" className="flex items-center gap-2.5">
          <YggLogoIcon size={26} />
          <span className="font-bold text-xs tracking-[0.15em] uppercase text-foreground">
            Yggdrasil
          </span>
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
