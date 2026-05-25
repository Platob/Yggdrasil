"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { YggdrasilBrand } from "./logo";

const NAV_ITEMS = [
  { 
    href: "/", 
    label: "Dashboard", 
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
    ),
    description: "Node overview"
  },
  { 
    href: "/execute", 
    label: "Execute", 
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="4 17 10 11 4 5" />
        <line x1="12" y1="19" x2="20" y2="19" />
      </svg>
    ),
    description: "Run code"
  },
  { 
    href: "/chat", 
    label: "Chat", 
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
      </svg>
    ),
    description: "Messaging"
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-full w-60 bg-card border-r border-border flex flex-col z-50">
      {/* Logo Section */}
      <div className="p-5 border-b border-border">
        <Link href="/" className="block">
          <YggdrasilBrand size={28} />
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-3">
        <div className="space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-all duration-200 ${
                  active
                    ? "bg-primary/10 text-primary border border-primary/20"
                    : "text-muted hover:text-foreground hover:bg-card-hover border border-transparent"
                }`}
              >
                <span className={`transition-colors ${active ? "text-primary" : "text-muted group-hover:text-foreground"}`}>
                  {item.icon}
                </span>
                <div className="flex flex-col">
                  <span className="font-medium">{item.label}</span>
                  <span className={`text-[10px] ${active ? "text-primary/70" : "text-muted"}`}>
                    {item.description}
                  </span>
                </div>
              </Link>
            );
          })}
        </div>
      </nav>

      {/* Decorative runic divider */}
      <div className="px-5 py-2">
        <div className="flex items-center gap-2 text-border-accent">
          <div className="flex-1 h-px bg-border" />
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted">
            <path d="M12 2L15 8L22 9L17 14L18 21L12 18L6 21L7 14L2 9L9 8L12 2Z" />
          </svg>
          <div className="flex-1 h-px bg-border" />
        </div>
      </div>

      {/* Status Section */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-2 mb-3">
          <div className="status-dot online" />
          <span className="text-xs text-muted">Connected</span>
        </div>
        <div className="flex items-center justify-between">
          <p className="text-[10px] text-muted font-mono tracking-wider">YGGDRASIL</p>
          <p className="text-[10px] text-primary font-mono">v0.1.0</p>
        </div>
      </div>
    </aside>
  );
}
