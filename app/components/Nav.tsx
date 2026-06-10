"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, MessageSquare, Activity, FolderOpen } from "lucide-react";

const links = [
  { href: "/", label: "Dashboard", icon: Activity },
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/analysis", label: "Analysis", icon: BarChart2 },
  { href: "/files", label: "Files", icon: FolderOpen },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav className="fixed left-0 top-0 h-full w-16 flex flex-col items-center py-6 gap-6 border-r border-[#1e1e2e] bg-[#0d0d14] z-50">
      <div className="mb-2">
        <span className="text-[#3b82f6] font-bold text-lg font-mono">YGG</span>
      </div>
      {links.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            title={label}
            className={`flex flex-col items-center gap-1 p-2 rounded-lg transition-colors group ${
              active
                ? "text-[#3b82f6] bg-[#1e1e2e]"
                : "text-gray-500 hover:text-gray-300 hover:bg-[#1a1a24]"
            }`}
          >
            <Icon size={20} />
            <span className="text-[9px] font-mono uppercase tracking-wider">{label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
