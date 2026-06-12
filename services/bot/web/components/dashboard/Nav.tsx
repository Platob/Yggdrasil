"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { TrendingUp, BarChart3, BrainCircuit, Briefcase } from "lucide-react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/",          label: "Dashboard",  icon: BarChart3 },
  { href: "/portfolio", label: "Portfolio",  icon: Briefcase },
  { href: "/signals",   label: "Signals",    icon: TrendingUp },
  { href: "/ai",        label: "AI Analysis",icon: BrainCircuit },
];

export function Nav() {
  const path = usePathname();
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-slate-800/60 bg-[#0a0f1a]/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
        <Link href="/" className="flex items-center gap-2 text-indigo-400 font-bold text-lg">
          <TrendingUp className="h-5 w-5" />
          <span>YGG<span className="text-slate-300 font-normal"> Bot</span></span>
        </Link>
        <div className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
                path === href
                  ? "bg-indigo-500/20 text-indigo-300"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              )}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
