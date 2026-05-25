"use client";

import { useEffect, useState } from "react";
import { GlobalSidebar } from "./global-sidebar";

export function ServiceLayout({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  // Read initial state synchronously to avoid layout shift
  useEffect(() => {
    const saved = localStorage.getItem("ygg-sidebar-collapsed");
    if (saved === "true") setCollapsed(true);
  }, []);

  // Observe DOM mutations on the aside width as a shared signal
  useEffect(() => {
    const check = () => {
      const saved = localStorage.getItem("ygg-sidebar-collapsed");
      setCollapsed(saved === "true");
    };
    // Poll on animation frame — only active while tab is visible, very cheap
    let raf: number;
    const loop = () => { check(); raf = requestAnimationFrame(loop); };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);

  const marginLeft = collapsed ? "56px" : "224px";

  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <GlobalSidebar onCollapse={setCollapsed} />
      <main
        className="flex-1 min-w-0 p-6 overflow-auto"
        style={{ marginLeft, transition: "margin-left 200ms ease" }}
      >
        {children}
      </main>
    </div>
  );
}
