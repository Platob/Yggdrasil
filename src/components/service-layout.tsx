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

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "ygg-sidebar-collapsed") setCollapsed(e.newValue === "true");
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
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
