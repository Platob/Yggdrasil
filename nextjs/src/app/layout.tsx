import type { Metadata, Viewport } from "next";
import "./globals.css";
import { AppShell } from "@/components/AppShell";

export const metadata: Metadata = {
  title: "Yggdrasil",
  description: "Yggdrasil - A Living Brain of Distributed Computing",
  keywords: ["yggdrasil", "distributed", "nodes", "python", "computing"],
};

export const viewport: Viewport = {
  themeColor: "#050510",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-foreground antialiased noise-bg">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
