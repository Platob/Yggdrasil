import type { Metadata, Viewport } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Yggdrasil",
  description: "Yggdrasil - The World Tree of Distributed Computing",
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
        <Sidebar />
        <main className="ml-[200px] min-h-screen">
          {children}
        </main>
      </body>
    </html>
  );
}
