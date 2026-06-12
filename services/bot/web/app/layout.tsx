import type { Metadata } from "next";
import "./globals.css";
import { Providers } from "./providers";
import { Nav } from "@/components/dashboard/Nav";

export const metadata: Metadata = {
  title: "YGG Bot — Trading Dashboard",
  description: "AI-powered trading signals and portfolio management",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#0a0f1a] text-slate-100">
        <Providers>
          <Nav />
          <main className="pt-16">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
