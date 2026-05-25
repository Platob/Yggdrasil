import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/sidebar";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Yggdrasil",
  description: "Yggdrasil - Distributed Bot Framework Dashboard",
  keywords: ["yggdrasil", "bot", "distributed", "python", "dashboard"],
};

export const viewport: Viewport = {
  themeColor: "#0c0c0f",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full bg-background`}>
      <body className="min-h-full flex bg-background text-foreground antialiased">
        <Sidebar />
        <main className="flex-1 ml-60 p-6 overflow-auto">
          {children}
        </main>
      </body>
    </html>
  );
}
