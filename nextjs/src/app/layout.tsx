import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "Yggdrasil Trading",
  description: "Real-time trading dashboard powered by Yggdrasil + Loki AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ display: "flex", minHeight: "100vh" }}>
        <Nav />
        <main style={{ flex: 1, overflow: "auto" }}>{children}</main>
      </body>
    </html>
  );
}
