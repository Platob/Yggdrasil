import type { Metadata } from "next";
import "./globals.css";
import Nav from "@/components/Nav";

export const metadata: Metadata = {
  title: "YGG Trading",
  description: "Yggdrasil trading dashboard — FX rates, analysis, AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div style={{ display: "flex", minHeight: "100vh" }}>
          <Nav />
          <main style={{ flex: 1, padding: "24px", overflowY: "auto" }}>{children}</main>
        </div>
      </body>
    </html>
  );
}
