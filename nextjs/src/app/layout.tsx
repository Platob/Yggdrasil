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
      <head>
        {/* Apply the saved theme before paint to avoid a flash. Default dark. */}
        <script
          dangerouslySetInnerHTML={{
            __html: `try{if(localStorage.getItem('ygg-theme')==='light')document.documentElement.setAttribute('data-theme','light');}catch(e){}`,
          }}
        />
      </head>
      <body className="min-h-screen bg-background text-foreground antialiased noise-bg">
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
