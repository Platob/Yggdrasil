import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "YGG Trading",
  description: "Yggdrasil trading terminal",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-200 antialiased">
        <Sidebar />
        <main className="md:ml-[220px] min-h-screen">{children}</main>
      </body>
    </html>
  );
}
