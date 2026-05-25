import { MsgSidebar } from "@/components/msg-sidebar";

export const metadata = {
  title: "Yggdrasil — Messaging",
  description: "Real-time messaging across Yggdrasil channels",
};

export default function MsgLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen" style={{ background: "#050507" }}>
      <MsgSidebar />
      <main className="flex-1 ml-60 p-6 min-h-screen">
        {children}
      </main>
    </div>
  );
}
