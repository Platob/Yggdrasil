import { Sidebar } from "@/components/sidebar";

export default function BotLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex bg-background text-foreground">
      <Sidebar />
      <main className="flex-1 ml-60 p-6 overflow-auto">
        {children}
      </main>
    </div>
  );
}
