import ChatPanel from "@/components/ChatPanel";

export default function ChatPage() {
  return (
    <div className="h-full max-w-5xl mx-auto flex flex-col">
      <h1 className="text-zinc-100 text-xl font-semibold mb-5">Chat</h1>
      <div className="flex-1 overflow-hidden">
        <ChatPanel />
      </div>
    </div>
  );
}
