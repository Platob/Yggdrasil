"use client";

import { useState, useRef, useCallback } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { getChannels, getMessages, sendMessage } from "@/lib/api";
import ChannelList from "@/components/ChannelList";
import ChatThread from "@/components/ChatThread";
import { Send } from "lucide-react";

export default function ChatPage() {
  const [activeChannel, setActiveChannel] = useState<string>("general");
  const [text, setText] = useState("");
  const [sender, setSender] = useState("user");
  const inputRef = useRef<HTMLInputElement>(null);
  const queryClient = useQueryClient();

  const channelsQ = useQuery({
    queryKey: ["channels"],
    queryFn: getChannels,
    refetchInterval: 10000,
    retry: false,
  });

  const messagesQ = useQuery({
    queryKey: ["messages", activeChannel],
    queryFn: () => getMessages(activeChannel, 50),
    refetchInterval: 3000,
    retry: false,
  });

  const sendMut = useMutation({
    mutationFn: () => sendMessage(text.trim(), sender || "user", activeChannel),
    onSuccess: () => {
      setText("");
      queryClient.invalidateQueries({ queryKey: ["messages", activeChannel] });
      inputRef.current?.focus();
    },
  });

  const handleSend = useCallback(() => {
    if (!text.trim()) return;
    sendMut.mutate();
  }, [text, sendMut]);

  const channels = channelsQ.data ?? [];
  // Ensure activeChannel appears in the list (fallback if API returns none)
  const displayChannels =
    channels.length > 0
      ? channels
      : [{ name: "general" }, { name: "alerts" }, { name: "system" }];

  return (
    <div className="h-full flex overflow-hidden">
      {/* Sidebar */}
      <aside className="w-52 border-r border-[#1e1e2e] bg-[#0d0d14] flex flex-col shrink-0">
        <ChannelList
          channels={displayChannels}
          active={activeChannel}
          onSelect={setActiveChannel}
          loading={channelsQ.isLoading}
        />
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Channel header */}
        <div className="h-12 border-b border-[#1e1e2e] flex items-center px-4 gap-2 shrink-0">
          <span className="text-gray-500 text-sm">#</span>
          <span className="font-mono font-medium text-white text-sm">{activeChannel}</span>
          {messagesQ.isFetching && (
            <span className="text-[10px] font-mono text-gray-600 ml-auto">refreshing...</span>
          )}
        </div>

        {/* Error banner */}
        {messagesQ.isError && (
          <div className="mx-4 mt-3 rounded-lg border border-red-800 bg-red-950/30 px-3 py-2 text-red-400 text-xs font-mono">
            Failed to load messages — backend may be unreachable
          </div>
        )}

        {/* Messages */}
        <ChatThread
          messages={messagesQ.data ?? []}
          loading={messagesQ.isLoading}
          error={null}
        />

        {/* Input area */}
        <div className="border-t border-[#1e1e2e] p-4 flex flex-col gap-2 shrink-0">
          <div className="flex gap-2 items-center">
            <input
              type="text"
              value={sender}
              onChange={(e) => setSender(e.target.value)}
              placeholder="sender"
              className="w-24 px-2 py-1.5 rounded-lg bg-[#1a1a24] border border-[#1e1e2e] text-xs font-mono text-gray-300 placeholder-gray-600 focus:outline-none focus:border-[#3b82f6]"
            />
            <span className="text-gray-600 text-sm">in #{activeChannel}</span>
          </div>
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Type a message… (Enter to send)"
              className="flex-1 px-4 py-2.5 rounded-xl bg-[#1a1a24] border border-[#1e1e2e] text-sm font-mono text-gray-200 placeholder-gray-600 focus:outline-none focus:border-[#3b82f6] transition-colors"
            />
            <button
              onClick={handleSend}
              disabled={!text.trim() || sendMut.isPending}
              className="px-4 py-2.5 rounded-xl bg-[#3b82f6] text-white text-sm font-mono font-medium hover:bg-[#2563eb] transition-colors disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Send size={15} />
              {sendMut.isPending ? "…" : "Send"}
            </button>
          </div>
          {sendMut.isError && (
            <span className="text-red-400 text-xs font-mono">
              Failed to send: backend unreachable
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
