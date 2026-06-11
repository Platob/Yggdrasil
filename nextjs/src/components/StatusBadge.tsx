type Status = "online" | "offline" | "loading";

export function StatusBadge({ status, label }: { status: Status; label?: string }) {
  const color =
    status === "online" ? "bg-emerald-400" : status === "offline" ? "bg-red-400" : "bg-yellow-400";
  return (
    <div className="flex items-center gap-2">
      <span className={`w-2 h-2 rounded-full ${color}`} />
      <span className="text-sm text-gray-400">{label ?? `Node ${status}`}</span>
    </div>
  );
}
