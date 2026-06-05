"use client";

export function ConfirmModal({
  open,
  title,
  message,
  onConfirm,
  onCancel,
  danger = false,
}: {
  open: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  danger?: boolean;
}) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[var(--modal-scrim)] backdrop-blur-sm">
      <div className="modal-surface p-6 max-w-md w-full mx-4">
        <h2 className="text-lg font-semibold mb-2">{title}</h2>
        <p className="text-sm text-foreground-dim mb-6">{message}</p>
        <div className="flex justify-end gap-2">
          <button
            onClick={onCancel}
            className="px-4 py-2 text-sm rounded-lg bg-white/[0.04] hover:bg-white/[0.06] border border-white/[0.08]"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`px-4 py-2 text-sm rounded-lg ${
              danger
                ? "bg-rose-500/20 text-rose-300 border border-rose-500/30 hover:bg-rose-500/30"
                : "bg-frost/20 text-frost border border-frost/30 hover:bg-frost/30"
            }`}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  );
}
