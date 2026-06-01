// Shared formatting + column-type helpers for the tabular / analyze surfaces.

// Column dtype → numeric? Covers polars/arrow spellings: int*, float*, double,
// decimal, number, real (``num`` already matches ``number``).
export const NUMERIC = /int|float|double|decimal|num|real/i;

// Compact cell formatting for grids/pivots: null as a dot, numbers
// thousands-grouped to 4 dp, everything else stringified.
export function fmtCell(v: unknown): string {
  if (v == null) return "·";
  if (typeof v === "number") return Number(v).toLocaleString(undefined, { maximumFractionDigits: 4 });
  return String(v);
}

// Trigger a browser download of a Blob under ``filename`` (anchor + objectURL).
export function downloadBlob(blob: Blob, filename: string): void {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}
