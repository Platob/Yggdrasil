// Decode an Arrow IPC stream (the backend's fast tabular wire) into a plain
// {columns, rows} shape the grid can render. Int64 comes back as BigInt in
// arrow-js — coerced to Number when safe, else a string, so React/JSON stay
// happy.
import { tableFromIPC } from "apache-arrow";
import type { TabularCell } from "./api";

export interface DecodedTable {
  columns: { name: string; type: string }[];
  rows: TabularCell[][];
  numRows: number;
}

function normalizeCell(v: unknown): TabularCell {
  if (v === null || v === undefined) return null;
  if (typeof v === "bigint") return Number.isSafeInteger(Number(v)) ? Number(v) : v.toString();
  if (typeof v === "number" || typeof v === "string" || typeof v === "boolean") return v;
  if (v instanceof Date) return v.toISOString();
  return String(v);
}

export async function fetchArrowTable(url: string): Promise<DecodedTable> {
  const res = await fetch(url);
  if (!res.ok) {
    let detail = "";
    try { detail = (await res.json())?.detail ?? ""; } catch { /* not json */ }
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ""}`);
  }
  const table = tableFromIPC(new Uint8Array(await res.arrayBuffer()));
  const columns = table.schema.fields.map((f) => ({ name: f.name, type: String(f.type) }));
  const colVectors = columns.map((_, i) => table.getChildAt(i));
  const rows: TabularCell[][] = [];
  for (let r = 0; r < table.numRows; r++) {
    const row: TabularCell[] = [];
    for (let c = 0; c < colVectors.length; c++) row.push(normalizeCell(colVectors[c]?.get(r)));
    rows.push(row);
  }
  return { columns, rows, numRows: table.numRows };
}
