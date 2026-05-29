// Client helpers for the Excel task-pane add-in.
//
// The add-in runs inside the Excel process (loaded from the node's
// frontend origin) and talks to a Yggdrasil node directly over HTTP —
// the node's CORS is wide open. Tabular payloads come back as Arrow IPC
// and are decoded here with apache-arrow into a {headers, rows} grid
// the Office.js layer can drop straight onto a worksheet.

import { tableFromIPC } from "apache-arrow";

export interface NodeInfo {
  node_id: string;
  node_name: string;
  version: string;
  table_formats: string[];
  capabilities: string[];
}

export interface FsNode {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  children: FsNode[];
}

export interface Grid {
  headers: string[];
  rows: unknown[][];
}

const EXCEL_PREFIX = "/api/v2/excel";

export function normalizeBase(url: string): string {
  return (url || "http://127.0.0.1:8100").replace(/\/+$/, "");
}

export async function getInfo(base: string): Promise<NodeInfo> {
  const res = await fetch(`${normalizeBase(base)}${EXCEL_PREFIX}/info`);
  if (!res.ok) throw new Error(`node /info -> HTTP ${res.status}`);
  return res.json();
}

export async function getTree(base: string, path = "", depth = 1): Promise<FsNode[]> {
  const url = new URL(`${normalizeBase(base)}${EXCEL_PREFIX}/fs/tree`);
  url.searchParams.set("path", path);
  url.searchParams.set("depth", String(depth));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`node /fs/tree -> HTTP ${res.status}`);
  return (await res.json()).tree as FsNode[];
}

// Decode an Arrow IPC response body into a printable grid.
async function arrowToGrid(res: Response): Promise<Grid> {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  const table = tableFromIPC(buf);
  const headers = table.schema.fields.map((f) => f.name);
  const rows = table.toArray().map((r: Record<string, unknown>) =>
    headers.map((h) => {
      const v = r[h];
      // Arrow returns BigInt for 64-bit ints; Excel wants Number.
      return typeof v === "bigint" ? Number(v) : v;
    })
  );
  return { headers, rows };
}

export async function runPython(
  base: string,
  body: {
    code: string;
    env?: string | null;
    df_name?: string;
    packages?: string[];
    max_rows?: number | null;
    env_vars?: Record<string, string>;
  },
): Promise<Grid> {
  const res = await fetch(`${normalizeBase(base)}${EXCEL_PREFIX}/python?format=arrow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return arrowToGrid(res);
}

export async function readFile(base: string, path: string): Promise<Grid> {
  const url = new URL(`${normalizeBase(base)}${EXCEL_PREFIX}/fs/read`);
  url.searchParams.set("path", path);
  url.searchParams.set("format", "arrow");
  return arrowToGrid(await fetch(url.toString()));
}

// Write a grid back to a node file. We ship CSV (trivial to build in JS,
// and the node parses it) rather than encoding parquet in the browser.
export async function writeFileCsv(base: string, path: string, grid: Grid): Promise<{ rows: number; columns: number }> {
  const esc = (v: unknown) => {
    const s = v == null ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const csv = [grid.headers.map(esc).join(","), ...grid.rows.map((r) => r.map(esc).join(","))].join("\n");
  const url = new URL(`${normalizeBase(base)}${EXCEL_PREFIX}/fs/write`);
  url.searchParams.set("path", path);
  const res = await fetch(url.toString(), {
    method: "POST",
    headers: { "Content-Type": "text/csv" },
    body: csv,
  });
  if (!res.ok) throw new Error(`node /fs/write -> HTTP ${res.status}`);
  return res.json();
}

// ── Office.js worksheet bridge ──────────────────────────────────────────

// Sanitize + uniquify a worksheet name (Excel: <=31 chars, no \ / ? * [ ] :,
// and `add` throws on a duplicate name).
function sheetName(base: string, taken: string[]): string {
  const clean = (base.replace(/[\\/?*[\]:]/g, "_").trim() || "Yggdrasil").slice(0, 31);
  if (!taken.includes(clean)) return clean;
  for (let i = 2; i < 1000; i++) {
    const suffix = ` (${i})`;
    const candidate = clean.slice(0, 31 - suffix.length) + suffix;
    if (!taken.includes(candidate)) return candidate;
  }
  return clean.slice(0, 27) + " " + String(Date.now()).slice(-3);
}

// Drop a grid onto a fresh worksheet and autofit.
export async function gridToNewSheet(grid: Grid, name: string): Promise<void> {
  await Excel.run(async (ctx: any) => {
    const sheets = ctx.workbook.worksheets;
    sheets.load("items/name");
    await ctx.sync();
    const taken = (sheets.items ?? []).map((s: any) => s.name as string);
    const sheet = sheets.add(sheetName(name, taken));
    const nRows = grid.rows.length + 1;
    const nCols = Math.max(grid.headers.length, 1);
    const values = [grid.headers.length ? grid.headers : [""], ...grid.rows];
    const range = sheet.getRangeByIndexes(0, 0, nRows, nCols);
    range.values = values as unknown[][];
    sheet.getUsedRange().format.autofitColumns();
    sheet.activate();
    await ctx.sync();
  });
}

// Read the active worksheet's used range into a grid (first row = headers).
// Tolerates an empty sheet (getUsedRange throws otherwise).
export async function activeSheetToGrid(): Promise<Grid> {
  return Excel.run(async (ctx: any) => {
    const range = ctx.workbook.worksheets.getActiveWorksheet().getUsedRangeOrNullObject();
    range.load(["values", "isNullObject"]);
    await ctx.sync();
    if (range.isNullObject) return { headers: [], rows: [] };
    const values: unknown[][] = range.values ?? [];
    if (values.length === 0) return { headers: [], rows: [] };
    const [headers, ...rows] = values;
    return { headers: (headers as unknown[]).map(String), rows };
  });
}
