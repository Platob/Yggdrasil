// Client-side port of ``yggdrasil.arrow.cast`` — the "any source -> Arrow"
// entry points.
//
// PARITY: python/src/yggdrasil/arrow/cast.py. Coerces the shapes a JS caller
// has — an Arrow Table, column arrays (``{ id: Int32Array, name: string[] }``),
// or row objects (``[{ id, name }, …]``) — into an Apache Arrow Table, then
// applies CastOptions (projection + target + row limit). polars frames are
// accepted when nodejs-polars is present (its ``.toArrow()``); that adapter is
// the documented compute/casting bridge.

import * as arrow from "apache-arrow";
import { CastOptions, type CastOptionsArg } from "../data/options";

type RowList = Record<string, unknown>[];
type ColumnArrays = Record<string, unknown>;

function isArrowTable(o: unknown): o is arrow.Table {
  return o instanceof arrow.Table;
}
function hasToArrow(o: unknown): o is { toArrow(): arrow.Table } {
  return !!o && typeof (o as { toArrow?: unknown }).toArrow === "function";
}

/** Coerce any supported source into an Arrow Table, then apply ``options``. */
export function anyToArrowTable(obj: unknown, options?: CastOptionsArg): arrow.Table {
  const opts = CastOptions.check(options);
  let table: arrow.Table;

  if (isArrowTable(obj)) {
    table = obj;
  } else if (obj instanceof arrow.RecordBatch) {
    table = new arrow.Table(obj.schema, obj);
  } else if (hasToArrow(obj)) {
    // polars DataFrame / yggdrasil Tabular / anything Arrow-yielding.
    table = obj.toArrow();
  } else if (Array.isArray(obj)) {
    table = tableFromRows(obj as RowList);
  } else if (obj && typeof obj === "object") {
    table = arrow.tableFromArrays(obj as ColumnArrays as Parameters<typeof arrow.tableFromArrays>[0]);
  } else {
    throw new Error("anyToArrowTable: unsupported source");
  }

  return opts.castArrowTable(table);
}

/** Stream the source as Arrow record batches (rechunked by ``rowSize``). */
export function* anyToArrowBatches(obj: unknown, options?: CastOptionsArg): Generator<arrow.RecordBatch> {
  const opts = CastOptions.check(options);
  yield* opts.rechunk(anyToArrowTable(obj, options));
}

/** Apply a target schema (structural) + projection + row limit to a Table. */
export function castArrowTable(table: arrow.Table, options?: CastOptionsArg): arrow.Table {
  return CastOptions.check(options).castArrowTable(table);
}

/** Build an Arrow Table from row objects (``[{ id: 1, name: "a" }, …]``). */
export function tableFromRows(rows: RowList): arrow.Table {
  if (rows.length === 0) return new arrow.Table();
  const keys = Object.keys(rows[0]);
  const cols: ColumnArrays = {};
  for (const k of keys) cols[k] = rows.map((r) => r[k]);
  return arrow.tableFromArrays(cols as Parameters<typeof arrow.tableFromArrays>[0]);
}
