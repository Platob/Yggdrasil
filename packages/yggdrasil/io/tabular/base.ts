// Client-side port of ``yggdrasil.io.tabular.base`` — the Tabular core.
//
// PARITY: python/src/yggdrasil/io/tabular/base.py. ``Tabular`` is the in-memory
// table value over an Apache Arrow ``Table`` (yggdrasil moves tabular data as
// Arrow IPC everywhere); ``TabularSource`` is the read/write interface the
// format leaves in ``../primitive`` and ``../nested`` implement.

import * as arrow from "apache-arrow";
import { Schema } from "../../data/schema";
import { CastOptions, type CastOptionsArg } from "../../data/options";
import { anyToArrowTable } from "../../arrow/cast";
import { MimeTypes, type MimeType } from "../../enums";

export class Tabular {
  constructor(readonly table: arrow.Table) {}

  /** Coerce any supported source (Arrow / arrays / rows / Tabular) + cast. */
  static from(obj: unknown, options?: CastOptionsArg): Tabular {
    return new Tabular(anyToArrowTable(obj, options));
  }

  /** Decode an Arrow IPC buffer (file or stream framing). */
  static fromArrowIPC(bytes: Uint8Array): Tabular {
    return new Tabular(arrow.tableFromIPC(bytes));
  }

  /** Wrap column arrays directly (``{ id: Int32Array, name: string[] }``). */
  static fromArrays(arrays: Record<string, unknown>): Tabular {
    return new Tabular(arrow.tableFromArrays(arrays as Parameters<typeof arrow.tableFromArrays>[0]));
  }

  /** yggdrasil Schema (struct-shaped) of the table. */
  get schema(): Schema { return new Schema(this.table.schema.fields); }
  get arrowSchema(): arrow.Schema { return this.table.schema; }

  /** The underlying Arrow Table — lets a Tabular flow through any ``toArrow()`` sink. */
  toArrow(): arrow.Table { return this.table; }

  get numRows(): number { return this.table.numRows; }
  get numCols(): number { return this.table.numCols; }
  get columnNames(): string[] { return this.table.schema.fields.map((f) => f.name); }

  /** The Arrow IPC stream media type — the transport contract. */
  get mediaType(): MimeType { return MimeTypes.ARROW_STREAM; }

  column(name: string) { return this.table.getChild(name); }
  slice(begin: number, end?: number): Tabular { return new Tabular(this.table.slice(begin, end)); }
  select(names: string[]): Tabular { return new Tabular(this.table.select(names)); }

  /** Apply CastOptions (structural projection/reorder to target + row limit). */
  cast(options?: CastOptionsArg): Tabular {
    return new Tabular(CastOptions.check(options).castArrowTable(this.table));
  }

  /** Serialize to Arrow IPC stream framing (the transport default). */
  toArrowIPC(): Uint8Array { return arrow.tableToIPC(this.table, "stream"); }
  /** Arrow IPC *file* framing (``ARROW1`` header, random access). */
  toArrowFile(): Uint8Array { return arrow.tableToIPC(this.table, "file"); }

  /** Materialise rows as plain objects (bounded — for small previews). */
  toArray(): Record<string, unknown>[] { return this.table.toArray().map((r) => ({ ...r })); }
}

/** Read/write interface for a tabular source (a file format leaf). */
export interface TabularSource {
  readArrowTable(options?: CastOptionsArg): Tabular;
}
