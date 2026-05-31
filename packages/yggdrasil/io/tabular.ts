// Client-side port of ``yggdrasil.io`` tabular core — the unified data-typing
// entry point.
//
// PARITY: python/src/yggdrasil/io/tabular/. Yggdrasil moves tabular data as
// Arrow IPC everywhere (``application/vnd.apache.arrow.stream``); this is the
// JS side of that contract — a thin ``Tabular`` over an Apache Arrow ``Table``.
// ``apache-arrow`` is a peer dependency (the consumer provides it — the Next app
// already does). A polars adapter (nodejs-polars / WASM) is a follow-on; Arrow
// is the stable cross-language wire format both speak.

import { Table, Schema, tableFromIPC, tableFromArrays, tableToIPC } from "apache-arrow";
import { MimeTypes, type MimeType } from "../enums";

export class Tabular {
  constructor(readonly table: Table) {}

  /** Decode an Arrow IPC buffer (file or stream framing) into a Tabular. */
  static fromArrowIPC(bytes: Uint8Array): Tabular {
    return new Tabular(tableFromIPC(bytes));
  }

  /** Wrap column arrays directly (``{ id: Int32Array, name: string[] }``). */
  static fromArrays(arrays: Record<string, unknown>): Tabular {
    return new Tabular(tableFromArrays(arrays as Parameters<typeof tableFromArrays>[0]));
  }

  /** Serialize to Arrow IPC stream framing (the yggdrasil transport default). */
  toArrowIPC(): Uint8Array {
    return tableToIPC(this.table, "stream");
  }

  /** Arrow IPC *file* framing (random-access, ``ARROW1`` header). */
  toArrowFile(): Uint8Array {
    return tableToIPC(this.table, "file");
  }

  get numRows(): number { return this.table.numRows; }
  get numCols(): number { return this.table.numCols; }
  get schema(): Schema { return this.table.schema; }
  get columnNames(): string[] { return this.table.schema.fields.map((f) => f.name); }

  /** The Arrow IPC stream media type — the transport contract. */
  get mediaType(): MimeType { return MimeTypes.ARROW_STREAM; }

  column(name: string) { return this.table.getChild(name); }

  slice(begin: number, end?: number): Tabular {
    return new Tabular(this.table.slice(begin, end));
  }

  select(names: string[]): Tabular {
    return new Tabular(this.table.select(names));
  }

  /** Materialise rows as plain objects (bounded — for small previews). */
  toArray(): Record<string, unknown>[] {
    return this.table.toArray().map((r) => ({ ...r }));
  }
}
