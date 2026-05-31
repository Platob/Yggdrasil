// Client-side port of ``yggdrasil.data.options.CastOptions``.
//
// PARITY: python/src/yggdrasil/data/options.py. Drives how a source is shaped
// into the target schema on read/write: projection, target schema, row limit,
// rechunking by row/byte size.
//
// NOTE: Apache Arrow's JS build ships no compute/cast kernels (unlike pyarrow),
// so ``castArrowTable`` performs the *structural* cast — project + reorder to
// the target columns and apply the row limit. Value-level type coercion
// (int→float, string→date, …) routes through polars and is a documented
// follow-on; the structural + projection + limit passes cover the IO read path.

import * as arrow from "apache-arrow";
import { Schema } from "./schema";

export interface CastOptionsArgs {
  target?: Schema | arrow.Schema | (string | arrow.Field)[] | null;
  source?: Schema | null;
  safe?: boolean;
  byteSize?: number | null;
  rowSize?: number | null;
  rowLimit?: number | null;
  columns?: string[] | null;
}

export class CastOptions {
  readonly target: Schema | null;
  readonly source: Schema | null;
  readonly safe: boolean;
  readonly byteSize: number | null;
  readonly rowSize: number | null;
  readonly rowLimit: number | null;
  readonly columns: string[] | null;

  constructor(a: CastOptionsArgs = {}) {
    this.target = a.target != null ? Schema.from(a.target) : null;
    this.source = a.source ?? null;
    this.safe = a.safe ?? false;
    this.byteSize = a.byteSize ?? null;
    this.rowSize = a.rowSize ?? null;
    this.rowLimit = a.rowLimit ?? null;
    this.columns = a.columns ?? null;
  }

  /** Normalize ``CastOptions | args | null`` into a CastOptions (mirrors ``check``). */
  static check(a?: CastOptions | CastOptionsArgs | null): CastOptions {
    return a instanceof CastOptions ? a : new CastOptions(a ?? {});
  }

  copy(overrides: CastOptionsArgs): CastOptions {
    return new CastOptions({
      target: overrides.target !== undefined ? overrides.target : this.target,
      source: overrides.source !== undefined ? overrides.source : this.source,
      safe: overrides.safe ?? this.safe,
      byteSize: overrides.byteSize !== undefined ? overrides.byteSize : this.byteSize,
      rowSize: overrides.rowSize !== undefined ? overrides.rowSize : this.rowSize,
      rowLimit: overrides.rowLimit !== undefined ? overrides.rowLimit : this.rowLimit,
      columns: overrides.columns !== undefined ? overrides.columns : this.columns,
    });
  }

  withTarget(target: Schema | arrow.Schema | (string | arrow.Field)[]): CastOptions {
    return this.copy({ target });
  }

  /** Columns a source reader should keep — explicit projection, else the target's. */
  readColumns(): string[] | null {
    return this.columns ?? this.target?.names ?? null;
  }

  /** Structural cast to the target (project + reorder) and apply the row limit. */
  castArrowTable(table: arrow.Table): arrow.Table {
    let out = table;
    const want = this.readColumns();
    if (want) {
      const present = want.filter((n) => table.schema.fields.some((f) => f.name === n));
      if (present.length) out = out.select(present);
    }
    return this.applyRowLimit(out);
  }

  castArrowBatch(batch: arrow.RecordBatch): arrow.RecordBatch {
    const want = this.readColumns();
    if (!want) return batch;
    const present = want.filter((n) => batch.schema.fields.some((f) => f.name === n));
    return present.length ? batch.select(present) : batch;
  }

  /** Post-read projection + row-limit pass (mirrors ``apply_post_read_table``). */
  applyPostReadTable(table: arrow.Table): arrow.Table {
    return this.castArrowTable(table);
  }

  private applyRowLimit(table: arrow.Table): arrow.Table {
    if (this.rowLimit != null && table.numRows > this.rowLimit) return table.slice(0, this.rowLimit);
    return table;
  }

  /** Rechunk a table into row-bounded batches (``rowSize``). */
  *rechunk(table: arrow.Table): Generator<arrow.RecordBatch> {
    const n = this.rowSize ?? table.numRows;
    if (!n || n >= table.numRows) { yield* table.batches; return; }
    for (let i = 0; i < table.numRows; i += n) {
      yield* table.slice(i, Math.min(i + n, table.numRows)).batches;
    }
  }
}

export type CastOptionsArg = CastOptions | CastOptionsArgs | null | undefined;
