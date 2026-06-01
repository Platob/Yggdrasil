// Client-side port of ``yggdrasil.data.schema`` — Schema (a struct-typed Field).
//
// PARITY: python/src/yggdrasil/data/schema.py (Schema == StructField). The
// table shape: an ordered list of fields, round-tripping an Arrow Schema.

import * as arrow from "apache-arrow";
import { Field, field, fieldFromArrow, fieldFromDict } from "./field";
import { DataType, StructType } from "./types";

export class Schema {
  readonly fields: Field[];

  constructor(fields: Iterable<Field | arrow.Field | string>) {
    this.fields = [...fields].map((f) =>
      f instanceof Field ? f : typeof f === "string" ? field(f, "string") : fieldFromArrow(f));
  }

  static from(obj: unknown): Schema {
    if (obj instanceof Schema) return obj;
    if (obj instanceof arrow.Schema) return new Schema(obj.fields);
    if (Array.isArray(obj)) return new Schema(obj as (Field | arrow.Field | string)[]);
    throw new Error(`Cannot parse Schema from ${String(obj)}`);
  }

  /** Rebuild from the canonical dict (a struct Field — parity with Python ``Schema.from_dict``). */
  static fromDict(d: Record<string, unknown>): Schema {
    const dtype = d.dtype as Record<string, unknown>;
    return new Schema((dtype.fields as Record<string, unknown>[]).map(fieldFromDict));
  }

  get names(): string[] { return this.fields.map((f) => f.name); }
  get length(): number { return this.fields.length; }

  field(nameOrIndex: string | number): Field | null {
    if (typeof nameOrIndex === "number") return this.fields[nameOrIndex] ?? null;
    return this.fields.find((f) => f.name === nameOrIndex) ?? null;
  }

  /** Project to a subset of columns (by name or index), preserving order given. */
  select(columns: (string | number)[]): Schema {
    return new Schema(columns.map((c) => {
      const f = this.field(c);
      if (!f) throw new Error(`column ${JSON.stringify(c)} not in schema`);
      return f;
    }));
  }

  /** As a struct DataType (the table's row type). */
  get dtype(): StructType { return new StructType(this.fields); }

  /** Canonical dict — a struct Field with empty name (parity with Python ``Schema.to_dict``). */
  toDict(): Record<string, unknown> {
    return { name: "", dtype: this.dtype.toDict(), nullable: false };
  }

  toArrow(): arrow.Schema { return new arrow.Schema(this.fields.map((f) => f.toArrow())); }

  toString(): string { return `Schema<${this.fields.map((f) => f.toString()).join(", ")}>`; }
}

/** Ergonomic factory (``schema([field("id","int64"), "name"])``). */
export function schema(fields: Iterable<Field | arrow.Field | string>): Schema {
  return new Schema(fields);
}

export { Field, field, DataType };
