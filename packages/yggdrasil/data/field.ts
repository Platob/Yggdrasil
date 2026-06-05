// Client-side port of ``yggdrasil.data.field`` — the Field value type.
//
// PARITY: python/src/yggdrasil/data/data_field.py. A named, typed, nullable
// column with optional metadata; round-trips an Apache Arrow Field both ways.

import * as arrow from "apache-arrow";
import { DataType, _registerFieldBridge } from "./types";

export class Field {
  constructor(
    readonly name: string,
    readonly dtype: DataType,
    readonly nullable = true,
    readonly metadata: Map<string, string> | null = null,
  ) {}

  static from(v: unknown): Field {
    if (v instanceof Field) return v;
    if (v instanceof arrow.Field) return fieldFromArrow(v);
    throw new Error(`Cannot parse Field from ${String(v)}`);
  }

  get children(): Field[] { return this.dtype.children; }

  toArrow(): arrow.Field {
    return new arrow.Field(this.name, this.dtype.toArrow(), this.nullable, this.metadata ?? undefined);
  }

  /** Canonical dict (parity with Python ``Field.to_dict``); ``metadata`` omitted when empty. */
  toDict(): Record<string, unknown> {
    const d: Record<string, unknown> = { name: this.name, dtype: this.dtype.toDict(), nullable: this.nullable };
    if (this.metadata && this.metadata.size) d.metadata = Object.fromEntries(this.metadata);
    return d;
  }

  static fromDict(d: Record<string, unknown>): Field { return fieldFromDict(d); }

  withName(name: string): Field { return new Field(name, this.dtype, this.nullable, this.metadata); }
  withDtype(dtype: DataType): Field { return new Field(this.name, DataType.from(dtype), this.nullable, this.metadata); }

  equals(other: Field, opts: { checkNames?: boolean; checkNullable?: boolean } = {}): boolean {
    if ((opts.checkNames ?? true) && this.name !== other.name) return false;
    if ((opts.checkNullable ?? false) && this.nullable !== other.nullable) return false;
    return this.dtype.equals(other.dtype);
  }

  toString(): string { return `${this.name}: ${this.dtype.name}`; }
}

/** Ergonomic factory (``field("id", "int64")`` / ``field("v", Float64Type())``). */
export function field(
  name: string,
  dtype: DataType | string | arrow.DataType,
  opts: { nullable?: boolean; metadata?: Record<string, string> | Map<string, string> } = {},
): Field {
  const meta = opts.metadata
    ? opts.metadata instanceof Map ? opts.metadata : new Map(Object.entries(opts.metadata))
    : null;
  return new Field(name, DataType.from(dtype), opts.nullable ?? true, meta);
}

export function fieldFromArrow(f: arrow.Field): Field {
  const meta = f.metadata && f.metadata.size ? new Map(f.metadata) : null;
  return new Field(f.name, DataType.fromArrow(f.type), f.nullable, meta);
}

export function fieldFromDict(d: Record<string, unknown>): Field {
  const meta = d.metadata ? new Map(Object.entries(d.metadata as Record<string, string>)) : null;
  return new Field(d.name as string, DataType.fromDict(d.dtype as Record<string, unknown>), (d.nullable as boolean) ?? true, meta);
}

export function makeField(name: string, dtype: DataType, nullable = true): Field {
  return new Field(name, dtype, nullable);
}

// Break the type<->field cycle: hand the bridge to ./types.
_registerFieldBridge({ fieldFromArrow, fieldFromDict, makeField });
