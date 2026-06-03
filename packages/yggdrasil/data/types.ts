// Client-side port of ``yggdrasil.data.types`` — the DataType hierarchy.
//
// PARITY: python/src/yggdrasil/data/types/. A unified, named, parametrized type
// system that round-trips Apache Arrow types both ways, so a column typed
// ``int64`` / ``decimal(38,18)`` / ``list<int32>`` means the same thing in
// Python and JS. ``apache-arrow`` is a peer dependency. Type-id integer codes
// match the Python ``DataTypeId`` enum for stable cross-language serialization.

import * as arrow from "apache-arrow";
import type { Field } from "./field";

export enum DataTypeId {
  OBJECT = 0, NULL = 1, BOOL = 10,
  INTEGER = 20, INT8 = 21, INT16 = 22, INT32 = 23, INT64 = 24,
  UINT8 = 26, UINT16 = 27, UINT32 = 28, UINT64 = 29,
  FLOAT = 40, FLOAT16 = 42, FLOAT32 = 43, FLOAT64 = 44,
  DECIMAL = 50,
  DATE = 60, TIME = 61, TIMESTAMP = 62, DURATION = 63,
  BINARY = 70, STRING = 71,
  ARRAY = 100, MAP = 101, STRUCT = 102,
}

export type TimeUnit = "s" | "ms" | "us" | "ns";
const ARROW_UNIT: Record<TimeUnit, arrow.TimeUnit> = {
  s: arrow.TimeUnit.SECOND, ms: arrow.TimeUnit.MILLISECOND,
  us: arrow.TimeUnit.MICROSECOND, ns: arrow.TimeUnit.NANOSECOND,
};
const UNIT_NAME: Record<number, TimeUnit> = {
  [arrow.TimeUnit.SECOND]: "s", [arrow.TimeUnit.MILLISECOND]: "ms",
  [arrow.TimeUnit.MICROSECOND]: "us", [arrow.TimeUnit.NANOSECOND]: "ns",
};

export abstract class DataType {
  abstract readonly id: DataTypeId;
  abstract readonly name: string;
  abstract toArrow(): arrow.DataType;

  get children(): Field[] { return []; }
  get isNumeric(): boolean { return this.id >= 20 && this.id < 60; }
  get isNested(): boolean { return this.id >= 100; }

  /** The DataTypeId enum member name (``INT64``) — the cross-language dict ``name``. */
  get typeName(): string { return DataTypeId[this.id]; }

  toString(): string { return this.name; }

  equals(other: DataType): boolean {
    return this.name === other.name;
  }

  /**
   * The canonical, language-neutral dict (``{ id, name, ...params }``) — the
   * exact shape of Python ``DataType.to_dict``; round-trips via ``fromDict``.
   * This (not the JS-side display ``name``) is the cross-language wire form.
   */
  toDict(): Record<string, unknown> {
    return { id: this.id, name: this.typeName, ...this.dictParams() };
  }

  /** Per-type parameters appended to ``toDict`` (byte_size, unit, fields, …). */
  protected dictParams(): Record<string, unknown> { return {}; }

  /** Coerce a string / arrow type / DataType into a DataType. */
  static from(v: unknown): DataType {
    if (v instanceof DataType) return v;
    if (typeof v === "string") return DataType.fromString(v);
    if (v instanceof arrow.DataType) return DataType.fromArrow(v);
    throw new Error(`Cannot parse DataType from ${String(v)}`);
  }

  /** Rebuild a DataType from its canonical dict (parity with Python ``from_dict``). */
  static fromDict(d: Record<string, unknown>): DataType {
    const id = d.id as DataTypeId;
    switch (id) {
      case DataTypeId.NULL: return new NullType();
      case DataTypeId.BOOL: return new BoolType();
      case DataTypeId.INT8: case DataTypeId.INT16: case DataTypeId.INT32: case DataTypeId.INT64:
      case DataTypeId.UINT8: case DataTypeId.UINT16: case DataTypeId.UINT32: case DataTypeId.UINT64:
        return new IntegerType(d.byte_size as number, d.signed as boolean);
      case DataTypeId.FLOAT16: case DataTypeId.FLOAT32: case DataTypeId.FLOAT64:
        return new FloatType(d.byte_size as number);
      case DataTypeId.DECIMAL: return new DecimalType(d.precision as number, d.scale as number);
      case DataTypeId.STRING: return new StringType();
      case DataTypeId.BINARY: return new BinaryType();
      case DataTypeId.DATE: return new DateType();
      case DataTypeId.TIME: return new TimeType(d.unit as TimeUnit);
      case DataTypeId.TIMESTAMP: return new TimestampType(d.unit as TimeUnit, (d.tz as string | null) ?? null);
      case DataTypeId.DURATION: return new DurationType(d.unit as TimeUnit);
      case DataTypeId.ARRAY: return new ListType(bridge().fieldFromDict(d.item_field as Record<string, unknown>));
      case DataTypeId.STRUCT: return new StructType((d.fields as Record<string, unknown>[]).map(bridge().fieldFromDict));
      case DataTypeId.MAP: {
        const entries = (d.item_field as Record<string, unknown>).dtype as Record<string, unknown>;
        const [k, v] = entries.fields as Record<string, unknown>[];
        return new MapType(bridge().fieldFromDict(k), bridge().fieldFromDict(v));
      }
      default: throw new Error(`Cannot rebuild DataType from id ${String(id)}`);
    }
  }

  /** Build a DataType from an Apache Arrow type. */
  static fromArrow(t: arrow.DataType): DataType {
    switch (t.typeId) {
      case arrow.Type.Null: return new NullType();
      case arrow.Type.Bool: return new BoolType();
      case arrow.Type.Int: {
        const it = t as arrow.Int;
        return new IntegerType(it.bitWidth / 8, it.isSigned);
      }
      case arrow.Type.Float: {
        const ft = t as arrow.Float;
        return new FloatType(ft.precision === arrow.Precision.HALF ? 2 : ft.precision === arrow.Precision.SINGLE ? 4 : 8);
      }
      case arrow.Type.Decimal: {
        const dt = t as arrow.Decimal;
        return new DecimalType(dt.precision, dt.scale);
      }
      case arrow.Type.Utf8: case arrow.Type.LargeUtf8: return new StringType();
      case arrow.Type.Binary: case arrow.Type.LargeBinary: case arrow.Type.FixedSizeBinary: return new BinaryType();
      case arrow.Type.Date: return new DateType();
      case arrow.Type.Time: return new TimeType(UNIT_NAME[(t as arrow.Time).unit] ?? "us");
      case arrow.Type.Timestamp: {
        const ts = t as arrow.Timestamp;
        return new TimestampType(UNIT_NAME[ts.unit] ?? "us", ts.timezone ?? null);
      }
      case arrow.Type.Duration: return new DurationType(UNIT_NAME[(t as arrow.Duration).unit] ?? "us");
      case arrow.Type.List: case arrow.Type.FixedSizeList:
        return new ListType(bridge().fieldFromArrow((t as arrow.List).children[0]));
      case arrow.Type.Struct:
        return new StructType((t as arrow.Struct).children.map(bridge().fieldFromArrow));
      case arrow.Type.Map: {
        const entries = (t as arrow.Map_).children[0].type as arrow.Struct;
        return new MapType(bridge().fieldFromArrow(entries.children[0]), bridge().fieldFromArrow(entries.children[1]));
      }
      default: return new StringType();
    }
  }

  /** Parse a type token: ``int64``, ``decimal(38,18)``, ``timestamp[us,UTC]``, ``list<int32>``. */
  static fromString(s: string): DataType {
    const t = s.trim().toLowerCase();
    const prim = PRIMITIVE_PARSERS[t];
    if (prim) return prim();
    let m = /^decimal\(?\s*(\d+)\s*,\s*(\d+)\s*\)?$/.exec(t);
    if (m) return new DecimalType(parseInt(m[1], 10), parseInt(m[2], 10));
    m = /^timestamp\[?\s*([a-z]+)\s*(?:,\s*([^\]]+))?\]?$/.exec(t);
    if (m) return new TimestampType(m[1] as TimeUnit, m[2] ? m[2].trim() : null);
    m = /^(?:list|array)<(.+)>$/.exec(t);
    if (m) return new ListType(bridge().makeField("item", DataType.fromString(m[1])));
    throw new Error(`Cannot parse DataType from ${JSON.stringify(s)}`);
  }
}

// Small bridge to ../field — breaks the type<->field import cycle without a
// dynamic import. ``./field`` registers itself at load time.
interface FieldBridge {
  fieldFromArrow(f: arrow.Field): Field;
  fieldFromDict(d: Record<string, unknown>): Field;
  makeField(name: string, dtype: DataType, nullable?: boolean): Field;
}
let _fieldBridge: FieldBridge | null = null;
export function _registerFieldBridge(b: FieldBridge): void { _fieldBridge = b; }
function bridge(): FieldBridge {
  if (!_fieldBridge) throw new Error("yggdrasil/data: ./field not loaded (import it before nested DataType ops)");
  return _fieldBridge;
}

// --- primitives --------------------------------------------------------------

export class NullType extends DataType { readonly id = DataTypeId.NULL; readonly name = "null"; toArrow() { return new arrow.Null(); } }
export class BoolType extends DataType { readonly id = DataTypeId.BOOL; readonly name = "bool"; toArrow() { return new arrow.Bool(); } }

export class IntegerType extends DataType {
  readonly id: DataTypeId; readonly name: string;
  constructor(readonly byteSize = 8, readonly signed = true) {
    super();
    const bits = byteSize * 8;
    this.name = `${signed ? "" : "u"}int${bits}`;
    this.id = signed
      ? ({ 1: DataTypeId.INT8, 2: DataTypeId.INT16, 4: DataTypeId.INT32, 8: DataTypeId.INT64 } as Record<number, DataTypeId>)[byteSize]
      : ({ 1: DataTypeId.UINT8, 2: DataTypeId.UINT16, 4: DataTypeId.UINT32, 8: DataTypeId.UINT64 } as Record<number, DataTypeId>)[byteSize];
  }
  toArrow() { return new arrow.Int(this.signed, (this.byteSize * 8) as 8 | 16 | 32 | 64); }
  protected dictParams() { return { byte_size: this.byteSize, signed: this.signed }; }
}
export const Int8Type = () => new IntegerType(1, true);
export const Int16Type = () => new IntegerType(2, true);
export const Int32Type = () => new IntegerType(4, true);
export const Int64Type = () => new IntegerType(8, true);
export const UInt8Type = () => new IntegerType(1, false);
export const UInt16Type = () => new IntegerType(2, false);
export const UInt32Type = () => new IntegerType(4, false);
export const UInt64Type = () => new IntegerType(8, false);

export class FloatType extends DataType {
  readonly id: DataTypeId; readonly name: string;
  constructor(readonly byteSize = 8) {
    super();
    this.name = `float${byteSize * 8}`;
    this.id = byteSize === 2 ? DataTypeId.FLOAT16 : byteSize === 4 ? DataTypeId.FLOAT32 : DataTypeId.FLOAT64;
  }
  toArrow() { return new arrow.Float(this.byteSize === 2 ? arrow.Precision.HALF : this.byteSize === 4 ? arrow.Precision.SINGLE : arrow.Precision.DOUBLE); }
  protected dictParams() { return { byte_size: this.byteSize }; }
}
export const Float32Type = () => new FloatType(4);
export const Float64Type = () => new FloatType(8);

export class DecimalType extends DataType {
  readonly id = DataTypeId.DECIMAL; readonly name: string;
  constructor(readonly precision = 38, readonly scale = 18) { super(); this.name = `decimal(${precision},${scale})`; }
  toArrow() { return new arrow.Decimal(this.scale, this.precision, 128); }
  protected dictParams() { return { byte_size: 16, precision: this.precision, scale: this.scale }; }
}

export class StringType extends DataType { readonly id = DataTypeId.STRING; readonly name = "string"; toArrow() { return new arrow.Utf8(); } }
export class BinaryType extends DataType { readonly id = DataTypeId.BINARY; readonly name = "binary"; toArrow() { return new arrow.Binary(); } }
export class DateType extends DataType {
  readonly id = DataTypeId.DATE; readonly name = "date";
  toArrow() { return new arrow.DateDay(); }
  protected dictParams() { return { byte_size: 4, unit: "d", tz: null }; }
}

export class TimeType extends DataType {
  readonly id = DataTypeId.TIME; readonly name: string;
  constructor(readonly unit: TimeUnit = "us") { super(); this.name = `time[${unit}]`; }
  get byteSize(): number { return this.unit === "s" || this.unit === "ms" ? 4 : 8; }
  toArrow() { return new arrow.Time(ARROW_UNIT[this.unit], (this.byteSize * 8) as 32 | 64); }
  protected dictParams() { return { byte_size: this.byteSize, unit: this.unit, tz: null }; }
}
export class TimestampType extends DataType {
  readonly id = DataTypeId.TIMESTAMP; readonly name: string;
  constructor(readonly unit: TimeUnit = "us", readonly tz: string | null = "Etc/UTC") {
    super(); this.name = `timestamp[${unit}${tz ? "," + tz : ""}]`;
  }
  toArrow() { return new arrow.Timestamp(ARROW_UNIT[this.unit], this.tz ?? undefined); }
  protected dictParams() { return { byte_size: 8, unit: this.unit, tz: this.tz }; }
}
export class DurationType extends DataType {
  readonly id = DataTypeId.DURATION; readonly name: string;
  constructor(readonly unit: TimeUnit = "us") { super(); this.name = `duration[${unit}]`; }
  toArrow() { return new arrow.Duration(ARROW_UNIT[this.unit]); }
  protected dictParams() { return { byte_size: 8, unit: this.unit, tz: null }; }
}

// --- nested ------------------------------------------------------------------

export class ListType extends DataType {
  readonly id = DataTypeId.ARRAY; readonly name: string;
  constructor(readonly itemField: Field) { super(); this.name = `list<${itemField.dtype.name}>`; }
  get children(): Field[] { return [this.itemField]; }
  toArrow() { return new arrow.List(this.itemField.toArrow()); }
  protected dictParams() { return { item_field: this.itemField.toDict() }; }
}
export class StructType extends DataType {
  readonly id = DataTypeId.STRUCT; readonly name: string;
  constructor(readonly fields: Field[]) { super(); this.name = `struct<${fields.map((f) => `${f.name}:${f.dtype.name}`).join(",")}>`; }
  get children(): Field[] { return this.fields; }
  toArrow() { return new arrow.Struct(this.fields.map((f) => f.toArrow())); }
  protected dictParams() { return { fields: this.fields.map((f) => f.toDict()) }; }
}
export class MapType extends DataType {
  readonly id = DataTypeId.MAP; readonly name: string;
  constructor(readonly keyField: Field, readonly valueField: Field) { super(); this.name = `map<${keyField.dtype.name},${valueField.dtype.name}>`; }
  get children(): Field[] { return [this.keyField, this.valueField]; }
  /** The Arrow ``entries`` struct field that backs a Map (key not-null, value nullable). */
  get entriesField(): Field { return bridge().makeField("entries", new StructType([this.keyField, this.valueField]), false); }
  toArrow() { return new arrow.Map_(this.entriesField.toArrow()); }
  protected dictParams() { return { item_field: this.entriesField.toDict() }; }
}

const PRIMITIVE_PARSERS: Record<string, () => DataType> = {
  null: () => new NullType(), bool: () => new BoolType(), boolean: () => new BoolType(),
  int8: Int8Type, int16: Int16Type, int32: Int32Type, int64: Int64Type, int: Int64Type,
  uint8: UInt8Type, uint16: UInt16Type, uint32: UInt32Type, uint64: UInt64Type,
  float16: () => new FloatType(2), float32: Float32Type, float64: Float64Type, float: Float64Type, double: Float64Type,
  string: () => new StringType(), utf8: () => new StringType(), binary: () => new BinaryType(),
  date: () => new DateType(),
};
