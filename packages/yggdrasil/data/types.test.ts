// Cross-language parity tests for the DataType hierarchy.
//
// The fixtures below are the canonical ``DataType.to_dict()`` output captured
// from the Python reference (``yggdrasil.data.types``). They are the wire
// contract: the JS port must produce — and round-trip — the exact same dicts,
// so an ``int64`` / ``decimal(38,18)`` / ``map<string,int64>`` means the same
// thing on both sides. Regenerate with:
//   python -c "import pyarrow as pa, json; from yggdrasil.data.types import DataType; ..."

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import {
  DataType, DataTypeId,
  NullType, BoolType, BinaryType, StringType, DecimalType,
  DateType, TimeType, TimestampType, DurationType, ListType, StructType, MapType,
  Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
  FloatType, Float32Type, Float64Type,
} from "./types";
import { makeField } from "./field";

// --- Python-captured canonical dicts (the cross-language contract) -----------
const PARITY: Record<string, Record<string, unknown>> = {
  null: { id: 1, name: "NULL" },
  bool: { id: 10, name: "BOOL" },
  int8: { id: 21, name: "INT8", byte_size: 1, signed: true },
  int16: { id: 22, name: "INT16", byte_size: 2, signed: true },
  int32: { id: 23, name: "INT32", byte_size: 4, signed: true },
  int64: { id: 24, name: "INT64", byte_size: 8, signed: true },
  uint8: { id: 26, name: "UINT8", byte_size: 1, signed: false },
  uint16: { id: 27, name: "UINT16", byte_size: 2, signed: false },
  uint32: { id: 28, name: "UINT32", byte_size: 4, signed: false },
  uint64: { id: 29, name: "UINT64", byte_size: 8, signed: false },
  float16: { id: 42, name: "FLOAT16", byte_size: 2 },
  float32: { id: 43, name: "FLOAT32", byte_size: 4 },
  float64: { id: 44, name: "FLOAT64", byte_size: 8 },
  decimal: { id: 50, name: "DECIMAL", byte_size: 16, precision: 38, scale: 18 },
  string: { id: 71, name: "STRING" },
  binary: { id: 70, name: "BINARY" },
  date: { id: 60, name: "DATE", byte_size: 4, unit: "d", tz: null },
  time: { id: 61, name: "TIME", byte_size: 8, unit: "us", tz: null },
  ts_utc: { id: 62, name: "TIMESTAMP", byte_size: 8, unit: "us", tz: "UTC" },
  ts_naive: { id: 62, name: "TIMESTAMP", byte_size: 8, unit: "ns", tz: null },
  duration: { id: 63, name: "DURATION", byte_size: 8, unit: "us", tz: null },
  list: {
    id: 100, name: "ARRAY",
    item_field: { name: "item", dtype: { id: 23, name: "INT32", byte_size: 4, signed: true }, nullable: true },
  },
  struct: {
    id: 102, name: "STRUCT",
    fields: [
      { name: "a", dtype: { id: 24, name: "INT64", byte_size: 8, signed: true }, nullable: true },
      { name: "b", dtype: { id: 71, name: "STRING" }, nullable: true },
    ],
  },
  map: {
    id: 101, name: "MAP",
    item_field: {
      name: "entries", nullable: false,
      dtype: {
        id: 102, name: "STRUCT",
        fields: [
          { name: "key", dtype: { id: 71, name: "STRING" }, nullable: false },
          { name: "value", dtype: { id: 24, name: "INT64", byte_size: 8, signed: true }, nullable: true },
        ],
      },
    },
  },
};

// The matching JS-constructed types (same identity, built from the API).
const JS_TYPES: Record<string, DataType> = {
  null: new NullType(), bool: new BoolType(),
  int8: Int8Type(), int16: Int16Type(), int32: Int32Type(), int64: Int64Type(),
  uint8: UInt8Type(), uint16: UInt16Type(), uint32: UInt32Type(), uint64: UInt64Type(),
  float16: new FloatType(2), float32: Float32Type(), float64: Float64Type(),
  decimal: new DecimalType(38, 18), string: new StringType(), binary: new BinaryType(),
  date: new DateType(), time: new TimeType("us"),
  ts_utc: new TimestampType("us", "UTC"), ts_naive: new TimestampType("ns", null),
  duration: new DurationType("us"),
  list: new ListType(makeField("item", Int32Type())),
  struct: new StructType([makeField("a", Int64Type()), makeField("b", new StringType())]),
  map: new MapType(makeField("key", new StringType(), false), makeField("value", Int64Type())),
};

describe("DataType ↔ Python to_dict parity", () => {
  for (const [key, expected] of Object.entries(PARITY)) {
    it(`${key}: JS toDict matches Python`, () => {
      expect(JS_TYPES[key].toDict()).toEqual(expected);
    });
    it(`${key}: fromDict round-trips the canonical dict`, () => {
      expect(DataType.fromDict(expected).toDict()).toEqual(expected);
    });
  }
});

describe("DataTypeId integer codes are stable cross-language", () => {
  it("matches the Python DataTypeId enum", () => {
    expect(DataTypeId.INT64).toBe(24);
    expect(DataTypeId.UINT8).toBe(26);
    expect(DataTypeId.FLOAT64).toBe(44);
    expect(DataTypeId.DECIMAL).toBe(50);
    expect(DataTypeId.TIMESTAMP).toBe(62);
    expect(DataTypeId.STRING).toBe(71);
    expect(DataTypeId.ARRAY).toBe(100);
    expect(DataTypeId.MAP).toBe(101);
    expect(DataTypeId.STRUCT).toBe(102);
  });
});

describe("Apache Arrow round-trip", () => {
  const arrowCases: arrow.DataType[] = [
    new arrow.Null(), new arrow.Bool(),
    new arrow.Int(true, 8), new arrow.Int(true, 64), new arrow.Int(false, 16),
    new arrow.Float(arrow.Precision.SINGLE), new arrow.Float(arrow.Precision.DOUBLE),
    new arrow.Decimal(18, 38, 128), new arrow.Utf8(), new arrow.Binary(),
    new arrow.Timestamp(arrow.TimeUnit.MICROSECOND, "UTC"),
  ];
  it("fromArrow(t).toArrow() preserves the typeId", () => {
    for (const t of arrowCases) {
      const back = DataType.fromArrow(t).toArrow();
      expect(back.typeId).toBe(t.typeId);
    }
  });
});

describe("DataType.fromString", () => {
  it("parses primitive + parametrized tokens", () => {
    expect(DataType.fromString("int64").toDict()).toEqual(PARITY.int64);
    expect(DataType.fromString("float64").toDict()).toEqual(PARITY.float64);
    expect(DataType.fromString("decimal(38,18)").toDict()).toEqual(PARITY.decimal);
    expect(DataType.fromString("string").id).toBe(DataTypeId.STRING);
  });
});

describe("DataType helpers", () => {
  it("isNumeric / isNested / typeName", () => {
    expect(Int64Type().isNumeric).toBe(true);
    expect(new StringType().isNumeric).toBe(false);
    expect(new StructType([]).isNested).toBe(true);
    expect(Int64Type().typeName).toBe("INT64");
    expect(Int64Type().name).toBe("int64");
  });
});
