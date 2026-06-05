// Field parity + round-trip tests. Dicts captured from Python ``Field.to_dict``.

import { describe, it, expect } from "vitest";
import * as arrow from "apache-arrow";
import { Field, field, fieldFromArrow, fieldFromDict } from "./field";
import { Int64Type, StringType, DataTypeId } from "./types";

describe("Field ↔ Python to_dict parity", () => {
  it("plain field (no metadata key when empty)", () => {
    const f = new Field("id", Int64Type(), false);
    expect(f.toDict()).toEqual({
      name: "id",
      dtype: { id: 24, name: "INT64", byte_size: 8, signed: true },
      nullable: false,
    });
  });

  it("field with metadata", () => {
    const f = new Field("x", new StringType(), true, new Map([["k", "v"]]));
    expect(f.toDict()).toEqual({
      name: "x",
      dtype: { id: 71, name: "STRING" },
      nullable: true,
      metadata: { k: "v" },
    });
  });

  it("fromDict round-trips", () => {
    const d = { name: "x", dtype: { id: 71, name: "STRING" }, nullable: true, metadata: { k: "v" } };
    expect(fieldFromDict(d).toDict()).toEqual(d);
  });
});

describe("field() factory", () => {
  it("accepts string / DataType dtypes and metadata", () => {
    expect(field("id", "int64").dtype.id).toBe(DataTypeId.INT64);
    expect(field("v", Int64Type(), { nullable: false }).nullable).toBe(false);
    const f = field("v", "string", { metadata: { a: "b" } });
    expect(f.metadata?.get("a")).toBe("b");
  });
});

describe("Arrow round-trip", () => {
  it("fieldFromArrow(f).toArrow() preserves name/nullable/type", () => {
    const af = new arrow.Field("amount", new arrow.Float(arrow.Precision.DOUBLE), false);
    const back = fieldFromArrow(af).toArrow();
    expect(back.name).toBe("amount");
    expect(back.nullable).toBe(false);
    expect(back.typeId).toBe(af.typeId);
  });
});

describe("Field helpers", () => {
  it("withName / withDtype / equals / toString", () => {
    const f = new Field("a", Int64Type());
    expect(f.withName("b").name).toBe("b");
    expect(f.withDtype(new StringType()).dtype.id).toBe(DataTypeId.STRING);
    expect(f.equals(new Field("a", Int64Type()))).toBe(true);
    expect(f.equals(new Field("c", Int64Type()))).toBe(false);
    expect(f.toString()).toBe("a: int64");
  });
});
