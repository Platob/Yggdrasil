import { describe, it, expect } from "vitest";
import { Path } from "./path";

describe("Path (URL-backed pathlib value type)", () => {
  it("exposes pathlib accessors", () => {
    const p = Path.from("s3://b/data/archive.csv.zst");
    expect(p.name).toBe("archive.csv.zst");
    expect(p.stem).toBe("archive.csv");
    expect(p.suffix).toBe(".zst");
    expect(p.suffixes).toEqual([".csv", ".zst"]);
    expect(p.parts).toEqual(["data", "archive.csv.zst"]);
    expect(p.parent.toString()).toBe("s3://b/data");
    expect(p.isAbsolute).toBe(true);
    expect(p.mediaType?.mimeType.value).toBe("text/csv"); // codec-aware
  });

  it("with_name / with_suffix / with_stem", () => {
    const p = Path.from("/a/b/c.csv");
    expect(p.withName("d.json").toString()).toBe("/a/b/d.json");
    expect(p.withSuffix(".parquet").toString()).toBe("/a/b/c.parquet");
    expect(p.withStem("renamed").toString()).toBe("/a/b/renamed.csv");
    expect(() => p.withSuffix("parquet")).toThrow();
    expect(() => p.withName("x/y")).toThrow();
  });

  it("joinpath", () => {
    expect(Path.from("npfs://n/dir").joinpath("a", "b/c.arrow").toString()).toBe("npfs://n/dir/a/b/c.arrow");
  });
});
