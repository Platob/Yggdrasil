import { describe, it, expect } from "vitest";
import { Tabular } from "./tabular";

describe("Tabular (Arrow IPC round-trip)", () => {
  it("builds, serializes and reloads an Arrow table", () => {
    const t = Tabular.fromArrays({
      id: Int32Array.from([1, 2, 3]),
      name: ["alice", "bob", "carol"],
    });
    expect(t.numRows).toBe(3);
    expect(t.numCols).toBe(2);
    expect(t.columnNames).toEqual(["id", "name"]);
    expect(t.mediaType.value).toBe("application/vnd.apache.arrow.stream");

    const ipc = t.toArrowIPC();
    expect(ipc).toBeInstanceOf(Uint8Array);
    const back = Tabular.fromArrowIPC(ipc);
    expect(back.numRows).toBe(3);
    expect(back.columnNames).toEqual(["id", "name"]);
    expect(back.toArray().map((r) => r.name)).toEqual(["alice", "bob", "carol"]);
  });

  it("slice + select", () => {
    const t = Tabular.fromArrays({ id: Int32Array.from([1, 2, 3, 4]), v: Float64Array.from([1, 2, 3, 4]) });
    expect(t.slice(1, 3).numRows).toBe(2);
    expect(t.select(["id"]).columnNames).toEqual(["id"]);
  });
});
