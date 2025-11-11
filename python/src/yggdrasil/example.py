"""Example module demonstrating shared utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pyarrow as pa


def greet(name: str) -> str:
    """Return a friendly greeting for *name*.

    Parameters
    ----------
    name:
        The person or system to greet.
    """
    return f"Hail, {name}! Welcome to Yggdrasil."


@dataclass(frozen=True)
class DataFormat:
    """Describe a data artifact exchanged between language layers."""

    name: str
    mime_type: str
    version: str = "1.0"

    def as_metadata(self) -> dict[str, str]:
        """Serialize the format metadata to a mapping suitable for PyArrow metadata."""

        return {"name": self.name, "mime_type": self.mime_type, "version": self.version}

    def attach_to(self, table: pa.Table) -> pa.Table:
        """Attach the format metadata to an Arrow table and return a new table."""

        encoded_metadata = {key: value.encode("utf-8") for key, value in self.as_metadata().items()}
        metadata = table.schema.metadata or {}
        metadata = {**metadata, **encoded_metadata}
        return table.replace_schema_metadata(metadata)


def demo_table(rows: Iterable[tuple[str, int]]) -> pa.Table:
    """Build a tiny Arrow table from *rows* and stamp it with a :class:`DataFormat`."""

    table = pa.table({"name": [name for name, _ in rows], "rank": [rank for _, rank in rows]})
    format_hint = DataFormat(name="character-roster", mime_type="application/x-yggdrasil+json")
    return format_hint.attach_to(table)


def _main() -> None:
    print(greet("Wanderer"))
    roster = demo_table([("Freya", 1), ("Odin", 2)])
    print(roster)
    print("Metadata:", {k.decode(): v.decode() for k, v in (roster.schema.metadata or {}).items()})


if __name__ == "__main__":
    _main()
