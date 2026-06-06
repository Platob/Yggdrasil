"""Token-efficient data interchange for talking to a model.

Sharing a table with an LLM is a token-budget problem: the model reads *text*,
and verbose encodings (markdown grids, JSON records) spend most of their tokens
on punctuation the model doesn't need — the same 10×3 frame costs ~3–4× more
tokens as a markdown table or JSON-records blob than as CSV. So Loki shares
tabular data with a model as **CSV with a one-line schema header**: the most
compact encoding that a model still reads natively.

Two distinct channels, deliberately split:

- **Into a model's context** (a prompt / tool observation) → text, token-tight
  → :func:`encode` (CSV + schema header).
- **Between tools / agents / the cache**, where no model reads it → bytes,
  size-tight → :func:`to_ipc` / :func:`from_ipc` (compressed **Arrow IPC**).
  Arrow IPC is the smallest *byte* wire, but base64-ing binary into a prompt is
  token-hostile (no tokenizer compresses random base64), so it never goes to a
  model — that's what :func:`compare` demonstrates.

:func:`compare` measures the token cost of every candidate encoding for a given
frame so the choice is evidenced, not asserted — point it at a real tokenizer
(``tiktoken`` / a HF tokenizer) or let it fall back to the cheap char estimate.
"""
from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    import polars as pl

__all__ = [
    "encode",
    "compare",
    "best_text_format",
    "to_ipc",
    "from_ipc",
    "as_csv",
    "as_markdown",
]

#: Default rows put in front of a model — enough to convey shape/units without
#: blowing the budget; the full frame stays in the Parquet cache.
DEFAULT_MAX_ROWS = 50


def as_csv(df: "pl.DataFrame") -> str:
    return df.write_csv()


def as_markdown(df: "pl.DataFrame") -> str:
    """A GitHub-flavored markdown table (the verbose baseline we avoid)."""
    cols = df.columns
    rows = df.rows()
    head = "| " + " | ".join(cols) + " |"
    rule = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join([head, rule, *body])


def encode(
    df: "pl.DataFrame",
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_cols: Optional[int] = None,
) -> str:
    """A token-efficient text view of *df* for an LLM: schema header + CSV.

    The header (``# N rows × M cols | col:dtype, …``) gives the model the shape
    and types in one line; the body is CSV of up to *max_rows* rows (the most
    compact legible tabular encoding). Truncation is noted so the model knows
    the sample isn't the whole frame.
    """
    full_rows, full_cols = df.height, df.width
    view = df
    if max_cols is not None and full_cols > max_cols:
        view = view.select(view.columns[:max_cols])
    truncated = full_rows > max_rows
    if truncated:
        view = view.head(max_rows)

    schema = ", ".join(f"{c}:{_dtype(t)}" for c, t in zip(view.columns, view.dtypes))
    header = f"# {full_rows} rows × {full_cols} cols | {schema}"
    parts = [header, as_csv(view).rstrip("\n")]
    if truncated:
        parts.append(f"# … {full_rows - max_rows} more rows (sample only)")
    if max_cols is not None and full_cols > max_cols:
        parts.append(f"# … {full_cols - max_cols} more cols omitted")
    return "\n".join(parts)


def _dtype(t: Any) -> str:
    """Short dtype tag (``Int64`` → ``i64``-ish) — keep the header compact."""
    s = str(t)
    return {
        "Int64": "int", "Int32": "int", "Float64": "f64", "Float32": "f32",
        "Utf8": "str", "String": "str", "Boolean": "bool", "Date": "date",
        "Datetime": "datetime", "Datetime(time_unit='us', time_zone=None)": "datetime",
    }.get(s, s)


def to_ipc(df: "pl.DataFrame", *, compression: str = "zstd") -> bytes:
    """Serialize *df* to compressed **Arrow IPC** bytes — the binary inter-agent
    / cache wire (smallest on the wire; not for a model's context)."""
    import io

    buf = io.BytesIO()
    df.write_ipc(buf, compression=compression)
    return buf.getvalue()


def from_ipc(data: bytes) -> "pl.DataFrame":
    """Read back an Arrow IPC payload produced by :func:`to_ipc`."""
    import io

    from yggdrasil.lazy_imports import polars as pl

    return pl.read_ipc(io.BytesIO(data))


def compare(
    df: "pl.DataFrame",
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> dict[str, dict[str, int]]:
    """Measure each candidate encoding's cost, so the format choice is evidenced.

    Returns ``{format: {"bytes": …, "tokens": …}}`` for the LLM-facing text
    encodings (``csv`` / ``tsv`` / ``markdown`` / ``json``) plus ``ipc_b64``
    (base64'd compressed Arrow IPC — the binary channel, shown for contrast).
    *tokenizer* counts tokens for a string; default is the cheap ~4-chars/token
    estimate, but pass a real tokenizer (``tiktoken`` / HF) for exact counts.
    """
    from .usage import estimate_tokens

    count = tokenizer or estimate_tokens
    sample = df.head(max_rows)
    encodings = {
        "csv": as_csv(sample),
        "tsv": sample.write_csv(separator="\t"),
        "markdown": as_markdown(sample),
        "json": json.dumps(sample.to_dicts(), separators=(",", ":")),
        "ipc_b64": base64.b64encode(to_ipc(sample)).decode("ascii"),
    }
    return {
        name: {"bytes": len(text.encode("utf-8")), "tokens": count(text)}
        for name, text in encodings.items()
    }


def best_text_format(
    df: "pl.DataFrame",
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    tokenizer: Optional[Callable[[str], int]] = None,
) -> str:
    """The fewest-token *legible* text format for *df* (binary excluded)."""
    measured = compare(df, max_rows=max_rows, tokenizer=tokenizer)
    text_only = {k: v for k, v in measured.items() if k != "ipc_b64"}
    return min(text_only, key=lambda k: text_only[k]["tokens"])
