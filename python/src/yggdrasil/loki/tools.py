"""Loki tools — the file/shell capabilities the agent acts *through*.

A :class:`Tool` is one concrete thing the agentic loop can do with its
hands: list a directory, read a file, write or edit one, search the tree,
run a command. The agent's engine (:mod:`yggdrasil.loki.engine`, the
"brain") emits a JSON tool call; Loki looks the tool up in a
:class:`Toolbox` and runs it, feeding the result back as the next
observation. That reason → act → observe loop lives on
:meth:`yggdrasil.loki.Loki.act`.

Every filesystem tool is **confined to a root** (the agent's working tree):
a path that resolves outside the root is refused, so an autonomous agent
can read and modify the project it was pointed at and nothing above it.
Mutating tools record what they touched in :attr:`Toolbox.changed`, so a
run can report exactly which files it created or edited.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["Tool", "Toolbox", "filesystem_toolbox"]

#: Hard ceilings so one observation can't blow the engine's context window.
MAX_READ_BYTES = 64_000
MAX_LIST_ENTRIES = 400
MAX_GREP_HITS = 200


@dataclass
class Tool:
    """One named capability the agent can invoke by emitting a JSON call.

    ``run`` takes the decoded ``args`` as keyword arguments and returns a
    string *observation* the agent reads on its next turn. ``params`` maps
    each argument name to a one-line description — it's rendered into the
    system prompt so the model knows the tool's shape.
    """

    name: str
    description: str
    params: dict[str, str]
    run: Callable[..., str]
    mutates: bool = False


@dataclass
class Toolbox:
    """A named set of tools plus the run state the loop reports on."""

    tools: dict[str, Tool] = field(default_factory=dict)
    #: Repo-relative paths the run created or edited, in first-touch order.
    changed: list[str] = field(default_factory=list)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)

    def names(self) -> list[str]:
        return list(self.tools)

    def spec(self) -> str:
        """The tool catalog as prompt text — one block per tool."""
        blocks = []
        for t in self.tools.values():
            args = ", ".join(f"{k}: {v}" for k, v in t.params.items()) or "(no args)"
            blocks.append(f"- {t.name}: {t.description}\n    args: {args}")
        return "\n".join(blocks)

    def call(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool by name; surface any error as a readable observation."""
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool {name!r}; available: {', '.join(self.tools)}"
        try:
            return tool.run(**args)
        except TypeError as exc:  # bad/missing args from the model
            return f"ERROR: bad arguments for {name!r}: {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": [
                {"name": t.name, "description": t.description,
                 "params": t.params, "mutates": t.mutates}
                for t in self.tools.values()
            ],
            "changed": list(self.changed),
        }


def filesystem_toolbox(
    root: str | pathlib.Path = ".",
    *,
    read_only: bool = False,
    allow_shell: bool = False,
) -> Toolbox:
    """Build the default toolbox rooted at *root*.

    The read tools (``list_dir``, ``read_file``, ``find``, ``grep``) are
    always present — discovery. The write tools (``write_file``,
    ``edit_file``) are added unless ``read_only``. ``run`` (a shell command
    inside the root) is added only when ``allow_shell`` — it's the sharpest
    tool, so it's opt-in.
    """
    base = pathlib.Path(root).resolve()
    box = Toolbox()

    def resolve(path: str) -> pathlib.Path:
        """Resolve *path* under the root, refusing anything that escapes it."""
        target = (base / path).resolve()
        if target != base and base not in target.parents:
            raise ValueError(f"path {path!r} escapes the agent root {base}")
        return target

    def rel(p: pathlib.Path) -> str:
        try:
            return str(p.relative_to(base)) or "."
        except ValueError:
            return str(p)

    def list_dir(path: str = ".") -> str:
        target = resolve(path)
        if not target.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = []
        for child in sorted(target.iterdir())[:MAX_LIST_ENTRIES]:
            if child.is_dir():
                entries.append(f"{rel(child)}/")
            else:
                entries.append(f"{rel(child)}  ({child.stat().st_size}b)")
        if not entries:
            return f"(empty directory: {rel(target)})"
        return "\n".join(entries)

    def read_file(path: str, start: int = 1, end: int | None = None) -> str:
        target = resolve(path)
        if not target.is_file():
            return f"ERROR: not a file: {path}"
        data = target.read_bytes()
        truncated = len(data) > MAX_READ_BYTES
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        lines = text.splitlines()
        lo = max(start, 1)
        hi = end if end is not None else len(lines)
        window = lines[lo - 1 : hi]
        numbered = "\n".join(f"{lo + i}\t{ln}" for i, ln in enumerate(window))
        if truncated:
            numbered += f"\n… (truncated at {MAX_READ_BYTES} bytes)"
        return numbered or "(empty file)"

    def find(pattern: str = "*", path: str = ".") -> str:
        target = resolve(path)
        hits = [
            rel(p) for p in sorted(target.rglob(pattern))
            if p.is_file() and ".git/" not in rel(p)
        ][:MAX_LIST_ENTRIES]
        return "\n".join(hits) if hits else f"(no files match {pattern!r} under {rel(target)})"

    def grep(pattern: str, path: str = ".", glob: str = "*") -> str:
        target = resolve(path)
        rx = re.compile(pattern)
        files = [target] if target.is_file() else sorted(target.rglob(glob))
        out: list[str] = []
        for f in files:
            if not f.is_file() or ".git/" in rel(f):
                continue
            try:
                text = f.read_text("utf-8", errors="replace")
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    out.append(f"{rel(f)}:{n}: {line.strip()[:200]}")
                    if len(out) >= MAX_GREP_HITS:
                        return "\n".join(out) + "\n… (more hits truncated)"
        return "\n".join(out) if out else f"(no matches for {pattern!r})"

    box.add(Tool("list_dir", "List the entries of a directory.",
                 {"path": "directory, default '.'"}, list_dir))
    box.add(Tool("read_file", "Read a text file (optionally a line range).",
                 {"path": "file path", "start": "first line (1-based, optional)",
                  "end": "last line (optional)"}, read_file))
    box.add(Tool("find", "Find files by glob pattern under a directory.",
                 {"pattern": "glob, e.g. '*.py'", "path": "root, default '.'"}, find))
    box.add(Tool("grep", "Search file contents for a regex.",
                 {"pattern": "regex", "path": "root, default '.'",
                  "glob": "file glob, default '*'"}, grep))

    if not read_only:
        def write_file(path: str, content: str) -> str:
            target = resolve(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            existed = target.exists()
            target.write_text(content, "utf-8")
            r = rel(target)
            if r not in box.changed:
                box.changed.append(r)
            verb = "overwrote" if existed else "created"
            return f"{verb} {r} ({len(content)} bytes)"

        def edit_file(path: str, old: str, new: str) -> str:
            target = resolve(path)
            if not target.is_file():
                return f"ERROR: not a file: {path}"
            text = target.read_text("utf-8")
            count = text.count(old)
            if count == 0:
                return f"ERROR: `old` text not found in {rel(target)} — read it first"
            if count > 1:
                return (f"ERROR: `old` text appears {count}× in {rel(target)} — "
                        "add surrounding context to make it unique")
            target.write_text(text.replace(old, new, 1), "utf-8")
            r = rel(target)
            if r not in box.changed:
                box.changed.append(r)
            return f"edited {r} (1 replacement)"

        box.add(Tool("write_file", "Create or overwrite a file with content.",
                     {"path": "file path", "content": "full file content"},
                     write_file, mutates=True))
        box.add(Tool("edit_file", "Replace one unique occurrence of `old` with `new`.",
                     {"path": "file path", "old": "exact text to replace (must be unique)",
                      "new": "replacement text"}, edit_file, mutates=True))

    if allow_shell:
        def run_cmd(command: str, timeout: float = 60.0) -> str:
            proc = subprocess.run(
                command, shell=True, cwd=str(base),
                capture_output=True, text=True, timeout=timeout,
            )
            parts = [f"exit={proc.returncode}"]
            if proc.stdout:
                parts.append("stdout:\n" + proc.stdout[:MAX_READ_BYTES])
            if proc.stderr:
                parts.append("stderr:\n" + proc.stderr[:MAX_READ_BYTES])
            return "\n".join(parts)

        box.add(Tool("run", "Run a shell command in the agent root and capture output.",
                     {"command": "shell command", "timeout": "seconds, default 60"},
                     run_cmd, mutates=True))

    return box
