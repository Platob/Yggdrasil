"""``ygg loki`` — drive the global yggdrasil agent from the terminal.

```text
ygg loki                 # status: identity + reachable backends + behaviors
ygg loki capabilities    # the detected backends and why
ygg loki behaviors       # the registered behavior catalog
ygg loki tools           # the tools the autonomous agent acts through
ygg loki reason "..."    # one-shot reasoning with the best engine
ygg loki do "..."        # act autonomously: discover + modify files
ygg loki token --probe   # the Databricks credentials Loki provides
ygg loki run NAME --kwarg k=v ...
```
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygg loki", description="Loki — the global yggdrasil agent.")
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("status", help="Identity + reachable backends + engines + behaviors (default).")
    sub.add_parser("capabilities", help="The detected backends and their signals.")
    sub.add_parser("behaviors", help="The registered behavior catalog.")
    sub.add_parser("engines", help="The reasoning engines and which are available.")

    reason = sub.add_parser("reason", help="Reason about a prompt with the best (or named) engine.")
    reason.add_argument("prompt", help="The prompt to reason about.")
    reason.add_argument("--system", default=None, help="Optional system instruction.")
    reason.add_argument("--engine", default=None, help="Force an engine (claude/openai/databricks).")

    do = sub.add_parser("do", help="Act autonomously: discover + modify files to do a task.")
    do.add_argument("task", help="What Loki should accomplish.")
    do.add_argument("--root", default=".", help="Working tree the agent is confined to (default '.').")
    do.add_argument("--engine", default=None, help="Force an engine (claude/openai/databricks).")
    do.add_argument("--max-steps", type=int, default=12, help="Tool-call budget (default 12).")
    do.add_argument("--read-only", action="store_true", help="Discovery only — no file writes.")
    do.add_argument("--allow-shell", action="store_true", help="Also give the agent a shell tool.")
    do.add_argument("--json", action="store_true", help="Print the full transcript as JSON.")

    tools = sub.add_parser("tools", help="The tools the autonomous agent can act through.")
    tools.add_argument("--root", default=".", help="Working tree to root the tools at (default '.').")
    tools.add_argument("--read-only", action="store_true", help="Show the read-only toolbox.")
    tools.add_argument("--allow-shell", action="store_true", help="Include the shell tool.")

    tok = sub.add_parser("token", help="The Databricks credentials Loki provides (non-secret).")
    tok.add_argument("--probe", action="store_true", help="Make one network call to resolve the user.")

    run = sub.add_parser("run", help="Run a behavior by name.")
    run.add_argument("name", help="Behavior name (see `ygg loki behaviors`).")
    run.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE",
                     help="Keyword argument (JSON-decoded; repeatable).")
    run.add_argument("--json", action="store_true", help="Print the raw result as JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import os

    from yggdrasil.cli import style

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    action = args.action or "status"

    # On the Databricks runtime (a deployed agent job) ``ygg loki`` *is* the
    # specialized DatabricksLoki — same single entry point, workspace-aware
    # agent. Everywhere else it's the global Loki.
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        from yggdrasil.databricks.loki import DatabricksLoki

        loki = DatabricksLoki.current()
    else:
        from yggdrasil.loki import Loki

        loki = Loki.current()

    style.print_logo("YGGLOKI")

    if action == "status":
        return _status(loki, style)
    if action == "capabilities":
        return _capabilities(loki, style)
    if action == "behaviors":
        return _behaviors(loki, style)
    if action == "engines":
        return _engines(loki, style)
    if action == "reason":
        return _reason(loki, style, args)
    if action == "do":
        return _do(loki, style, args)
    if action == "tools":
        return _tools(style, args)
    if action == "token":
        return _token(loki, style, probe=args.probe)
    if action == "run":
        return _run(loki, style, args)
    parser.print_help()
    return 0


def _status(loki: Any, style: Any) -> int:
    style.out(f"  {style.cyan('agent')}   {style.bold(loki.name)}  {style.dim('#' + str(loki.agent_id))}\n")
    style.out(f"  {style.cyan('user')}    {loki.user}@{loki.host}\n\n")
    style.out(f"  {style.bold('backends')}\n")
    for b in loki.backends():
        glyph = style.green("●") if b.available else style.dim("○")
        extra = b.detail.get("host") or b.detail.get("home") or b.detail.get("user") or ""
        style.out(f"    {glyph} {b.name.ljust(11)} {style.dim(str(extra))}\n")
    style.out(f"\n  {style.bold('engines')}\n")
    best = loki.engine()
    for eng in loki.engines():
        glyph = style.green("●") if eng.available() else style.dim("○")
        star = style.dim(" (default)") if best is not None and eng.name == best.name else ""
        style.out(f"    {glyph} {eng.name.ljust(11)} {style.dim(str(eng.model))}{star}\n")
    style.out(f"\n  {style.bold('behaviors')}\n")
    for beh in loki.behaviors():
        ok = beh.available(loki)
        glyph = style.green("●") if ok else style.dim("○")
        style.out(f"    {glyph} {beh.name.ljust(11)} {style.dim(beh.description)}\n")
    if not loki.behaviors():
        style.out(f"    {style.dim('(none registered)')}\n")
    return 0


def _engines(loki: Any, style: Any) -> int:
    best = loki.engine()
    for eng in loki.engines():
        glyph = style.green("●") if eng.available() else style.dim("○")
        star = style.dim(" (default)") if best is not None and eng.name == best.name else ""
        style.out(f"  {glyph} {style.bold(eng.name)}  {style.dim(str(eng.model))}{star}\n")
    if best is None:
        style.warn("no engine available — log into Claude Code, or set ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a Databricks session")
    return 0


def _reason(loki: Any, style: Any, args: Any) -> int:
    try:
        out = loki.reason(args.prompt, system=args.system, engine=args.engine)
    except (KeyError, RuntimeError) as exc:
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1
    style.out(f"\n{out}\n")
    return 0


def _do(loki: Any, style: Any, args: Any) -> int:
    style.out(f"  {style.brand('▸')} {style.bold(args.task)}\n")
    mode = "read-only" if args.read_only else "read-write"
    if args.allow_shell:
        mode += "+shell"
    style.out(f"  {style.dim('root ' + args.root + '  ·  ' + mode)}\n\n")

    def on_step(rec: dict) -> None:
        if rec.get("done"):
            return
        call = f"{rec['tool']}({', '.join(f'{k}={_short(v)}' for k, v in rec['args'].items())})"
        style.out(f"  {style.dim(str(rec['n']).rjust(2))} {style.brand(call)}\n")
        if rec.get("thought"):
            style.out(f"     {style.dim(_short(rec['thought']))}\n")
        first = rec["observation"].splitlines()[0] if rec["observation"] else ""
        style.out(f"     {style.dim('→ ' + _short(first))}\n")

    try:
        result = loki.act(
            args.task, root=args.root, engine=args.engine, max_steps=args.max_steps,
            read_only=args.read_only, allow_shell=args.allow_shell,
            on_step=None if args.json else on_step,
        )
    except (KeyError, RuntimeError) as exc:
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1

    if args.json:
        from yggdrasil.pickle import json as yjson

        style.out(yjson.dumps(result, indent=2) + "\n")
        return 0

    style.out("\n")
    if result["files_changed"]:
        style.out(f"  {style.bold('files changed')}\n")
        for f in result["files_changed"]:
            style.out(f"    {style.good('✎')} {f}\n")
        style.out("\n")
    if result["completed"]:
        style.ok(result["answer"] or "done")
    else:
        style.warn(result["answer"] or "did not finish")
    return 0 if result["completed"] else 1


def _tools(style: Any, args: Any) -> int:
    from yggdrasil.loki.tools import filesystem_toolbox

    box = filesystem_toolbox(args.root, read_only=args.read_only, allow_shell=args.allow_shell)
    for t in box.tools.values():
        mark = style.brand("✎") if t.mutates else style.dim("○")
        style.out(f"  {mark} {style.bold(t.name)}  {style.dim(t.description)}\n")
        for k, v in t.params.items():
            style.out(f"      {style.dim(k.ljust(8))} {style.dim(v)}\n")
    return 0


def _capabilities(loki: Any, style: Any) -> int:
    for b in loki.backends(refresh=True):
        glyph = style.green("●") if b.available else style.dim("○")
        style.out(f"  {glyph} {style.bold(b.name)}\n")
        for k, v in b.detail.items():
            style.out(f"      {style.dim(k.ljust(10))} {v}\n")
    return 0


def _behaviors(loki: Any, style: Any) -> int:
    behaviors = loki.behaviors()
    if not behaviors:
        style.out(f"  {style.dim('no behaviors registered')}\n")
        return 0
    for beh in behaviors:
        glyph = style.green("●") if beh.available(loki) else style.dim("○")
        req = f" {style.dim('requires=' + beh.requires)}" if beh.requires else ""
        style.out(f"  {glyph} {style.bold(beh.name)}{req}\n")
        if beh.description:
            style.out(f"      {style.dim(beh.description)}\n")
    return 0


def _token(loki: Any, style: Any, *, probe: bool) -> int:
    info = loki.token_info()
    if not info.get("available"):
        style.warn("no Databricks session detected — Loki has no token to provide")
        return 1
    style.out(f"    {style.dim('host')}      {info.get('host')}\n")
    style.out(f"    {style.dim('auth')}      {info.get('auth_type')}\n")
    style.out(f"    {style.dim('catalog')}   {info.get('catalog') or style.dim('(unset)')}\n")
    style.out(f"    {style.dim('schema')}    {info.get('schema') or style.dim('(unset)')}\n")
    if probe:
        user = loki.whoami(probe=True)
        style.out(f"    {style.dim('user')}      {user or style.dim('(unresolved)')}\n")
    style.ok("Databricks credentials available")
    return 0


def _run(loki: Any, style: Any, args: Any) -> int:
    kwargs: dict[str, Any] = {}
    for kv in args.kwarg:
        if "=" not in kv:
            style.fail(f"--kwarg must be KEY=VALUE, got {kv!r}")
            return 2
        k, v = kv.split("=", 1)
        try:
            kwargs[k] = json.loads(v)
        except json.JSONDecodeError:
            kwargs[k] = v
    try:
        result = loki.run(args.name, **kwargs)
    except (KeyError, RuntimeError) as exc:
        # KeyError stringifies with surrounding quotes; show the clean message.
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1
    if args.json:
        from yggdrasil.pickle import json as yjson

        style.out(yjson.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result))
        style.out("\n")
    else:
        style.ok(f"behavior {args.name!r} completed")
        _print_result(result, style)
    return 0


def _print_result(result: Any, style: Any) -> None:
    if isinstance(result, dict):
        for k, v in result.items():
            style.out(f"    {style.dim(k.ljust(14))} {_short(v)}\n")
    else:
        style.out(f"    {_short(result)}\n")


def _short(v: Any) -> str:
    s = str(v)
    return s if len(s) <= 200 else s[:197] + "…"


if __name__ == "__main__":
    import sys

    sys.exit(main())
