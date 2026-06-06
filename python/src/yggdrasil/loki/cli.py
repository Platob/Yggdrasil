"""``ygg loki`` — drive the global yggdrasil agent from the terminal.

```text
ygg loki                 # interactive session (modern REPL) on a terminal
ygg loki status          # identity + reachable backends + engines + behaviors
ygg loki capabilities    # the detected backends and why
ygg loki engines         # the reasoning engines and which are available
ygg loki usage           # live token usage + USD KPIs, per model and global
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

    sub.add_parser("chat", aliases=["repl"], help="Interactive session (the default on a terminal).")
    sub.add_parser("status", help="Identity + reachable backends + engines + behaviors.")
    sub.add_parser("capabilities", help="The detected backends and their signals.")
    sub.add_parser("behaviors", help="The registered behavior catalog.")
    sub.add_parser("engines", help="The reasoning engines and which are available.")
    sub.add_parser("usage", help="Live token usage + USD KPIs, per model and global.")

    reason = sub.add_parser("reason", help="Reason about a prompt with the best (or named) engine.")
    reason.add_argument("prompt", help="The prompt to reason about.")
    reason.add_argument("--system", default=None, help="Optional system instruction.")
    reason.add_argument("--engine", default=None, help="Force an engine (claude/openai/databricks).")
    reason.add_argument("--tier", default=None, choices=["fast", "deep"], help="Force a model tier (default: adaptive).")

    do = sub.add_parser("do", help="Act autonomously: discover + modify files to do a task.")
    do.add_argument("task", help="What Loki should accomplish.")
    do.add_argument("--root", default=".", help="Working tree the agent is confined to (default '.').")
    do.add_argument("--engine", default=None, help="Force an engine (claude/openai/databricks).")
    do.add_argument("--tier", default=None, choices=["fast", "deep"], help="Force a model tier (default: adaptive).")
    do.add_argument("--max-steps", type=int, default=12, help="Tool-call budget (default 12).")
    do.add_argument("--read-only", action="store_true", help="Discovery only — no file writes.")
    do.add_argument("--allow-shell", action="store_true", help="Also give the agent a shell tool.")
    do.add_argument("--allow-web", action="store_true", help="Also give the agent web fetch/table/image tools.")
    do.add_argument("--json", action="store_true", help="Print the full transcript as JSON.")

    tools = sub.add_parser("tools", help="The tools the autonomous agent can act through.")
    tools.add_argument("--root", default=".", help="Working tree to root the tools at (default '.').")
    tools.add_argument("--read-only", action="store_true", help="Show the read-only toolbox.")
    tools.add_argument("--allow-shell", action="store_true", help="Include the shell tool.")
    tools.add_argument("--allow-web", action="store_true", help="Include the web tools.")

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
    import sys

    from yggdrasil.cli import style

    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    action = args.action
    if action == "repl":
        action = "chat"
    if action is None:
        # No subcommand: open the interactive session on a real terminal,
        # otherwise (pipe, CI, Databricks job) fall back to a static status.
        interactive = (
            sys.stdin.isatty()
            and sys.stdout.isatty()
            and not os.getenv("DATABRICKS_RUNTIME_VERSION")
        )
        action = "chat" if interactive else "status"

    # On the Databricks runtime (a deployed agent job) ``ygg loki`` *is* the
    # specialized DatabricksLoki — same single entry point, workspace-aware
    # agent. Everywhere else it's the global Loki.
    if os.getenv("DATABRICKS_RUNTIME_VERSION"):
        from yggdrasil.databricks.loki import DatabricksLoki

        loki = DatabricksLoki.current()
    else:
        from yggdrasil.loki import Loki

        loki = Loki.current()

    # Register the specialized behavior fleets for every reachable backend
    # (databricks-* / aws-*), so they show up and dispatch here.
    loki.load_specialists()

    style.print_logo("YGGLOKI")

    if action == "chat":
        return _repl(loki, style)
    if action == "usage":
        return _usage_table(style)
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
        style.out(f"    {glyph} {eng.name.ljust(11)} {style.dim(str(eng.model_label))}{star}\n")
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
        style.out(f"  {glyph} {style.bold(eng.name)}  {style.dim(str(eng.model_label))}{star}\n")
    if best is None:
        style.warn("no engine available — log into Claude Code, or set ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a Databricks session")
    return 0


# -- interactive session ---------------------------------------------------

def _repl(loki: Any, style: Any) -> int:
    """A modern interactive session: route → reason/act, live token KPIs, budget."""
    from yggdrasil.loki.usage import METER

    METER.set_limit(METER.DEFAULT_LIMIT)
    state: dict[str, Any] = {"tier": None, "root": ".", "engine": None}

    style.out(f"  {style.bold('interactive session')}  "
              f"{style.dim('· live prompts, streamed replies · /help · /quit')}\n")
    _select_engine(loki, style, state)
    style.out(f"  {style.dim('budget')} {style.brand(f'{METER.limit:,}')} {style.dim('tokens')}\n\n")

    while True:
        try:
            raw = input(_prompt(style, state))
        except (EOFError, KeyboardInterrupt):
            style.out("\n")
            break
        line = raw.strip()
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line.startswith("/"):
            if not _repl_command(loki, style, state, line):
                break
            continue
        # Pre-turn budget gate: at/over the cap, offer to raise before spending.
        if METER.over_budget() and not _budget_prompt(style):
            continue
        _repl_turn(loki, style, state, line)

    style.out("\n")
    _usage_line(style, prefix="session")
    style.ok("session ended")
    return 0


def _prompt(style: Any, state: dict) -> str:
    tag = f"{state.get('engine') or 'auto'}·{state['tier'] or 'auto'}"
    return f"  {style.brand('⟢')} {style.dim(tag)} {style.bold('›')} "


def _select_engine(loki: Any, style: Any, state: dict) -> None:
    """Detect the configured/available engines and pick a default for the session.

    Auto-selects the single available engine; when several are configured
    (Claude key/login, a Databricks session, OpenAI key, …) it lists them and
    lets the user choose. ``/engine`` switches mid-session.
    """
    available = [e for e in loki.engines() if e.available()]
    if not available:
        style.warn("no engine configured — set ANTHROPIC_API_KEY, log into Claude Code, "
                   "or run with a Databricks session")
        style.out(f"  {style.dim('(web fetches and `run` behaviors still work without one)')}\n")
        state["engine"] = None
        return

    best = loki.engine()
    state["engine"] = best.name if best is not None else available[0].name
    if len(available) == 1:
        only = available[0]
        style.out(f"  {style.dim('engine')} {style.brand(only.name)} "
                  f"{style.dim('· ' + only.model_label)}\n")
        return

    style.out(f"  {style.bold('configured engines')} "
              f"{style.dim('— pick a default for this session')}\n")
    for i, e in enumerate(available, 1):
        mark = style.dim(" (best)") if e.name == state["engine"] else ""
        style.out(f"    {style.brand(str(i))}  {e.name.ljust(11)} "
                  f"{style.dim(e.model_label)}{mark}\n")
    try:
        ans = input(f"  choose [1-{len(available)}] or Enter for "
                    f"{style.brand(state['engine'])}: ").strip()
    except (EOFError, KeyboardInterrupt):
        ans = ""
    if ans.isdigit() and 1 <= int(ans) <= len(available):
        state["engine"] = available[int(ans) - 1].name
    elif ans and any(e.name == ans for e in available):
        state["engine"] = ans
    style.ok(f"engine → {state['engine']}")


def _stream_reply(agent: Any, style: Any, line: str, state: dict) -> str:
    """Stream a reasoned reply to the terminal token-by-token; return the full text."""
    style.out("\n  ")
    parts: list[str] = []
    for chunk in agent.reason_stream(line, engine=state.get("engine"), tier=state["tier"]):
        style.out(chunk)
        parts.append(chunk)
    style.out("\n\n")
    return "".join(parts)


def _repl_turn(loki: Any, style: Any, state: dict, line: str) -> None:
    from yggdrasil.loki.usage import METER

    before = METER.total_tokens
    plan = loki.route(line)

    agent, tail = loki, ""
    if plan["specialist"]:
        spec = loki.specialist(plan["specialist"])
        if spec is not None:
            agent, tail = spec, f"  →  {style.brand(spec.name)} {style.dim('(isolated)')}"
    style.out(f"  {style.dim('▹ ' + plan['category'] + ' · ' + plan['why'])}{tail}\n")

    streamed = False
    try:
        if plan["action"] == "web" and plan.get("url"):
            res = agent.run("web", url=plan["url"], question=line)
            _print_web(style, res)
            reply = res.get("answer") or style.dim(f"fetched {plan['url']}")
        elif plan["action"] == "act":
            res = agent.act(line, root=state["root"], tier=state["tier"], allow_web=True,
                            on_step=lambda r: _act_step(style, r))
            reply = res["answer"]
            if res.get("files_changed"):
                style.out(f"  {style.good('✎')} {', '.join(res['files_changed'])}\n")
        elif plan["action"] == "genie":
            res = agent.run("genie", question=line)
            reply = res.get("text", "") if isinstance(res, dict) else str(res)
        else:
            # chat (or a web verb with no URL) → stream the reply live
            reply = _stream_reply(agent, style, line, state)
            streamed = True
    except Exception as exc:  # never let one turn kill the session
        style.out("\n")
        style.fail(exc.args[0] if exc.args else str(exc))
        return
    if not streamed:
        style.out(f"\n  {reply}\n\n")
    _usage_line(style, delta=METER.total_tokens - before)


def _act_step(style: Any, rec: dict) -> None:
    if rec.get("done"):
        return
    call = f"{rec['tool']}({', '.join(f'{k}={_short(v)}' for k, v in rec['args'].items())})"
    style.out(f"  {style.dim(str(rec['n']).rjust(2))} {style.brand(call)}\n")


def _print_web(style: Any, res: dict) -> None:
    """Pretty-print a `web` behavior result by kind: table / image / json / page."""
    kind = res.get("action")
    style.out(f"  {style.dim('🌐 ' + str(res.get('url', '')))}\n")
    if kind == "table":
        style.out(f"  {style.good('▦')} {style.bold(str(tuple(res['shape'])))} "
                  f"{style.dim('· ' + ', '.join(res['columns']))}\n")
        style.out(style.dim("  " + res["preview"].replace("\n", "\n  ")) + "\n")
    elif kind == "image":
        dims = (f"{res.get('width')}×{res.get('height')}"
                if res.get("width") else "?")
        style.out(f"  {style.good('🖼')} {res.get('content_type')}  "
                  f"{res.get('bytes'):,} bytes  {dims}"
                  + (f"  → {res['saved_to']}" if res.get("saved_to") else "") + "\n")
    elif kind == "json":
        style.out(style.dim("  " + _short(res.get("data"))) + "\n")
    else:
        for link in res.get("links", [])[:8]:
            style.out(f"  {style.dim('↪')} {style.dim(link['text'][:48])} {style.dim(link['href'])}\n")
        body = (res.get("text") or "").strip()
        if body:
            style.out(style.dim("  " + body[:600].replace("\n", "\n  ")) + "\n")


def _repl_command(loki: Any, style: Any, state: dict, line: str) -> bool:
    """Handle a /slash command. Returns False only to end the session."""
    from yggdrasil.loki.usage import METER

    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("help", "h", "?"):
        style.out(
            f"  {style.bold('commands')}\n"
            f"    {style.brand('/engine')}   pick the session engine (claude/databricks/openai/auto)\n"
            f"    {style.brand('/engines')}  reasoning engines and adaptive models\n"
            f"    {style.brand('/status')}   identity + backends + engines + behaviors\n"
            f"    {style.brand('/usage')}    token KPIs (per model + global, USD)\n"
            f"    {style.brand('/tier')}     fast | deep | auto  (model tier for this session)\n"
            f"    {style.brand('/root')}     set the working tree for file tasks\n"
            f"    {style.brand('/budget')}   show, set N, +N step, or off\n"
            f"    {style.brand('/reset')}    zero the usage meter\n"
            f"    {style.brand('/quit')}     leave\n"
            f"  {style.dim('plain text is routed automatically — reason (streamed), act on files, web, or a specialist')}\n"
        )
    elif cmd == "engine":
        if not arg or arg == "auto":
            best = loki.engine()
            state["engine"] = best.name if best else None
            style.ok(f"engine → {state['engine'] or 'none available'}")
        elif any(e.name == arg for e in loki.engines()):
            eng = loki.engine(arg)
            if not eng.available():
                style.warn(f"engine {arg!r} is not configured/available")
            else:
                state["engine"] = arg
                style.ok(f"engine → {arg}")
        else:
            style.warn(f"unknown engine {arg!r} — see /engines")
    elif cmd == "status":
        _status(loki, style)
    elif cmd == "engines":
        _engines(loki, style)
    elif cmd == "usage":
        _usage_table(style)
    elif cmd == "tier":
        state["tier"] = None if arg in ("", "auto") else arg
        style.ok(f"tier → {state['tier'] or 'auto (adaptive)'}")
    elif cmd == "root":
        state["root"] = arg or "."
        style.ok(f"root → {state['root']}")
    elif cmd == "budget":
        if arg in ("off", "none"):
            METER.set_limit(None); style.ok("budget off (unlimited)")
        elif arg.startswith("+") and arg[1:].isdigit():
            METER.raise_limit(int(arg[1:])); style.ok(f"budget → {METER.limit:,}")
        elif arg.isdigit():
            METER.set_limit(int(arg)); style.ok(f"budget → {METER.limit:,}")
        else:
            lim = "off" if METER.limit is None else f"{METER.limit:,}"
            style.out(f"  {style.dim('budget')} {style.brand(lim)}  "
                      f"{style.dim('used')} {METER.total_tokens:,}\n")
    elif cmd == "reset":
        METER.reset(); style.ok("usage meter reset")
    else:
        style.warn(f"unknown command /{cmd} — try /help")
    return True


def _budget_prompt(style: Any) -> bool:
    """At/over budget: ask to raise step-by-step, set a custom cap, or stop."""
    from yggdrasil.loki.usage import METER

    style.warn(f"token budget reached — {METER.total_tokens:,} ≥ {METER.limit:,}")
    try:
        ans = input(
            f"  raise by {METER.step:,} [Enter] · set N · off · stop [s]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans in ("s", "stop", "n", "no"):
        return False
    if ans in ("off", "none"):
        METER.set_limit(None); style.ok("budget off (unlimited)"); return True
    if ans.isdigit():
        METER.set_limit(int(ans)); style.ok(f"budget → {METER.limit:,}"); return True
    METER.raise_limit(); style.ok(f"budget → {METER.limit:,}"); return True


def _usage_line(style: Any, *, delta: int | None = None, prefix: str = "usage") -> None:
    from yggdrasil.loki.usage import METER

    t = METER.total()
    bits = [
        f"{style.dim('↑')}{t.input_tokens:,} {style.dim('↓')}{t.output_tokens:,}",
        f"{style.bold(f'{t.total_tokens:,}')} {style.dim('tok')}",
        style.good(f"${METER.total_cost:.4f}"),
    ]
    if delta:
        bits.append(style.dim(f"(+{delta:,})"))
    if METER.limit is not None:
        rem = METER.remaining() or 0
        bits.append((style.good if rem > 0 else style.bad)(f"{rem:,} left"))
    style.out(f"  {style.dim(prefix)}  " + "  ".join(bits) + "\n")


def _usage_table(style: Any) -> int:
    from yggdrasil.loki.usage import METER

    rows = METER.rows()
    style.out(f"\n  {style.bold('token usage')}  {style.dim('· live per model · in/out · USD')}\n")
    head = (f"  {'engine · model'.ljust(32)}{'calls'.rjust(6)}{'in'.rjust(9)}"
            f"{'out'.rjust(9)}{'tokens'.rjust(10)}{'usd'.rjust(11)}")
    style.out(style.dim(head) + "\n")
    if not rows:
        style.out(f"  {style.dim('(no completions yet — reason or act to populate)')}\n")
    for r in rows:
        name = f"{r.engine} · {r.model}"
        style.out(
            f"  {style.brand(name.ljust(32))}{str(r.calls).rjust(6)}"
            f"{f'{r.input_tokens:,}'.rjust(9)}{f'{r.output_tokens:,}'.rjust(9)}"
            f"{f'{r.total_tokens:,}'.rjust(10)}{f'${r.cost_usd:.4f}'.rjust(11)}\n"
        )
    t = METER.total()
    style.out("  " + style.dim("─" * 77) + "\n")
    style.out(
        f"  {style.bold('global'.ljust(32))}{str(t.calls).rjust(6)}"
        f"{f'{t.input_tokens:,}'.rjust(9)}{f'{t.output_tokens:,}'.rjust(9)}"
        f"{style.bold(f'{t.total_tokens:,}'.rjust(10))}"
        f"{style.good(f'${METER.total_cost:.4f}'.rjust(11))}\n"
    )
    if METER.limit is not None:
        rem = METER.remaining() or 0
        g = style.good if rem > 0 else style.bad
        style.out(f"  {style.dim('budget')} {style.brand(f'{METER.limit:,}')}  "
                  f"{g(f'{rem:,} left')}\n")
    style.out(f"  {style.dim('pricing USD/1M tok — defaults in yggdrasil.loki.usage.PRICING')}\n")
    return 0


def _reason(loki: Any, style: Any, args: Any) -> int:
    try:
        out = loki.reason(args.prompt, system=args.system, engine=args.engine, tier=args.tier)
    except (KeyError, RuntimeError) as exc:
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1
    style.out(f"\n{out}\n\n")
    _usage_line(style)
    return 0


def _do(loki: Any, style: Any, args: Any) -> int:
    style.out(f"  {style.brand('▸')} {style.bold(args.task)}\n")
    mode = "read-only" if args.read_only else "read-write"
    if args.allow_shell:
        mode += "+shell"
    if args.allow_web:
        mode += "+web"
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
            args.task, root=args.root, engine=args.engine, tier=args.tier, max_steps=args.max_steps,
            read_only=args.read_only, allow_shell=args.allow_shell, allow_web=args.allow_web,
            on_step=None if args.json else on_step,
        )
    except (KeyError, RuntimeError) as exc:
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1

    if args.json:
        style.out(_json(result) + "\n")
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
    style.out("\n")
    _usage_line(style)
    return 0 if result["completed"] else 1


def _tools(style: Any, args: Any) -> int:
    from yggdrasil.loki.tools import filesystem_toolbox

    box = filesystem_toolbox(args.root, read_only=args.read_only, allow_shell=args.allow_shell, allow_web=args.allow_web)
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
        style.out(_json(result) if isinstance(result, (dict, list)) else str(result))
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


def _json(result: Any) -> str:
    """JSON-encode a result for the CLI (orjson emits bytes → decode to str)."""
    from yggdrasil.pickle import json as yjson

    out = yjson.dumps(result, indent=2)
    return out.decode() if isinstance(out, (bytes, bytearray)) else out


if __name__ == "__main__":
    import sys

    sys.exit(main())
