"""``ygg loki`` — drive the global yggdrasil agent from the terminal.

```text
ygg loki                 # interactive session (modern REPL) on a terminal
ygg loki status          # identity + reachable backends + engines + skills
ygg loki capabilities    # the detected backends and why
ygg loki engines         # the reasoning engines and which are available
ygg loki setup [model]   # bootstrap a free local model, sized to this box
ygg loki usage           # live token usage + USD KPIs, per model and global
ygg loki tools           # the tools the autonomous agent acts through
ygg loki reason "..."    # one-shot reasoning with the best engine
ygg loki do "..."        # act autonomously: discover + modify files
ygg loki guide "..."     # the optimized yggdrasil way to build something
ygg loki token --probe   # the Databricks credentials Loki provides
ygg loki run NAME --kwarg k=v ...
```
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Any, Sequence

#: Tool-call budget for an autonomous turn inside the interactive session (the
#: ``ygg loki do`` CLI takes its own ``--max-steps``). Kept here so the live
#: step-budget bar and the ``act`` call agree on the denominator.
_REPL_ACT_STEPS = 12


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ygg loki", description="Loki — the global yggdrasil agent.")
    sub = parser.add_subparsers(dest="action")

    sub.add_parser("chat", aliases=["repl"], help="Interactive session (the default on a terminal).")
    sub.add_parser("status", help="Identity + reachable backends + engines + skills.")
    sub.add_parser("capabilities", help="The detected backends and their signals.")
    sub.add_parser("skills", help="The registered skill catalog.")
    sub.add_parser("engines", help="The reasoning engines and which are available.")
    setup = sub.add_parser("setup", help="Bootstrap a free local model (lazy-install on demand).")
    setup.add_argument("model", nargs="?", default=None, help="A specific local model to ready.")
    guide = sub.add_parser("guide", help="The optimized yggdrasil way to build something.")
    guide.add_argument("task", help="What you want to build.")
    guide.add_argument("--plan", action="store_true", help="Also synthesize a grounded plan (uses an engine).")
    sub.add_parser("usage", help="Live token usage + USD KPIs, per model and global.")
    sub.add_parser("mcp", help="Run Loki as an MCP server (stdio) — expose it to MCP clients.")

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
    do.add_argument("--budget", type=float, default=None, help="USD cost cap — stop before exceeding it.")
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

    run = sub.add_parser("run", help="Run a skill by name.")
    run.add_argument("name", help="Skill name (see `ygg loki skills`).")
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

    # Register the specialized skill fleets for every reachable backend
    # (databricks-* / aws-*) only for the actions that surface or run them right
    # away — the import is heavy (pulls the Databricks SDK). Lightweight reads
    # (engines/usage/capabilities/token) and pure-reasoning actions stay fast.
    # The REPL warms the fleet *asynchronously* (see _repl) so the logo is
    # instant. On the Databricks runtime ``ygg loki`` *is* the specialist.
    if action in ("status", "skills", "run") or os.getenv("DATABRICKS_RUNTIME_VERSION"):
        loki.load_specialists()

    # The MCP server speaks JSON-RPC on stdio — no logo / chatter on stdout.
    if action == "mcp":
        from yggdrasil.loki import mcp as loki_mcp

        return loki_mcp.main()

    style.print_logo("YGGLOKI")

    # Surface local-model progress in the terminal: the transformers engine
    # logs its (otherwise silent) weight download + model load through the
    # `yggdrasil.loki` logger, so a long first run on a CPU box isn't a black
    # box. Stderr-bound + styled — never tangles with streamed output on stdout.
    import logging

    style.install_logging(logger=logging.getLogger("yggdrasil.loki"), show_name=False)

    if action == "chat":
        return _repl(loki, style)
    if action == "usage":
        return _usage_table(style)
    if action == "status":
        return _status(loki, style)
    if action == "capabilities":
        return _capabilities(loki, style)
    if action == "skills":
        return _skills(loki, style)
    if action == "engines":
        return _engines(loki, style)
    if action == "setup":
        return _setup(loki, style, args.model or "")
    if action == "guide":
        res = loki.run("guide", task=args.task, plan=args.plan)
        _print_guide(style, res)
        if res.get("plan"):
            style.out(f"\n  {res['plan']}\n")
        return 0
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
    style.out(f"  {style.cyan('user')}    {loki.user}@{loki.host}\n")
    _print_hardware(style)
    style.out("\n")
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
        kind = style.dim("local " if eng.local else "remote")
        style.out(f"    {glyph} {eng.name.ljust(13)} {kind}  {style.dim(str(eng.model_label))}{star}\n")
    style.out(f"\n  {style.bold('skills')}\n")
    for beh in loki.skills():
        ok = beh.available(loki)
        glyph = style.green("●") if ok else style.dim("○")
        style.out(f"    {glyph} {beh.name.ljust(11)} {style.dim(beh.description)}\n")
    if not loki.skills():
        style.out(f"    {style.dim('(none registered)')}\n")
    return 0


def _print_hardware(style: Any) -> None:
    """One line of the box's compute: cores, RAM, and the detected accelerator —
    NVIDIA cuda / **Intel GPU** xpu / Apple mps, plus an Intel NPU flag — so the
    user can see what a local model will run on."""
    from yggdrasil.loki import resources

    snap = resources.snapshot()
    accel = {"cuda": "NVIDIA GPU (cuda)", "xpu": "Intel GPU (xpu)",
             "mps": "Apple GPU (mps)"}.get(snap.accelerator or "", None)
    if accel is None:
        # Present but torch can't drive it yet — name it and point at the fix,
        # so the GPU shows instead of a flat "CPU only".
        accel = (style.brand("Intel GPU") + style.dim(" (present · pip install intel-extension-for-pytorch to use)")
                 if snap.intel_gpu else "CPU only")
    bits = [f"{snap.cpu} cores", f"{snap.ram_gb:g} GB", accel]
    if snap.npu:
        bits.append(style.brand("Intel NPU"))
    style.out(f"  {style.cyan('compute')} {style.dim(' · '.join(str(b) for b in bits))}\n")


def _engines(loki: Any, style: Any) -> int:
    best = loki.engine()
    for eng in loki.engines():
        glyph = style.green("●") if eng.available() else style.dim("○")
        star = style.dim(" (default)") if best is not None and eng.name == best.name else ""
        kind = style.dim("local " if eng.local else "remote")
        style.out(f"  {glyph} {style.bold(eng.name.ljust(13))} {kind}  {style.dim(str(eng.model_label))}{star}\n")
    if best is None:
        style.warn("no engine available — log into Claude Code, or set ANTHROPIC_API_KEY / OPENAI_API_KEY, or run with a Databricks session")
    return 0


def _pull_bar(style: Any) -> Any:
    """A progress callback for a model download → renders a live byte bar.

    Ollama's pull streams ``{status, completed, total}`` events; show a filled
    bar with a short byte readout while bytes flow, and just the status line
    (manifest / verifying / writing) for the non-byte phases."""
    def on_progress(ev: dict) -> None:
        status = str(ev.get("status", "")).split()[0] if ev.get("status") else ""
        total, completed = ev.get("total"), ev.get("completed")
        if total and completed is not None:
            mb = f"{completed / 1e6:.0f}/{total / 1e6:.0f} MB"
            style.progress(int(completed), int(total), label=f"{status} {mb}")
        elif status:
            style.clear_line()
            style.out(f"  {style.dim('▹ ' + status)}\r")
    return on_progress


def _setup(loki: Any, style: Any, arg: str) -> int:
    """Bootstrap a free local model — lazily install it on demand.

    A lightweight, smart-enough brain for basic setup/config that knows when
    to hand harder work up to a remote model. ``/setup <model>`` targets a
    specific local model.
    """
    model = arg.strip() or None
    style.out(f"  {style.dim('readying a free local model — this may download on first run…')}\n")
    res = loki.bootstrap_local(model=model, on_progress=_pull_bar(style))
    style.clear_line()
    if res.get("ready"):
        was = res.get("was_present")
        verb = "already installed" if was else "installed"
        style.ok(f"local engine {style.brand(res['engine'])} ready "
                 f"· {res['model']} ({verb})")
        style.out(f"  {style.dim('light work now runs locally for free; heavy tasks escalate to a remote model (you are asked first)')}\n")
    else:
        style.warn("no local engine yet — install one of:")
        for hint in res.get("install", []):
            style.out(f"    {style.brand('›')} {style.dim(hint)}\n")
    _enable_intel_npu(style)
    _enable_intel_gpu(style)
    return 0


def _enable_intel_npu(style: Any) -> None:
    """Turn a *detected* Intel NPU into a *usable* one — the ``openvino`` engine.

    The NPU (AI Boost) runs LLMs through OpenVINO / optimum-intel, not torch.
    When the NPU is present but those packages aren't installed, offer to install
    ``optimum[openvino]`` so ``/engine openvino`` can run a model on it; when they
    are, just point at the engine.
    """
    import importlib.util
    import subprocess
    import sys

    from yggdrasil.loki import resources

    if not resources.has_npu():
        return
    if (importlib.util.find_spec("openvino") is not None
            and importlib.util.find_spec("optimum") is not None):
        style.out(f"  {style.brand('▸ Intel NPU')} {style.dim('detected — run a model on it with')} "
                  f"{style.brand('/engine openvino')}\n")
        return
    cmd = [sys.executable, "-m", "pip", "install", "optimum[openvino]"]
    style.out(f"  {style.brand('▸ Intel NPU')} {style.dim('detected — enable the NPU (openvino) engine:')}\n")
    style.out(f"    {style.brand(' '.join(cmd))}\n")
    try:
        ans = input("  install the NPU engine now? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans not in ("y", "yes"):
        return
    with style.Spinner("installing optimum[openvino] for the NPU…"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "see pip output"
        style.fail(f"NPU engine install failed — {_short(tail)}")
        return
    style.ok("NPU engine installed — use /engine openvino (restart `ygg loki` if it's not listed yet)")


def _enable_intel_gpu(style: Any) -> None:
    """Turn a *detected* Intel GPU into a *usable* one.

    The Intel GPU shows in ``status`` as soon as the OS sees it, but a local
    model only runs on it (through torch) once torch can target the XPU backend —
    which the stock CPU/CUDA wheel can't. When the GPU is present but not yet
    usable, offer to install the GPU (XPU) torch build from the dedicated PyTorch
    index. (The NPU is handled by :func:`_enable_intel_npu`.)
    """
    import subprocess
    import sys

    from yggdrasil.loki import resources

    if not resources.intel_gpu_present() or resources.accelerator() == "xpu":
        return                                   # no Intel GPU, or it's already usable
    cmd = [sys.executable, "-m", "pip", "install", "--index-url",
           resources.XPU_TORCH_INDEX, "torch"]
    style.out(f"  {style.brand('▸ Intel GPU')} {style.dim('detected but not yet usable by torch (CPU build)')}\n")
    style.out(f"  {style.dim('enable it with:')}\n    {style.brand(' '.join(cmd))}\n")
    try:
        ans = input(f"  install the GPU torch build now? {style.dim('(~1 GB download)')} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if ans not in ("y", "yes"):
        return
    with style.Spinner("installing GPU torch (xpu) — this downloads ~1 GB…"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "see pip output"
        style.fail(f"GPU torch install failed — {_short(tail)}")
        return
    # torch may already be imported (CPU) in this process — a reinstall can't
    # swap the live module, so a fresh `ygg loki` is what actually picks it up.
    if "torch" not in sys.modules and resources.accelerator() == "xpu":
        style.ok("Intel GPU enabled — local models now run on the GPU (xpu)")
    else:
        style.ok("GPU torch installed — restart `ygg loki` to run local models on the Intel GPU")


# -- interactive session ---------------------------------------------------

def _repl(loki: Any, style: Any) -> int:
    """A modern interactive session: route → reason/act, live token KPIs, budget."""
    from yggdrasil.loki.memory import LokiMemory
    from yggdrasil.loki.session import LokiSession
    from yggdrasil.loki.usage import METER

    METER.set_limit(METER.DEFAULT_COST_LIMIT)

    # Warm in the background while the user picks a session and types the first
    # prompt: register the specialist skill fleets (heavy Databricks import) and
    # build the chosen engine's client. Overlapping this with the input wait
    # keeps the logo instant and the first turn's submit fast.
    import threading

    def _warm() -> None:
        try:
            loki.load_specialists()
        except Exception:
            pass
        try:
            eng = loki.engine()
            warm = getattr(eng, "warm", None)
            if callable(warm):
                warm()
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True).start()

    session = _choose_session(style)  # resume a prior session for this user, or start new
    state: dict[str, Any] = {
        "tier": None, "engine": None, "session": session,
        "root": str(session.workspace),
        "memory": LokiMemory(session.memory_file),
        "named": bool(session.name),
    }

    style.out(f"  {style.bold('interactive session')} {style.dim('#' + session.label)} "
              f"{style.dim('· ' + session.user)}\n")
    style.out(f"  {style.dim('workspace ' + str(session.workspace))}\n")
    _select_engine(loki, style, state)
    # Pre-load the *chosen* engine in the background (the startup _warm thread
    # only warms the best-by-preference one). For a local transformers model
    # this overlaps the slow weights download with the capability probe and the
    # user's first keystrokes, so the first turn isn't a multi-minute black box.
    if state.get("engine"):
        def _warm_chosen(name: str = state["engine"]) -> None:
            try:
                warm = getattr(loki.engine(name), "warm", None)
                if callable(warm):
                    warm()
            except Exception:
                pass

        threading.Thread(target=_warm_chosen, daemon=True).start()
    # Warm the local-capability probe now (it imports torch the first time —
    # slow on a box that has it) so the first turn's engine selection doesn't
    # stall silently. Only matters when a local engine is actually reachable.
    if any(e.local and e.available() for e in loki.engines()):
        with style.Spinner("checking local model capacity…"):
            loki.can_run_local()
    style.out(f"  {style.dim('cost budget')} {style.brand(f'${METER.cost_limit:.2f}')} "
              f"{style.dim('· raised in $1 steps when reached · live KPIs in the title bar')}\n\n")
    style.set_title(_usage_title())   # pin the KPIs in the terminal head

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
        # Name the session from its first prompt; mark used (resets purge clock).
        if not state["named"]:
            name = session.name_from_prompt(line)
            state["named"] = True
            style.out(f"  {style.dim('session named')} {style.brand(name)}\n")
        session.touch()
        _repl_turn(loki, style, state, line)

    session.touch()
    style.out("\n")
    _usage_line(style, prefix="session")
    style.ok(f"session #{session.label} saved")
    return 0


def _ago(ts: float) -> str:
    import time as _t

    d = max(0.0, _t.time() - ts)
    for n, u in ((86400, "d"), (3600, "h"), (60, "m")):
        if d >= n:
            return f"{int(d // n)}{u}"
    return f"{int(d)}s"


def _choose_session(style: Any) -> Any:
    """Offer to resume one of this user's recent sessions, or start a new one."""
    from yggdrasil.loki.session import LokiSession

    recent = LokiSession.list()
    if recent:
        style.out(f"  {style.bold('resume a session')} {style.dim('or Enter for a new one')}\n")
        shown = recent[:6]
        for i, s in enumerate(shown, 1):
            style.out(f"    {style.brand(str(i))}  {(s.name or '(unnamed)').ljust(40)} "
                      f"{style.dim(s.id)} {style.dim(_ago(s.last_used_at) + ' ago')}\n")
        try:
            ans = input(f"  [1-{len(shown)}] resume · Enter new: ").strip()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans.isdigit() and 1 <= int(ans) <= len(shown):
            resumed = LokiSession.resume(shown[int(ans) - 1].id)
            if resumed is not None:
                style.ok(f"resumed #{resumed.label}")
                return resumed
    return LokiSession.start()


def _prompt(style: Any, state: dict) -> str:
    tag = f"{state.get('engine') or 'auto'}·{state['tier'] or 'auto'}"
    model = state.get("model")           # show a pinned model (from /model) compactly
    if model:
        short = model.split("/")[-1]
        tag += "·" + (short if len(short) <= 18 else short[:17] + "…")
    return f"  {style.brand('⟢')} {style.dim(tag)} {style.bold('›')} "


def _select_engine(loki: Any, style: Any, state: dict) -> None:
    """Detect the configured/available engines and pick a default for the session.

    Auto-selects the single available engine; when several are configured
    (Claude key/login, a Databricks session, OpenAI key, …) it lists them and
    lets the user choose. ``/engine`` switches mid-session.
    """
    # Probe all engines' availability in parallel — several gate on a network
    # round-trip (Ollama liveness, Databricks), so a serial scan stacks their
    # latencies onto the startup path. (The probes are memoized, so the
    # `loki.engine()` best-pick just below reuses these results.)
    avail = loki.available_engines()
    available = list(avail.values())
    if not available:
        style.warn("no engine configured — set ANTHROPIC_API_KEY, log into Claude Code, "
                   "or run with a Databricks session")
        style.out(f"  {style.dim('(web fetches and `run` skills still work without one)')}\n")
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


def _choose_model(loki: Any, style: Any, state: dict, arg: str = "") -> None:
    """Pick the model for the session's engine — pin it for every later turn.

    ``/model <id>`` pins directly. With no argument it lists the engine's preset
    models (the resource-sized ladder for a local engine, the fast/deep tiers
    for a remote one) plus any already-pulled Ollama models, marks the current
    and recommended ones, and lets you pick by number or type a custom id. For a
    local engine it then offers to download the choice now, with a progress bar.
    """
    name = state.get("engine")
    if not name:
        style.warn("no engine selected — pick one with /engine first")
        return
    eng = loki.engine(name)
    if arg:
        eng.model = arg
        state["model"] = arg
        style.ok(f"model → {style.brand(arg)} {style.dim('(' + name + ')')}")
        return

    current = None
    try:
        current = eng.resolve_model()
    except Exception:
        pass
    # The engine's preset ids (resource ladder or tier map), then any models the
    # local Ollama server has already pulled — deduped, first note wins.
    notes: dict[str, str] = {}
    for tier, mid in {**getattr(eng, "RESOURCE_MODELS", {}), **getattr(eng, "MODELS", {})}.items():
        notes.setdefault(mid, tier)
    if hasattr(eng, "installed_models"):
        try:
            for mid in eng.installed_models():
                notes.setdefault(mid, "installed")
        except Exception:
            pass
    options = list(notes.items())
    if not options:
        style.out(f"  {style.dim(f'engine {name} has no presets — pin one with /model <id>')}\n")
        return

    recommended = eng.bootstrap_model if eng.local else (getattr(eng, "MODELS", {}) or {}).get("deep")
    style.out(f"  {style.bold('models')} {style.dim('· ' + name + ' · current ' + (current or 'auto'))}\n")
    for i, (mid, note) in enumerate(options, 1):
        marks = [m for m, on in (("current", mid == current), ("recommended", mid == recommended)) if on]
        tail = style.dim(" (" + ", ".join(marks) + ")") if marks else ""
        style.out(f"    {style.brand(str(i))}  {mid.ljust(28)} {style.dim(note)}{tail}\n")
    try:
        ans = input(f"  pick [1-{len(options)}], type an id, or Enter to keep: ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not ans:
        return
    chosen = options[int(ans) - 1][0] if ans.isdigit() and 1 <= int(ans) <= len(options) else ans
    eng.model = chosen
    state["model"] = chosen
    style.ok(f"model → {style.brand(chosen)} {style.dim('(' + name + ')')}")
    if eng.local:
        try:
            go = input("  download / ready it now? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            go = ""
        if go in ("y", "yes"):
            _ready_model(loki, style, name, chosen)


def _ready_model(loki: Any, style: Any, engine: str, model: str) -> None:
    """Download/ready *model* now, with a progress bar — Ollama streams a byte
    bar; the HF transformers engine shows its own download bar as it fetches."""
    if engine == "ollama":
        res = loki.bootstrap_local(model=model, on_progress=_pull_bar(style))
        style.clear_line()
        if res.get("ready"):
            style.ok(f"{res['engine']} ready · {res['model']} ({res.get('status', 'ok')})")
        else:
            style.warn("could not ready it — see /setup")
        return
    # transformers: warm() builds the pipeline, downloading weights on first use
    # (HF prints the download progress); it swallows errors, so confirm after.
    style.out(f"  {style.dim('downloading ' + model + ' — progress below…')}\n")
    eng = loki.engine(engine)
    eng.warm(model)
    if getattr(eng, "ready", lambda _m: False)(model):
        style.ok(f"{engine} ready · {model}")
    else:
        style.warn(f"could not ready {model} — check the log above or try /setup")


def _local_load_notice(agent: Any, style: Any, engine: "str | None") -> None:
    """Warn that a local model is about to load (slow + silent on a first run).

    The first turn on a fresh box downloads weights and builds the pipeline
    before a single token streams; say so up front so the wait isn't a black
    box (the engine then logs the load itself via the routed loki logger). Covers
    both local pipeline engines — ``transformers`` (torch) and ``openvino``."""
    if engine not in ("transformers", "openvino"):
        return
    eng = agent.engine(engine)
    model = eng.resolve_model()
    if not eng.ready(model):
        # Name the device it'll actually load on so the accelerator is visible:
        # torch's cuda/xpu/mps, or OpenVINO's NPU/GPU/CPU.
        device = None
        try:
            device = eng.resolve_device()
        except Exception:
            pass
        where = {"cuda": "the NVIDIA GPU", "xpu": "the Intel GPU", "mps": "the Apple GPU",
                 "NPU": "the Intel NPU", "GPU": "the Intel GPU", "CPU": "CPU"}.get(device or "", "CPU")
        extra = " (and converts to OpenVINO IR)" if engine == "openvino" else ""
        style.out(f"  {style.dim('▹ loading local model')} {style.brand(model)} "
                  f"{style.dim(f'· first run downloads weights{extra}, then runs on {where} — this can take a while')}\n")


def _stream_reply(agent: Any, style: Any, line: str, state: dict,
                  *, engine: "str | None" = None, system: "str | None" = None) -> str:
    """Submit the reply asynchronously and yield it to the terminal as it lands.

    The reasoning stream runs on a worker thread that feeds a queue; the main
    thread shows a spinner until the first token, then drains the queue,
    printing tokens the instant they arrive. Decoupling production from the UI
    keeps the spinner smooth across token gaps and the terminal responsive,
    and surfaces a stream error back on the main thread."""
    import queue
    import threading

    _local_load_notice(agent, style, engine)
    q: "queue.Queue[Any]" = queue.Queue()
    done = object()
    error: list[BaseException] = []

    def produce() -> None:
        try:
            for chunk in agent.reason_stream(line, engine=engine,
                                             tier=state["tier"], system=system):
                q.put(chunk)
        except BaseException as exc:           # surfaced + re-raised on the main thread
            error.append(exc)
        finally:
            q.put(done)

    threading.Thread(target=produce, daemon=True).start()   # submit

    parts: list[str] = []
    spin = style.Spinner(f"{engine} · thinking…" if engine else "thinking…").start()
    try:
        while True:
            chunk = q.get()                    # yield as tokens land
            if chunk is done:
                break
            if spin is not None:               # first token — drop the spinner, open output
                spin.stop()
                spin = None
                style.out("  ")
            style.out(chunk)
            parts.append(chunk)
    finally:
        if spin is not None:                   # nothing streamed (empty or error)
            spin.stop()
    if error:
        raise error[0]
    style.out("\n\n")
    return "".join(parts)


def _escalation_confirm(style: Any):
    """A confirm callback for :meth:`Loki.select` — asks before auto-switching
    from a free local model up to a paid remote one for a heavy task."""
    def ask(engine: Any, model: "str | None") -> bool:
        label = f"{engine.name} {model}" if model else engine.name
        try:
            ans = input(f"  {style.amber('⤴ escalate')} this looks heavy — switch up to "
                        f"{style.brand(label)} {style.dim('(remote, paid)')}? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("", "y", "yes")
    return ask


def _turn_engine(loki: Any, style: Any, state: dict, line: str) -> "str | None":
    """The engine to use for this turn: complexity drives it **both ways** —
    light work drops to a free local model, heavy work climbs to a remote
    (asking first). Resolves once, up front (so any escalation prompt fires
    before output), notes a silent demotion, and returns the engine to pin.
    """
    base = state.get("engine")
    chosen = loki.select(line, tier=state["tier"], base=base,
                         confirm=_escalation_confirm(style))
    if chosen is None:
        return base
    # Surface a remote→local demotion (the cost-saving switch is silent
    # otherwise; an escalation already announced itself via the confirm).
    if chosen.name != base and chosen.local:
        style.out(f"  {style.dim('↓ local')} {style.brand(chosen.name)} "
                  f"{style.dim('· lighter task, kept free/on-box')}\n")
    return chosen.name


def _short_text(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


class _ActMonitor:
    """Live view of the autonomous ``act`` loop.

    A spinner + step-budget bar runs through the model's otherwise-silent
    "thinking" (``on_think``), then each finished turn is *committed* as a clean
    line to the scrollback (``on_step``). The cursor line always shows what's
    happening **now**; the transcript above it accumulates the important steps —
    each with the one-line *thought* that drove it. ``confirm`` pauses the
    spinner so an approval prompt never fights the animation.
    """

    def __init__(self, style: Any, max_steps: int, *, verbose: bool = False) -> None:
        self.style = style
        self.max = max_steps
        self.verbose = verbose          # the `do` CLI shows observations; the REPL stays terse
        self.count = 0                  # tool steps committed (the visible "important steps")
        self._spin: Any = None

    def think(self, n: int) -> None:
        """Turn *n* is about to call the model — spin on it, advancing the bar."""
        label = f"{self.style.dim(f'step {n}/{self.max}')} {self.style.dim('· thinking…')}"
        if self._spin is None:
            self._spin = self.style.Spinner(label).start()
        else:
            self._spin.update(label)
        self._spin.set_progress(n - 1, self.max)

    def step(self, rec: dict) -> None:
        """A turn finished — drop the spinner and commit the step (cumulative)."""
        self._drop()
        if rec.get("done"):
            return                       # the final answer is printed by the caller
        self.count += 1
        call = f"{rec['tool']}({', '.join(f'{k}={_short(v)}' for k, v in rec['args'].items())})"
        self.style.out(f"  {self.style.dim(str(rec['n']).rjust(2))} {self.style.brand(call)}\n")
        if rec.get("thought"):           # the thinking behind the step — the bit worth seeing
            self.style.out(f"     {self.style.dim('↳ ' + _short_text(rec['thought'], 110))}\n")
        if self.verbose and rec.get("observation"):
            first = rec["observation"].splitlines()[0]
            self.style.out(f"     {self.style.dim('→ ' + _short(first))}\n")

    def confirm(self, action: str) -> bool:
        """Approve a destructive op — pausing the spinner so the prompt is clean."""
        self._drop()
        return _confirm(self.style, action)

    def close(self) -> None:
        self._drop()

    def _drop(self) -> None:
        if self._spin is not None:
            self._spin.stop()
            self._spin = None


#: Spinner frames for the live fleet dashboard (one running agent per line).
_FLEET_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _fleet_lines(style: Any, agents: list, frame: str) -> list[str]:
    """Render one dashboard line per process agent — glyph, task, status, time."""
    lines: list[str] = []
    for a in agents:
        if a.running:
            glyph, st = style.brand(frame), style.dim("running")
        elif a.ok:
            glyph, st = style.good("✓"), style.good("done")
        elif a.status == "timeout":
            glyph, st = style.amber("⏱"), style.amber("timeout")
        else:
            glyph, st = style.bad("✗"), style.bad("failed")
        if a.ok and a.files_changed:
            tail = style.dim("✎ " + ", ".join(a.files_changed[:3]))
        elif not a.running and not a.ok and a.stderr_tail:
            tail = style.dim(_short_text(a.stderr_tail, 40))
        else:
            tail = ""
        # Validation badge — did this agent self-check with a passing smoke test?
        if not a.running:
            badge = ({True: style.good("✓test"), False: style.bad("✗test")}
                     .get(a.validated, style.amber("untested")))
        else:
            badge = ""
        lines.append(
            f"  {glyph} {style.dim('#' + str(a.id))} {_short_text(a.task, 40).ljust(41)} "
            f"{st.ljust(8)} {badge.ljust(8)} {style.dim(f'{a.elapsed:4.1f}s')}  {tail}"
        )
    return lines


def _fleet_kpi_line(style: Any, fleet: Any) -> str:
    """A one-line KPI footer for the live fleet dashboard."""
    k = fleet.kpis()
    bits = [style.good(f"{k.done} done")]
    if k.failed:
        bits.append(style.bad(f"{k.failed} failed"))
    if k.running:
        bits.append(style.brand(f"{k.running} running"))
    if k.queued:
        bits.append(style.dim(f"{k.queued} queued"))
    if k.validated:
        bits.append(style.good(f"{k.validated} tested"))
    cost = f"${k.cost:.4f}" + (f"/${fleet.cost_cap:.2f}" if fleet.cost_cap else "")
    tail = f"{k.steps} steps · {k.tokens:,} tok · {cost} · {k.elapsed:.1f}s"
    return f"  {style.dim('agents')} {'  '.join(bits)}  {style.dim('· ' + tail)}"


def _fleet_cap_prompt(style: Any, kpis: Any) -> "float | None":
    """The fleet hit its cost cap with work queued — raise it, or stop. Returns
    the new cap, or ``None`` to stop launching and let runners finish."""
    style.warn(f"fleet cost ${kpis.cost:.4f} reached the cap — "
               f"{kpis.done} done, {kpis.queued} still queued")
    try:
        ans = input("  go further? raise cap to $N · Enter for +$1 · stop [s]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if ans in ("s", "stop", "n", "no"):
        return None
    amt = _usd(ans)
    return amt if amt is not None else round(kpis.cost + 1.0, 2)


def _run_fleet(loki: Any, style: Any, state: dict, tasks: list, *,
               max_parallel: "int | None" = None) -> Any:
    """Spawn a process agent per task and live-monitor the mesh to completion."""
    from yggdrasil.loki.fleet import Fleet
    from yggdrasil.loki.usage import METER

    note = (f" · ≤{max_parallel} at once" if max_parallel else "") + " · local×1"
    style.out(f"  {style.brand('⛓ delegating')} {style.dim(str(len(tasks)) + ' mesh agents')} "
              f"{style.dim('· ' + (state.get('engine') or 'auto') + ' · root ' + state['root'] + note)}\n")
    # Shared workspace (mesh files + mesh.json roster); aggregate cost cap from the
    # session budget — the fleet asks before going past it.
    fleet = Fleet(max_parallel=max_parallel, cost_cap=METER.cost_limit, mesh_dir=state["root"])
    fleet.spawn_all(tasks, root=state["root"], engine=state.get("engine"),
                    tier=state["tier"], allow_web=True)
    live = style.LiveDisplay()
    tick = {"i": 0}

    def render(agents: list) -> None:
        tick["i"] += 1
        frame = _FLEET_FRAMES[tick["i"] % len(_FLEET_FRAMES)]
        live.update(_fleet_lines(style, agents, frame) + [_fleet_kpi_line(style, fleet)])

    try:
        fleet.monitor(render, interval=0.12, on_cap=lambda k: _fleet_cap_prompt(style, k))
    except KeyboardInterrupt:
        fleet.cancel_all()
        render(fleet.agents)
        style.warn("fleet cancelled")
    finally:
        live.stop()

    done = sum(1 for a in fleet.agents if a.ok)
    failed = len(fleet.agents) - done
    bits = [style.good(f"{done} done")] + ([style.bad(f"{failed} failed")] if failed else [])
    style.out(f"  {style.dim('—')} {'  '.join(bits)}\n")
    changed = sorted({f for a in fleet.agents for f in a.files_changed})
    if changed:
        style.out(f"  {style.good('✎')} {', '.join(changed)}\n")
    return fleet


def _scaffold_args(text: str) -> dict:
    """Extract scaffold kwargs (name, languages, preset, cloud) from free text."""
    from yggdrasil.loki import scaffold

    # "called X" / "named X" are reliable; "project"/"app"/"repo" are usually
    # nouns ("a python project"), so they don't introduce the name.
    m = re.search(r"\b(?:called|named)\s+['\"]?([A-Za-z][\w.-]{1,40})['\"]?", text)
    if not m:
        m = re.search(r"['\"]([A-Za-z][\w .-]{1,40})['\"]", text)        # a quoted name
    name = (m.group(1).strip().replace(" ", "-") if m else "new-project")
    return {
        "name": name,
        "languages": scaffold.resolve_languages(text) or ["python"],
        "preset": scaffold.resolve_preset(text),
        "cloud": scaffold.resolve_cloud(text),
        "description": text.strip(),
    }


def _delegate_tasks(agent: Any, style: Any, state: dict, text: str) -> list:
    """Turn a delegate/swarm request into a task list — split an explicit list,
    else let the engine decompose a single high-level goal."""
    body = re.sub(r"^\s*(please\s+)?(delegate|swarm|spawn|launch|run|do)\b[:,]?\s*", "", text, flags=re.I)
    body = re.sub(r"\b(in parallel|concurrently|at the same time|simultaneously|"
                  r"as (parallel|multiple|several|background) agents|fan out)\b", "", body, flags=re.I)
    body = body.strip(" .,;:")
    parts = [p.strip(" .,;:") for p in re.split(r"\s*;\s*|\s+and\s+|\s*,\s*", body) if p.strip(" .,;:")]
    if len(parts) >= 2:
        return parts
    with style.Spinner("decomposing into parallel subtasks…"):
        return agent.decompose(body or text, engine=state.get("engine"), tier=state["tier"])


def _print_scaffold(style: Any, res: dict) -> None:
    kind = res["preset"] if res["preset"] != "lib" else ", ".join(res["languages"])
    cloud = (" · cloud " + ", ".join(res["cloud"])) if res.get("cloud") else ""
    style.out(f"  {style.good('✦')} {style.bold(res['name'])} {style.dim('· ' + kind + cloud)}\n")
    style.out(f"  {style.dim('at ' + res['path'])}\n")
    for f in res["files"][:14]:
        style.out(f"    {style.good('+')} {style.dim(f)}\n")
    if len(res["files"]) > 14:
        style.out(f"    {style.dim('… +' + str(len(res['files']) - 14) + ' more')}\n")
    git = (style.good("git ✓ initial commit on main") if res["git"]
           else style.amber("git not initialised — commit when ready"))
    style.out(f"  {git}\n")
    style.out(f"  {style.brand('→ push')} {style.dim(res['push'])}\n")


def _repl_turn(loki: Any, style: Any, state: dict, line: str) -> None:
    plan = loki.plan(line)

    agent, tail = loki, ""
    if plan["specialist"]:
        spec = loki.specialist(plan["specialist"])
        if spec is not None:
            agent, tail = spec, f"  →  {style.brand(spec.name)} {style.dim('(isolated)')}"
    persona = plan.get("persona", "assistant")
    badge = f"{plan['category']} · {style.brand(persona)}" if persona != "assistant" else plan["category"]
    style.out(f"  {style.dim('▹ ')}{style.dim(badge)}{style.dim(' · ' + plan['why'])}{tail}\n")

    streamed = False
    try:
        if plan["action"] == "tabular" and plan.get("url"):
            cache_dir = getattr(state.get("session"), "cache_dir", None)
            with style.Spinner(f"fetching {_short_text(plan['url'], 52)}…"):
                res = agent.run("tabular", url=plan["url"],
                                cache_dir=str(cache_dir) if cache_dir else None)
            _print_tabular(style, res)
            reply = (f"Fetched {res['rows']}×{len(res['columns'])} from {res['source']}; "
                     f"cached → {res['cached_to']}")
        elif plan["action"] == "web" and plan.get("url"):
            with style.Spinner(f"fetching {_short_text(plan['url'], 52)}…"):
                res = agent.run("web", url=plan["url"], question=line)
            _print_web(style, res)
            reply = res.get("answer") or style.dim(f"fetched {plan['url']}")
        elif plan["action"] == "act":
            eng_name = _turn_engine(agent, style, state, line)
            mon = _ActMonitor(style, _REPL_ACT_STEPS)
            try:
                res = agent.act(line, root=state["root"], engine=eng_name, tier=state["tier"],
                                max_steps=_REPL_ACT_STEPS, allow_web=True,
                                confirm=mon.confirm, on_think=mon.think, on_step=mon.step)
            finally:
                mon.close()
            reply = res["answer"]
            if mon.count:
                style.out(f"  {style.dim(f'· {mon.count} step' + ('s' if mon.count != 1 else ''))}\n")
            if res.get("files_changed"):
                style.out(f"  {style.good('✎')} {', '.join(res['files_changed'])}\n")
        elif plan["action"] == "scaffold":
            args = _scaffold_args(line)
            kind = args["preset"] if args["preset"] != "lib" else ", ".join(args["languages"])
            with style.Spinner(f"scaffolding {args['name']} ({kind})…"):
                res = agent.run("scaffold", **args)
            _print_scaffold(style, res)
            reply = f"Scaffolded {res['name']} ({', '.join(res['languages'])}) → {res['path']}"
            streamed = True
        elif plan["action"] == "delegate":
            tasks = _delegate_tasks(agent, style, state, line)
            if not tasks:
                reply = style.dim("nothing to delegate")
            else:
                style.out(f"  {style.bold('plan')} {style.dim(str(len(tasks)) + ' parallel subtasks')}\n")
                for t in tasks:
                    style.out(f"    {style.brand('•')} {style.dim(_short_text(t, 70))}\n")
                fleet = _run_fleet(loki, style, state, tasks)
                done = sum(1 for a in fleet.agents if a.ok)
                reply = f"{done}/{len(fleet.agents)} agents completed"
            streamed = True
        elif plan["action"] == "genie":
            with style.Spinner("asking Genie…"):
                res = agent.run("genie", question=line)
            reply = res.get("text", "") if isinstance(res, dict) else str(res)
        elif plan["action"] == "skill" and plan.get("skill"):
            # A precise databricks request → dispatch the specialized skill
            # (already printed by _print_dbx, so don't echo the reply again).
            reply = _dispatch_skill(agent, style, plan["skill"], plan.get("skill_kwargs") or {})
            streamed = True
        elif plan["action"] == "guide":
            with style.Spinner("composing the yggdrasil recipe…"):
                res = agent.run("guide", task=line, plan=True)
            _print_guide(style, res)
            reply = res.get("plan") or style.dim("see the yggdrasil recipes above")
            if res.get("plan"):
                style.out(f"\n  {res['plan']}\n")
        else:
            # chat (or a web verb with no URL) → stream the reply live, with
            # the persona system prompt + session memory as context. The engine
            # is resolved up front: light → base/local, heavy → remote (asked).
            memory = state.get("memory")
            parts = [p for p in (plan.persona_prompt(),
                                 memory.system_context() if memory is not None else None) if p]
            system = "\n\n".join(parts) or None
            eng_name = _turn_engine(agent, style, state, line)
            reply = _stream_reply(agent, style, line, state, engine=eng_name, system=system)
            streamed = True
    except Exception as exc:  # never let one turn kill the session
        style.out("\n")
        style.fail(exc.args[0] if exc.args else str(exc))
        return
    if not streamed:
        style.out(f"\n  {reply}\n\n")
    # Record the turn in memory and auto-compress when it grows.
    memory = state.get("memory")
    if memory is not None:
        memory.add("user", line)
        memory.add("assistant", _short_text(reply, 1200))
        # Compression is a real (slow) model call — spin only when it'll run.
        if memory.should_compress():
            with style.Spinner("compressing session memory…"):
                compressed = memory.maybe_compress(loki, engine=state.get("engine"))
            if compressed:
                note = f"· memory compressed → {memory.chars()} chars"
                style.out(f"  {style.dim(note)}\n")
    # Token/cost KPIs ride the terminal title bar (a static "head"), not a fresh
    # line each turn — local turns are free, so the spend only moves on a remote.
    style.set_title(_usage_title())


def _confirm(style: Any, action: str) -> bool:
    """Ask the user to approve a destructive op on a non-temporary asset."""
    try:
        ans = input(f"  {style.amber('⚠ confirm')} {action}? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


def _print_tabular(style: Any, res: dict) -> None:
    """Print a fetched-and-cached tabular frame + the proposed next steps."""
    style.out(f"  {style.dim('▦ ' + str(res['source']))}\n")
    shape = f"{res['rows']} × {len(res['columns'])}"
    style.out(f"  {style.good('▦')} {style.bold(shape)} "
              f"{style.dim('· ' + ', '.join(res['columns'][:8]))}\n")
    style.out(style.dim("  " + res["preview"].replace("\n", "\n  ")) + "\n")
    style.out(f"  {style.good('✎ cached')} {style.dim(res['cached_to'])}  "
              f"{style.dim('(Parquet, via io)')}\n")
    if res.get("stored"):
        style.out(f"  {style.good('✎ stored')} {style.dim(res['stored'])}\n")
    style.out(f"  {style.bold('next steps')} {style.dim('— reuse · store · load')}\n")
    for step in res["next_steps"]:
        style.out(f"    {style.brand('›')} {style.dim(step)}\n")


def _print_web(style: Any, res: dict) -> None:
    """Pretty-print a `web` skill result by kind: table / image / json / page."""
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


def _dispatch_skill(agent: Any, style: Any, skill: str, kwargs: dict) -> str:
    """Run a specialized skill the planner picked, print it, return a memory line."""
    shown = f" {style.dim(str(kwargs))}" if kwargs else ""
    style.out(f"  {style.dim('▹ dispatched')} {style.brand(skill)}{shown}\n")
    try:
        with style.Spinner(f"running {skill}…"):
            res = agent.run(skill, **kwargs)
    except Exception as exc:
        style.fail(f"{skill}: {exc.args[0] if exc.args else exc}")
        return f"{skill} failed: {exc}"
    _print_dbx(style, res)
    return _short(_jsonable(res))


def _print_dbx(style: Any, res: Any) -> None:
    """Compact print of a Databricks skill result.

    Row sets render through the object's own preview — ``Tabular.display()`` for
    a statement result / io leaf, polars' own aligned repr for a frame — never a
    hand-rolled table.
    """
    if not isinstance(res, dict):
        style.out(f"    {_short(res)}\n")
        return
    for k, v in res.items():
        if callable(getattr(v, "display", None)):        # a yggdrasil Tabular
            block = v.display().replace("\n", "\n    ")
            style.out(f"    {style.good('▦')} {style.bold(k)}\n    {style.dim(block)}\n")
        elif hasattr(v, "to_dicts") and hasattr(v, "head"):   # a polars frame
            block = str(v.head(10)).replace("\n", "\n    ")
            style.out(f"    {style.good('▦')} {style.bold(k)}\n    {style.dim(block)}\n")
        elif isinstance(v, list):
            style.out(f"    {style.dim(k)} {style.dim('(' + str(len(v)) + ')')}\n")
            for item in v[:30]:
                style.out(f"      {style.brand('·')} {style.dim(_short(item))}\n")
        else:
            style.out(f"    {style.dim(k.ljust(12))} {_short(v)}\n")


def _print_guide(style: Any, res: dict) -> None:
    """Pretty-print the `guide` skill — the matched yggdrasil recipes."""
    style.out(f"  {style.bold('yggdrasil recipes')} {style.dim('— the optimized path')}\n")
    for g in res.get("guides", []):
        style.out(f"  {style.brand('▸')} {style.bold(g['title'])}\n")
        style.out(f"    {style.dim(g['summary'])}\n")
        for u in g["use"][:3]:
            style.out(f"      {style.good('use')} {style.dim(u)}\n")
        for a in g["avoid"][:2]:
            style.out(f"      {style.amber('avoid')} {style.dim(a)}\n")


def _repl_command(loki: Any, style: Any, state: dict, line: str) -> bool:
    """Handle a /slash command. Returns False only to end the session."""
    from yggdrasil.loki.usage import METER

    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("help", "h", "?"):
        style.out(
            f"  {style.bold('commands')}\n"
            f"    {style.brand('/engine')}   pick the session base engine (claude/databricks/openai/ollama/auto)\n"
            f"    {style.brand('/model')}    pick the model for this engine (lists presets · downloads with a bar)\n"
            f"    {style.brand('/fleet')}    delegate tasks to parallel background agents:  /fleet t1 ;; t2 ;; t3\n"
            f"    {style.brand('/swarm')}    autonomous: decompose a goal into subtasks + run them as a monitored fleet\n"
            f"    {style.brand('/engines')}  reasoning engines and adaptive models\n"
            f"    {style.brand('/setup')}    bootstrap a free local model (lazy-install on demand)\n"
            f"    {style.brand('/status')}   identity + backends + engines + skills\n"
            f"    {style.brand('/usage')}    token KPIs (per model + global, USD)\n"
            f"    {style.brand('/tier')}     fast | deep | auto  (model tier for this session)\n"
            f"    {style.brand('/root')}     set the working tree for file tasks\n"
            f"    {style.brand('/budget')}   cost cap — show, set $N, +$N step, or off\n"
            f"    {style.brand('/memory')}   show the session's synthesized memory\n"
            f"    {style.brand('/sessions')} list session workspaces (under ~/.loki)\n"
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
    elif cmd == "model":
        _choose_model(loki, style, state, arg)
    elif cmd == "fleet":
        tasks = [t.strip() for t in arg.split(";;") if t.strip()]
        if not tasks:
            style.warn("usage: /fleet <task1> ;; <task2> ;; …  (parallel background agents)")
        else:
            _run_fleet(loki, style, state, tasks)
    elif cmd == "swarm":
        eng = loki.engine(state["engine"]) if state.get("engine") else loki.engine()
        if not arg:
            style.warn("usage: /swarm <high-level goal>  (Loki decomposes + delegates it)")
        elif eng is None or not eng.available():
            style.warn("no engine available to decompose the goal — set one with /engine")
        else:
            with style.Spinner("decomposing the goal into parallel subtasks…"):
                tasks = loki.decompose(arg, engine=state.get("engine"), tier=state["tier"])
            style.out(f"  {style.bold('plan')} {style.dim(str(len(tasks)) + ' parallel subtasks')}\n")
            for t in tasks:
                style.out(f"    {style.brand('•')} {style.dim(_short_text(t, 70))}\n")
            _run_fleet(loki, style, state, tasks)
    elif cmd == "status":
        loki.load_specialists()   # ensure the specialist fleet is listed even
        _status(loki, style)      # if the background warmer hasn't finished yet
    elif cmd == "engines":
        _engines(loki, style)
    elif cmd == "setup":
        _setup(loki, style, arg)
    elif cmd == "usage":
        _usage_table(style)
    elif cmd == "tier":
        state["tier"] = None if arg in ("", "auto") else arg
        style.ok(f"tier → {state['tier'] or 'auto (adaptive)'}")
    elif cmd == "root":
        state["root"] = arg or "."
        style.ok(f"root → {state['root']}")
    elif cmd == "budget":
        amt = _usd(arg.lstrip("+"))
        if arg in ("off", "none"):
            METER.set_limit(None); style.ok("cost budget off (unlimited)")
        elif arg.startswith("+") and amt is not None:
            METER.raise_limit(amt); style.ok(f"cost budget → ${METER.cost_limit:.2f}")
        elif amt is not None:
            METER.set_limit(amt); style.ok(f"cost budget → ${METER.cost_limit:.2f}")
        else:
            lim = "off" if METER.cost_limit is None else f"${METER.cost_limit:.2f}"
            style.out(f"  {style.dim('cost budget')} {style.brand(lim)}  "
                      f"{style.dim('spent')} ${METER.total_cost:.4f}\n")
    elif cmd == "memory":
        memory = state.get("memory")
        if memory is None or (not memory.synthesis and not memory.turns):
            style.out(f"  {style.dim('(memory is empty)')}\n")
        else:
            if memory.synthesis:
                style.out(f"  {style.bold('synthesis')}\n  {style.dim(memory.synthesis)}\n")
            stat = f"{len(memory.turns)} recent turns · {memory.chars()} chars"
            style.out(f"  {style.dim(stat)}\n")
    elif cmd == "sessions":
        from yggdrasil.loki.session import LokiSession

        sessions = LokiSession.list()
        current_id = getattr(state.get("session"), "id", None)
        if not sessions:
            style.out(f"  {style.dim('(no sessions yet)')}\n")
        for s in sessions[:20]:
            cur = style.good(" ← current") if s.id == current_id else ""
            style.out(f"  {style.bold((s.name or '(unnamed)').ljust(40))} "
                      f"{style.dim(s.id)} {style.dim(_ago(s.last_used_at) + ' ago')}{cur}\n")
        style.out(f"  {style.dim('per-user · auto-purged: keep 20 most-recent, drop >14 days idle')}\n")
    elif cmd == "reset":
        METER.reset(); style.ok("usage meter reset")
    else:
        style.warn(f"unknown command /{cmd} — try /help")
    return True


def _budget_prompt(style: Any) -> bool:
    """At/over the cost cap (checked between actions): raise by a step, set a
    custom cap, or stop. Returns whether to continue."""
    from yggdrasil.loki.usage import METER

    style.warn(f"cost budget reached — ${METER.total_cost:.4f} ≥ ${METER.cost_limit:.2f}")
    try:
        ans = input(
            f"  raise by ${METER.cost_step:.2f} [Enter] · set $N · off · stop [s]: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans in ("s", "stop", "n", "no"):
        return False
    if ans in ("off", "none"):
        METER.set_limit(None); style.ok("cost budget off (unlimited)"); return True
    amt = _usd(ans)
    if amt is not None:
        METER.set_limit(amt); style.ok(f"cost budget → ${METER.cost_limit:.2f}"); return True
    METER.raise_limit(); style.ok(f"cost budget → ${METER.cost_limit:.2f}"); return True


def _usd(s: str) -> "float | None":
    """Parse a USD amount like ``1``, ``1.50``, ``$2`` → float, else None."""
    try:
        return float(s.strip().lstrip("$"))
    except (ValueError, AttributeError):
        return None


def _usage_title() -> str:
    """A compact one-line KPI string for the terminal title bar (no color).

    Shows global tokens, USD spent, and budget left — the same numbers as the
    per-turn ``usage`` line, but pinned statically in the terminal head so the
    transcript isn't padded with a usage line after every (often free) turn."""
    from yggdrasil.loki.usage import METER

    t = METER.total()
    bits = ["ygg loki", f"↑{t.input_tokens:,} ↓{t.output_tokens:,}",
            f"{t.total_tokens:,} tok", f"${METER.total_cost:.4f}"]
    if METER.cost_limit is not None:
        bits.append(f"${(METER.remaining() or 0.0):.2f} left")
    return "  ·  ".join(bits)


def _usage_line(style: Any, *, delta: int | None = None, prefix: str = "usage") -> None:
    from yggdrasil.loki.usage import METER

    t = METER.total()
    bits = [
        f"{style.dim('↑')}{t.input_tokens:,} {style.dim('↓')}{t.output_tokens:,}",
        f"{style.bold(f'{t.total_tokens:,}')} {style.dim('tok')}",
        style.good(f"${METER.total_cost:.4f}"),
    ]
    if delta:
        bits.append(style.dim(f"(+{delta:,} tok)"))
    if METER.cost_limit is not None:
        rem = METER.remaining() or 0.0
        bits.append((style.good if rem > 0 else style.bad)(f"${rem:.4f} left"))
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
    if METER.cost_limit is not None:
        rem = METER.remaining() or 0.0
        g = style.good if rem > 0 else style.bad
        style.out(f"  {style.dim('cost budget')} {style.brand(f'${METER.cost_limit:.2f}')}  "
                  f"{g(f'${rem:.4f} left')}\n")
    style.out(f"  {style.dim('pricing USD/1M tok — defaults in yggdrasil.loki.usage.PRICING')}\n")
    return 0


def _reason(loki: Any, style: Any, args: Any) -> int:
    try:
        with style.Spinner("reasoning…"):
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

    if args.budget is not None:                       # self-cap this agent's spend
        from yggdrasil.loki.usage import METER

        METER.set_limit(args.budget)

    mon = _ActMonitor(style, args.max_steps, verbose=True)
    try:
        result = loki.act(
            args.task, root=args.root, engine=args.engine, tier=args.tier, max_steps=args.max_steps,
            read_only=args.read_only, allow_shell=args.allow_shell, allow_web=args.allow_web,
            on_think=None if args.json else mon.think,
            on_step=None if args.json else mon.step,
        )
    except (KeyError, RuntimeError) as exc:
        style.fail(exc.args[0] if exc.args else str(exc))
        return 1
    finally:
        mon.close()

    if args.json:
        # Attach this agent's token/cost usage so a parent fleet can aggregate
        # live KPIs across the agents it delegated to.
        from yggdrasil.loki.usage import METER

        t = METER.total()
        result["usage"] = {"input_tokens": t.input_tokens, "output_tokens": t.output_tokens,
                           "total_tokens": t.total_tokens, "cost_usd": round(METER.total_cost, 6)}
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


def _skills(loki: Any, style: Any) -> int:
    skills = loki.skills()
    if not skills:
        style.out(f"  {style.dim('no skills registered')}\n")
        return 0
    for beh in skills:
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
        style.out(_json(result))
        style.out("\n")
    else:
        style.ok(f"skill {args.name!r} completed")
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


def _jsonable(obj: Any) -> Any:
    """Coerce a skill result into JSON-serializable shapes.

    Skill results carry rich objects — chiefly polars/pandas frames (a SQL or
    Genie row set) — that orjson can't encode. Frames become records (capped),
    dicts/lists recurse, models/SDK objects fall back to ``to_dict()`` or a
    string. Keeps ``ygg loki run … --json`` working whatever a skill returns.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # Row sets convert through their own method — Tabular.to_pylist (statement
    # result / io leaf) or polars to_dicts — capped; never re-serialized by hand.
    for method in ("to_pylist", "to_dicts"):
        fn = getattr(obj, method, None)
        if callable(fn):
            try:
                rows = list(fn())
                return _jsonable(rows[:1000] if len(rows) > 1000 else rows)
            except Exception:
                return str(obj)
    to_dict = getattr(obj, "to_dict", None)          # pandas / Pydantic models
    if callable(to_dict):
        try:
            return _jsonable(to_dict())
        except Exception:
            pass
    return str(obj)


def _json(result: Any) -> str:
    """JSON-encode a result for the CLI (orjson emits bytes → decode to str)."""
    from yggdrasil.pickle import json as yjson

    out = yjson.dumps(_jsonable(result), indent=2)
    return out.decode() if isinstance(out, (bytes, bytearray)) else out


if __name__ == "__main__":
    import sys

    sys.exit(main())
