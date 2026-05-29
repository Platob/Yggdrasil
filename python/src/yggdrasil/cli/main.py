"""``ygg`` — unified CLI entry point for yggdrasil.

Subcommands::

    ygg node start      Start node (background daemon, public by default)
    ygg node stop       Stop the running node
    ygg node serve      Start node + frontend (foreground)
    ygg node back       Start backend only (Uvicorn API, foreground)
    ygg node front      Start frontend only (Next.js dev server, foreground)
    ygg node status     Show running node status
    ygg node watch      Live TTY dashboard — auto-refreshing node stats
    ygg node logs       Tail the node log file (-f to follow)
    ygg node ps         List active runs
    ygg node procs      Live per-run process metrics (CPU/RAM/duration)
    ygg node mesh       Live cluster health: this node + every peer
    ygg node call       Run a function by name and print result
    ygg node health     Run health checks on the node
    ygg node excel      Check Excel service + create/update the add-in manifest
    ygg node create     Create a new named node
    ygg node install    Install node as boot service (systemd/launchd)
    ygg node uninstall  Remove boot service (--purge to delete data)
    ygg node run        Call a @remote function
    ygg node chat       Open YGGCHAT terminal
    ygg databricks      YGGDBKS Databricks management CLI
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ygg",
        description="Yggdrasil CLI — distributed node framework.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")

    sub = parser.add_subparsers(dest="command")

    # -- node --------------------------------------------------------------
    node = sub.add_parser("node", help="Yggdrasil node lifecycle and execution.")
    node_sub = node.add_subparsers(dest="node_action")

    # start (background daemon)
    start = node_sub.add_parser("start", help="Start node as a background daemon (public by default).")
    start.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    start.add_argument("--port", type=int, default=None, help="Bind port (default: 8100, auto-scans if busy).")
    start.add_argument("--name", default=None, help="Node ID override.")
    start.add_argument("--persist", action="store_true", default=False, help="Also create/update the boot auto-start service (Task Scheduler on Windows).")
    start.set_defaults(handler=_node_start)

    # stop
    stop = node_sub.add_parser("stop", help="Stop the running node.")
    stop.set_defaults(handler=_node_stop)

    # serve (foreground)
    serve = node_sub.add_parser("serve", help="Start node + frontend in foreground.")
    serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    serve.add_argument("--port", type=int, default=None, help="Bind port (default: 8100).")
    serve.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload.")
    serve.add_argument("--no-front", action="store_true", default=False, help="Skip launching the frontend.")
    serve.add_argument("--name", default=None, help="Node ID override.")
    serve.add_argument("--persist", action="store_true", default=False, help="Create/update the boot auto-start service (Task Scheduler on Windows).")
    serve.set_defaults(handler=_node_serve)

    # status
    status = node_sub.add_parser("status", help="Show node status.")
    status.set_defaults(handler=_node_status)

    # watch — live TTY dashboard
    watch = node_sub.add_parser("watch", help="Live TTY dashboard — auto-refreshing node stats.")
    watch.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds.")
    watch.add_argument("--url", default=None, help="Node URL (default: local).")
    watch.set_defaults(handler=_node_watch)

    # logs — tail the node log
    logs = node_sub.add_parser("logs", help="Tail the node log file.")
    logs.add_argument("--follow", "-f", action="store_true", help="Follow log output.")
    logs.add_argument("--lines", "-n", type=int, default=50, help="Show last N lines.")
    logs.set_defaults(handler=_node_logs)

    # ps — list active runs
    ps = node_sub.add_parser("ps", help="List active runs (like docker ps).")
    ps.add_argument("--all", "-a", action="store_true", help="Show all runs, not just active.")
    ps.set_defaults(handler=_node_ps)

    # procs — live per-run process metrics
    procs = node_sub.add_parser("procs", help="Live per-run process metrics (CPU/RAM/duration).")
    procs.add_argument("--interval", type=float, default=1.5, help="Refresh interval in seconds.")
    procs.add_argument("--url", default=None, help="Node URL (default: local).")
    procs.set_defaults(handler=_node_procs)

    # mesh — live cluster health
    mesh = node_sub.add_parser("mesh", help="Live cluster health: this node + every peer.")
    mesh.add_argument("--interval", type=float, default=2.0, help="Refresh interval in seconds.")
    mesh.add_argument("--url", default=None, help="Node URL (default: local).")
    mesh.set_defaults(handler=_node_mesh)

    # call — run a function by name
    call = node_sub.add_parser("call", help="Run a function by name and print result.")
    call.add_argument("name", help="Function name.")
    call.add_argument("--arg", action="append", default=[], help="Positional arg (JSON-decoded).")
    call.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE", help="Keyword arg.")
    call.add_argument("--wait", type=float, default=60.0, help="Wait for completion (seconds).")
    call.set_defaults(handler=_node_call)

    # health — quick health check
    health_cmd = node_sub.add_parser("health", help="Run health checks on the node.")
    health_cmd.set_defaults(handler=_node_health)

    # excel — check + (re)generate the Excel integration artifacts
    excel_cmd = node_sub.add_parser(
        "excel",
        help="Check the Excel service and create/update the add-in manifest + Power Query connector.",
    )
    excel_cmd.add_argument("--check", action="store_true", default=False, help="Only check; don't write files.")
    excel_cmd.add_argument("--host", default=None, help="Node base URL to check (default: local node).")
    excel_cmd.add_argument("--install", action="store_true", default=False, help="Sideload the add-in into Excel's trusted folder (where supported).")
    excel_cmd.set_defaults(handler=_node_excel)

    # create
    create = node_sub.add_parser("create", help="Create a new named node (initializes ~/.node/<name>/).")
    create.add_argument("name", help="Node name/ID.")
    create.add_argument("--start", action="store_true", default=False, help="Start the node after creation.")
    create.add_argument("--port", type=int, default=None, help="Bind port.")
    create.set_defaults(handler=_node_create)

    # back
    back = node_sub.add_parser("back", help="Start backend API only (foreground).")
    back.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    back.add_argument("--port", type=int, default=None, help="Bind port (default: 8100).")
    back.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload.")
    back.add_argument("--name", default=None, help="Node ID override.")
    back.add_argument("--persist", action="store_true", default=False, help="Create/update the boot auto-start service (Task Scheduler on Windows).")
    back.set_defaults(handler=_node_back)

    # front
    front = node_sub.add_parser("front", help="Start frontend dev server only.")
    front.add_argument("--port", type=int, default=None, help="Frontend port (default: 3000).")
    front.add_argument("--node-port", type=int, default=None, help="Node API port to proxy to.")
    front.add_argument("--persist", action="store_true", default=False, help="Create/update the boot auto-start service (Task Scheduler on Windows).")
    front.set_defaults(handler=_node_front)

    # run
    run = node_sub.add_parser("run", help="Call a @remote function.")
    run.add_argument("func", help="Function key (e.g. 'mymodule:my_func').")
    run.add_argument("args", nargs="*", default=[], help="Positional arguments.")
    run.add_argument("--url", default=None, help="Node server URL (default: auto).")
    run.add_argument("--kwarg", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument("--timeout", type=float, default=600.0)
    run.add_argument("--stream", action="store_true", default=False)
    run.set_defaults(handler=_node_run)

    # chat
    chat = node_sub.add_parser("chat", help="Open YGGCHAT terminal.")
    chat.add_argument("--url", default=None, help="Node server URL (default: auto).")
    chat.add_argument("--user", default=None, help="Display name.")
    chat.add_argument("--channel", default="general", help="Initial channel.")
    chat.set_defaults(handler=_node_chat)

    # install
    install = node_sub.add_parser("install", help="Install node as a boot service (systemd/launchd).")
    install.add_argument("--no-front", action="store_true", default=False, help="Skip frontend service.")
    install.add_argument("--port", type=int, default=None, help="Override node port.")
    install.add_argument("--front-port", type=int, default=None, help="Override frontend port.")
    install.set_defaults(handler=_node_install)

    # uninstall
    uninstall = node_sub.add_parser("uninstall", help="Remove node boot service and stop.")
    uninstall.add_argument("--purge", action="store_true", default=False, help="Also remove all data in ~/.node/.")
    uninstall.set_defaults(handler=_node_uninstall)

    # -- databricks --------------------------------------------------------
    dbks = sub.add_parser("databricks", help="YGGDBKS Databricks management.", add_help=False)
    dbks.set_defaults(handler=_databricks)

    return parser


# ── helpers ──────────────────────────────────────────────────────

def _apply_node_env(args: argparse.Namespace) -> None:
    """Push CLI flags into env vars before settings are read."""
    import os
    name = getattr(args, "name", None)
    if name:
        os.environ["YGG_NODE_NODE_ID"] = name
    host = getattr(args, "host", None)
    if host:
        os.environ["YGG_NODE_HOST"] = host
    port = getattr(args, "port", None)
    if port:
        os.environ["YGG_NODE_PORT"] = str(port)
    os.environ["YGG_NODE_ALLOW_REMOTE"] = "1"


def _ensure_node_running() -> str:
    try:
        from yggdrasil.node.daemon import spawn_node
        _, port = spawn_node()
        return f"http://127.0.0.1:{port}"
    except Exception:
        return "http://127.0.0.1:8100"


def _start_frontend(settings, *, node_port: int, front_port: int | None = None, quiet: bool = True):
    import os
    import shutil
    import subprocess

    front_home = settings.front_home
    if not (front_home / "package.json").exists():
        from yggdrasil.cli.style import dim, out, yellow
        out(f"  {yellow('skip')}  frontend not found at {dim(str(front_home))}\n")
        return None

    npm = shutil.which("npm")
    if npm is None:
        npm = _ensure_nodejs()
    if npm is None:
        from yggdrasil.cli.style import dim, out, yellow
        out(f"  {yellow('skip')}  Node.js/npm unavailable and auto-install failed — "
            f"install Node.js to serve the frontend\n")
        return None

    if not (front_home / "node_modules").exists():
        from yggdrasil.cli.style import Spinner
        with Spinner("installing frontend dependencies...", color="33") as sp:
            subprocess.run(
                [npm, "install"],
                cwd=str(front_home),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            sp.stop()

    from yggdrasil.node.config import _find_open_port
    port = front_port or _find_open_port(settings.front_port, settings.front_port + 100)

    env = {**os.environ, "YGG_NODE_PORT": str(node_port), "PORT": str(port)}
    proc = subprocess.Popen(
        [npm, "run", "dev", "--", "--hostname", "0.0.0.0", "--port", str(port)],
        cwd=str(front_home),
        env=env,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
    )

    from yggdrasil.cli.style import bold, cyan, out
    out(f"  {cyan('front')} {bold(f'http://0.0.0.0:{port}')}\n")
    return proc


def _ensure_nodejs() -> "str | None":
    """Best-effort: make ``npm`` available, auto-installing Node.js via the
    platform package manager when missing. Returns the npm path or None.

    Tries one manager per platform (Homebrew on macOS; apt/dnf/yum on
    Linux; winget/choco on Windows). Anything else — no manager, no
    privileges, network blocked — falls through to None and the caller
    prints install guidance.
    """
    import platform
    import shutil
    import subprocess
    from yggdrasil.cli.style import dim, green, out, yellow

    npm = shutil.which("npm")
    if npm:
        return npm

    system = platform.system()
    # (probe binary, install argv) candidates, first available wins.
    candidates: list[tuple[str, list[str]]] = []
    if system == "Darwin":
        candidates = [("brew", ["brew", "install", "node"])]
    elif system == "Windows":
        candidates = [
            ("winget", ["winget", "install", "-e", "--id", "OpenJS.NodeJS", "--silent"]),
            ("choco", ["choco", "install", "nodejs", "-y"]),
        ]
    else:  # Linux / other POSIX
        if shutil.which("apt-get"):
            candidates = [("apt-get", ["sudo", "apt-get", "install", "-y", "nodejs", "npm"])]
        elif shutil.which("dnf"):
            candidates = [("dnf", ["sudo", "dnf", "install", "-y", "nodejs"])]
        elif shutil.which("yum"):
            candidates = [("yum", ["sudo", "yum", "install", "-y", "nodejs"])]

    mgr = next(((m, cmd) for m, cmd in candidates if shutil.which(m)), None)
    if mgr is None:
        out(f"  {yellow('!')} Node.js missing and no known package manager found "
            f"({dim('install from https://nodejs.org')})\n")
        return None

    name, cmd = mgr
    out(f"  {yellow('…')} Node.js missing — attempting install via {name}\n")
    try:
        subprocess.run(cmd, check=True, timeout=600)
    except Exception as exc:  # noqa: BLE001 — best-effort, report and bail
        out(f"  {yellow('!')} auto-install via {name} failed: {dim(str(exc))}\n")
        return None
    npm = shutil.which("npm")
    if npm:
        out(f"  {green('✓')} Node.js installed\n")
    return npm


def _persist_service(settings, *, no_front: bool = False) -> None:
    """Create/update the boot auto-start service for ``--persist``.

    Idempotent (install_service rewrites the unit/agent/scheduled-task).
    Best-effort: a failure to install — no systemd, locked-down
    Task Scheduler, missing privileges — is logged as a warning and does
    NOT abort the deploy; the foreground process still runs.
    """
    from yggdrasil.cli.style import dim, green, out, yellow
    try:
        from yggdrasil.node.service import install_service, is_windows
        ok, msg = install_service(settings, no_front=no_front)
    except Exception as exc:  # noqa: BLE001 — never fail the deploy on this
        out(f"  {yellow('warn')}  could not install auto-start service: {dim(str(exc))}\n")
        return
    if ok:
        kind = "scheduled task" if is_windows() else "boot service"
        out(f"  {green('✓')} persisted ({kind}): {dim(msg)}\n")
    else:
        out(f"  {yellow('warn')}  auto-start not installed: {dim(msg)}\n")


# ── handlers ─────────────────────────────────────────────────────

def _node_start(args: argparse.Namespace) -> int:
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, orange, out, print_logo
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.daemon import spawn_node

    _apply_node_env(args)
    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")

    host = getattr(args, "host", "0.0.0.0")

    out(f"  starting node...\n")
    pid, port = spawn_node(settings, host=host)

    out(f"  {green('✓')} node running\n")
    out(f"  {cyan('node')}    {orange(settings.node_id)}\n")
    out(f"  {cyan('bind')}    {green(f'{host}:{port}')}\n")
    out(f"  {cyan('home')}    {blue(str(settings.node_home))}\n")
    out(f"  {cyan('pid')}     {dim(str(pid))}\n")

    # Persisting (auto-start on boot) is opt-in — `start` just starts.
    if getattr(args, "persist", False):
        _persist_service(settings, no_front=True)

    out(f"\n  {dim('Public access enabled. Stop with:')} {bold('ygg node stop')}\n")
    return 0


def _node_stop(args: argparse.Namespace) -> int:
    from yggdrasil.node.daemon import stop_node
    from yggdrasil.node.config import get_settings
    from yggdrasil.cli.style import dim, green, orange, out, print_logo, red

    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")
    if stop_node(settings):
        out(f"  {green('✓')} {orange(settings.node_id)} stopped\n")
    else:
        out(f"  {red('✗')} no running node found\n")
    return 0


def _node_serve(args: argparse.Namespace) -> int:
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, orange, out, print_logo
    from yggdrasil.node.config import _find_open_port, get_settings
    from yggdrasil.node.daemon import cleanup_old_logs, ensure_directories

    _apply_node_env(args)
    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")
    ensure_directories(settings)
    cleanup_old_logs(settings)

    port = args.port or _find_open_port(settings.port, settings.port + 100)
    host = args.host or settings.host

    out(f"  {cyan('node')}    {orange(settings.node_id)}\n")
    out(f"  {cyan('home')}    {blue(str(settings.node_home))}\n")
    out(f"  {cyan('bind')}    {green(f'{host}:{port}')}\n")
    out(f"  {cyan('mode')}    {dim('public — remote access enabled')}\n")

    if getattr(args, "persist", False):
        _persist_service(settings, no_front=args.no_front)

    front_proc = None
    if not args.no_front:
        front_proc = _start_frontend(settings, node_port=port)
    out("\n")

    import uvicorn
    try:
        uvicorn.run("yggdrasil.node.app:app", host=host, port=port, reload=args.reload)
    finally:
        if front_proc is not None:
            front_proc.terminate()
            front_proc.wait(timeout=5)
    return 0


def _node_status(args: argparse.Namespace) -> int:
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.daemon import _is_node_running, ensure_directories
    from yggdrasil.node.service import service_status
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, magenta, orange, out, print_logo, red, yellow

    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")
    ensure_directories(settings)
    running, pid, port = _is_node_running(settings)

    out(f"  {cyan('node')}    {orange(settings.node_id)}\n")
    out(f"  {cyan('home')}    {blue(str(settings.node_home))}\n")
    if running:
        out(f"  {cyan('status')}  {green('● running')} {dim(f'pid={pid}')}\n")
        out(f"  {cyan('url')}     {green(f'http://0.0.0.0:{port}')}\n")
    else:
        out(f"  {cyan('status')}  {red('● stopped')}\n")
        out(f"\n  Start with: {bold('ygg node start')}\n")

    svc = service_status(settings)
    if svc:
        out(f"\n  {magenta('boot services:')}\n")
        for name, state in svc.items():
            color = green if state in ("active", "running") else (red if state == "not installed" else yellow)
            out(f"    {dim(name)}  {color(state)}\n")

    return 0


def _node_excel(args: argparse.Namespace) -> int:
    """Check the Excel service and create/update its integration artifacts.

    1. Probes ``/api/v2/excel/info`` and prints node identity + caps.
    2. (Re)writes the Office.js add-in ``manifest.xml`` under the
       frontend's ``public/excel-addin/`` with URLs pinned to the
       configured frontend port — auto-created if missing.
    3. Points at the Power Query connector sources + the ``/excel`` page.
    """
    import json as _json
    import urllib.request
    from pathlib import Path
    from yggdrasil.node.config import get_settings
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, orange, out, print_logo, red, yellow

    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")

    base = (args.host or f"http://127.0.0.1:{settings.port}").rstrip("/")
    front_port = settings.front_port

    # 1. check the Excel service
    out(f"  {cyan('excel service')}\n")
    try:
        with urllib.request.urlopen(f"{base}/api/v2/excel/info", timeout=5) as resp:
            info = _json.loads(resp.read().decode())
        out(f"    {green('● online')}  {dim(base)}\n")
        out(f"    {cyan('node')}     {orange(info.get('node_id', '?'))} {dim('v' + str(info.get('version', '?')))}\n")
        out(f"    {cyan('formats')}  {', '.join(info.get('table_formats', []))}\n")
        out(f"    {cyan('caps')}     {', '.join(info.get('capabilities', []))}\n")
    except Exception as exc:
        out(f"    {red('● unreachable')}  {dim(str(exc))}\n")
        out(f"\n  Start the node first: {bold('ygg node serve')}\n")

    # 2. (re)generate the Office add-in manifest
    front = Path(settings.front_home)
    manifest = front / "public" / "excel-addin" / "manifest.xml"
    taskpane = f"http://localhost:{front_port}/excel/taskpane"
    icon = f"http://localhost:{front_port}/favicon.ico"
    content = _excel_manifest_xml(taskpane=taskpane, icon=icon)

    out(f"\n  {cyan('office add-in')}\n")
    if args.check:
        state = "present" if manifest.exists() else "missing"
        out(f"    {dim('manifest')}  {yellow(state)} {dim(str(manifest))}\n")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        existed = manifest.exists()
        manifest.write_text(content)
        out(f"    {green('✓')} {'updated' if existed else 'created'} {dim(str(manifest))}\n")
        out(f"    {dim('taskpane')}  {blue(taskpane)}\n")

    # 3. deploy prerequisites — Node.js (frontend hosts the task-pane)
    import platform
    import shutil
    node = shutil.which("node")
    npm = shutil.which("npm")
    out(f"\n  {cyan('node.js')}\n")
    if node and npm:
        out(f"    {green('✓')} node {dim(node)}\n")
    else:
        out(f"    {yellow('!')} not found — run {bold('ygg node serve')} to auto-install, "
            f"or get it from {dim('https://nodejs.org')}\n")

    # 4. direct add-in install into Excel's trusted sideload folder
    system = platform.system()
    if system == "Darwin":
        wef = Path.home() / "Library" / "Containers" / "com.microsoft.Excel" / "Data" / "Documents" / "wef"
        out(f"\n  {cyan('sideload')}    {dim('macOS trusted folder available')}\n")
        if args.install and not args.check:
            wef.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest, wef / "yggdrasil.manifest.xml")
            out(f"    {green('✓')} installed → {dim(str(wef / 'yggdrasil.manifest.xml'))}\n")
            out(f"    {dim('Restart Excel; find it under Insert → Add-ins → Developer Add-ins.')}\n")
        else:
            out(f"    {dim('run with --install to copy the manifest there')}\n")
    elif system == "Windows":
        out(f"\n  {cyan('sideload')}    {dim('Windows needs a shared-folder trusted catalog')}\n")
        out(f"    {dim('Share the manifest folder, then File → Options → Trust Center → Trusted Add-in Catalogs.')}\n")
        if args.install:
            out(f"    {yellow('!')} direct install unsupported on Windows (catalog is registry-based)\n")
    else:
        out(f"\n  {cyan('sideload')}    {dim('Excel desktop not present on this OS; use Upload My Add-in on Win/Mac')}\n")

    # 5. Power Query + page pointers
    out(f"\n  {cyan('power query')}  {dim('powerquery/YggdrasilExcel.pq (paste) · Yggdrasil.pq (.mez)')}\n")
    out(f"  {cyan('manage')}       {blue(f'http://localhost:{front_port}/excel')}\n")
    out(f"\n  Manual sideload: Excel → Insert → Add-ins → Upload My Add-in → manifest.xml\n")
    return 0


def _excel_manifest_xml(*, taskpane: str, icon: str) -> str:
    """Office task-pane manifest (task-pane + Home-tab ribbon button)
    with all URLs pinned to this frontend."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<OfficeApp
  xmlns="http://schemas.microsoft.com/office/appforoffice/1.1"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xmlns:bt="http://schemas.microsoft.com/office/officeappbasictypes/1.0"
  xmlns:ov="http://schemas.microsoft.com/office/taskpaneappversionoverrides"
  xsi:type="TaskPaneApp">
  <Id>2b9d6a1e-9b1a-4c5e-9d3f-7a1ce0fda001</Id>
  <Version>1.0.0.0</Version>
  <ProviderName>Yggdrasil</ProviderName>
  <DefaultLocale>en-US</DefaultLocale>
  <DisplayName DefaultValue="Yggdrasil for Excel" />
  <Description DefaultValue="Run Python on a Yggdrasil node, read/write remote files, and walk remote filesystems." />
  <IconUrl DefaultValue="{icon}" />
  <HighResolutionIconUrl DefaultValue="{icon}" />
  <SupportUrl DefaultValue="https://github.com/Platob/Yggdrasil" />
  <Hosts><Host Name="Workbook" /></Hosts>
  <DefaultSettings><SourceLocation DefaultValue="{taskpane}" /></DefaultSettings>
  <Permissions>ReadWriteDocument</Permissions>
  <VersionOverrides xmlns="http://schemas.microsoft.com/office/taskpaneappversionoverrides" xsi:type="VersionOverridesV1_0">
    <Hosts>
      <Host xsi:type="Workbook">
        <DesktopFormFactor>
          <ExtensionPoint xsi:type="PrimaryCommandSurface">
            <OfficeTab id="TabHome">
              <Group id="Ygg.Group">
                <Label resid="Ygg.Group.Label" />
                <Control xsi:type="Button" id="Ygg.Open">
                  <Label resid="Ygg.Open.Label" />
                  <Supertip>
                    <Title resid="Ygg.Open.Label" />
                    <Description resid="Ygg.Desc" />
                  </Supertip>
                  <Icon><bt:Image size="16" resid="Ygg.Icon" /><bt:Image size="32" resid="Ygg.Icon" /><bt:Image size="80" resid="Ygg.Icon" /></Icon>
                  <Action xsi:type="ShowTaskpane">
                    <TaskpaneId>YggTaskpane</TaskpaneId>
                    <SourceLocation resid="Ygg.Taskpane.Url" />
                  </Action>
                </Control>
              </Group>
            </OfficeTab>
          </ExtensionPoint>
        </DesktopFormFactor>
      </Host>
    </Hosts>
    <Resources>
      <bt:Images><bt:Image id="Ygg.Icon" DefaultValue="{icon}" /></bt:Images>
      <bt:Urls><bt:Url id="Ygg.Taskpane.Url" DefaultValue="{taskpane}" /></bt:Urls>
      <bt:ShortStrings>
        <bt:String id="Ygg.Group.Label" DefaultValue="Yggdrasil" />
        <bt:String id="Ygg.Open.Label" DefaultValue="Open Yggdrasil" />
      </bt:ShortStrings>
      <bt:LongStrings>
        <bt:String id="Ygg.Desc" DefaultValue="Run Python on a node, read/write remote files, and walk remote filesystems into your sheet." />
      </bt:LongStrings>
    </Resources>
  </VersionOverrides>
</OfficeApp>
"""


def _node_create(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, magenta, orange, out, print_logo, yellow
    from yggdrasil.node.daemon import ensure_directories, spawn_node
    from yggdrasil.node.service import install_service

    print_logo("YGGNODE")

    os.environ["YGG_NODE_NODE_ID"] = args.name
    os.environ["YGG_NODE_ALLOW_REMOTE"] = "1"
    if args.port:
        os.environ["YGG_NODE_PORT"] = str(args.port)

    from yggdrasil.node.config import get_settings
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")
    ensure_directories(settings)

    id_file = Path.home() / ".node" / ".ygg_node_id"
    id_file.parent.mkdir(parents=True, exist_ok=True)
    id_file.write_text(args.name)

    out(f"  {green('✓')} node created\n")
    out(f"  {cyan('node')}    {orange(args.name)}\n")
    out(f"  {cyan('home')}    {blue(str(settings.node_home))}\n")

    if args.start:
        pid, port = spawn_node(settings, host="0.0.0.0")
        out(f"  {green('✓')} running on {green(f'0.0.0.0:{port}')} {dim(f'pid={pid}')}\n")
        ok, msg = install_service(settings, no_front=True)
        if ok:
            out(f"  {green('✓')} {magenta('boot service')} installed\n")
        else:
            out(f"  {yellow('skip')} {dim(msg)}\n")
    else:
        out(f"\n  Start with: {bold('ygg node start')}\n")
    return 0


def _node_back(args: argparse.Namespace) -> int:
    from yggdrasil.cli.style import blue, cyan, dim, green, orange, out, print_logo
    from yggdrasil.node.config import _find_open_port, get_settings
    from yggdrasil.node.daemon import cleanup_old_logs, ensure_directories

    _apply_node_env(args)
    print_logo("YGGNODE")
    settings = get_settings()
    out(f"  {dim(f'v{settings.app_version}')}\n\n")
    ensure_directories(settings)
    cleanup_old_logs(settings)

    port = args.port or _find_open_port(settings.port, settings.port + 100)
    host = args.host or settings.host

    out(f"  {cyan('node')}    {orange(settings.node_id)}\n")
    out(f"  {cyan('home')}    {blue(str(settings.node_home))}\n")
    out(f"  {cyan('bind')}    {green(f'{host}:{port}')}\n")
    out(f"  {cyan('mode')}    {dim('backend only')}\n\n")

    if getattr(args, "persist", False):
        _persist_service(settings, no_front=True)

    import uvicorn
    uvicorn.run("yggdrasil.node.app:app", host=host, port=port, reload=args.reload)
    return 0


def _node_front(args: argparse.Namespace) -> int:
    import signal
    from yggdrasil.node.config import get_settings
    from yggdrasil.cli.style import dim, out, print_logo

    print_logo("YGGNODE")
    settings = get_settings()

    node_port = args.node_port or settings.port
    if getattr(args, "persist", False):
        _persist_service(settings)

    proc = _start_frontend(settings, node_port=node_port, front_port=args.port, quiet=False)
    if proc is None:
        return 1

    out(f"  {dim('Press Ctrl+C to stop.')}\n\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    return 0


def _node_run(args: argparse.Namespace) -> int:
    from yggdrasil.node.client import NodeClient
    from yggdrasil.cli.style import Spinner

    url = args.url or _ensure_node_running()
    kwargs = {}
    for kv in args.kwarg:
        if "=" not in kv:
            print(f"Error: --kwarg must be KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 1
        k, v = kv.split("=", 1)
        kwargs[k] = v

    client = NodeClient(url, timeout=args.timeout)

    if args.stream:
        try:
            for batch in client.call_stream(args.func, *args.args, **kwargs):
                print(batch.to_pandas().to_string(index=False))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        return 0

    with Spinner(f"calling {args.func}...", color="33"):
        try:
            result = client.call(args.func, *args.args, **kwargs)
        except Exception as exc:
            print(f"\nError: {exc}", file=sys.stderr)
            return 1

    import pyarrow as pa
    if isinstance(result, pa.Table):
        print(result.to_pandas().to_string(index=False))
    else:
        print(result)
    return 0


def _node_chat(args: argparse.Namespace) -> int:
    from yggdrasil.node.chat import run_chat
    url = args.url or _ensure_node_running()
    return run_chat(url=url, username=args.user, channel=args.channel)


def _node_install(args: argparse.Namespace) -> int:
    import os
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red, yellow
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.service import install_service, service_status

    print_logo("YGGNODE")

    os.environ["YGG_NODE_ALLOW_REMOTE"] = "1"
    if args.port:
        os.environ["YGG_NODE_PORT"] = str(args.port)
    if args.front_port:
        os.environ["YGG_NODE_FRONT_PORT"] = str(args.front_port)

    settings = get_settings()

    out(f"  {cyan('install')} registering boot service...\n")
    out(f"  {cyan('node')}    {bold(settings.node_id)} on port {bold(str(settings.port))}\n")
    if not args.no_front:
        out(f"  {cyan('front')}   port {bold(str(settings.front_port))}\n")
    out(f"  {cyan('home')}    {dim(str(settings.node_home))}\n\n")

    ok, msg = install_service(settings, no_front=args.no_front)

    if ok:
        out(f"  {green('✓')} {msg}\n\n")
        status = service_status(settings)
        for name, state in status.items():
            color = green if state in ("active", "running") else yellow
            out(f"  {cyan(name)}  {color(state)}\n")
        out(f"\n  Node will start automatically on boot.\n")
        out(f"  Uninstall with: {bold('ygg node uninstall')}\n")
    else:
        out(f"  {red('✗')} {msg}\n")
        return 1
    return 0


def _node_uninstall(args: argparse.Namespace) -> int:
    from yggdrasil.cli.style import bold, cyan, dim, green, out, print_logo, red, yellow
    from yggdrasil.node.config import get_settings
    from yggdrasil.node.service import uninstall_service

    print_logo("YGGNODE")

    settings = get_settings()

    if args.purge:
        out(f"  {yellow('purge')} will remove all data in {dim(str(settings.node_home))}\n\n")

    ok, msg = uninstall_service(settings, purge=args.purge)

    if ok:
        out(f"  {green('✓')} {msg}\n")
        if args.purge:
            out(f"  {dim('All node data has been removed.')}\n")
    else:
        out(f"  {red('✗')} {msg}\n")
        return 1
    return 0


def _node_watch(args: argparse.Namespace) -> int:
    """Live auto-refreshing TTY dashboard of node stats."""
    import json
    import time
    import urllib.request
    from yggdrasil.cli.style import blue, bold, cyan, dim, green, magenta, orange, out, red, yellow, _CSI, _RESET

    url = args.url or _ensure_node_running()
    interval = args.interval

    def fetch(path):
        try:
            with urllib.request.urlopen(f"{url}{path}", timeout=3) as resp:
                return json.loads(resp.read())
        except Exception:
            return None

    def bar(pct, width=30):
        filled = int(width * min(100, max(0, pct)) / 100)
        color = "31" if pct > 90 else "33" if pct > 70 else "36"
        return f"{_CSI}{color}m{'█' * filled}{_RESET}{dim('░' * (width - filled))}"

    def format_uptime(seconds):
        if seconds < 60: return f"{int(seconds)}s"
        if seconds < 3600: return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        if seconds < 86400: return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60)}m"
        return f"{int(seconds // 86400)}d {int(seconds % 86400 // 3600)}h"

    # Clear screen and hide cursor
    sys.stdout.write("\033[?25l\033[2J\033[H")
    try:
        while True:
            stats = fetch("/api/v2/stats")
            health = fetch("/api/v2/health")
            metrics = fetch("/api/v2/metrics")
            audit = fetch("/api/v2/audit?limit=5")

            # Move to top, don't clear (less flicker)
            sys.stdout.write("\033[H\033[J")

            # Header
            out(f"  {cyan('═' * 70)}\n")
            if stats:
                out(f"  {bold('YGGDRASIL NODE')}  {orange(stats.get('node_id', '?'))}  ")
                out(f"{dim('uptime')} {green(format_uptime(stats.get('uptime', 0)))}  ")
                if health:
                    status = health.get('status', 'unknown')
                    color_fn = green if status == 'healthy' else yellow if status == 'degraded' else red
                    out(f"{dim('status')} {color_fn('● ' + status)}\n")
                else:
                    out("\n")
            else:
                out(f"  {red('● node unreachable')}  retrying...\n")
            out(f"  {cyan('═' * 70)}\n\n")

            if stats:
                # Resources
                cpu = stats.get('cpu_percent', 0)
                mem = stats.get('memory_percent', 0)
                out(f"  {cyan('CPU')}     {bar(cpu)}  {bold(f'{cpu:5.1f}%')}\n")
                out(f"  {cyan('Memory')}  {bar(mem)}  {bold(f'{mem:5.1f}%')}\n")
                out("\n")

                # Counts
                out(f"  {magenta('Assets')}\n")
                out(f"    {dim('functions')}    {bold(str(stats.get('func_count', 0)))}\n")
                out(f"    {dim('environments')} {bold(str(stats.get('env_count', 0)))}\n")
                out(f"    {dim('DAGs')}         {bold(str(stats.get('dag_count', 0)))}  ")
                if stats.get('scheduled_dags', 0):
                    out(f"{green('●')} {stats['scheduled_dags']} scheduled")
                out("\n\n")

                # Runs
                active = stats.get('active_runs', 0)
                total = stats.get('total_runs', 0)
                out(f"  {magenta('Execution')}\n")
                active_color = yellow if active > 0 else dim
                out(f"    {dim('active')}     {active_color(str(active))}\n")
                out(f"    {dim('total')}      {bold(str(total))}\n\n")

                # Network
                out(f"  {magenta('Network')}\n")
                out(f"    {dim('peers')}      {bold(str(stats.get('peer_count', 0)))}\n")
                out(f"    {dim('GPUs')}       {bold(str(stats.get('gpu_count', 0)))}\n\n")

            # Top functions
            if metrics and metrics.get('top_by_runs'):
                out(f"  {magenta('Top Functions')}\n")
                for f in metrics['top_by_runs'][:5]:
                    name = f['name'][:24].ljust(24)
                    out(f"    {cyan(name)} {dim('runs')} {bold(str(f['runs']))}\n")
                out("\n")

            # Recent activity
            if audit and audit.get('entries'):
                out(f"  {magenta('Recent Activity')}\n")
                for e in audit['entries'][-5:]:
                    op = e.get('operation', '?')
                    op_color = green if op == 'create' else red if op == 'delete' else cyan
                    ts = e.get('timestamp', '')[11:19]  # HH:MM:SS
                    out(f"    {dim(ts)}  {op_color(op.ljust(7))} {e.get('asset_type', '?')} {dim('#' + str(e.get('asset_id', '?'))[:10])}\n")

            out(f"\n  {dim('refreshing every ' + str(interval) + 's — Ctrl+C to exit')}\n")
            sys.stdout.flush()

            time.sleep(interval)
    except KeyboardInterrupt:
        # Show cursor again
        sys.stdout.write("\033[?25h\n")
    return 0


def _node_logs(args: argparse.Namespace) -> int:
    import datetime as dt
    import time
    from yggdrasil.cli.style import bold, cyan, dim, out, print_logo, red
    from yggdrasil.node.config import get_settings

    print_logo("YGGNODE")
    settings = get_settings()
    log_file = settings.logs_root / f"node-{dt.date.today().isoformat()}.log"

    if not log_file.exists():
        out(f"  {red('no log file found at')} {dim(str(log_file))}\n")
        return 1

    out(f"  {cyan('log')}  {dim(str(log_file))}\n\n")
    with open(log_file) as f:
        # Print last N lines
        all_lines = f.readlines()
        for line in all_lines[-args.lines:]:
            out(line)

        if args.follow:
            out(f"\n  {dim('following — Ctrl+C to exit')}\n")
            try:
                while True:
                    line = f.readline()
                    if line:
                        out(line)
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass
    return 0


def _node_ps(args: argparse.Namespace) -> int:
    import json
    import urllib.request
    from yggdrasil.cli.style import bold, cyan, dim, green, out, red, yellow

    url = _ensure_node_running()
    try:
        with urllib.request.urlopen(f"{url}/api/v2/pyfuncrun", timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        out(f"  {red('error:')} {exc}\n")
        return 1

    runs = data.get("runs", [])
    if not args.all:
        runs = [r for r in runs if r.get("status") in ("pending", "running")]

    if not runs:
        out(f"  {dim('no')} {('active ' if not args.all else '')}{dim('runs')}\n")
        return 0

    out(f"  {bold('RUN ID'.ljust(22))} {bold('FUNC'.ljust(20))} {bold('STATUS'.ljust(11))} {bold('DURATION'.ljust(10))} {bold('STARTED')}\n")
    out(f"  {dim('─' * 90)}\n")
    for r in runs:
        rid = str(r.get("id", ""))[:20].ljust(22)
        fid = str(r.get("func_id", ""))[:18].ljust(20)
        status = r.get("status", "?")
        status_color = green if status == "completed" else red if status == "failed" else yellow if status == "running" else cyan
        duration = r.get("duration")
        dur_str = f"{duration:.2f}s" if duration else "-"
        started = (r.get("started_at") or "")[:19]
        out(f"  {dim(rid)} {dim(fid)} {status_color(status.ljust(11))} {dim(dur_str.ljust(10))} {dim(started)}\n")
    return 0


def _node_procs(args: argparse.Namespace) -> int:
    """Live TTY view of running PyFuncRuns with per-process CPU/RAM."""
    import json
    import time
    import urllib.request
    from yggdrasil.cli.style import bold, cyan, dim, green, magenta, orange, out, red, yellow, _CSI, _RESET

    url = args.url or _ensure_node_running()
    interval = args.interval

    try:
        import psutil  # type: ignore
    except ImportError:
        psutil = None

    sys.stdout.write("\033[?25l\033[2J\033[H")
    try:
        while True:
            try:
                with urllib.request.urlopen(f"{url}/api/v2/pyfuncrun?status=running,pending", timeout=3) as r:
                    runs = json.loads(r.read()).get("runs", [])
            except Exception as e:
                runs = []
                err = str(e)
            else:
                err = ""

            sys.stdout.write("\033[H\033[J")
            out(f"  {cyan('═' * 90)}\n")
            out(f"  {bold('YGG PROCESSES')}  {dim('active and pending runs')}  {dim(f'every {interval}s')}\n")
            out(f"  {cyan('═' * 90)}\n\n")

            if err:
                out(f"  {red('node unreachable:')} {err}\n")
            elif not runs:
                out(f"  {dim('no active runs')}\n")
            else:
                out(f"  {bold('PID'.ljust(8))} {bold('RUN'.ljust(20))} {bold('FUNC'.ljust(20))} ")
                out(f"{bold('STATUS'.ljust(10))} {bold('CPU%'.rjust(7))} {bold('MEM(MB)'.rjust(10))} {bold('DUR(s)'.rjust(8))}\n")
                out(f"  {dim('-' * 90)}\n")
                for r in runs:
                    rid = str(r.get("id", ""))[:18].ljust(20)
                    fid = str(r.get("func_id", ""))[:18].ljust(20)
                    status = r.get("status", "?")
                    sc = green if status == "running" else yellow if status == "pending" else dim
                    pid = r.get("pid")
                    cpu_pct = "-"
                    mem_mb = "-"
                    if psutil is not None and pid:
                        try:
                            p = psutil.Process(int(pid))
                            cpu_pct = f"{p.cpu_percent(interval=0.0):.1f}"
                            mem_mb = f"{p.memory_info().rss / 1024 / 1024:.1f}"
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    duration = r.get("duration") or (
                        (time.time() - time.mktime(time.strptime((r.get("started_at") or "")[:19], "%Y-%m-%dT%H:%M:%S")))
                        if r.get("started_at") else None
                    )
                    dur_s = f"{duration:.1f}" if isinstance(duration, (int, float)) else "-"
                    pid_s = str(pid) if pid else "-"
                    out(f"  {dim(pid_s.ljust(8))} {cyan(rid)} {dim(fid)} {sc(status.ljust(10))} ")
                    out(f"{cyan(cpu_pct.rjust(7))} {cyan(mem_mb.rjust(10))} {dim(dur_s.rjust(8))}\n")

            out(f"\n  {dim('Ctrl+C to exit')}\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n")
    return 0


def _node_mesh(args: argparse.Namespace) -> int:
    """Live cluster mesh view: self + every peer, side-by-side health."""
    import json
    import time
    import urllib.request
    from yggdrasil.cli.style import bold, cyan, dim, green, magenta, orange, out, red, yellow, _CSI, _RESET

    url = args.url or _ensure_node_running()
    interval = args.interval

    sys.stdout.write("\033[?25l\033[2J\033[H")
    try:
        while True:
            try:
                with urllib.request.urlopen(f"{url}/api/v2/topology", timeout=3) as r:
                    topo = json.loads(r.read())
            except Exception as e:
                topo, err = None, str(e)
            else:
                err = ""

            sys.stdout.write("\033[H\033[J")
            out(f"  {cyan('═' * 90)}\n")
            out(f"  {bold('YGG MESH')}  {dim('cluster health')}  {dim(f'every {interval}s')}\n")
            out(f"  {cyan('═' * 90)}\n\n")

            if err or not topo:
                out(f"  {red('node unreachable:')} {err or 'no topology'}\n")
            else:
                nodes = topo.get("nodes", [])
                cpu_avg = topo.get("total_cpu_percent", 0)
                out(f"  {dim('cluster')}  ")
                out(f"{cyan('cpu_avg')} {bold(f'{cpu_avg:.1f}%')}  ")
                out(f"{cyan('runs')} {bold(str(topo.get('total_active_runs', 0)))}  ")
                out(f"{cyan('gpus')} {bold(str(topo.get('total_gpus', 0)))}  ")
                out(f"{cyan('nodes')} {bold(str(len(nodes)))}\n\n")

                hdr_fmt = "  {:<24} {:<8} {:<22} {:>7} {:>7} {:>7} {:>5}\n"
                out(hdr_fmt.format("NODE", "ROLE", "ADDRESS", "CPU%", "MEM%", "RUNS", "GPUs"))
                out(f"  {dim('-' * 88)}\n")
                for n in nodes:
                    cpu = n.get("cpu_percent", 0)
                    mem = n.get("memory_percent", 0)
                    cpu_c = red if cpu > 80 else yellow if cpu > 50 else green
                    mem_c = red if mem > 80 else yellow if mem > 50 else green
                    self_flag = "self" if n.get("self") else ""
                    nid = (n.get("node_id") or "?")[:22].ljust(24)
                    role = (str(n.get("role") or "?"))[:6].ljust(8)
                    addr = f"{n.get('host', '?')}:{n.get('port', '?')}"[:20].ljust(22)
                    cpu_s = cpu_c(f"{cpu:.1f}".rjust(7))
                    mem_s = mem_c(f"{mem:.1f}".rjust(7))
                    runs = str(n.get("active_runs", 0)).rjust(7)
                    gpus = str(n.get("gpu_count", 0)).rjust(5)
                    out(f"  {orange(nid) if n.get('self') else dim(nid)} {dim(role)} {dim(addr)} {cpu_s} {mem_s} {dim(runs)} {dim(gpus)}\n")

            out(f"\n  {dim('Ctrl+C to exit')}\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n")
    return 0


def _node_call(args: argparse.Namespace) -> int:
    import json
    import urllib.request
    from yggdrasil.cli.style import bold, cyan, dim, green, magenta, out, red

    url = _ensure_node_running()

    # Parse args
    call_args = []
    for a in args.arg:
        try:
            call_args.append(json.loads(a))
        except json.JSONDecodeError:
            call_args.append(a)

    call_kwargs = {}
    for kv in args.kwarg:
        if "=" not in kv:
            out(f"  {red('error:')} --kwarg must be KEY=VALUE\n")
            return 1
        k, v = kv.split("=", 1)
        try:
            call_kwargs[k] = json.loads(v)
        except json.JSONDecodeError:
            call_kwargs[k] = v

    payload = {"args": call_args, "kwargs": call_kwargs}
    req = urllib.request.Request(
        f"{url}/api/v2/pyfunc/by-name/{args.name}/run",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    out(f"  {cyan('calling')} {bold(args.name)}\n")
    try:
        with urllib.request.urlopen(req, timeout=args.wait + 10) as resp:
            result = json.loads(resp.read())
    except Exception as exc:
        out(f"  {red('error:')} {exc}\n")
        return 1

    run = result.get("run", {})
    status = run.get("status", "?")
    status_color = green if status == "completed" else red
    out(f"  {status_color('●')} {status}  {dim(str(run.get('duration', 0)) + 's')}\n")

    if run.get("stdout"):
        out(f"\n  {magenta('stdout:')}\n")
        for line in run["stdout"].splitlines():
            out(f"    {line}\n")
    if run.get("stderr"):
        out(f"\n  {magenta('stderr:')}\n")
        for line in run["stderr"].splitlines():
            out(f"    {dim(line)}\n")
    if run.get("result") is not None:
        out(f"\n  {magenta('result:')} {json.dumps(run['result'], indent=2)}\n")
    return 0


def _node_health(args: argparse.Namespace) -> int:
    import json
    import urllib.request
    from yggdrasil.cli.style import bold, cyan, dim, green, magenta, out, print_logo, red, yellow

    print_logo("YGGNODE")
    url = _ensure_node_running()
    try:
        with urllib.request.urlopen(f"{url}/api/v2/health", timeout=5) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        out(f"  {red('node unreachable:')} {exc}\n")
        return 1

    status = data.get("status", "unknown")
    color_fn = green if status == "healthy" else yellow if status == "degraded" else red
    out(f"  {cyan('overall')}   {color_fn('● ' + status)}\n")
    out(f"  {cyan('node')}      {bold(data.get('node_id', '?'))}\n\n")

    out(f"  {magenta('subsystems:')}\n")
    for name, check in data.get("checks", {}).items():
        check_status = check.get("status", "?")
        check_color = green if check_status == "ok" else red
        extra = []
        if "cpu" in check: extra.append(f"cpu={check['cpu']:.1f}%")
        if "count" in check: extra.append(f"count={check['count']}")
        if "peers" in check: extra.append(f"peers={check['peers']}")
        if "error" in check: extra.append(f"error={check['error']}")
        out(f"    {check_color('●')} {dim(name.ljust(12))}  {check_color(check_status)}  {dim(' '.join(extra))}\n")
    return 0


def _databricks(args: argparse.Namespace) -> int:
    from yggdrasil.databricks.cli import main as dbks_main
    remaining = sys.argv[2:] if len(sys.argv) > 2 else []
    return dbks_main(remaining)


# ── entry point ──────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]

    args, remaining = parser.parse_known_args(argv)

    if args.debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.command is None:
        from yggdrasil.cli.style import print_logo
        print_logo("YGG")
        parser.print_help()
        return 0

    handler = getattr(args, "handler", None)
    if handler is None:
        if args.command == "node":
            parser.parse_args(["node", "--help"])
        parser.print_help()
        return 0

    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
