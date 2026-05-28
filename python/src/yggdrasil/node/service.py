"""System service management — install/uninstall yggdrasil node as a boot service.

Linux: systemd user service (~/.config/systemd/user/yggdrasil-node.service)
macOS: launchd user agent (~/Library/LaunchAgents/com.yggdrasil.node.plist)
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from yggdrasil.node.config import Settings, get_settings

_SYSTEMD_UNIT = "yggdrasil-node.service"
_LAUNCHD_LABEL = "com.yggdrasil.node"


def _python_executable() -> str:
    return sys.executable


def _systemd_dir() -> Path:
    return Path.home() / ".config" / "systemd" / "user"


def _launchd_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def is_service_installed(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if is_linux():
        return (_systemd_dir() / _SYSTEMD_UNIT).exists()
    if is_macos():
        return (_launchd_dir() / f"{_LAUNCHD_LABEL}.plist").exists()
    return False


def _systemd_unit_content(settings: Settings) -> str:
    python = _python_executable()

    lines = [
        "[Unit]",
        "Description=Yggdrasil Node Server",
        "After=network.target",
        "",
        "[Service]",
        "Type=simple",
        f"ExecStart={python} -m yggdrasil.node.main",
        f"Environment=YGG_NODE_PORT={settings.port}",
        f"Environment=YGG_NODE_HOST={settings.host}",
        f"Environment=YGG_NODE_HOME={settings.node_home}",
        f"Environment=YGG_NODE_NODE_ID={settings.node_id}",
        f"Environment=YGG_NODE_FRONT_HOME={settings.front_home}",
        "Environment=YGG_NODE_ALLOW_REMOTE=1",
        f"Environment=PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
        "Restart=on-failure",
        "RestartSec=5",
        "",
        "[Install]",
        "WantedBy=default.target",
    ]
    return "\n".join(lines) + "\n"


def _launchd_plist_content(settings: Settings) -> str:
    python = _python_executable()
    node_home = settings.node_home
    front_home = settings.front_home
    log_out = settings.logs_root / "launchd-stdout.log"
    log_err = settings.logs_root / "launchd-stderr.log"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>yggdrasil.node.main</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>YGG_NODE_PORT</key>
        <string>{settings.port}</string>
        <key>YGG_NODE_HOST</key>
        <string>{settings.host}</string>
        <key>YGG_NODE_HOME</key>
        <string>{node_home}</string>
        <key>YGG_NODE_FRONT_HOME</key>
        <string>{front_home}</string>
        <key>YGG_NODE_ALLOW_REMOTE</key>
        <string>1</string>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '/usr/bin:/bin')}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
</dict>
</plist>
"""


def _front_systemd_unit_content(settings: Settings) -> str:
    npm = shutil.which("npm") or "/usr/bin/npm"
    front_home = settings.front_home

    lines = [
        "[Unit]",
        "Description=Yggdrasil Frontend (Next.js)",
        f"After=network.target {_SYSTEMD_UNIT}",
        f"Requires={_SYSTEMD_UNIT}",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={front_home}",
        f"ExecStart={npm} run dev -- --hostname 0.0.0.0 --port {settings.front_port}",
        f"Environment=YGG_NODE_PORT={settings.port}",
        f"Environment=PORT={settings.front_port}",
        f"Environment=PATH={os.environ.get('PATH', '/usr/bin:/bin')}",
        f"Environment=NODE_API_URL=http://127.0.0.1:{settings.port}",
        "Restart=on-failure",
        "RestartSec=5",
        "",
        "[Install]",
        "WantedBy=default.target",
    ]
    return "\n".join(lines) + "\n"


_FRONT_SYSTEMD_UNIT = "yggdrasil-front.service"
_FRONT_LAUNCHD_LABEL = "com.yggdrasil.front"


def _front_launchd_plist_content(settings: Settings) -> str:
    npm = shutil.which("npm") or "/usr/local/bin/npm"
    front_home = settings.front_home
    log_out = settings.logs_root / "launchd-front-stdout.log"
    log_err = settings.logs_root / "launchd-front-stderr.log"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{_FRONT_LAUNCHD_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{npm}</string>
        <string>run</string>
        <string>dev</string>
        <string>--</string>
        <string>--hostname</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>{settings.front_port}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{front_home}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>YGG_NODE_PORT</key>
        <string>{settings.port}</string>
        <key>PORT</key>
        <string>{settings.front_port}</string>
        <key>NODE_API_URL</key>
        <string>http://127.0.0.1:{settings.port}</string>
        <key>PATH</key>
        <string>{os.environ.get('PATH', '/usr/bin:/bin')}</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_out}</string>
    <key>StandardErrorPath</key>
    <string>{log_err}</string>
</dict>
</plist>
"""


def is_linux() -> bool:
    return platform.system() == "Linux"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def install_service(settings: Settings | None = None, *, no_front: bool = False) -> tuple[bool, str]:
    """Install node (and optionally frontend) as a boot service. Returns (success, message)."""
    settings = settings or get_settings()

    from yggdrasil.node.daemon import ensure_directories
    ensure_directories(settings)

    if is_linux():
        return _install_systemd(settings, no_front=no_front)
    elif is_macos():
        return _install_launchd(settings, no_front=no_front)
    else:
        return False, f"Unsupported platform: {platform.system()}. Only Linux (systemd) and macOS (launchd) are supported."


def _install_systemd(settings: Settings, *, no_front: bool = False) -> tuple[bool, str]:
    unit_dir = _systemd_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)

    unit_path = unit_dir / _SYSTEMD_UNIT
    unit_path.write_text(_systemd_unit_content(settings))

    messages = [f"wrote {unit_path}"]

    if not no_front and (settings.front_home / "package.json").exists():
        front_path = unit_dir / _FRONT_SYSTEMD_UNIT
        front_path.write_text(_front_systemd_unit_content(settings))
        messages.append(f"wrote {front_path}")

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, capture_output=True)
        subprocess.run(["systemctl", "--user", "enable", _SYSTEMD_UNIT], check=True, capture_output=True)
        messages.append(f"enabled {_SYSTEMD_UNIT}")

        if not no_front and (settings.front_home / "package.json").exists():
            subprocess.run(["systemctl", "--user", "enable", _FRONT_SYSTEMD_UNIT], check=True, capture_output=True)
            messages.append(f"enabled {_FRONT_SYSTEMD_UNIT}")

        subprocess.run(["systemctl", "--user", "start", _SYSTEMD_UNIT], check=True, capture_output=True)
        messages.append("started node")

        if not no_front and (settings.front_home / "package.json").exists():
            subprocess.run(["systemctl", "--user", "start", _FRONT_SYSTEMD_UNIT], check=True, capture_output=True)
            messages.append("started frontend")

    except FileNotFoundError:
        return False, "systemctl not found — systemd is required on Linux."
    except subprocess.CalledProcessError as e:
        return False, f"systemctl failed: {e.stderr.decode().strip() if e.stderr else str(e)}"

    _enable_linger()

    return True, "; ".join(messages)


def _enable_linger() -> None:
    """Enable loginctl linger so user services persist after logout."""
    try:
        subprocess.run(
            ["loginctl", "enable-linger", os.environ.get("USER", "")],
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass


def _install_launchd(settings: Settings, *, no_front: bool = False) -> tuple[bool, str]:
    agent_dir = _launchd_dir()
    agent_dir.mkdir(parents=True, exist_ok=True)

    plist_path = agent_dir / f"{_LAUNCHD_LABEL}.plist"
    plist_path.write_text(_launchd_plist_content(settings))
    messages = [f"wrote {plist_path}"]

    if not no_front and (settings.front_home / "package.json").exists():
        front_plist = agent_dir / f"{_FRONT_LAUNCHD_LABEL}.plist"
        front_plist.write_text(_front_launchd_plist_content(settings))
        messages.append(f"wrote {front_plist}")

    try:
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
        subprocess.run(["launchctl", "load", "-w", str(plist_path)], check=True, capture_output=True)
        messages.append("loaded node agent")

        if not no_front and (settings.front_home / "package.json").exists():
            front_plist = agent_dir / f"{_FRONT_LAUNCHD_LABEL}.plist"
            subprocess.run(["launchctl", "unload", str(front_plist)], capture_output=True)
            subprocess.run(["launchctl", "load", "-w", str(front_plist)], check=True, capture_output=True)
            messages.append("loaded frontend agent")

    except FileNotFoundError:
        return False, "launchctl not found."
    except subprocess.CalledProcessError as e:
        return False, f"launchctl failed: {e.stderr.decode().strip() if e.stderr else str(e)}"

    return True, "; ".join(messages)


def uninstall_service(settings: Settings | None = None, *, purge: bool = False) -> tuple[bool, str]:
    """Uninstall the boot service. If purge=True, also removes ~/.ygg data."""
    settings = settings or get_settings()

    if is_linux():
        ok, msg = _uninstall_systemd(settings)
    elif is_macos():
        ok, msg = _uninstall_launchd(settings)
    else:
        return False, f"Unsupported platform: {platform.system()}."

    if purge and settings.node_home.exists():
        shutil.rmtree(settings.node_home, ignore_errors=True)
        msg += f"; purged {settings.node_home}"

    return ok, msg


def _uninstall_systemd(settings: Settings) -> tuple[bool, str]:
    messages = []

    try:
        for unit in (_FRONT_SYSTEMD_UNIT, _SYSTEMD_UNIT):
            subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
            subprocess.run(["systemctl", "--user", "disable", unit], capture_output=True)
            messages.append(f"stopped+disabled {unit}")

        subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True)
    except FileNotFoundError:
        pass

    unit_dir = _systemd_dir()
    for unit in (_SYSTEMD_UNIT, _FRONT_SYSTEMD_UNIT):
        path = unit_dir / unit
        if path.exists():
            path.unlink()
            messages.append(f"removed {path}")

    from yggdrasil.node.daemon import stop_node
    stop_node(settings)

    return True, "; ".join(messages) if messages else "no services found"


def _uninstall_launchd(settings: Settings) -> tuple[bool, str]:
    messages = []
    agent_dir = _launchd_dir()

    for label, filename in (
        (_FRONT_LAUNCHD_LABEL, f"{_FRONT_LAUNCHD_LABEL}.plist"),
        (_LAUNCHD_LABEL, f"{_LAUNCHD_LABEL}.plist"),
    ):
        plist_path = agent_dir / filename
        if plist_path.exists():
            try:
                subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)
            except FileNotFoundError:
                pass
            plist_path.unlink()
            messages.append(f"unloaded+removed {filename}")

    from yggdrasil.node.daemon import stop_node
    stop_node(settings)

    return True, "; ".join(messages) if messages else "no agents found"


def service_status(settings: Settings | None = None) -> dict[str, str]:
    """Return status of installed services."""
    settings = settings or get_settings()
    result = {}

    if is_linux():
        for unit in (_SYSTEMD_UNIT, _FRONT_SYSTEMD_UNIT):
            path = _systemd_dir() / unit
            if path.exists():
                try:
                    r = subprocess.run(
                        ["systemctl", "--user", "is-active", unit],
                        capture_output=True, text=True,
                    )
                    result[unit] = r.stdout.strip()
                except FileNotFoundError:
                    result[unit] = "unknown (no systemctl)"
            else:
                result[unit] = "not installed"

    elif is_macos():
        for label in (_LAUNCHD_LABEL, _FRONT_LAUNCHD_LABEL):
            plist = _launchd_dir() / f"{label}.plist"
            if plist.exists():
                try:
                    r = subprocess.run(
                        ["launchctl", "list", label],
                        capture_output=True, text=True,
                    )
                    result[label] = "running" if r.returncode == 0 else "stopped"
                except FileNotFoundError:
                    result[label] = "unknown (no launchctl)"
            else:
                result[label] = "not installed"

    return result
