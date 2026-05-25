"""Terminal chat client for the yggdrasil bot messenger endpoints.

Pure-stdlib implementation -- no external dependencies beyond Python 3.10+.
Uses ANSI escape codes for color, a background thread for long-poll message
reception, and ``urllib.request`` for HTTP.
"""
from __future__ import annotations

import getpass
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

# -- ANSI helpers (no external deps) ----------------------------------------

_CSI = "\033["

def _color(text: str, code: str) -> str:  return f"{_CSI}{code}m{text}{_CSI}0m"
def _bold(text: str) -> str:   return _color(text, "1")
def _dim(text: str) -> str:    return _color(text, "2")
def _cyan(text: str) -> str:   return _color(text, "36")
def _green(text: str) -> str:  return _color(text, "32")
def _yellow(text: str) -> str: return _color(text, "33")
def _magenta(text: str) -> str: return _color(text, "35")
def _red(text: str) -> str:    return _color(text, "31")
def _blue(text: str) -> str:   return _color(text, "34")

def _clear_line() -> None:
    sys.stdout.write(f"{_CSI}2K\r"); sys.stdout.flush()

def _move_up(n: int = 1) -> None:
    sys.stdout.write(f"{_CSI}{n}A"); sys.stdout.flush()

_USERNAME_CODES = (31, 32, 33, 34, 35, 36)

def _username_color(name: str) -> int:
    return _USERNAME_CODES[hash(name) % len(_USERNAME_CODES)]

def _colored_name(name: str) -> str:
    return _color(name, str(_username_color(name)))

# -- HTTP helpers (stdlib only) ---------------------------------------------

def _http_get(url: str, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def _http_post(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else b""
    headers = {"Content-Type": "application/json"} if body is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

# -- Banner & typing animation ---------------------------------------------

_BANNER = r"""
  {g}  __  __              {c}    _            _ _ {r}
  {g}  \ \/ /__ _ __ _     {c}   | |__ ___  __| | |{r}
  {g}   \  / _` / _` |    {c}   | '_ \/ _ \/ _` |_|{r}
  {g}   / / (_| \__, |    {c}   |_.__/\___/\__,_(_){r}
  {g}  /_/ \__, |___/     {c}          chat{r}
  {g}      |___/          {r}
"""

def _print_banner() -> None:
    sys.stdout.write(_BANNER.format(g=_CSI + "32m", c=_CSI + "36m", r=_CSI + "0m"))
    sys.stdout.flush()

def _typing_dots(duration: float = 0.6, frames: int = 3) -> None:
    """Show a brief typing indicator before rendering a message."""
    delay = duration / (frames + 1)
    for i in range(1, frames + 1):
        sys.stdout.write(f"\r  {_dim('.' * i)}"); sys.stdout.flush()
        time.sleep(delay)
    _clear_line()

# -- ChatClient -------------------------------------------------------------

class ChatClient:
    """Interactive terminal chat client for the bot messenger API."""

    def __init__(self, base_url: str, username: str, channel: str = "general"):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.channel = channel
        self.last_id: str = "0"
        self._stop = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def _prompt(self) -> str:
        return f"[{_cyan(self.channel)}] {_colored_name(self.username)}> "

    def _write_prompt(self) -> None:
        sys.stdout.write(self._prompt); sys.stdout.flush()

    def _api(self, path: str) -> str:
        return f"{self.base_url}/api/messenger{path}"

    # -- API calls -------------------------------------------------------

    def send_message(self, text: str) -> dict | None:
        try:
            resp = _http_post(
                self._api(""),
                {"text": text, "sender": self.username, "channel": self.channel},
            )
            if "id" in resp:  # track own id so poller skips it
                self.last_id = resp["id"]
            return resp
        except urllib.error.URLError as exc:
            self._print_system(f"send failed: {exc.reason}")
            return None

    def list_channels(self) -> list[dict]:
        try:
            return _http_get(self._api("/channels")).get("channels", [])
        except urllib.error.URLError as exc:
            self._print_system(f"cannot list channels: {exc.reason}")
            return []

    def fetch_messages(self, channel: str | None = None) -> list[dict]:
        try:
            return _http_get(
                self._api(f"/channels/{channel or self.channel}/messages"),
            ).get("messages", [])
        except urllib.error.URLError:
            return []

    def poll_messages(self) -> list[dict]:
        try:
            return _http_get(
                self._api(
                    f"/channels/{self.channel}/poll"
                    f"?after_id={self.last_id}&timeout=25"
                ),
                timeout=30.0,
            ).get("messages", [])
        except (urllib.error.URLError, TimeoutError, OSError):
            return []

    def create_channel(self, name: str) -> dict | None:
        # Server expects name as a query parameter, not a JSON body
        try:
            encoded = urllib.request.quote(name, safe="")
            return _http_post(self._api(f"/channels?name={encoded}"))
        except urllib.error.URLError as exc:
            self._print_system(f"create channel failed: {exc.reason}")
            return None

    # -- display ---------------------------------------------------------

    @staticmethod
    def _format_ts(ts: str) -> str:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%H:%M")
        except (ValueError, TypeError):
            return ts[:5] if len(ts) >= 5 else ts

    def _print_message(self, msg: dict, *, animate: bool = False) -> None:
        sender, text = msg.get("sender", "???"), msg.get("text", "")
        ts = self._format_ts(msg.get("timestamp", ""))
        if animate and sender != self.username:
            _typing_dots(duration=0.35)
        sys.stdout.write(f"  {_dim(ts)} {_colored_name(sender)}: {text}\n")
        sys.stdout.flush()

    def _print_system(self, text: str) -> None:
        sys.stdout.write(f"  {_dim('*')} {_yellow(text)}\n"); sys.stdout.flush()

    # -- background poller -----------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            messages = self.poll_messages()
            if not messages:
                continue
            with self._lock:
                _clear_line()
                for msg in messages:
                    if msg.get("sender") != self.username:
                        self._print_message(msg, animate=True)
                    self.last_id = msg.get("id", self.last_id)
                self._write_prompt()

    def _start_poller(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._poll_loop, daemon=True, name="chat-poll")
        t.start()
        self._poll_thread = t

    def _stop_poller(self) -> None:
        self._stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None

    # -- slash commands ---------------------------------------------------

    def _cmd_join(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._print_system("usage: /join <channel>"); return
        self._stop_poller()
        self.channel = name
        self._seed_last_id()
        self._print_system(f"joined #{name}")
        self._start_poller()

    def _cmd_channels(self) -> None:
        channels = self.list_channels()
        if not channels:
            self._print_system("no channels (or server unreachable)"); return
        self._print_system("channels:")
        for ch in channels:
            n, c = ch.get("name", "?"), ch.get("message_count", 0)
            m = len(ch.get("members", []))
            mark = _green(" <--") if n == self.channel else ""
            sys.stdout.write(f"    {_bold('#' + n)}  {_dim(f'{c} msgs, {m} members')}{mark}\n")
        sys.stdout.flush()

    def _cmd_create(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._print_system("usage: /create <channel>"); return
        if self.create_channel(name):
            self._print_system(f"channel #{name} created")

    def _cmd_users(self) -> None:
        for ch in self.list_channels():
            if ch.get("name") == self.channel:
                self._print_system(f"members of #{self.channel}:")
                for m in ch.get("members", []):
                    sys.stdout.write(f"    {_colored_name(m)}\n")
                sys.stdout.flush(); return
        self._print_system(f"channel #{self.channel} not found")

    def _cmd_help(self) -> None:
        self._print_system("commands:")
        for cmd, desc in [
            ("/join <channel>", "switch to a channel"),
            ("/channels", "list all channels"),
            ("/create <channel>", "create a new channel"),
            ("/users", "show members in current channel"),
            ("/help", "show this help"),
            ("/quit", "exit the chat"),
        ]:
            sys.stdout.write(f"    {_bold(cmd):30s} {_dim(desc)}\n")
        sys.stdout.flush()

    def _handle_command(self, line: str) -> bool:
        """Handle a ``/command``.  Returns *True* to quit."""
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if cmd == "/quit":
            return True
        handler = {
            "/join": lambda: self._cmd_join(arg),
            "/channels": self._cmd_channels,
            "/create": lambda: self._cmd_create(arg),
            "/users": self._cmd_users,
            "/help": self._cmd_help,
        }.get(cmd)
        if handler:
            handler()
        else:
            self._print_system(f"unknown command {cmd} -- try /help")
        return False

    def _seed_last_id(self) -> None:
        messages = self.fetch_messages()
        if messages:
            self.last_id = messages[-1].get("id", "0")

    # -- main REPL -------------------------------------------------------

    def run(self) -> int:
        _print_banner()
        self._print_system(
            f"connecting to {_bold(self.base_url)} as {_colored_name(self.username)}",
        )
        self._cmd_channels()
        self._print_system(f"joined #{self.channel} -- type /help for commands\n")
        self._seed_last_id()
        for msg in self.fetch_messages()[-20:]:
            self._print_message(msg)
        sys.stdout.write("\n")
        self._start_poller()
        try:
            while True:
                try:
                    self._write_prompt()
                    line = input()
                except EOFError:
                    break
                line = line.strip()
                if not line:
                    continue
                if line.startswith("/"):
                    if self._handle_command(line):
                        break
                    continue
                self.send_message(line)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_poller()
            sys.stdout.write("\n")
            self._print_system("disconnected")
        return 0


# -- Entry point ------------------------------------------------------------

def run_chat(
    url: str = "http://127.0.0.1:8100",
    username: str | None = None,
    channel: str = "general",
) -> int:
    """Launch the interactive terminal chat client."""
    if username is None:
        try:
            username = getpass.getuser()
        except Exception:
            username = "anon"
    return ChatClient(url, username, channel).run()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="yggdrasil bot chat client")
    p.add_argument("--url", default="http://127.0.0.1:8100", help="bot server URL")
    p.add_argument("--user", default=None, help="display name")
    p.add_argument("--channel", default="general", help="initial channel")
    a = p.parse_args()
    raise SystemExit(run_chat(url=a.url, username=a.user, channel=a.channel))
