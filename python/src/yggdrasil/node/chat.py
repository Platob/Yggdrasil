"""Terminal chat client for the yggdrasil bot messenger."""
from __future__ import annotations

import getpass
import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone

from yggdrasil.cli.style import (
    Spinner, bold, clear_line, colored_name, cyan, dim, green,
    out, print_logo, typing_dots, yellow,
)


# -- HTTP helpers (stdlib only) ---------------------------------------------

def _http_get(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return json.loads(r.read())


def _http_post(url: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    data = json.dumps(body).encode() if body is not None else b""
    hdrs = {"Content-Type": "application/json"} if body is not None else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, data=data, headers=hdrs), timeout=timeout,
    ) as r:
        return json.loads(r.read())


# -- ChatClient -------------------------------------------------------------

class ChatClient:

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
        return f"[{cyan(self.channel)}] {colored_name(self.username)}> "

    def _write_prompt(self) -> None:
        out(self._prompt)

    def _api(self, path: str) -> str:
        return f"{self.base_url}/api/messenger{path}"

    # -- API calls ---------------------------------------------------------

    def send_message(self, text: str) -> dict | None:
        try:
            resp = _http_post(
                self._api(""),
                {"text": text, "sender": self.username, "channel": self.channel},
            )
            if "id" in resp:
                self.last_id = resp["id"]
            return resp
        except urllib.error.URLError as exc:
            self._sys(f"send failed: {exc.reason}")
            return None

    def list_channels(self) -> list[dict]:
        try:
            return _http_get(self._api("/channels")).get("channels", [])
        except urllib.error.URLError as exc:
            self._sys(f"cannot list channels: {exc.reason}")
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
        try:
            encoded = urllib.request.quote(name, safe="")
            return _http_post(self._api(f"/channels?name={encoded}"))
        except urllib.error.URLError as exc:
            self._sys(f"create channel failed: {exc.reason}")
            return None

    # -- display -----------------------------------------------------------

    @staticmethod
    def _format_ts(ts: str) -> str:
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%H:%M")
        except (ValueError, TypeError):
            return ts[:5] if len(ts) >= 5 else ts

    def _print_msg(self, msg: dict, *, animate: bool = False) -> None:
        sender, text = msg.get("sender", "???"), msg.get("text", "")
        ts = self._format_ts(msg.get("timestamp", ""))
        if animate and sender != self.username:
            typing_dots()
        out(f"  {dim(ts)} {colored_name(sender)}: {text}\n")

    def _sys(self, text: str) -> None:
        out(f"  {dim('*')} {yellow(text)}\n")

    # -- background poller -------------------------------------------------

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            messages = self.poll_messages()
            if not messages:
                continue
            with self._lock:
                clear_line()
                for msg in messages:
                    if msg.get("sender") != self.username:
                        self._print_msg(msg, animate=True)
                    self.last_id = msg.get("id", self.last_id)
                self._write_prompt()

    def _start_poller(self) -> None:
        self._stop.clear()
        t = threading.Thread(target=self._poll_loop, daemon=True, name="chat-poll")
        t.start()
        self._poll_thread = t

    def _stop_poller(self) -> None:
        self._stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None

    # -- slash commands ----------------------------------------------------

    def _handle_command(self, line: str) -> bool:
        parts = line.split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
        if cmd == "/quit":
            return True
        handlers = {
            "/join": lambda: self._cmd_join(arg),
            "/channels": self._cmd_channels,
            "/create": lambda: self._cmd_create(arg),
            "/users": self._cmd_users,
            "/help": self._cmd_help,
        }
        h = handlers.get(cmd)
        if h:
            h()
        else:
            self._sys(f"unknown command {cmd} -- try /help")
        return False

    def _cmd_join(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._sys("usage: /join <channel>"); return
        self._stop_poller()
        self.channel = name
        self._seed_last_id()
        self._sys(f"joined #{name}")
        self._start_poller()

    def _cmd_channels(self) -> None:
        channels = self.list_channels()
        if not channels:
            self._sys("no channels (or server unreachable)"); return
        self._sys("channels:")
        for ch in channels:
            n = ch.get("name", "?")
            c, m = ch.get("message_count", 0), len(ch.get("members", []))
            mark = green(" <--") if n == self.channel else ""
            out(f"    {bold('#' + n)}  {dim(f'{c} msgs, {m} members')}{mark}\n")

    def _cmd_create(self, arg: str) -> None:
        name = arg.strip()
        if not name:
            self._sys("usage: /create <channel>"); return
        if self.create_channel(name):
            self._sys(f"channel #{name} created")

    def _cmd_users(self) -> None:
        for ch in self.list_channels():
            if ch.get("name") == self.channel:
                self._sys(f"members of #{self.channel}:")
                for m in ch.get("members", []):
                    out(f"    {colored_name(m)}\n")
                return
        self._sys(f"channel #{self.channel} not found")

    def _cmd_help(self) -> None:
        self._sys("commands:")
        for c, d in [
            ("/join <channel>", "switch to a channel"),
            ("/channels", "list all channels"),
            ("/create <channel>", "create a new channel"),
            ("/users", "show members"),
            ("/help", "show this help"),
            ("/quit", "exit"),
        ]:
            out(f"    {bold(c):30s} {dim(d)}\n")

    def _seed_last_id(self) -> None:
        messages = self.fetch_messages()
        if messages:
            self.last_id = messages[-1].get("id", "0")

    # -- REPL --------------------------------------------------------------

    def run(self) -> int:
        print_logo("YGGCHAT")
        with Spinner("connecting...", color="33") as sp:
            channels = self.list_channels()
            sp.stop(f"{yellow('connected')} to {bold(self.base_url)} as {colored_name(self.username)}")
        self._cmd_channels()
        self._sys(f"joined #{self.channel} -- type /help for commands\n")
        self._seed_last_id()
        for msg in self.fetch_messages()[-20:]:
            self._print_msg(msg)
        out("\n")
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
            out("\n")
            self._sys("disconnected")
        return 0


def run_chat(
    url: str = "http://127.0.0.1:8100",
    username: str | None = None,
    channel: str = "general",
) -> int:
    if username is None:
        try:
            username = getpass.getuser()
        except Exception:
            username = "anon"
    return ChatClient(url, username, channel).run()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="YGGCHAT")
    p.add_argument("--url", default="http://127.0.0.1:8100")
    p.add_argument("--user", default=None)
    p.add_argument("--channel", default="general")
    a = p.parse_args()
    raise SystemExit(run_chat(url=a.url, username=a.user, channel=a.channel))
