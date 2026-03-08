# test_http_proxy_uses_proxy.py
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread
from queue import Queue, Empty

import pytest
import requests


def _find_http_proxy_py() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("yggdrasil.web.http_proxy")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot find module yggdrasil.web.http_proxy. Is it installed / on PYTHONPATH?")
    return Path(spec.origin)


def _wait_port(host: str, port: int, timeout_s: float = 10.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError as e:
            last_err = e
            time.sleep(0.05)
    raise TimeoutError(f"Proxy did not open {host}:{port} within {timeout_s}s. Last error: {last_err!r}")


def _terminate_process(p: subprocess.Popen, grace_s: float = 3.0) -> None:
    if p.poll() is not None:
        return

    try:
        if os.name == "nt":
            p.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            p.send_signal(signal.SIGINT)
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.05)

    try:
        p.terminate()
    except Exception:
        pass

    t0 = time.time()
    while time.time() - t0 < grace_s:
        if p.poll() is not None:
            return
        time.sleep(0.05)

    try:
        p.kill()
    except Exception:
        pass


def _start_stdout_pump(proc: subprocess.Popen):
    """
    Read proxy stdout continuously so it doesn't buffer/hang and so we can assert logs.
    """
    q: Queue[str] = Queue()

    def _pump():
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                q.put(line)
        except Exception:
            pass

    t = Thread(target=_pump, daemon=True)
    t.start()
    return q


def _drain_logs(q: Queue[str]) -> str:
    chunks = []
    while True:
        try:
            chunks.append(q.get_nowait())
        except Empty:
            break
    return "".join(chunks)


def _wait_for_log(q: Queue[str], needle: str, timeout_s: float = 5.0) -> str:
    """
    Wait until `needle` appears in the streamed stdout.
    Returns collected logs (for debugging).
    """
    deadline = time.time() + timeout_s
    collected = ""
    while time.time() < deadline:
        collected += _drain_logs(q)
        if needle in collected:
            return collected
        time.sleep(0.05)
    return collected


@pytest.fixture(scope="session")
def proxy_process():
    host = os.getenv("PROXY_HOST", "127.0.0.1")
    port = int(os.getenv("PROXY_PORT", "8888"))

    proxy_py = _find_http_proxy_py()

    # IMPORTANT: if your proxy ignores these args, the port check will fail (good!),
    # and then we know the real issue is your proxy's CLI.
    cmd = [sys.executable, str(proxy_py), "--host", host, "--port", str(port)]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    env = os.environ.copy()
    # avoid weird proxy bypass env that can confuse debugging
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=creationflags,
        bufsize=1,  # line buffered
    )
    q = _start_stdout_pump(p)

    try:
        _wait_port(host, port, timeout_s=12.0)
        yield {"host": host, "port": port, "proc": p, "logq": q}
    finally:
        _terminate_process(p)
        logs = _drain_logs(q)
        if logs:
            print("\n--- proxy process output ---\n" + logs + "\n--- end proxy output ---\n")


def _session_for_proxy(host: str, port: int) -> requests.Session:
    proxy_url = f"http://{host}:{port}"
    s = requests.Session()
    s.trust_env = False  # <- KEY: ignore env proxies/no_proxy
    s.proxies = {"http": proxy_url, "https": proxy_url}
    s.headers.update({"User-Agent": "yggdrasil-proxy-unit-test/1.0"})
    return s


def test_proxy_http_is_used(proxy_process):
    host, port, q = proxy_process["host"], proxy_process["port"], proxy_process["logq"]
    s = _session_for_proxy(host, port)

    # pick a stable domain and check proxy logs mention it
    r = s.get("http://example.com", timeout=15)
    assert r.status_code == 200

    logs = _wait_for_log(q, "example.com", timeout_s=5.0)
    assert "example.com" in logs, f"Proxy stdout did not mention example.com. Logs:\n{logs}"


def test_proxy_https_connect_is_used(proxy_process):
    host, port, q = proxy_process["host"], proxy_process["port"], proxy_process["logq"]
    s = _session_for_proxy(host, port)

    r = s.get("https://example.com", timeout=25)
    assert r.status_code == 200

    # Most proxies log CONNECT or at least example.com:443
    logs = _wait_for_log(q, "CONNECT", timeout_s=5.0)
    if "CONNECT" not in logs:
        logs = _wait_for_log(q, "example.com", timeout_s=5.0)

    assert ("CONNECT" in logs) or ("example.com" in logs), (
        "Proxy stdout did not show HTTPS tunneling.\n"
        f"Logs:\n{logs}"
    )