# test_userinfo.py
from __future__ import annotations

import subprocess
from typing import Any

import pytest

import yggdrasil.environ.userinfo as userinfo


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    # Ensure tests don't bleed cache state into each other
    userinfo._clear_cache()


def test_current_happy_path_windows_upn_email(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host1")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        if cmd == ["whoami"]:
            return "DOMAIN\\nika\n"
        if cmd == ["whoami", "/UPN"]:
            return "nika@corp.com\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    ui = userinfo.UserInfo.current()
    assert ui.hostname == "host1"
    assert ui.sam == "DOMAIN\\nika"
    assert ui.email == "nika@corp.com"


def test_current_upn_null_becomes_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host2")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        if cmd == ["whoami"]:
            return "nika\n"
        if cmd == ["whoami", "/UPN"]:
            return "null\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    ui = userinfo.UserInfo.current()
    assert ui.sam == "nika"
    assert ui.email is None


def test_current_upn_without_at_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host3")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        if cmd == ["whoami"]:
            return "nika\n"
        if cmd == ["whoami", "/UPN"]:
            return "CORP\\nika\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    ui = userinfo.UserInfo.current()
    assert ui.sam == "nika"
    assert ui.email is None


def test_current_when_upn_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host4")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        if cmd == ["whoami"]:
            return "nika\n"
        if cmd == ["whoami", "/UPN"]:
            raise subprocess.CalledProcessError(1, cmd)
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    ui = userinfo.UserInfo.current()
    assert ui.sam == "nika"
    assert ui.email is None


def test_username_fallback_env_when_whoami_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host5")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        raise OSError("no whoami")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)
    monkeypatch.setenv("USER", "envuser")

    ui = userinfo.UserInfo.current()
    assert ui.sam == "envuser"
    assert ui.email is None
    assert ui.hostname == "host5"


def test_username_fallback_unknown_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host6")

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        raise OSError("no whoami")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    monkeypatch.delenv("USERNAME", raising=False)
    monkeypatch.delenv("USER", raising=False)
    monkeypatch.delenv("LOGNAME", raising=False)

    ui = userinfo.UserInfo.current()
    assert ui.sam == "UNKNOWN"
    assert ui.email is None
    assert ui.hostname == "host6"


def test_get_user_info_is_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        userinfo.UserInfo,
        "current",
        classmethod(lambda cls, *, refresh=False: cls(email=None, sam="x", hostname="h")),
    )
    ui = userinfo.get_user_info()
    assert (ui.email, ui.sam, ui.hostname) == (None, "x", "h")


def test_cache_returns_same_object_and_avoids_second_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host7")

    calls: list[list[str]] = []

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        calls.append(cmd)
        if cmd == ["whoami"]:
            return "nika\n"
        if cmd == ["whoami", "/UPN"]:
            return "nika@corp.com\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    a = userinfo.get_user_info()
    b = userinfo.get_user_info()

    assert a is b  # exact same cached object
    assert calls == [["whoami"], ["whoami", "/UPN"]]  # only once each


def test_refresh_bypasses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(userinfo.socket, "gethostname", lambda: "host8")

    n = {"i": 0}

    def fake_check_output(cmd: list[str], text: bool, stderr: Any) -> str:
        if cmd == ["whoami"]:
            n["i"] += 1
            return f"nika{n['i']}\n"
        if cmd == ["whoami", "/UPN"]:
            return "nika@corp.com\n"
        raise AssertionError(f"unexpected cmd: {cmd}")

    monkeypatch.setattr(subprocess, "check_output", fake_check_output)

    a = userinfo.get_user_info()
    b = userinfo.get_user_info(refresh=True)

    assert a is not b
    assert a.sam != b.sam
