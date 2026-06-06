"""Tests for Loki's interactive browser automation — fill forms, click, type.

The browser layer is Playwright-backed; these exercise the whole control flow
(navigate → fill → submit → read) against a fake Playwright so no real browser
or network is needed.
"""
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from yggdrasil.loki import web


class _FakePage:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.url = "https://site/start"
        self.keyboard = MagicMock()

    def set_default_timeout(self, t): self.calls.append(("timeout", t))
    def goto(self, u): self.calls.append(("goto", u)); self.url = "https://site/after"
    def fill(self, s, v): self.calls.append(("fill", s, v))
    def type(self, s, v, delay=0): self.calls.append(("type", s, v))
    def click(self, s): self.calls.append(("click", s))
    def check(self, s): self.calls.append(("check", s))
    def uncheck(self, s): self.calls.append(("uncheck", s))
    def select_option(self, s, v): self.calls.append(("select", s, v))
    def press(self, s, k): self.calls.append(("press", s, k))
    def wait_for_selector(self, s): self.calls.append(("wait", s))
    def title(self): return "Result Page"
    def input_value(self, s): return "typed"
    def inner_text(self, s): return "the resulting page body"
    def screenshot(self, path=None): self.calls.append(("shot", path))


def _install_fake_playwright():
    """Inject a fake ``playwright.sync_api`` and return the shared page."""
    page = _FakePage()
    ctx = MagicMock(); ctx.new_page.return_value = page
    browser = MagicMock(); browser.new_context.return_value = ctx
    chromium = MagicMock()
    chromium.launch.return_value = browser
    chromium.executable_path = "/usr/bin/chromium"
    pw = MagicMock(); pw.chromium = chromium
    handle = MagicMock()
    handle.start.return_value = pw
    handle.__enter__ = lambda s: pw       # for `with sync_playwright() as pw`
    handle.__exit__ = lambda s, *a: False
    sync_api = types.ModuleType("playwright.sync_api")
    sync_api.sync_playwright = lambda: handle
    root = types.ModuleType("playwright")
    root.sync_api = sync_api
    return page, {"playwright": root, "playwright.sync_api": sync_api}


class TestBrowser(unittest.TestCase):
    def test_fill_form_navigates_fills_and_submits(self):
        page, mods = _install_fake_playwright()
        with patch.dict(sys.modules, mods), \
             patch("importlib.util.find_spec", return_value=object()):
            out = web.fill_form(
                "https://site/login",
                {"#user": "me", "#pass": "secret"},
                submit="button[type=submit]",
                wait_for="#dashboard",
            )
        self.assertEqual(out["url"], "https://site/after")
        self.assertEqual(out["title"], "Result Page")
        self.assertEqual(out["filled"], ["#user", "#pass"])
        # The control sequence the browser actually performed.
        self.assertEqual(page.calls, [
            ("timeout", 30000),
            ("goto", "https://site/login"),
            ("fill", "#user", "me"),
            ("fill", "#pass", "secret"),
            ("click", "button[type=submit]"),
            ("wait", "#dashboard"),
        ])

    def test_interact_runs_each_step_in_order(self):
        page, mods = _install_fake_playwright()
        steps = [
            {"type": ["#q", "headphones"]},
            {"press": ["#q", "Enter"]},
            {"wait_for": ".results"},
            {"check": "#in-stock"},
            {"select": ["#sort", "price"]},
            {"click": ".results a"},
        ]
        with patch.dict(sys.modules, mods):
            out = web.interact("https://shop/search", steps)
        self.assertEqual(len(out["steps"]), 6)
        kinds = [c[0] for c in page.calls]
        self.assertEqual(kinds, ["timeout", "goto", "type", "press", "wait",
                                 "check", "select", "click"])

    def test_interact_rejects_unknown_step(self):
        _, mods = _install_fake_playwright()
        with patch.dict(sys.modules, mods):
            with self.assertRaises(ValueError):
                web.interact("https://x", [{"teleport": "#y"}])

    def test_submit_without_selector_presses_enter(self):
        page, mods = _install_fake_playwright()
        with patch.dict(sys.modules, mods):
            web.interact("https://x", [{"submit": None}])
        page.keyboard.press.assert_called_once_with("Enter")


class TestWebSkillAutomation(unittest.TestCase):
    def test_form_action_delegates_to_fill_form(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.capability import Backend

        loki = Loki(); loki._backends = [Backend("local", True)]
        with patch.object(web, "browser_available", return_value=True), \
             patch.object(web, "fill_form", return_value={"url": "u", "title": "t"}) as ff:
            res = loki.run("web", url="https://site/form", action="form",
                           fields={"#a": "1"}, submit="#go")
        self.assertEqual(res["action"], "form")
        ff.assert_called_once()
        self.assertEqual(ff.call_args.kwargs["submit"], "#go")

    def test_form_action_reports_install_when_unavailable(self):
        from yggdrasil.loki import Loki
        from yggdrasil.loki.capability import Backend

        loki = Loki(); loki._backends = [Backend("local", True)]
        with patch.object(web, "browser_available", return_value=False):
            res = loki.run("web", url="https://x", action="interact", steps=[])
        self.assertIn("install", res)
        self.assertIn("playwright", res["install"])


if __name__ == "__main__":
    unittest.main()
