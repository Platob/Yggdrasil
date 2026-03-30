from __future__ import annotations

from importlib import import_module


mod = import_module("yggdrasil.mongoengine.http_proxy.app.main")


def test_databricks_app_entrypoint_builds_listen_from_env(monkeypatch):
    captured = {}

    def fake_proxy_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9000")
    monkeypatch.setattr(mod, "proxy_main", fake_proxy_main, raising=True)

    assert mod.main() == 0
    assert captured["argv"] == ["--listen", "0.0.0.0:9000"]
