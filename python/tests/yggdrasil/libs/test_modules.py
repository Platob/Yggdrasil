from yggdrasil.libs import modules


def test_check_modules_retry_and_install(monkeypatch):
    calls = []
    attempts = {"count": 0}

    monkeypatch.setattr(modules, "install_package", lambda pkg, **_: calls.append(pkg))

    @modules.check_modules("missing_mod")
    def needs_install():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ImportError("No module named 'missing_mod'")
        return "ok"

    assert needs_install() == "ok"
    assert calls == ["missing_mod"]
    assert attempts["count"] == 2


def test_check_modules_supports_package_argument(monkeypatch):
    calls = []
    attempts = {"count": 0}

    monkeypatch.setattr(modules, "install_package", lambda pkg, **_: calls.append(pkg))

    @modules.check_modules(package="polars")
    def needs_polars():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ImportError("polars not found")
        return "ok"

    assert needs_polars() == "ok"
    assert calls == ["polars"]
    assert attempts["count"] == 2


def test_check_modules_bare_decorator(monkeypatch):
    calls = []

    monkeypatch.setattr(modules, "install_package", lambda pkg, **_: calls.append(pkg))

    @modules.check_modules
    def bare():
        return "ok"

    assert bare() == "ok"
    assert calls == []
