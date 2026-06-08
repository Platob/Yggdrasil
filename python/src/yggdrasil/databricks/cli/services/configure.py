"""``ygg databricks configure`` — set up a Databricks profile, the way
``databricks configure`` does, then remember it as the current session.

Three things in one command:

1. **Write a profile** into ``~/.databrickscfg`` (the same INI file the
   Databricks CLI / SDK read). Host + token are taken from flags, or
   prompted for interactively (token hidden) when omitted — mirroring
   ``databricks configure --token``. Existing profiles in the file are
   preserved; only the named section is rewritten.
2. **Verify** the credentials by building a :class:`DatabricksClient`
   against the freshly written profile and resolving the current user
   (skippable with ``--no-verify``). With ``--sso`` this runs the
   interactive browser / CLI flow.
3. **Remember the session** — the verified client becomes the process
   *current* client (:meth:`DatabricksClient.set_current`) and a small
   metadata snapshot of the latest session (profile, host, user,
   workspace/account ids, timestamp) is dumped into the session folder
   ``~/.config/databricks-sdk-py/sessions/`` as ``<hostname>.json`` (the
   per-machine default) so later tooling can default to "the workspace I
   last configured on this host". For an **SSO** login the credential is
   not on disk, so the resolved session bearer token is captured into the
   snapshot too (and the file is locked to owner-only).

Sub-actions::

    ygg databricks configure                 # write + verify + remember
    ygg databricks configure --profile prod --host https://… --token dapi…
    ygg databricks configure --sso --host https://…   # SSO: dump the token
    ygg databricks configure list            # list profiles in ~/.databrickscfg
    ygg databricks configure session         # show the remembered session
"""
from __future__ import annotations

import configparser
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class ConfigureCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "configure",
            help="Set up a Databricks profile (~/.databrickscfg) and remember it as the current session.",
        )
        sub = parser.add_subparsers(dest="configure_action")

        # Bare ``configure`` writes/updates a profile. The client flags are
        # re-declared here (distinct dests) so they work *after* the
        # sub-command — ``configure --host … --token …`` — falling back to the
        # top-level ``ygg databricks --host …`` flags when given there instead.
        parser.add_argument("--profile", dest="configure_profile", default=None,
                            help="Profile name to write (default: DEFAULT).")
        parser.add_argument("--host", dest="configure_host", default=None,
                            help="Workspace URL (prompted if omitted).")
        parser.add_argument("--token", dest="configure_token", default=None,
                            help="Personal access token (prompted, hidden, if omitted).")
        parser.add_argument("--client-id", dest="configure_client_id", default=None,
                            help="OAuth client id (service principal) — written instead of a token.")
        parser.add_argument("--client-secret", dest="configure_client_secret", default=None,
                            help="OAuth client secret.")
        parser.add_argument("--sso", dest="sso", action="store_true",
                            help="Authenticate via SSO (interactive browser) — no static credential "
                                 "is written; the resolved session token is dumped into the session.")
        parser.add_argument("--auth-type", dest="configure_auth_type", default=None,
                            help="Explicit auth type (external-browser, azure-cli, databricks-cli, …); "
                                 "implies SSO when no token/secret is given.")
        parser.add_argument("--account-id", dest="configure_account_id", default=None,
                            help="Databricks account id (for account-level profiles).")
        parser.add_argument("--config-file", dest="configure_config_file", default=None,
                            help="Config file path (default: $DATABRICKS_CONFIG_FILE or ~/.databrickscfg).")
        parser.add_argument("--no-verify", dest="no_verify", action="store_true",
                            help="Skip the credential check (don't call the workspace).")
        parser.add_argument("--no-session", dest="no_session", action="store_true",
                            help="Write the profile but don't remember it as the current session.")
        parser.set_defaults(handler=cls._configure)

        ls = sub.add_parser("list", help="List profiles in ~/.databrickscfg.")
        ls.add_argument("--config-file", dest="configure_config_file", default=None,
                        help="Config file path (default: $DATABRICKS_CONFIG_FILE or ~/.databrickscfg).")
        ls.set_defaults(handler=cls._list)

        session = sub.add_parser("session", help="Show the remembered latest session metadata.")
        session.set_defaults(handler=cls._session)

    # ------------------------------------------------------------------ #
    # paths
    # ------------------------------------------------------------------ #
    @staticmethod
    def _config_file(args: Any) -> Path:
        """Resolve the ``~/.databrickscfg`` path (flag > env > default)."""
        explicit = getattr(args, "configure_config_file", None)
        path = explicit or os.environ.get("DATABRICKS_CONFIG_FILE") or "~/.databrickscfg"
        return Path(path).expanduser()

    @staticmethod
    def _session_dir() -> Path:
        """The session folder — per-host snapshots live here, beside the SDK's
        own ``~/.config/databricks-sdk-py`` cache folder."""
        return Path.home() / ".config" / "databricks-sdk-py" / "sessions"

    @staticmethod
    def _session_file() -> Path:
        """The default session file for this machine: ``sessions/<hostname>.json``.

        Keying the latest-session snapshot by the local hostname keeps each
        workstation's "last workspace I configured" separate when the home
        directory is shared (NFS / synced profiles)."""
        import socket
        host = socket.gethostname() or "default"
        # Sanitise to a filename-safe token (FQDNs, odd hostnames).
        host = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in host)
        return ConfigureCommand._session_dir() / f"{host}.json"

    # ------------------------------------------------------------------ #
    # configure
    # ------------------------------------------------------------------ #
    @classmethod
    def _configure(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        # Resolve inputs: sub-command flag wins, then the shared top-level
        # ``ygg databricks --host/--token/--profile`` flags.
        profile = getattr(args, "configure_profile", None) or getattr(args, "profile", None) or "DEFAULT"
        host = getattr(args, "configure_host", None) or getattr(args, "host", None)
        token = getattr(args, "configure_token", None) or getattr(args, "token", None)
        client_id = getattr(args, "configure_client_id", None)
        client_secret = getattr(args, "configure_client_secret", None)
        auth_type = getattr(args, "configure_auth_type", None)
        account_id = getattr(args, "configure_account_id", None)

        interactive = sys.stdin.isatty()
        style.out(f"\n  {style.bold('ygg databricks configure')}  "
                  f"{style.dim('profile ' + profile)}\n\n")

        # -- gather host -------------------------------------------------
        if not host and interactive:
            host = input("  Databricks host (https://...): ").strip()
        if not host:
            style.fail("a host is required (--host or DATABRICKS_HOST)")
            return 1
        if "://" not in host:
            host = "https://" + host
        host = host.rstrip("/")

        # -- pick the credential mode -----------------------------------
        # OAuth M2M (client id/secret) wins when supplied; SSO (a browser /
        # CLI flow, no static secret) when ``--sso`` / ``--auth-type`` is set;
        # otherwise a PAT, prompted hidden when interactive.
        oauth = bool(client_id or client_secret)
        sso = bool(getattr(args, "sso", False) or auth_type) and not oauth and not token
        if sso:
            auth_type = auth_type or "external-browser"
            kind = "sso"
        elif oauth:
            if not (client_id and client_secret):
                style.fail("OAuth needs both --client-id and --client-secret")
                return 1
            kind = "oauth"
        else:
            if not token and interactive:
                import getpass
                token = getpass.getpass("  Token (hidden): ").strip()
            if not token:
                style.fail("a credential is required: --token, --client-id/--client-secret, or --sso")
                return 1
            kind = "pat"

        # -- write the profile ------------------------------------------
        config_file = cls._config_file(args)
        fields: dict[str, str] = {"host": host}
        if kind == "oauth":
            fields["client_id"] = client_id
            fields["client_secret"] = client_secret
        elif kind == "sso":
            # SSO carries no static secret — only the host + auth type so the
            # SDK runs the right interactive flow on use.
            fields["auth_type"] = auth_type
        else:
            fields["token"] = token
        if account_id:
            fields["account_id"] = account_id

        existed = config_file.exists()
        cls._write_profile(config_file, profile, fields)
        style.ok(f"profile {style.bold(profile)} {'updated' if existed else 'written'} → {config_file}")

        # -- verify + remember session ----------------------------------
        from yggdrasil.databricks.client import DatabricksClient

        # Build straight off the profile we just wrote so the verify reflects
        # exactly what's on disk (not the in-process env / flags).
        if kind == "sso":
            # SSO is a U2M **browser** login — force ``external-browser`` and
            # explicitly carry *no* static credential so an ambient
            # ``DATABRICKS_TOKEN`` (or a profile token) can't shortcut the flow
            # or trip the SDK's "more than one auth method" guard. ``token=None``
            # (not the sentinel) keeps the env default out.
            client = DatabricksClient(
                profile=profile, config_file=str(config_file),
                auth_type=auth_type or "external-browser",
                token=None, client_id=None, client_secret=None,
            )
        else:
            client = DatabricksClient(profile=profile, config_file=str(config_file))

        # SSO must authenticate to mint a token to remember — force the flow
        # even if the user passed --no-verify.
        verify = (not args.no_verify) or (kind == "sso" and not args.no_session)
        user: Optional[str] = None
        if verify:
            try:
                with style.Spinner(
                    "signing in via SSO (external browser)..." if kind == "sso"
                    else "verifying credentials...",
                    color="33",
                ):
                    user = client.workspace_client().current_user.me().user_name
                style.ok(f"authenticated as {style.bold(user)}")
            except Exception as exc:  # noqa: BLE001
                # Could not connect → don't leave a broken profile to fail
                # silently later. Invalidate it: drop the section we wrote and
                # forget any remembered session that pointed at it.
                cls._remove_profile(config_file, profile)
                cls._forget_session(profile)
                style.fail(f"could not connect with profile {style.bold(profile)} — "
                           f"config invalidated and removed: {exc}")
                return 1

        if not args.no_session:
            DatabricksClient.set_current(client)
            path = cls._dump_session(client, profile=profile, host=host,
                                     config_file=config_file, user=user, kind=kind,
                                     auth_type=auth_type)
            extra = style.dim(" (incl. SSO token)") if kind == "sso" else ""
            style.ok(f"remembered as current session → {path}{extra}")

        style.out("\n")
        style.out(f"  Use it: {style.dim('ygg databricks --profile ' + profile + ' <command>')}\n")
        return 0

    @staticmethod
    def _write_profile(config_file: Path, profile: str, fields: dict[str, str]) -> None:
        """Upsert ``[profile]`` into the INI file, preserving the rest.

        ``configparser`` round-trips every other section verbatim; only the
        named profile is rewritten. ``DEFAULT`` lands in configparser's
        special default section — exactly how ``databricks configure``
        stores its default profile.
        """
        parser = configparser.ConfigParser()
        if config_file.exists():
            parser.read(config_file)

        if profile.upper() == "DEFAULT":
            target = parser["DEFAULT"]
        else:
            if not parser.has_section(profile):
                parser.add_section(profile)
            target = parser[profile]
        for key, value in fields.items():
            target[key] = value

        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as fh:
            parser.write(fh)
        # Credentials live here — keep it owner-only (best-effort; no-op on
        # filesystems / platforms without POSIX modes).
        try:
            config_file.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def _remove_profile(config_file: Path, profile: str) -> None:
        """Drop ``[profile]`` from the INI file (invalidate a broken config).

        Preserves every other section, mirroring :meth:`_write_profile`. A
        ``DEFAULT`` profile is emptied (its keys cleared) since configparser's
        special default section can't be "removed".
        """
        if not config_file.exists():
            return
        parser = configparser.ConfigParser()
        parser.read(config_file)
        if profile.upper() == "DEFAULT":
            for key in list(parser["DEFAULT"].keys()):
                del parser["DEFAULT"][key]
        elif parser.has_section(profile):
            parser.remove_section(profile)
        with open(config_file, "w") as fh:
            parser.write(fh)

    @classmethod
    def _forget_session(cls, profile: str) -> None:
        """Remove the remembered session **iff** it points at *profile* — a
        config that just failed to connect shouldn't stay the current session."""
        meta = cls._read_session()
        if meta and meta.get("profile") == profile:
            try:
                cls._session_file().unlink()
            except OSError:
                pass

    @classmethod
    def _dump_session(
        cls,
        client: Any,
        *,
        profile: str,
        host: str,
        config_file: Path,
        user: Optional[str],
        kind: str,
        auth_type: Optional[str] = None,
    ) -> Path:
        """Persist a snapshot of the just-configured session.

        A tiny JSON record of "the workspace the user last set up" — enough
        for later tooling to default to it without re-reading credentials.

        For **PAT / OAuth** profiles no secret is written (the credential
        already lives in ``~/.databrickscfg``); only non-sensitive identity +
        routing metadata is dumped. For **SSO** the credential is *not* on
        disk — it's an ephemeral, interactively-minted bearer — so the
        resolved session token is captured here, and the session file is
        locked down to owner-only.
        """
        import socket

        meta: dict[str, Any] = {
            "profile": profile,
            "host": host,
            "hostname": socket.gethostname() or None,
            "config_file": str(config_file),
            "auth_type": auth_type or kind,
            "user": user,
            "product": getattr(client, "product", None),
            "project": getattr(client, "project", None),
            "product_version": getattr(client, "product_version", None),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        # Best-effort enrichment — never let a slow / failing API call block
        # writing the session record.
        for key, getter in (("account_id", "account_id"), ("workspace_id", "workspace_id")):
            try:
                meta[key] = getattr(client, getter, None)
            except Exception:  # noqa: BLE001
                meta[key] = None

        has_secret = False
        if kind == "sso":
            # Resolve the bearer the SSO flow just minted and stash it so the
            # session can be replayed without re-prompting the browser.
            try:
                header = client.files_authorization()  # "Bearer <token>"
                scheme, _, value = header.partition(" ")
                meta["token_type"] = scheme or "Bearer"
                meta["access_token"] = value or header
                has_secret = True
            except Exception as exc:  # noqa: BLE001 — token capture is best-effort
                meta["access_token"] = None
                meta["token_error"] = str(exc)

        path = cls._session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, default=str))
        # A captured SSO bearer is a secret — owner-only, like ~/.databrickscfg.
        if has_secret:
            try:
                path.chmod(0o600)
            except OSError:
                pass
        return path

    # ------------------------------------------------------------------ #
    # list / session
    # ------------------------------------------------------------------ #
    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        config_file = cls._config_file(args)
        if not config_file.exists():
            style.warn(f"no config file at {config_file}")
            return 1

        parser = configparser.ConfigParser()
        parser.read(config_file)

        # The DEFAULT profile is real in databrickscfg but lives in
        # configparser's special section — surface it explicitly.
        names = (["DEFAULT"] if dict(parser["DEFAULT"]) else []) + parser.sections()
        if not names:
            style.warn(f"no profiles in {config_file}")
            return 1

        remembered = cls._read_session()
        current = remembered.get("profile") if remembered else None
        for name in names:
            section = parser["DEFAULT"] if name == "DEFAULT" else parser[name]
            host = section.get("host", "")
            mark = style.brand(" ●") if name == current else ""
            sys.stdout.write(f"{name}\t{host}{mark}\n")
        return 0

    @classmethod
    def _session(cls, args: Any, build_client: Any) -> int:
        from yggdrasil.cli import style

        meta = cls._read_session()
        if not meta:
            style.warn(f"no remembered session at {cls._session_file()}")
            style.out(f"  Set one with: {style.dim('ygg databricks configure')}\n")
            return 1
        sys.stdout.write(json.dumps(meta, indent=2))
        sys.stdout.write("\n")
        return 0

    @classmethod
    def _read_session(cls) -> Optional[dict[str, Any]]:
        path = cls._session_file()
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None
