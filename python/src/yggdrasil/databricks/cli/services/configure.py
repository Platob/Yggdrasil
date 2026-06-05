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
   (skippable with ``--no-verify``).
3. **Remember the session** — the verified client becomes the process
   *current* client (:meth:`DatabricksClient.set_current`) and a small
   metadata snapshot of the latest session (profile, host, user,
   workspace/account ids, timestamp) is dumped to
   ``~/.config/databricks-sdk-py/ygg-session.json`` so later tooling can
   default to "the workspace I last configured".

Sub-actions::

    ygg databricks configure                 # write + verify + remember
    ygg databricks configure --profile prod --host https://… --token dapi…
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
    def _session_file() -> Path:
        """Where the latest-session snapshot is dumped — alongside the SDK's
        own ``~/.config/databricks-sdk-py`` cache folder."""
        return Path.home() / ".config" / "databricks-sdk-py" / "ygg-session.json"

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

        # -- gather credential ------------------------------------------
        # OAuth (client id/secret) takes precedence when supplied; otherwise
        # a PAT, prompted hidden when interactive.
        oauth = bool(client_id or client_secret)
        if not oauth and not token and interactive:
            import getpass
            token = getpass.getpass("  Token (hidden): ").strip()
        if oauth:
            if not (client_id and client_secret):
                style.fail("OAuth needs both --client-id and --client-secret")
                return 1
        elif not token:
            style.fail("a token is required (--token or DATABRICKS_TOKEN)")
            return 1

        # -- write the profile ------------------------------------------
        config_file = cls._config_file(args)
        fields: dict[str, str] = {"host": host}
        if oauth:
            fields["client_id"] = client_id
            fields["client_secret"] = client_secret
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
        client = DatabricksClient(profile=profile, config_file=str(config_file))

        user: Optional[str] = None
        if not args.no_verify:
            try:
                with style.Spinner("verifying credentials...", color="33"):
                    user = client.workspace_client().current_user.me().user_name
                style.ok(f"authenticated as {style.bold(user)}")
            except Exception as exc:  # noqa: BLE001 — profile is saved regardless
                style.warn(f"profile saved, but verification failed: {exc}")

        if not args.no_session:
            DatabricksClient.set_current(client)
            path = cls._dump_session(client, profile=profile, host=host,
                                     config_file=config_file, user=user, oauth=oauth)
            style.ok(f"remembered as current session → {path}")

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

    @classmethod
    def _dump_session(
        cls,
        client: Any,
        *,
        profile: str,
        host: str,
        config_file: Path,
        user: Optional[str],
        oauth: bool,
    ) -> Path:
        """Persist a snapshot of the just-configured session.

        A tiny JSON record of "the workspace the user last set up" — enough
        for later tooling to default to it without re-reading credentials.
        No secrets are written: token / client_secret are deliberately left
        out; only the non-sensitive identity + routing metadata is dumped.
        """
        meta: dict[str, Any] = {
            "profile": profile,
            "host": host,
            "config_file": str(config_file),
            "auth_type": "oauth" if oauth else "pat",
            "user": user,
            "product": getattr(client, "product", None),
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

        path = cls._session_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2, default=str))
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
