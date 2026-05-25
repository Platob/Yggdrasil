"""Abstract base for Databricks-CLI entry points.

:class:`DatabricksCLI` owns three things every concrete sub-service CLI
needs to do the same way:

1. A shared ``"Databricks client"`` argparse group — every workspace
   credential / discovery flag the :class:`DatabricksClient`
   constructor accepts is surfaced here so each sub-service can stop
   re-declaring them. Use ``ygg-<service> --help`` to see the full
   list.
2. A single :meth:`build_client` handshake that filters argparse fields
   down to ``DatabricksClient.__init__`` kwargs, builds the client, and
   exits cleanly on construction errors (returning exit code 2).
3. A :meth:`parse_and_run` driver that subclasses opt into by exposing
   a ``main(argv)`` module-level function — keeps every sub-service
   entry point a one-liner.

Subclasses implement:

- :meth:`add_service_arguments` to attach service-specific flag groups
  (the Genie CLI adds Genie defaults, agent options, REPL settings, …).
- :meth:`run` — the actual service workload. Receives the parsed
  argparse namespace and a live :class:`DatabricksClient`.

Both hooks are deliberately tiny so a new sub-service CLI rarely needs
more than a 30-line subclass plus a ``main`` wrapper.
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from yggdrasil.databricks.client import DatabricksClient


__all__ = ["DatabricksCLI", "CLIENT_FLAGS"]


# ---------------------------------------------------------------------------
# Shared ``DatabricksClient.__init__`` flags. Order: common credential
# settings first, escape hatches and cloud-specific flags after.
# ---------------------------------------------------------------------------
#: ``(flag, dest, argparse_kwargs)`` triples consumed by :meth:`DatabricksCLI.build_parser`.
CLIENT_FLAGS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("--host", "host", {"help": "Workspace URL or hostname (env: DATABRICKS_HOST)"}),
    ("--token", "token", {"help": "Personal access token (env: DATABRICKS_TOKEN)"}),
    ("--profile", "profile", {"help": "Profile in ~/.databrickscfg (env: DATABRICKS_CONFIG_PROFILE)"}),
    ("--config-file", "config_file", {"help": "Path to a databricks config file"}),
    ("--auth-type", "auth_type", {"help": "Auth method (pat, oauth, azure-cli, github-oidc, …)"}),
    ("--client-id", "client_id", {"help": "OAuth client id (service principal)"}),
    ("--client-secret", "client_secret", {"help": "OAuth client secret"}),
    ("--account-id", "account_id", {"help": "Databricks account id"}),
    ("--workspace-id", "workspace_id", {"help": "Databricks workspace id"}),
    ("--cluster-id", "cluster_id", {"help": "Default cluster id for compute fallbacks"}),
    ("--serverless-compute-id", "serverless_compute_id",
     {"help": "Serverless compute id (default 'auto' when neither cluster nor this is set)"}),
    ("--azure-tenant-id", "azure_tenant_id", {"help": "Azure tenant id"}),
    ("--azure-client-id", "azure_client_id", {"help": "Azure client id"}),
    ("--azure-client-secret", "azure_client_secret", {"help": "Azure client secret"}),
    ("--azure-workspace-resource-id", "azure_workspace_resource_id",
     {"help": "Azure workspace resource id"}),
    ("--google-credentials", "google_credentials", {"help": "Google credentials JSON / path"}),
    ("--google-service-account", "google_service_account",
     {"help": "Google service-account email"}),
)


class DatabricksCLI(ABC):
    """Abstract base for ``ygg-<service>`` console scripts.

    Concrete subclasses live in ``yggdrasil.cli.databricks.<service>``
    and expose a module-level ``main(argv=None)`` function that simply
    calls ``cls.parse_and_run(argv)``.

    Attributes
    ----------
    prog
        Program name shown in ``--help``. Set on the subclass.
    description
        One-line description shown in ``--help``.
    epilog
        Optional example block appended to ``--help`` output.
    """

    #: Console-script name. Subclasses MUST override.
    prog: ClassVar[str] = "ygg-databricks"

    #: One-line description shown above the flag list.
    description: ClassVar[str] = "Databricks sub-service CLI."

    #: Multiline epilog appended after the flag list. Use for examples.
    epilog: ClassVar[Optional[str]] = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, client: "DatabricksClient", args: argparse.Namespace):
        self.client = client
        self.args = args

    # ------------------------------------------------------------------ #
    # Parser
    # ------------------------------------------------------------------ #
    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Build the argparse tree: client group + service flags."""
        parser = argparse.ArgumentParser(
            prog=cls.prog,
            description=cls.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=textwrap.dedent(cls.epilog) if cls.epilog else None,
        )
        client_grp = parser.add_argument_group("Databricks client")
        for flag, dest, kwargs in CLIENT_FLAGS:
            client_grp.add_argument(flag, dest=dest, default=None, **kwargs)
        cls.add_service_arguments(parser)
        # Universal flags every sub-service shares.
        parser.add_argument(
            "--debug", action="store_true",
            help="Set yggdrasil logger to DEBUG.",
        )
        return parser

    @classmethod
    @abstractmethod
    def add_service_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Attach service-specific flag groups to ``parser``.

        Subclasses override to declare their own argument groups. The
        method runs once per parser build — keep it side-effect-free.
        """

    # ------------------------------------------------------------------ #
    # Client construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def client_kwargs(args: argparse.Namespace) -> dict[str, Any]:
        """Return the :class:`DatabricksClient` kwargs supplied on the CLI.

        Skips fields the caller left at ``None`` so the client falls
        back to its env defaults (``DATABRICKS_*``).
        """
        return {
            dest: value
            for _flag, dest, _meta in CLIENT_FLAGS
            if (value := getattr(args, dest, None)) is not None
        }

    @classmethod
    def build_client(cls, args: argparse.Namespace) -> "DatabricksClient":
        """Construct the :class:`DatabricksClient` from CLI args.

        Lazy-imports the client so ``ygg-<service> --help`` doesn't pay
        the SDK import cost.
        """
        # Lazy import keeps --help fast and lets test harnesses patch
        # the symbol at ``yggdrasil.databricks.client.DatabricksClient``.
        from yggdrasil.databricks.client import DatabricksClient

        return DatabricksClient(**cls.client_kwargs(args))

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    @classmethod
    def parse_and_run(cls, argv: Optional[Sequence[str]] = None) -> int:
        """End-to-end: build parser, parse, build client, dispatch :meth:`run`.

        Returns the process exit code so concrete entry points can
        ``return DatabricksCLI.parse_and_run(...)``.
        """
        parser = cls.build_parser()
        args = parser.parse_args(argv)

        if getattr(args, "debug", False):
            logging.basicConfig(level=logging.DEBUG)
            logging.getLogger("yggdrasil").setLevel(logging.DEBUG)

        try:
            client = cls.build_client(args)
        except Exception:
            from yggdrasil.databricks.client import DatabricksClient
            client = DatabricksClient()

        instance = cls(client=client, args=args)
        return instance.run()

    # ------------------------------------------------------------------ #
    # Service entry point
    # ------------------------------------------------------------------ #
    @abstractmethod
    def run(self) -> int:
        """Run the sub-service. Returns a process exit code."""
