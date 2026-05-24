"""``ygg-databricks clusters`` — manage Databricks clusters."""
from __future__ import annotations

import sys
from typing import Any


class ClustersCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser("clusters", help="Manage Databricks clusters.")
        sub = parser.add_subparsers(dest="clusters_action")

        ls = sub.add_parser("list", help="List clusters in the workspace.")
        ls.add_argument("--name", default=None, help="Filter by cluster name.")
        ls.set_defaults(handler=cls._list)

        get = sub.add_parser("get", help="Get a cluster by id or name.")
        get.add_argument("--id", dest="cluster_id", default=None)
        get.add_argument("--name", default=None)
        get.set_defaults(handler=cls._get)

        create = sub.add_parser("create", help="Create a cluster.")
        create.add_argument("--name", required=True, help="Cluster name.")
        create.add_argument("--node-type", dest="node_type_id", default=None)
        create.add_argument("--num-workers", type=int, default=None)
        create.add_argument("--spark-version", default=None)
        create.add_argument("--autotermination-minutes", type=int, default=None)
        create.add_argument("--single-user", dest="single_user_name", default=None)
        create.add_argument("-f", "--file", dest="config_file", default=None,
                            help="Cluster config YAML file (overrides other flags).")
        create.set_defaults(handler=cls._create)

        delete = sub.add_parser("delete", help="Delete a cluster.")
        delete.add_argument("--id", dest="cluster_id", required=True)
        delete.set_defaults(handler=cls._delete)

        start = sub.add_parser("start", help="Start a stopped cluster.")
        start.add_argument("--id", dest="cluster_id", default=None)
        start.add_argument("--name", default=None)
        start.set_defaults(handler=cls._start)

        stop = sub.add_parser("stop", help="Stop a running cluster.")
        stop.add_argument("--id", dest="cluster_id", default=None)
        stop.add_argument("--name", default=None)
        stop.set_defaults(handler=cls._stop)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        for cluster in client.compute.clusters.list(
            name=getattr(args, "name", None),
        ):
            state = ""
            if cluster.details and cluster.details.state:
                state = f"\t{cluster.details.state.value}"
            sys.stdout.write(
                f"{cluster.cluster_id}\t{cluster.cluster_name}{state}\n"
            )
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        cluster = client.compute.clusters.find_cluster(
            cluster_id=getattr(args, "cluster_id", None),
            cluster_name=getattr(args, "name", None),
            raise_error=True,
        )
        details = cluster.details
        sys.stdout.write(
            f"Cluster: {cluster.cluster_name}\n"
            f"ID:      {cluster.cluster_id}\n"
            f"State:   {details.state.value if details and details.state else 'unknown'}\n"
        )
        if details:
            if details.spark_version:
                sys.stdout.write(f"Runtime: {details.spark_version}\n")
            if details.node_type_id:
                sys.stdout.write(f"Node:    {details.node_type_id}\n")
            if details.num_workers is not None:
                sys.stdout.write(f"Workers: {details.num_workers}\n")
        return 0

    @classmethod
    def _create(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)

        spec: dict[str, Any] = {}
        if args.config_file:
            from pathlib import Path
            import yaml
            cfg = yaml.safe_load(Path(args.config_file).read_text(encoding="utf-8"))
            from ..bundle.resources import _build_cluster_spec
            spec = _build_cluster_spec(cfg)
        else:
            for attr, key in (
                ("node_type_id", "node_type_id"),
                ("num_workers", "num_workers"),
                ("spark_version", "spark_version"),
                ("autotermination_minutes", "autotermination_minutes"),
                ("single_user_name", "single_user_name"),
            ):
                val = getattr(args, attr, None)
                if val is not None:
                    spec[key] = val

        cluster = client.compute.clusters.create(
            cluster_name=args.name, wait=False, **spec,
        )
        sys.stdout.write(
            f"{cluster.cluster_id}\t{cluster.cluster_name}\n"
        )
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        client.compute.clusters.delete(args.cluster_id)
        sys.stderr.write("Deleted.\n")
        return 0

    @classmethod
    def _start(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        cluster = client.compute.clusters.find_cluster(
            cluster_id=getattr(args, "cluster_id", None),
            cluster_name=getattr(args, "name", None),
            raise_error=True,
        )
        cluster.start(wait=False)
        sys.stderr.write(f"Starting cluster {cluster.cluster_name!r}\n")
        return 0

    @classmethod
    def _stop(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        cluster = client.compute.clusters.find_cluster(
            cluster_id=getattr(args, "cluster_id", None),
            cluster_name=getattr(args, "name", None),
            raise_error=True,
        )
        cluster.stop(wait=False)
        sys.stderr.write(f"Stopping cluster {cluster.cluster_name!r}\n")
        return 0
