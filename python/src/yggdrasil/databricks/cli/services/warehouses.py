"""``ygg-databricks warehouses`` — manage SQL warehouses."""
from __future__ import annotations

import sys
from typing import Any


class WarehousesCommand:

    @classmethod
    def register(cls, subparsers: Any) -> None:
        parser = subparsers.add_parser(
            "warehouses", help="Manage SQL warehouses.",
        )
        sub = parser.add_subparsers(dest="warehouses_action")

        ls = sub.add_parser("list", help="List SQL warehouses.")
        ls.set_defaults(handler=cls._list)

        get = sub.add_parser("get", help="Get a warehouse by id or name.")
        get.add_argument("--id", dest="warehouse_id", default=None)
        get.add_argument("--name", default=None)
        get.set_defaults(handler=cls._get)

        create = sub.add_parser("create", help="Create a SQL warehouse.")
        create.add_argument("--name", required=True, help="Warehouse name.")
        create.add_argument("--cluster-size", default=None,
                            help="Size (2X-Small, X-Small, Small, Medium, Large, …).")
        create.add_argument("--type", dest="warehouse_type", default=None,
                            help="PRO or CLASSIC.")
        create.add_argument("--serverless", action="store_true", default=None,
                            help="Enable serverless compute.")
        create.add_argument("--auto-stop-mins", type=int, default=None)
        create.set_defaults(handler=cls._create)

        delete = sub.add_parser("delete", help="Delete a warehouse.")
        delete.add_argument("--id", dest="warehouse_id", required=True)
        delete.set_defaults(handler=cls._delete)

        start = sub.add_parser("start", help="Start a stopped warehouse.")
        start.add_argument("--id", dest="warehouse_id", default=None)
        start.add_argument("--name", default=None)
        start.set_defaults(handler=cls._start)

        stop = sub.add_parser("stop", help="Stop a running warehouse.")
        stop.add_argument("--id", dest="warehouse_id", default=None)
        stop.add_argument("--name", default=None)
        stop.set_defaults(handler=cls._stop)

        parser.set_defaults(handler=lambda args, bc: parser.print_help() or 1)

    @classmethod
    def _list(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        for wh in client.warehouses.list_warehouses():
            sys.stdout.write(
                f"{wh.warehouse_id}\t{wh.warehouse_name}\n"
            )
        return 0

    @classmethod
    def _get(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        wh = client.warehouses.find_warehouse(
            warehouse_id=getattr(args, "warehouse_id", None),
            warehouse_name=getattr(args, "name", None),
            create=False,
        )
        if wh is None:
            sys.stderr.write("Warehouse not found.\n")
            return 1
        sys.stdout.write(
            f"Warehouse: {wh.warehouse_name}\n"
            f"ID:        {wh.warehouse_id}\n"
        )
        return 0

    @classmethod
    def _create(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        kwargs: dict[str, Any] = {}
        if args.cluster_size:
            kwargs["cluster_size"] = args.cluster_size
        if args.warehouse_type:
            kwargs["warehouse_type"] = args.warehouse_type
        if args.serverless is not None:
            kwargs["enable_serverless_compute"] = args.serverless
        if args.auto_stop_mins is not None:
            kwargs["auto_stop_mins"] = args.auto_stop_mins

        wh = client.warehouses.create(name=args.name, wait=False, **kwargs)
        sys.stdout.write(f"{wh.warehouse_id}\t{wh.warehouse_name}\n")
        return 0

    @classmethod
    def _delete(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        client.warehouses.delete(args.warehouse_id)
        sys.stderr.write("Deleted.\n")
        return 0

    @classmethod
    def _start(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        wh = client.warehouses.find_warehouse(
            warehouse_id=getattr(args, "warehouse_id", None),
            warehouse_name=getattr(args, "name", None),
            create=False,
        )
        if wh is None:
            sys.stderr.write("Warehouse not found.\n")
            return 1
        wh.start(wait=False)
        sys.stderr.write(f"Starting warehouse {wh.warehouse_name!r}\n")
        return 0

    @classmethod
    def _stop(cls, args: Any, build_client: Any) -> int:
        client = build_client(args)
        wh = client.warehouses.find_warehouse(
            warehouse_id=getattr(args, "warehouse_id", None),
            warehouse_name=getattr(args, "name", None),
            create=False,
        )
        if wh is None:
            sys.stderr.write("Warehouse not found.\n")
            return 1
        wh.stop(wait=False)
        sys.stderr.write(f"Stopping warehouse {wh.warehouse_name!r}\n")
        return 0
