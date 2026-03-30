from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from pymongo import MongoClient

from .serde import BsonJsonSerde

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
HTTPMONGO_TOKEN = os.getenv("HTTPMONGO_TOKEN")

app = FastAPI(title="yggdrasil-httpmongo")
_client = MongoClient(MONGO_URI)
_serde = BsonJsonSerde()


class RpcRequest(BaseModel):
    database: str
    collection: Optional[str] = None
    operation: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


@app.get("/health")
def health() -> Dict[str, int]:
    return {"ok": 1}


@app.post("/rpc")
def rpc(req: RpcRequest, x_auth_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if HTTPMONGO_TOKEN and x_auth_token != HTTPMONGO_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    db = _client[req.database]
    coll = db[req.collection] if req.collection else None
    args = _serde.normalize(req.arguments)

    try:
        if req.operation == "ping":
            return {"ok": 1, "result": _serde.normalize(_client.admin.command("ping"))}
        if req.operation == "server_info":
            return {"ok": 1, "result": _serde.normalize(_client.server_info())}
        if req.operation == "command":
            return {"ok": 1, "result": _serde.normalize(db.command(args["command"]))}
        if req.operation == "list_collection_names":
            return {"ok": 1, "result": _serde.normalize(db.list_collection_names())}

        if coll is None:
            raise ValueError("collection is required for this operation")

        if req.operation == "aggregate":
            result = list(coll.aggregate(args.get("pipeline", []), **args.get("kwargs", {})))
            return {"ok": 1, "result": _serde.normalize(result)}
        if req.operation == "find":
            cursor = coll.find(args.get("filter", {}), args.get("projection"), **args.get("kwargs", {}))
            if args.get("sort"):
                cursor = cursor.sort(args["sort"])
            if args.get("skip"):
                cursor = cursor.skip(args["skip"])
            if args.get("limit"):
                cursor = cursor.limit(args["limit"])
            return {"ok": 1, "result": _serde.normalize(list(cursor))}
        if req.operation == "find_one":
            result = coll.find_one(args.get("filter", {}), args.get("projection"), **args.get("kwargs", {}))
            return {"ok": 1, "result": _serde.normalize(result)}
        if req.operation == "count_documents":
            return {"ok": 1, "result": coll.count_documents(args.get("filter", {}), **args.get("kwargs", {}))}
        if req.operation == "insert_one":
            res = coll.insert_one(args["document"])
            return {"ok": 1, "result": {"inserted_id": _serde.normalize(res.inserted_id), "acknowledged": res.acknowledged}}
        if req.operation == "insert_many":
            res = coll.insert_many(args["documents"], ordered=args.get("ordered", True))
            return {"ok": 1, "result": {"inserted_ids": _serde.normalize(res.inserted_ids), "acknowledged": res.acknowledged}}
        if req.operation == "update_one":
            res = coll.update_one(args["filter"], args["update"], upsert=args.get("upsert", False))
            return {"ok": 1, "result": {"matched_count": res.matched_count, "modified_count": res.modified_count, "upserted_id": _serde.normalize(res.upserted_id), "acknowledged": res.acknowledged}}
        if req.operation == "update_many":
            res = coll.update_many(args["filter"], args["update"], upsert=args.get("upsert", False))
            return {"ok": 1, "result": {"matched_count": res.matched_count, "modified_count": res.modified_count, "upserted_id": _serde.normalize(res.upserted_id), "acknowledged": res.acknowledged}}
        if req.operation == "delete_one":
            res = coll.delete_one(args["filter"])
            return {"ok": 1, "result": {"deleted_count": res.deleted_count, "acknowledged": res.acknowledged}}
        if req.operation == "delete_many":
            res = coll.delete_many(args["filter"])
            return {"ok": 1, "result": {"deleted_count": res.deleted_count, "acknowledged": res.acknowledged}}
        if req.operation == "create_index":
            return {"ok": 1, "result": coll.create_index(args["keys"], **args.get("kwargs", {}))}
        if req.operation == "index_information":
            return {"ok": 1, "result": _serde.normalize(coll.index_information())}
        if req.operation == "drop_index":
            coll.drop_index(args["name"])
            return {"ok": 1, "result": True}
        if req.operation == "drop":
            coll.drop()
            return {"ok": 1, "result": True}

        raise ValueError(f"unsupported operation: {req.operation}")
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": 0,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
        }
