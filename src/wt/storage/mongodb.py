"""MongoDB Atlas connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo import MongoClient
from pymongo.errors import ConfigurationError, OperationFailure, PyMongoError, ServerSelectionTimeoutError

from wt.config import AppSettings, load_settings


@dataclass(frozen=True, slots=True)
class MongoStatus:
    connected: bool
    message: str
    version: str | None = None


def _classify_mongo_error(exc: Exception) -> str:
    if isinstance(exc, ServerSelectionTimeoutError):
        message = str(exc)
        if "ReplicaSetNoPrimary" in message or "No replica set members found yet" in message:
            return "MongoDB cluster reachable by URI, but no primary was selectable. This usually indicates network access restrictions, a paused/provisioning cluster, or replica set reachability issues."
        return "MongoDB server selection timed out. This usually indicates network reachability or Atlas access list issues."

    if isinstance(exc, ConfigurationError):
        return f"MongoDB configuration error: {exc}"

    if isinstance(exc, OperationFailure):
        return f"MongoDB authentication/authorization failed: {exc}"

    if isinstance(exc, PyMongoError):
        return f"MongoDB client error: {exc}"

    return str(exc)


def check_mongodb_connection(settings: AppSettings | None = None) -> MongoStatus:
    cfg = settings or load_settings()
    if not cfg.mongodb_uri:
        return MongoStatus(connected=False, message="MongoDB URI is not configured")

    client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=10_000)
    try:
        result = client.admin.command("ping")
        build_info = client.admin.command("buildInfo")
        if result.get("ok") != 1:
            return MongoStatus(connected=False, message="MongoDB ping failed")
        return MongoStatus(connected=True, message="ok", version=str(build_info.get("version")))
    except Exception as exc:  # pragma: no cover - live connectivity branch
        return MongoStatus(connected=False, message=_classify_mongo_error(exc))
    finally:
        client.close()
