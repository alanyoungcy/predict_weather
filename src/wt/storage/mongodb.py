"""MongoDB Atlas connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pymongo import MongoClient

from wt.config import AppSettings, load_settings


@dataclass(frozen=True, slots=True)
class MongoStatus:
    connected: bool
    message: str
    version: str | None = None


def check_mongodb_connection(settings: AppSettings | None = None) -> MongoStatus:
    cfg = settings or load_settings()
    if not cfg.mongodb_uri:
        return MongoStatus(connected=False, message="MongoDB URI is not configured")

    client = MongoClient(cfg.mongodb_uri, serverSelectionTimeoutMS=10_000)
    try:
        result = client.admin.command("ping")
        build_info = client.admin.command("buildInfo")
    finally:
        client.close()
    if result.get("ok") != 1:
        return MongoStatus(connected=False, message="MongoDB ping failed")
    return MongoStatus(connected=True, message="ok", version=str(build_info.get("version")))
