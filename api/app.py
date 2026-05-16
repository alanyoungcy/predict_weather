from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qs, urlparse

from pymongo import MongoClient
from pymongo.errors import PyMongoError


def _safe_host(uri: str | None) -> str | None:
    if not uri:
        return None
    try:
        parsed = urlparse(uri)
        return parsed.hostname
    except Exception:
        return None


def _evaluate_request(headers: dict[str, str], query_params: dict[str, list[str]]) -> tuple[dict[str, Any], int]:
    token = os.getenv("MONGO_TEST_TOKEN")
    if token:
        supplied = headers.get("x-mongo-test-token")
        if not supplied:
            supplied_values = query_params.get("token", [])
            supplied = supplied_values[0] if supplied_values else None
        if supplied != token:
            return {"ok": False, "message": "unauthorized"}, 401

    uri = os.getenv("MONGODB_URI")
    payload: dict[str, Any] = {
        "ok": False,
        "checked_at": datetime.now(tz=UTC).isoformat(),
        "uri_present": bool(uri),
        "uri_host": _safe_host(uri),
        "topology_type": None,
        "servers": [],
        "message": None,
    }

    if not uri:
        payload["message"] = "MONGODB_URI is not configured"
        return payload, 500

    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    try:
        client.admin.command("ping")
        build_info = client.admin.command("buildInfo")
        topo = client.topology_description
        payload["ok"] = True
        payload["message"] = "ok"
        payload["version"] = str(build_info.get("version"))
        payload["topology_type"] = topo.topology_type_name
        payload["servers"] = [
            {"address": f"{addr[0]}:{addr[1]}", "type": desc.server_type_name}
            for addr, desc in topo.server_descriptions().items()
        ]
        return payload, 200
    except PyMongoError as exc:
        topo = client.topology_description
        payload["message"] = str(exc)
        payload["topology_type"] = topo.topology_type_name
        payload["servers"] = [
            {
                "address": f"{addr[0]}:{addr[1]}",
                "type": desc.server_type_name,
                "has_error": desc.error is not None,
            }
            for addr, desc in topo.server_descriptions().items()
        ]
        return payload, 503
    finally:
        client.close()


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        headers = {key.lower(): value for key, value in self.headers.items()}
        payload, status = _evaluate_request(headers, query_params)

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
