"""MotherDuck connectivity helpers via DuckDB."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from wt.config import AppSettings, load_settings


@dataclass(frozen=True, slots=True)
class MotherDuckStatus:
    connected: bool
    message: str
    database: str | None = None


def get_motherduck_dsn(settings: AppSettings | None = None, *, read_only: bool = False) -> str:
    cfg = settings or load_settings()
    token = cfg.motherduck_readonly_token if read_only and cfg.motherduck_readonly_token else cfg.motherduck_token
    if not token:
        raise ValueError("MotherDuck token is not configured")
    return f"md:{cfg.motherduck_database}?motherduck_token={token}"


def check_motherduck_connection(settings: AppSettings | None = None) -> MotherDuckStatus:
    cfg = settings or load_settings()
    try:
        dsn = get_motherduck_dsn(cfg)
    except ValueError as exc:
        return MotherDuckStatus(connected=False, message=str(exc))

    conn = duckdb.connect(dsn)
    try:
        conn.execute("select current_database()").fetchone()
    finally:
        conn.close()
    return MotherDuckStatus(connected=True, message="ok", database=cfg.motherduck_database)
