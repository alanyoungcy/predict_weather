"""Supabase/Postgres connectivity helpers."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from wt.config import AppSettings, load_settings


@dataclass(frozen=True, slots=True)
class SupabaseStatus:
    connected: bool
    message: str
    server_version: str | None = None


def get_supabase_dsn(settings: AppSettings | None = None, *, non_pooling: bool = False) -> str:
    cfg = settings or load_settings()
    if non_pooling and cfg.supabase_db_url_non_pooling:
        return cfg.supabase_db_url_non_pooling
    if cfg.supabase_db_url:
        return cfg.supabase_db_url
    raise ValueError("Supabase/Postgres DSN is not configured")


def check_supabase_connection(settings: AppSettings | None = None) -> SupabaseStatus:
    cfg = settings or load_settings()
    try:
        dsn = get_supabase_dsn(cfg)
    except ValueError as exc:
        return SupabaseStatus(connected=False, message=str(exc))

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("select version()")
            version = str(cur.fetchone()[0])
    return SupabaseStatus(connected=True, message="ok", server_version=version)
