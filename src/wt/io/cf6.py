"""NWS CF6 climate report ingestion and parsing."""

from __future__ import annotations

import calendar
import math
import re
from datetime import UTC, date, datetime
from functools import lru_cache
from typing import Any

import pandas as pd
import requests

TRACE_PRECIP_IN = 0.005
_IEM_CF6_URL = "https://mesonet.agron.iastate.edu/json/cf6.py"
_CF6_HEADER_RE = re.compile(r"MONTH:\s+(?P<month>[A-Z]+)\s+YEAR:\s+(?P<year>\d{4})")
_CF6_PRODUCT_RE = re.compile(r"^[A-Z]{4}\d{2}\s+[A-Z]{4}\s+(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})$")
_CF6_DAILY_ROW_RE = re.compile(
    r"^\s*(?P<day>\d{1,2})\s+"
    r"(?P<tmax>\S+)\s+(?P<tmin>\S+)\s+\S+\s+\S+\s+\S+\s+\S+\s+"
    r"(?P<precip>\S+)\s+(?P<snow>\S+)"
)
_PRODUCT_TS_RE = re.compile(r"(?P<ts>\d{12})-[A-Z]{4}-[A-Z0-9]{6,}-CF6[A-Z0-9]{3,}")


def _month_name_to_number(month_name: str) -> int:
    try:
        return list(calendar.month_name).index(month_name.title())
    except ValueError as exc:
        raise ValueError(f"Unsupported CF6 month name: {month_name!r}") from exc


def _coerce_cf6_number(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text == "M":
        return math.nan
    if text == "T":
        return TRACE_PRECIP_IN
    return float(text)


def _parse_product_timestamp(product: str | None) -> pd.Timestamp:
    if not product:
        return pd.NaT
    match = _PRODUCT_TS_RE.search(product)
    if not match:
        return pd.NaT
    return pd.Timestamp(datetime.strptime(match.group("ts"), "%Y%m%d%H%M").replace(tzinfo=UTC))


def parse_cf6_text(text: str) -> pd.DataFrame:
    """Parse a raw CF6 fixed-width text product into daily rows."""

    month_match = _CF6_HEADER_RE.search(text)
    if month_match is None:
        raise ValueError("CF6 text is missing MONTH/YEAR header")

    month = _month_name_to_number(month_match.group("month"))
    year = int(month_match.group("year"))

    issued_at = pd.NaT
    for line in text.splitlines():
        product_match = _CF6_PRODUCT_RE.match(line.strip())
        if not product_match:
            continue
        issued_at = pd.Timestamp(
            datetime(
                year=year,
                month=month,
                day=int(product_match.group("day")),
                hour=int(product_match.group("hour")),
                minute=int(product_match.group("minute")),
                tzinfo=UTC,
            )
        )
        break

    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _CF6_DAILY_ROW_RE.match(line)
        if match is None:
            continue

        day = int(match.group("day"))
        rows.append(
            {
                "local_date": date(year, month, day),
                "tmax_f": _coerce_cf6_number(match.group("tmax")),
                "tmin_f": _coerce_cf6_number(match.group("tmin")),
                "precip_in": _coerce_cf6_number(match.group("precip")),
                "snow_in": _coerce_cf6_number(match.group("snow")),
                "source": "CF6",
                "settled_at": issued_at,
            }
        )

    if not rows:
        raise ValueError("No daily CF6 rows parsed from text product")

    return pd.DataFrame(rows)


def _normalize_iem_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in results:
        rows.append(
            {
                "local_date": pd.to_datetime(item["valid"], utc=False).date(),
                "tmax_f": _coerce_cf6_number(item.get("high")),
                "tmin_f": _coerce_cf6_number(item.get("low")),
                "precip_in": _coerce_cf6_number(item.get("precip")),
                "snow_in": _coerce_cf6_number(item.get("snow")),
                "source": "CF6",
                "settled_at": _parse_product_timestamp(item.get("product")),
            }
        )

    return pd.DataFrame(rows)


@lru_cache(maxsize=256)
def _fetch_cf6_year_cached(station_icao: str, year: int) -> pd.DataFrame:
    response = requests.get(
        _IEM_CF6_URL,
        params={"station": station_icao, "year": year},
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError("Unexpected CF6 response shape: missing results list")

    frame = _normalize_iem_results(results)
    if frame.empty:
        return frame.reindex(
            columns=["local_date", "tmax_f", "tmin_f", "precip_in", "snow_in", "source", "settled_at"]
        )

    frame.sort_values("local_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def fetch_cf6_year(station_icao: str, year: int) -> pd.DataFrame:
    """Fetch one full CF6 year for a station."""

    return _fetch_cf6_year_cached(station_icao, year).copy()


def fetch_cf6(wfo: str, station_icao: str, year: int, month: int) -> pd.DataFrame:
    """Fetch one month of CF6 daily rows.

    The IEM JSON endpoint is used for reliable bulk historical retrieval and is
    still sourced from the underlying CF6 product stream.
    """

    frame = fetch_cf6_year(station_icao, year)
    frame = frame[pd.to_datetime(frame["local_date"]).dt.month == month].copy()
    frame.sort_values("local_date", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame
