"""NWS api.weather.gov forecast-grid fallback ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd

from wt.config import load_settings


NWS_BASE_URL = "https://api.weather.gov"


def _parse_valid_time(value: str) -> pd.Timestamp:
    start = value.split("/", 1)[0]
    ts = pd.Timestamp(start)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    return ts.tz_convert(UTC)


def _fahrenheit(value_c: float | int | None) -> float | None:
    if value_c is None:
        return None
    return float(value_c) * 9.0 / 5.0 + 32.0


def _inch(value_mm: float | int | None) -> float | None:
    if value_mm is None:
        return None
    return float(value_mm) * 0.0393701


def fetch_nws_grid_hourly(
    latitude: float,
    longitude: float,
    *,
    timeout: float = 30.0,
    user_agent: str | None = None,
) -> pd.DataFrame:
    """Fetch NWS grid forecast values near a station.

    NWS grid values are period-based. For this v1 feature fallback, each period
    is represented by its start time so daily aggregation can use it.
    """

    settings = load_settings()
    headers = {
        "User-Agent": user_agent or settings.nws_user_agent,
        "Accept": "application/geo+json",
    }
    with httpx.Client(timeout=timeout, headers=headers) as client:
        point = client.get(f"{NWS_BASE_URL}/points/{latitude:.4f},{longitude:.4f}")
        point.raise_for_status()
        grid_url = point.json()["properties"]["forecastGridData"]
        grid = client.get(grid_url)
        grid.raise_for_status()
        payload: dict[str, Any] = grid.json()

    props = payload.get("properties", {})
    temperature = props.get("temperature", {}).get("values", [])
    qpf = props.get("quantitativePrecipitation", {}).get("values", [])
    pop = props.get("probabilityOfPrecipitation", {}).get("values", [])

    rows: list[dict[str, object]] = []
    generated = datetime.now(tz=UTC)
    for item in temperature:
        value = _fahrenheit(item.get("value"))
        if value is not None:
            rows.append(
                {
                    "valid_time_utc": _parse_valid_time(item["validTime"]),
                    "var_name": "t2m_f",
                    "value": value,
                }
            )
    for item in qpf:
        value = _inch(item.get("value"))
        if value is not None:
            rows.append(
                {
                    "valid_time_utc": _parse_valid_time(item["validTime"]),
                    "var_name": "apcp_in",
                    "value": value,
                }
            )
    for item in pop:
        value = item.get("value")
        if value is not None:
            rows.append(
                {
                    "valid_time_utc": _parse_valid_time(item["validTime"]),
                    "var_name": "pop_pct",
                    "value": float(value),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "valid_time_utc",
                "var_name",
                "value",
                "model",
                "init_run_utc",
                "forecast_hour",
            ]
        )
    frame["model"] = "nws"
    frame["init_run_utc"] = generated
    frame["forecast_hour"] = (
        (
            pd.to_datetime(frame["valid_time_utc"], utc=True) - pd.Timestamp(generated)
        ).dt.total_seconds()
        // 3600
    ).astype("int64")
    return frame
