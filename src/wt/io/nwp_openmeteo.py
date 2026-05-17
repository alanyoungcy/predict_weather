"""Open-Meteo forecast fallback ingestion."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_openmeteo_hourly(
    latitude: float,
    longitude: float,
    *,
    forecast_days: int = 3,
    models: str = "gfs_hrrr,best_match",
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch hourly temperature and precipitation in UTC.

    Returns columns compatible with the Herbie point-forecast aggregation path:
    ``valid_time_utc``, ``var_name``, ``value``, ``model``.
    """

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation",
        "models": models,
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "UTC",
        "forecast_days": forecast_days,
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])
    rows: list[dict[str, object]] = []
    for idx, stamp in enumerate(times):
        valid_time = pd.Timestamp(stamp)
        if valid_time.tzinfo is None:
            valid_time = valid_time.tz_localize(UTC)
        else:
            valid_time = valid_time.tz_convert(UTC)
        if idx < len(temps) and temps[idx] is not None:
            rows.append(
                {
                    "valid_time_utc": valid_time,
                    "var_name": "t2m_f",
                    "value": float(temps[idx]),
                    "model": "openmeteo",
                    "init_run_utc": datetime.now(tz=UTC),
                    "forecast_hour": None,
                }
            )
        if idx < len(precip) and precip[idx] is not None:
            rows.append(
                {
                    "valid_time_utc": valid_time,
                    "var_name": "apcp_in",
                    "value": float(precip[idx]),
                    "model": "openmeteo",
                    "init_run_utc": datetime.now(tz=UTC),
                    "forecast_hour": None,
                }
            )
    return pd.DataFrame(rows)
