"""ASOS observation fallback ingestion via the Iowa State Mesonet service."""

from __future__ import annotations

from datetime import date
from io import StringIO

import httpx
import pandas as pd


IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"


def fetch_asos_observations(
    station_icao: str,
    start: date,
    end: date,
    *,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetch raw ASOS observations for a station/date range.

    This is a defensive observation source for diagnostics and future intraday
    features. It returns IEM CSV columns plus a normalized ``valid_time_utc``.
    """

    params = {
        "station": station_icao.removeprefix("K") if station_icao.startswith("K") else station_icao,
        "data": ["tmpf", "dwpf", "p01i"],
        "year1": start.year,
        "month1": start.month,
        "day1": start.day,
        "year2": end.year,
        "month2": end.month,
        "day2": end.day,
        "tz": "Etc/UTC",
        "format": "comma",
        "latlon": "no",
        "direct": "yes",
        "report_type": ["1", "2"],
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.get(IEM_ASOS_URL, params=params)
        response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text), comment="#")
    if "valid" in frame.columns:
        frame["valid_time_utc"] = pd.to_datetime(frame["valid"], utc=True)
    return frame
