"""GHCN-Daily fallback ingestion."""

from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import requests

_GHCND_URL = "https://www.ncei.noaa.gov/data/daily-summaries/access/{station_id}.csv"
_TENTHS_C_TO_F_SCALE = 9.0 / 50.0
_TENTHS_C_TO_F_OFFSET = 32.0
_TENTHS_MM_TO_IN = 0.00393701


def _tenths_c_to_f(series: pd.Series) -> pd.Series:
    return (series.astype("float64") * _TENTHS_C_TO_F_SCALE) + _TENTHS_C_TO_F_OFFSET


def _tenths_mm_to_in(series: pd.Series) -> pd.Series:
    return series.astype("float64") * _TENTHS_MM_TO_IN


def fetch_ghcnd_station(station_id: str, start: date, end: date) -> pd.DataFrame:
    """Download daily GHCN summaries for one station.

    NOAA daily-summaries access CSV exposes temperatures in tenths of degrees C
    and precipitation in tenths of millimeters for these station exports.
    """

    response = requests.get(_GHCND_URL.format(station_id=station_id), timeout=60)
    response.raise_for_status()

    frame = pd.read_csv(StringIO(response.text), usecols=["DATE", "TMAX", "TMIN", "PRCP"])
    frame["DATE"] = pd.to_datetime(frame["DATE"], utc=False).dt.date
    frame = frame[(frame["DATE"] >= start) & (frame["DATE"] <= end)].copy()
    frame.rename(columns={"DATE": "date"}, inplace=True)

    frame["tmax_f"] = _tenths_c_to_f(frame["TMAX"])
    frame["tmin_f"] = _tenths_c_to_f(frame["TMIN"])
    frame["prcp_in"] = _tenths_mm_to_in(frame["PRCP"])

    return frame[["date", "tmax_f", "tmin_f", "prcp_in"]].reset_index(drop=True)
