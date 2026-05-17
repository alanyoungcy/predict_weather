"""Live feature assembly for inference runs."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Callable

import pandas as pd

from wt.config import Station, load_settings
from wt.features.build_train import _aggregate_daily_point_forecast, _forecast_hours_for_window
from wt.features.transforms import add_doy_encoding, add_model_spread
from wt.io.nwp_herbie import get_model_point_forecast
from wt.io.nwp_openmeteo import fetch_openmeteo_hourly
from wt.io.nws_api import fetch_nws_grid_hourly
from wt.utils.time import kalshi_day_window_utc


LiveFetcher = Callable[[str, datetime, list[int], float, float, list[str]], pd.DataFrame]


RUN_SCHEDULES: dict[str, dict[str, int]] = {
    "morning": {"hrrr": 0, "gfs": 0, "nbm": 1, "aifs": 0},
    "evening": {"hrrr": 12, "gfs": 12, "nbm": 13},
}


def _latest_init_for_target(target_local_date: date, init_hour_utc: int) -> pd.Timestamp:
    # Forecasts for tomorrow's settlement day generally use today's UTC cycle.
    init_date = target_local_date - timedelta(days=1)
    return pd.Timestamp(datetime.combine(init_date, time(init_hour_utc, tzinfo=UTC)))


def _fallback_window_rows(
    station: Station,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, float]:
    rows: dict[str, float] = {}
    for model_name, fetcher in (
        ("openmeteo", fetch_openmeteo_hourly),
        ("nws", fetch_nws_grid_hourly),
    ):
        try:
            frame = fetcher(station.lat, station.lon)
        except Exception:
            continue
        if frame.empty:
            continue
        frame = frame[
            (pd.to_datetime(frame["valid_time_utc"], utc=True) >= pd.Timestamp(window_start))
            & (pd.to_datetime(frame["valid_time_utc"], utc=True) < pd.Timestamp(window_end))
        ]
        rows.update(_aggregate_daily_point_forecast(frame, model_name))
    return rows


def build_live_features(
    stations: list[Station],
    target_local_date: date,
    *,
    run_slot: str = "evening",
    forecast_fetcher: LiveFetcher | None = None,
    use_fallbacks: bool = True,
) -> pd.DataFrame:
    """Build one live feature row per station for the requested target date."""

    settings = load_settings()
    schedule = RUN_SCHEDULES.get(run_slot)
    if schedule is None:
        raise ValueError(
            f"Unsupported run_slot {run_slot!r}; expected one of {sorted(RUN_SCHEDULES)}"
        )

    fetcher = forecast_fetcher or (
        lambda model, init_run, hours, lat, lon, variables: get_model_point_forecast(
            model=model,
            init_run_utc=init_run,
            forecast_hours=hours,
            station_lat=lat,
            station_lon=lon,
            variables=variables,
            settings=settings,
        )
    )

    rows: list[dict[str, object]] = []
    for station in stations:
        window_start, window_end = kalshi_day_window_utc(target_local_date, station.zoneinfo)
        primary_init = _latest_init_for_target(target_local_date, min(schedule.values()))
        row: dict[str, object] = {
            "station": station.icao,
            "target_local_date": pd.Timestamp(target_local_date),
            "init_run_utc": primary_init,
            "lead_hours": int(
                (window_start - primary_init.to_pydatetime()).total_seconds() // 3600
            ),
        }
        for model_name, init_hour in schedule.items():
            init_run = _latest_init_for_target(target_local_date, init_hour)
            hours = _forecast_hours_for_window(init_run, window_start, window_end)
            try:
                forecast = fetcher(
                    model_name,
                    init_run.to_pydatetime(),
                    hours,
                    station.lat,
                    station.lon,
                    ["TMP:2 m", "APCP:surface"],
                )
            except Exception:
                forecast = pd.DataFrame()
            row.update(_aggregate_daily_point_forecast(forecast, model_name))

        if use_fallbacks and not any(key.endswith("_t2m_max_f") for key in row):
            row.update(_fallback_window_rows(station, window_start, window_end))
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame = add_doy_encoding(frame)
    frame = add_model_spread(frame)
    return frame
