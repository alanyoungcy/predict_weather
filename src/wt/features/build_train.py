"""Historical training feature assembly."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
import sys
from typing import Callable

import pandas as pd

from wt.config import Station, load_settings, load_yaml
from wt.features.transforms import add_climatology, add_doy_encoding, add_lag_features, add_model_spread, add_persistence_features
from wt.io.nwp_herbie import get_model_point_forecast
from wt.utils.paths import CONFIG_DIR
from wt.utils.time import kalshi_day_window_utc

ForecastFetcher = Callable[[str, datetime, list[int], float, float, list[str]], pd.DataFrame]


def _default_init_run(target_local_date: date, init_hour_utc: int) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(target_local_date - timedelta(days=1), time(init_hour_utc, tzinfo=UTC)))


def _forecast_hours_for_window(init_run_utc: pd.Timestamp, window_start_utc: datetime, window_end_utc: datetime) -> list[int]:
    start_hour = int((window_start_utc - init_run_utc.to_pydatetime()).total_seconds() // 3600)
    end_hour = int((window_end_utc - init_run_utc.to_pydatetime()).total_seconds() // 3600)
    return list(range(start_hour, end_hour))


def _aggregate_daily_point_forecast(frame: pd.DataFrame, model: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if frame.empty:
        return result
    by_var = {name: group["value"].astype("float64") for name, group in frame.groupby("var_name")}
    if "t2m_f" in by_var:
        result[f"{model}_t2m_max_f"] = float(by_var["t2m_f"].max())
        result[f"{model}_t2m_min_f"] = float(by_var["t2m_f"].min())
    if "apcp_in" in by_var:
        result[f"{model}_apcp_24h_in"] = float(by_var["apcp_in"].sum())
        result[f"{model}_qpf_24h_in"] = float(by_var["apcp_in"].sum())
    return result


def _enabled_models() -> list[str]:
    models_cfg = load_yaml(CONFIG_DIR / "models.yaml")
    return [name for name, cfg in models_cfg["models"].items() if cfg.get("enabled")]


def _model_historical_starts() -> dict[str, date]:
    models_cfg = load_yaml(CONFIG_DIR / "models.yaml")
    starts: dict[str, date] = {}
    for name, cfg in models_cfg["models"].items():
        if cfg.get("historical_start"):
            starts[name] = pd.Timestamp(cfg["historical_start"]).date()
    return starts


def _model_available(model: str, init_run_utc: pd.Timestamp, starts: dict[str, date]) -> bool:
    start = starts.get(model)
    return start is None or init_run_utc.date() >= start


def build_training_features(
    station: Station,
    start_date: date,
    end_date: date,
    init_hour_utc: int = 18,
    *,
    label_df: pd.DataFrame,
    climo_df: pd.DataFrame | None = None,
    forecast_fetcher: ForecastFetcher | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    """Build one historical row per target date using only pre-init-run data."""

    settings = load_settings()
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
    model_list = models or _enabled_models()
    model_starts = _model_historical_starts()

    labels = label_df.copy()
    labels["local_date"] = pd.to_datetime(labels["local_date"]).dt.date

    rows: list[dict[str, object]] = []
    target_dates = list(pd.date_range(start_date, end_date, freq="D").date)
    for idx, target_local_date in enumerate(target_dates, start=1):
        init_run_utc = _default_init_run(target_local_date, init_hour_utc)
        window_start_utc, window_end_utc = kalshi_day_window_utc(target_local_date, station.zoneinfo)
        forecast_hours = _forecast_hours_for_window(init_run_utc, window_start_utc, window_end_utc)
        row: dict[str, object] = {
            "station": station.icao,
            "target_local_date": pd.Timestamp(target_local_date),
            "init_run_utc": init_run_utc,
            "lead_hours": forecast_hours[0] if forecast_hours else None,
        }
        for model in model_list:
            if not _model_available(model, init_run_utc, model_starts):
                continue
            try:
                forecast = fetcher(
                    model,
                    init_run_utc.to_pydatetime(),
                    forecast_hours,
                    station.lat,
                    station.lon,
                    ["TMP:2 m", "APCP:surface"],
                )
            except Exception as exc:
                print(
                    f"[{station.icao}] warning: {model} failed for {target_local_date}: {exc}",
                    file=sys.stderr,
                )
                continue
            row.update(_aggregate_daily_point_forecast(forecast, model))
        rows.append(row)
        if idx == 1 or idx % 25 == 0 or idx == len(target_dates):
            print(
                f"[{station.icao}] feature backfill progress {idx}/{len(target_dates)} "
                f"through {target_local_date}",
                file=sys.stderr,
            )

    frame = pd.DataFrame(rows)
    frame = add_lag_features(frame, labels)
    frame = add_persistence_features(frame, labels)
    if climo_df is not None and not climo_df.empty:
        frame = add_climatology(frame, climo_df)
    frame = add_doy_encoding(frame)
    frame = add_model_spread(frame)
    return frame
