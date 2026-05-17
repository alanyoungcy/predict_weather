"""Historical feature backfill orchestration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from wt.config import Station, load_settings, load_stations
from wt.features.build_train import build_training_features
from wt.features.transforms import build_climatology


def select_stations(station_filter: str = "all") -> list[Station]:
    stations = load_stations()
    if station_filter == "all":
        return stations
    wanted = {part.strip().upper() for part in station_filter.split(",") if part.strip()}
    return [
        station
        for station in stations
        if station.icao in wanted or station.kalshi_city.upper() in wanted
    ]


def _station_labels(labels: pd.DataFrame, station: Station) -> pd.DataFrame:
    frame = labels[labels["station"] == station.icao].copy()
    if frame.empty:
        raise ValueError(f"No labels available for {station.icao}")
    frame["local_date"] = pd.to_datetime(frame["local_date"]).dt.date
    return frame


def backfill_training_features(
    *,
    start_date: date,
    end_date: date,
    station_filter: str = "all",
    labels_path: Path | None = None,
    output_path: Path | None = None,
    models: Iterable[str] | None = None,
    init_hour_utc: int = 18,
) -> pd.DataFrame:
    """Build and persist historical training features for selected stations."""

    settings = load_settings()
    label_source = labels_path or settings.data_dir / "labels" / "labels.parquet"
    labels = pd.read_parquet(label_source)
    labels["local_date"] = pd.to_datetime(labels["local_date"]).dt.date
    climo = build_climatology(labels)
    model_list = list(models) if models is not None else None

    parts: list[pd.DataFrame] = []
    for station in select_stations(station_filter):
        station_label_df = _station_labels(labels, station)
        features = build_training_features(
            station=station,
            start_date=start_date,
            end_date=end_date,
            init_hour_utc=init_hour_utc,
            label_df=station_label_df,
            climo_df=climo,
            models=model_list,
        )
        parts.append(features)

    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    destination = output_path or settings.data_dir / "features" / "features_train.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    return result
