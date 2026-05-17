"""Data access helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from wt.config import load_stations
from wt.utils.paths import DATA_DIR, MODELS_DIR


@dataclass(frozen=True, slots=True)
class DashboardArtifacts:
    labels: pd.DataFrame
    features_live: pd.DataFrame
    predictions: pd.DataFrame
    bucketed_predictions: pd.DataFrame
    markets: pd.DataFrame
    signals: pd.DataFrame
    runs: pd.DataFrame
    model_inventory: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def read_parquet_dataset(path: Path) -> pd.DataFrame:
    """Read a parquet file or partitioned parquet directory if present."""

    if not path.exists():
        return _empty()
    try:
        return pd.read_parquet(path)
    except Exception:
        return _empty()


def read_parquet_collection(path: Path) -> pd.DataFrame:
    """Read all parquet files under a directory into one dataframe."""

    if not path.exists():
        return _empty()
    if path.is_file():
        return read_parquet_dataset(path)
    files = sorted(path.rglob("*.parquet"), key=lambda item: item.stat().st_mtime, reverse=True)
    parts = [read_parquet_dataset(file) for file in files]
    parts = [part for part in parts if not part.empty]
    return pd.concat(parts, ignore_index=True) if parts else _empty()


def latest_parquet(path: Path) -> pd.DataFrame:
    """Read the newest parquet file under ``path`` or the dataset at ``path``."""

    if not path.exists():
        return _empty()
    if path.is_file() or path.name.endswith(".parquet"):
        return read_parquet_dataset(path)
    files = sorted(path.rglob("*.parquet"), key=lambda item: item.stat().st_mtime, reverse=True)
    return read_parquet_dataset(files[0]) if files else _empty()


def load_model_inventory(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Return available model files with version/station/target metadata."""

    if not models_dir.exists():
        return _empty()
    rows: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("model_*_*.pkl")):
        stem = path.stem.removeprefix("model_")
        station, _, target = stem.partition("_")
        rows.append(
            {
                "version": path.parent.name,
                "station": station,
                "target": target,
                "path": str(path),
                "modified_at": pd.Timestamp(path.stat().st_mtime, unit="s"),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
            }
        )
    return pd.DataFrame(rows)


def load_model_metrics(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Read metrics files emitted by model training."""

    if not models_dir.exists():
        return _empty()
    rows: list[dict[str, Any]] = []
    for path in sorted(models_dir.rglob("metrics*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if all(isinstance(value, dict) for value in data.values()):
            for key, metrics in data.items():
                station, _, target = key.partition("_")
                rows.append(
                    {"version": path.parent.name, "station": station, "target": target, **metrics}
                )
        else:
            stem = path.stem.removeprefix("metrics_")
            station, _, target = stem.partition("_")
            rows.append({"version": path.parent.name, "station": station, "target": target, **data})
    return pd.DataFrame(rows)


def load_feature_importance(models_dir: Path = MODELS_DIR) -> pd.DataFrame:
    """Read feature-importance CSVs emitted by model training."""

    if not models_dir.exists():
        return _empty()
    parts: list[pd.DataFrame] = []
    for path in sorted(models_dir.rglob("feature_importance.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        frame["version"] = path.parent.name
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else _empty()


def load_dashboard_artifacts(
    data_dir: Path = DATA_DIR,
    models_dir: Path = MODELS_DIR,
) -> DashboardArtifacts:
    """Load all local dashboard artifacts, returning empty frames when absent."""

    return DashboardArtifacts(
        labels=read_parquet_dataset(data_dir / "labels" / "labels.parquet"),
        features_live=read_parquet_collection(data_dir / "features" / "live"),
        predictions=read_parquet_collection(data_dir / "predictions"),
        bucketed_predictions=read_parquet_collection(data_dir / "predictions" / "bucketed"),
        markets=read_parquet_collection(data_dir / "markets"),
        signals=read_parquet_collection(data_dir / "signals"),
        runs=read_parquet_collection(data_dir / "runs"),
        model_inventory=load_model_inventory(models_dir),
        metrics=load_model_metrics(models_dir),
        feature_importance=load_feature_importance(models_dir),
    )


def station_options() -> list[str]:
    return [station.icao for station in load_stations()]


def summarize_labels(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return _empty()
    frame = labels.copy()
    frame["local_date"] = pd.to_datetime(frame["local_date"])
    grouped = frame.groupby("station", dropna=False)
    return (
        grouped.agg(
            days=("local_date", "count"),
            start=("local_date", "min"),
            end=("local_date", "max"),
            avg_tmax_f=("tmax_f", "mean"),
            avg_tmin_f=("tmin_f", "mean"),
            wet_days=("precip_in", lambda values: int((values > 0.01).sum())),
        )
        .reset_index()
        .sort_values("station")
    )


def filter_frame(
    frame: pd.DataFrame,
    *,
    stations: list[str] | None = None,
    venues: list[str] | None = None,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    if stations and "station" in filtered.columns:
        filtered = filtered[filtered["station"].isin(stations)]
    if venues and "venue" in filtered.columns:
        filtered = filtered[filtered["venue"].isin(venues)]
    if targets and "target" in filtered.columns:
        filtered = filtered[filtered["target"].isin(targets)]
    return filtered
