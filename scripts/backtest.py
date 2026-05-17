"""Backtest report generation for available prediction artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from wt.utils.paths import DATA_DIR


TARGET_COLUMNS = {"tmax": "tmax_f", "tmin": "tmin_f", "apcp24": "precip_in"}


def _metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["local_date"] = pd.to_datetime(labels["local_date"]).dt.date
    predictions = predictions.copy()
    predictions["target_local_date"] = pd.to_datetime(predictions["target_local_date"]).dt.date
    rows: list[dict[str, object]] = []
    for target, label_col in TARGET_COLUMNS.items():
        subset = predictions[predictions["target"] == target]
        if subset.empty:
            continue
        joined = subset.merge(
            labels[["station", "local_date", label_col]],
            left_on=["station", "target_local_date"],
            right_on=["station", "local_date"],
            how="inner",
        )
        if joined.empty:
            continue
        error = joined["point_pred"].astype("float64") - joined[label_col].astype("float64")
        rows.append(
            {
                "target": target,
                "rows": len(joined),
                "mae": float(np.abs(error).mean()),
                "rmse": float(np.sqrt(np.square(error).mean())),
                "bias": float(error.mean()),
                "avg_sigma": (
                    float(joined["sigma"].astype("float64").mean())
                    if "sigma" in joined
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


@click.command()
@click.option(
    "--predictions",
    "predictions_path",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "predictions" / "latest.parquet",
)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "labels" / "labels.parquet",
)
@click.option("--start", required=True)
@click.option("--end", required=True)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), required=True)
def main(predictions_path: Path, labels_path: Path, start: str, end: str, output_dir: Path) -> None:
    predictions = pd.read_parquet(predictions_path)
    labels = pd.read_parquet(labels_path)
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()
    predictions["target_local_date"] = pd.to_datetime(predictions["target_local_date"]).dt.date
    predictions = predictions[predictions["target_local_date"].between(start_date, end_date)]

    output_dir.mkdir(parents=True, exist_ok=True)
    point_metrics = _metrics(predictions, labels)
    point_metrics.to_csv(output_dir / "point_metrics.csv", index=False)
    summary = {
        "start": start,
        "end": end,
        "prediction_rows": int(len(predictions)),
        "metrics_rows": int(len(point_metrics)),
        "note": (
            "This report currently includes point metrics. Historical market replay "
            "and reliability plots are next."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    click.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
