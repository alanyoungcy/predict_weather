"""Train all station-target XGBoost models."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import pandas as pd

from wt.config import load_stations
from wt.models.registry import save_trained_model
from wt.models.train import TARGET_MAP, train_one
from wt.utils.paths import DATA_DIR, MODELS_DIR


def _parse_csv(value: str, valid: list[str]) -> list[str]:
    if value.lower() == "all":
        return valid
    requested = [part.strip().upper() for part in value.split(",") if part.strip()]
    return requested


def _date_splits(
    max_date: pd.Timestamp,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]]:
    test_end = max_date.normalize()
    test_start = test_end - pd.DateOffset(years=1)
    val_start = test_start - pd.DateOffset(years=1)
    return (val_start, test_start - pd.Timedelta(days=1)), (test_start, test_end)


@click.command()
@click.option(
    "--features",
    "features_path",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "features" / "features_train.parquet",
)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(path_type=Path),
    default=DATA_DIR / "labels" / "labels.parquet",
)
@click.option("--stations", "stations_arg", default="all", show_default=True)
@click.option("--targets", "targets_arg", default="all", show_default=True)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    default=MODELS_DIR / "current",
)
@click.option(
    "--promote-current",
    is_flag=True,
    help="Replace models/current symlink with the output directory.",
)
def main(
    features_path: Path,
    labels_path: Path,
    stations_arg: str,
    targets_arg: str,
    output_dir: Path,
    promote_current: bool,
) -> None:
    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    station_codes = [station.icao for station in load_stations()]
    requested_stations = _parse_csv(stations_arg, station_codes)
    requested_targets = [target.lower() for target in _parse_csv(targets_arg, list(TARGET_MAP))]

    max_date = pd.to_datetime(features["target_local_date"]).max()
    val_split, test_split = _date_splits(max_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict[str, float]] = {}
    feature_rows: list[dict[str, object]] = []
    for station in requested_stations:
        for target in requested_targets:
            model = train_one(features, labels, station, target, val_split, test_split)
            save_trained_model(model, output_dir)
            key = f"{station}_{target}"
            all_metrics[key] = model.metrics
            feature_rows.extend(
                {"station": station, "target": target, "feature": name, "importance": importance}
                for name, importance in model.feature_importance.items()
            )
            click.echo(f"trained {key}: test_mae={model.metrics.get('test_mae'):.3f}")

    (output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    pd.DataFrame(feature_rows).to_csv(output_dir / "feature_importance.csv", index=False)

    if promote_current:
        current = MODELS_DIR / "current"
        if output_dir.resolve() == current.resolve():
            click.echo("output is already models/current; no symlink promotion needed")
            return
        if current.exists() or current.is_symlink():
            if current.is_symlink() or current.is_file():
                current.unlink()
            else:
                shutil.rmtree(current)
        current.symlink_to(output_dir.resolve(), target_is_directory=True)


if __name__ == "__main__":
    main()
