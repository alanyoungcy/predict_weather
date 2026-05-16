"""Bootstrap historical ground-truth labels from CF6 and GHCN-D."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path
import sys

import click
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from wt.config import Station, load_settings, load_stations
from wt.io.cf6 import fetch_cf6_year
from wt.io.ghcnd import fetch_ghcnd_station

LABEL_COLUMNS = ["station", "local_date", "tmax_f", "tmin_f", "precip_in", "source", "settled_at"]


def _month_range(start: date, end: date) -> Iterable[tuple[int, int]]:
    current = date(start.year, start.month, 1)
    limit = date(end.year, end.month, 1)
    while current <= limit:
        yield current.year, current.month
        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


def _year_range(start: date, end: date) -> Iterable[int]:
    for year in range(start.year, end.year + 1):
        yield year


def build_station_labels(station: Station, start: date, end: date) -> pd.DataFrame:
    cf6_parts: list[pd.DataFrame] = []
    click.echo(f"[{station.icao}] fetching CF6 + GHCND for {start} -> {end}", err=True)
    for year in _year_range(start, end):
        yearly = fetch_cf6_year(station.icao, year=year)
        if not yearly.empty:
            cf6_parts.append(yearly)

    cf6_df = (
        pd.concat(cf6_parts, ignore_index=True)
        if cf6_parts
        else pd.DataFrame(columns=["local_date", "tmax_f", "tmin_f", "precip_in", "snow_in", "source", "settled_at"])
    )
    cf6_df = cf6_df[(cf6_df["local_date"] >= start) & (cf6_df["local_date"] <= end)].copy()

    ghcnd_df = fetch_ghcnd_station(station.ghcnd_id, start, end).rename(columns={"date": "local_date"})
    merged = ghcnd_df.merge(
        cf6_df[["local_date", "tmax_f", "tmin_f", "precip_in", "settled_at"]],
        on="local_date",
        how="outer",
        suffixes=("_ghcnd", "_cf6"),
    )
    merged.sort_values("local_date", inplace=True)

    labels = pd.DataFrame(
        {
            "station": station.icao,
            "local_date": merged["local_date"],
            "tmax_f": merged["tmax_f_cf6"].combine_first(merged["tmax_f_ghcnd"]),
            "tmin_f": merged["tmin_f_cf6"].combine_first(merged["tmin_f_ghcnd"]),
            "precip_in": merged["precip_in"].combine_first(merged["prcp_in"]),
            "source": merged["tmax_f_cf6"].notna().map({True: "CF6", False: "GHCND"}),
            "settled_at": merged["settled_at"],
        }
    )

    labels = labels[(labels["local_date"] >= start) & (labels["local_date"] <= end)].copy()
    labels.reset_index(drop=True, inplace=True)
    _validate_labels(station, labels, start, end)
    source_counts = labels["source"].value_counts(dropna=False).to_dict()
    click.echo(f"[{station.icao}] completed with source counts {source_counts}", err=True)
    return labels[LABEL_COLUMNS]


def _validate_labels(station: Station, labels: pd.DataFrame, start: date, end: date) -> None:
    expected_dates = pd.date_range(start, end, freq="D").date
    actual_dates = list(labels["local_date"])
    if actual_dates != list(expected_dates):
        missing = sorted(set(expected_dates) - set(actual_dates))
        raise ValueError(f"{station.icao}: missing label dates, first few={missing[:5]}")

    if (labels["tmin_f"] > labels["tmax_f"]).any():
        bad = labels.loc[labels["tmin_f"] > labels["tmax_f"], ["local_date", "tmin_f", "tmax_f"]]
        raise ValueError(f"{station.icao}: invalid min/max rows found: {bad.head().to_dict(orient='records')}")

    if (labels["precip_in"] < 0).any():
        bad = labels.loc[labels["precip_in"] < 0, ["local_date", "precip_in"]]
        raise ValueError(f"{station.icao}: negative precipitation rows found: {bad.head().to_dict(orient='records')}")


@click.command()
@click.option("--start", default="2015-01-01", show_default=True, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--end", default=None, type=click.DateTime(formats=["%Y-%m-%d"]))
@click.option("--station", "station_filter", default="all", show_default=True)
@click.option("--output", default=None, type=click.Path(path_type=Path))
def main(start: pd.Timestamp, end: pd.Timestamp | None, station_filter: str, output: Path | None) -> None:
    settings = load_settings()
    end_date = end.date() if end is not None else (date.today() - timedelta(days=1))
    start_date = start.date()

    stations = load_stations()
    if station_filter != "all":
        wanted = {part.strip().upper() for part in station_filter.split(",") if part.strip()}
        stations = [station for station in stations if station.icao in wanted]

    if not stations:
        raise click.ClickException("No stations selected")

    label_parts = [build_station_labels(station, start_date, end_date) for station in stations]
    labels = pd.concat(label_parts, ignore_index=True)
    labels["local_date"] = pd.to_datetime(labels["local_date"])

    destination = output or (settings.data_dir / "labels" / "labels.parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(destination, index=False, partition_cols=["station"])

    coverage = (
        labels.groupby(["station", "source"]).size().rename("days").reset_index()
        .merge(labels.groupby("station").size().rename("total_days").reset_index(), on="station")
    )
    coverage["coverage_pct"] = (coverage["days"] / coverage["total_days"]) * 100.0
    print(coverage.to_string(index=False))
    print(f"\nWrote {len(labels):,} labels to {destination}")


if __name__ == "__main__":
    main()
