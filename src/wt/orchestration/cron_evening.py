"""Evening live prediction orchestration."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import click
import pandas as pd

from wt.config import load_settings, load_stations
from wt.features.build_live import build_live_features
from wt.io.kalshi import KalshiClient
from wt.io.polymarket import PolymarketClient
from wt.markets.buckets import align_predictions_to_buckets
from wt.markets.signals import generate_signals
from wt.models.predict import NormalDistribution, QuantileDistribution, predict_distribution
from wt.models.registry import load_trained_model
from wt.ops.monitoring import build_run_summary
from wt.utils.paths import MODELS_DIR


TARGETS = ("tmax", "tmin", "apcp24")


def _run_id(now: datetime | None = None) -> str:
    return (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%SZ")


def _target_date(value: str | None) -> date:
    if value:
        return date.fromisoformat(value)
    return datetime.now(ZoneInfo("America/New_York")).date() + timedelta(days=1)


def _model_path(model_root: Path, station: str, target: str) -> Path:
    current = model_root / "current"
    root = current if current.exists() else model_root
    return root / f"model_{station}_{target}.pkl"


def _prediction_record(
    *,
    station: str,
    target_local_date: date,
    init_run_utc: Any,
    target: str,
    dist: Any,
    model_version: str,
) -> dict[str, Any]:
    quantiles = getattr(dist, "quantiles", {})
    if isinstance(dist, NormalDistribution):
        q = {
            0.05: dist.point - 1.6448536269514722 * dist.sigma,
            0.25: dist.point - 0.6744897501960817 * dist.sigma,
            0.5: dist.point,
            0.75: dist.point + 0.6744897501960817 * dist.sigma,
            0.95: dist.point + 1.6448536269514722 * dist.sigma,
        }
        family = "normal"
        sigma = dist.sigma
    elif isinstance(dist, QuantileDistribution):
        q = quantiles
        family = "quantile"
        sigma = dist.sigma
    else:
        q = quantiles
        family = "unknown"
        sigma = getattr(dist, "sigma", None)
    return {
        "station": station,
        "target_local_date": pd.Timestamp(target_local_date),
        "init_run_utc": pd.Timestamp(init_run_utc),
        "target": target,
        "point_pred": float(dist.point),
        "q05": float(q.get(0.05, q.get(0.1, dist.point))),
        "q25": float(q.get(0.25, dist.point)),
        "q50": float(q.get(0.5, dist.point)),
        "q75": float(q.get(0.75, dist.point)),
        "q95": float(q.get(0.95, q.get(0.9, dist.point))),
        "sigma": float(sigma or 0.0),
        "dist_family": family,
        "model_version": model_version,
        "generated_at": datetime.now(tz=UTC),
    }


def run_live_prediction(
    *,
    target_local_date: date,
    station_filter: str | None = None,
    run_slot: str = "evening",
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build live features and predictions, returning both dataframes."""

    settings = load_settings()
    stations = load_stations()
    if station_filter:
        selected = {part.strip().upper() for part in station_filter.split(",") if part.strip()}
        stations = [
            station
            for station in stations
            if station.icao in selected or station.kalshi_city.upper() in selected
        ]
    if not stations:
        raise ValueError("No stations selected")

    run_id = _run_id()
    features = build_live_features(stations, target_local_date, run_slot=run_slot)
    features_path = settings.data_dir / "features" / "live" / f"run_id={run_id}.parquet"
    if not dry_run:
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_path, index=False)

    records: list[dict[str, Any]] = []
    for row in features.to_dict(orient="records"):
        station = str(row["station"])
        for target in TARGETS:
            path = _model_path(settings.models_dir or MODELS_DIR, station, target)
            if not path.exists():
                click.echo(f"warning: missing model {path}", err=True)
                continue
            try:
                model = load_trained_model(path)
                dist = predict_distribution(model, row)
            except Exception as exc:
                click.echo(f"warning: prediction failed for {station} {target}: {exc}", err=True)
                continue
            records.append(
                _prediction_record(
                    station=station,
                    target_local_date=target_local_date,
                    init_run_utc=row["init_run_utc"],
                    target=target,
                    dist=dist,
                    model_version=path.parent.name,
                )
            )

    predictions = pd.DataFrame(records)
    predictions_path = settings.data_dir / "predictions" / f"run_id={run_id}.parquet"
    if not dry_run and not predictions.empty:
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(predictions_path, index=False)
    return features, predictions


def fetch_market_snapshot(
    *,
    target_local_date: date,
    station_filter: str | None = None,
    targets: list[str] | None = None,
    include_kalshi: bool = True,
    include_polymarket: bool = True,
) -> pd.DataFrame:
    """Fetch read-only market buckets for all configured venues."""

    stations = load_stations()
    if station_filter:
        selected = {part.strip().upper() for part in station_filter.split(",") if part.strip()}
        stations = [
            station
            for station in stations
            if station.icao in selected or station.kalshi_city.upper() in selected
        ]

    frames: list[pd.DataFrame] = []
    if include_kalshi:
        try:
            with KalshiClient() as client:
                frames.append(
                    client.fetch_relevant_markets(
                        stations,
                        target_local_date,
                        targets=targets,
                    )
                )
        except Exception as exc:
            click.echo(f"warning: Kalshi market fetch failed: {exc}", err=True)
    if include_polymarket:
        try:
            with PolymarketClient() as client:
                frames.append(
                    client.fetch_relevant_markets(
                        stations,
                        target_local_date,
                        targets=targets,
                    )
                )
        except Exception as exc:
            click.echo(f"warning: Polymarket market fetch failed: {exc}", err=True)

    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_signal_generation(
    predictions: pd.DataFrame,
    *,
    target_local_date: date,
    station_filter: str | None = None,
    min_edge_bps: int = 300,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch market buckets, align probabilities, and generate +EV signals."""

    settings = load_settings()
    run_id = _run_id()
    markets = fetch_market_snapshot(
        target_local_date=target_local_date,
        station_filter=station_filter,
    )
    if markets.empty or predictions.empty:
        return markets, pd.DataFrame(), generate_signals(pd.DataFrame(), pd.DataFrame())

    bucket_probs = align_predictions_to_buckets(predictions, markets)
    signals = generate_signals(bucket_probs, markets, min_edge_bps=min_edge_bps)

    if not dry_run:
        market_path = settings.data_dir / "markets" / f"run_id={run_id}.parquet"
        probs_path = settings.data_dir / "predictions" / "bucketed" / f"run_id={run_id}.parquet"
        signals_path = settings.data_dir / "signals" / f"run_id={run_id}.parquet"
        for path, frame in (
            (market_path, markets),
            (probs_path, bucket_probs),
            (signals_path, signals),
        ):
            if not frame.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
    return markets, bucket_probs, signals


@click.command()
@click.option(
    "--dry-run",
    is_flag=True,
    help="Build in memory and print counts without writing parquet.",
)
@click.option("--station", "station_filter", help="ICAO or Kalshi city code, comma-separated.")
@click.option(
    "--target-date",
    help="Target local date YYYY-MM-DD. Defaults to tomorrow in America/New_York.",
)
@click.option(
    "--run-slot",
    default="evening",
    type=click.Choice(["morning", "evening"]),
    show_default=True,
)
@click.option("--with-markets", is_flag=True, help="Fetch venues and generate signal rows.")
@click.option("--min-edge-bps", default=300, show_default=True, type=int)
def main(
    dry_run: bool,
    station_filter: str | None,
    target_date: str | None,
    run_slot: str,
    with_markets: bool,
    min_edge_bps: int,
) -> None:
    target = _target_date(target_date)
    features, predictions = run_live_prediction(
        target_local_date=target,
        station_filter=station_filter,
        run_slot=run_slot,
        dry_run=dry_run,
    )
    click.echo(
        f"run_slot={run_slot} target_date={target.isoformat()} "
        f"features={len(features)} predictions={len(predictions)}"
    )
    signals = pd.DataFrame()
    if with_markets:
        markets, bucket_probs, signals = run_signal_generation(
            predictions,
            target_local_date=target,
            station_filter=station_filter,
            min_edge_bps=min_edge_bps,
            dry_run=dry_run,
        )
        click.echo(
            f"markets={len(markets)} bucket_probs={len(bucket_probs)} signals={len(signals)}"
        )
        if dry_run and not signals.empty:
            click.echo(signals.head(20).to_string(index=False))
    if not dry_run:
        settings = load_settings()
        run_id = _run_id()
        summary = build_run_summary(
            run_id=run_id,
            features=features,
            predictions=predictions,
            signals=signals,
        )
        summary_path = settings.data_dir / "runs" / f"run_id={run_id}.parquet"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(summary_path, index=False)
    if dry_run and not predictions.empty:
        click.echo(predictions.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
