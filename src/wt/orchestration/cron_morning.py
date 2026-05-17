"""Morning live prediction orchestration."""

from __future__ import annotations

import click
import pandas as pd

from wt.orchestration.cron_evening import (
    _target_date,
    _run_id,
    run_live_prediction,
    run_signal_generation,
)
from wt.config import load_settings
from wt.ops.monitoring import build_run_summary


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
@click.option("--with-markets", is_flag=True, help="Fetch venues and generate signal rows.")
@click.option("--min-edge-bps", default=300, show_default=True, type=int)
def main(
    dry_run: bool,
    station_filter: str | None,
    target_date: str | None,
    with_markets: bool,
    min_edge_bps: int,
) -> None:
    target = _target_date(target_date)
    features, predictions = run_live_prediction(
        target_local_date=target,
        station_filter=station_filter,
        run_slot="morning",
        dry_run=dry_run,
    )
    click.echo(
        f"run_slot=morning target_date={target.isoformat()} "
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
